# play_pt_policy.py 
# MuJoCo position-servo policy playback with IsaacLab env.yaml parsing

from __future__ import annotations
import argparse, sys, time, re
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Any

import numpy as np
import torch
import mujoco
from mujoco import viewer
import yaml

# keycodes for teleop
GLFW_SPACE = 32
GLFW_S = 83
GLFW_ESCAPE = 256
GLFW_UP = 265
GLFW_DOWN = 264
GLFW_LEFT = 263
GLFW_RIGHT = 262
GLFW_Q = 81
GLFW_E = 69
GLFW_1 = 49
GLFW_2 = 50
GLFW_3 = 51
GLFW_4 = 52
GLFW_5 = 53
GLFW_6 = 54
GLFW_7 = 55
GLFW_8 = 56
GLFW_9 = 57
GLFW_0 = 48
GLFW_V = 86

@dataclass
class PolicyBundle:
    module: torch.nn.Module
    obs_mean: Optional[torch.Tensor] = None
    obs_std: Optional[torch.Tensor] = None

@dataclass
class HistoryCfg:
    base_ang_vel: int = 10
    base_ori: int = 10
    joint_pos: int = 10
    joint_vel: int = 10
    actions: int = 10
    velocity_commands: int = 10
    ee_pose_command: int = 10
    projected_gravity: int = 10

# ---------------- IsaacLab env.yaml parsing utilities ----------------

def _yaml_load_tolerant(path: str) -> dict:
    # 1. try safe_load
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        pass
    # 2. try unsafe_load (PyYAML >= 5.1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.unsafe_load(f) or {}
    except Exception:
        pass
    # 3. sanitize and safe_load: drop lines with python/object/apply slices
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        # remove blocks like:
        # joint_ids: !!python/object/apply:builtins.slice
        # null
        pat = re.compile(r"!!python/object/apply:builtins\.slice[\s\S]*?(?=(\n\S)|\Z)", re.MULTILINE)
        txt2 = pat.sub("", txt)
        return yaml.safe_load(txt2) or {}
    except Exception:
        return {}
    
def _parse_warmup_leg_targets(path: Optional[str]) -> Dict[str, float]:
    """Return dict[joint_name] -> target_angle (rad) from env.yaml warmup.leg_joint_targets."""
    if not path:
        return {}
    cfg = _yaml_load_tolerant(path)
    warm = (cfg.get("warmup", {}) or {}).get("leg_joint_targets", {}) or {}
    out: Dict[str, float] = {}
    for k, v in warm.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            pass
    return out

def _dfs_find_actions(node: Any) -> Optional[dict]:
    if isinstance(node, dict):
        if "joint_pos" in node or "arm_pose" in node:
            return node
        for v in node.values():
            res = _dfs_find_actions(v)
            if res is not None: return res
    elif isinstance(node, (list, tuple)):
        for v in node:
            res = _dfs_find_actions(v)
            if res is not None: return res
    return None

def _parse_history_cfg(path: Optional[str]) -> HistoryCfg:
    try:
        if not path: return HistoryCfg()
        cfg = _yaml_load_tolerant(path)
        pol = (cfg.get("observations", {}) or {}).get("policy", {}) or {}
        def _hist(name: str, default: int) -> int:
            h = (pol.get(name, {}) or {}).get("history_length")
            return int(h) if isinstance(h, int) and h > 0 else default
        return HistoryCfg(
            base_ang_vel=_hist("base_ang_vel", 10),
            base_ori=_hist("base_ori", 10),
            joint_pos=_hist("joint_pos", 10),
            joint_vel=_hist("joint_vel", 10),
            actions=_hist("actions", 10),
            velocity_commands=_hist("velocity_commands", 10),
            ee_pose_command=_hist("ee_pose_command", 10),
            projected_gravity=_hist("projected_gravity", 10),
        )

    except Exception:
        return HistoryCfg()

def _parse_action_joint_names(path: str) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    try:
        cfg = _yaml_load_tolerant(path)
        actions_root = (cfg.get("actions", {}) or {})
        actions = actions_root if actions_root else _dfs_find_actions(cfg)
        if not isinstance(actions, dict): return None
        leg_cfg = actions.get("joint_pos", {}) or {}
        arm_cfg = actions.get("arm_pose", {}) or {}
        leg_names = tuple(leg_cfg.get("joint_names", []) or [])
        arm_names = tuple(arm_cfg.get("joint_names", []) or [])
        if len(leg_names) + len(arm_names) == 0: return None
        return leg_names, arm_names
    except Exception:
        return None

def _parse_action_scales(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        cfg = _yaml_load_tolerant(path)
        actions_root = (cfg.get("actions", {}) or {})
        actions = actions_root if actions_root else _dfs_find_actions(cfg)
        if not isinstance(actions, dict): return out
        for grp_key in ("joint_pos", "arm_pose"):
            g = actions.get(grp_key, {}) or {}
            names = g.get("joint_names", []) or []
            scales = g.get("scale")
            if isinstance(scales, dict):
                for jn, sc in scales.items():
                    try: out[str(jn)] = float(sc)
                    except Exception: pass
            elif isinstance(scales, (int, float)):
                for jn in names: out[str(jn)] = float(scales)
    except Exception:
        pass
    return out

def _parse_control_rate(path: str, default_hz: float) -> float:
    try:
        cfg = _yaml_load_tolerant(path)
        dt = float(((cfg.get("sim", {}) or {}).get("dt")) or 0.0)
        decimation = int(cfg.get("decimation", 0) or 0)
        if dt > 0 and decimation > 0:
            return 1.0 / (dt * decimation)
    except Exception:
        pass
    return default_hz

# ---------------- Actuators / Indices ----------------

def build_actuator_maps(model: mujoco.MjModel, params_yaml: Optional[str]) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], np.ndarray]:
    joint_to_act: Dict[str, int] = {}
    for a in range(model.nu):
        j_id = model.actuator_trnid[a, 0]
        if j_id < 0: continue
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        if jn and jn not in joint_to_act: joint_to_act[jn] = a
    selected_act_ids: list[int] = []
    selected_names: list[str] = []

    parsed = _parse_action_joint_names(params_yaml) if params_yaml else None
    if params_yaml and parsed is None:
        print(f"[WARN] Could not parse joint order from params: {params_yaml}. Using heuristic fallback (legs first, then arm).")

    if parsed:
        leg_names, arm_names = parsed
        for jname in list(leg_names) + list(arm_names):
            a_id = joint_to_act.get(jname)
            if a_id is None:
                print(f"[WARN] Training joint '{jname}' not found in MuJoCo model; skipping.")
                continue
            selected_act_ids.append(a_id)
            aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_id)
            selected_names.append(aname if aname else f"actuator_{a_id}")
    else:
        # legs, arms, ignore gripper finger links
        leg_ids, arm_ids = [], []
        for a in range(model.nu):
            j_id = model.actuator_trnid[a, 0]
            jn = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or "").lower()
            an = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").lower()
            nm = jn + " " + an
            if "finger" in nm: continue
            if any(k in nm for k in ("fr_", "fl_", "rr_", "rl_")): leg_ids.append(a)
            elif "widow" in nm: arm_ids.append(a)
            else: leg_ids.append(a)
        selected_act_ids = leg_ids + arm_ids
        for a in selected_act_ids:
            an = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            selected_names.append(an if an else f"actuator_{a}")

    act_ids_arr = np.array(selected_act_ids, dtype=np.int32)
    qpos_indices = np.zeros(len(act_ids_arr), dtype=np.int32)
    qvel_indices = np.zeros(len(act_ids_arr), dtype=np.int32)
    for i, a in enumerate(act_ids_arr):
        j_id = model.actuator_trnid[a, 0]
        if j_id < 0: raise RuntimeError(f"Actuator {a} is not attached to a joint")
        qpos_indices[i] = model.jnt_qposadr[j_id]
        qvel_indices[i] = model.jnt_dofadr[j_id]
    return qpos_indices, qvel_indices, tuple(selected_names), act_ids_arr

def build_joint_limits(model: mujoco.MjModel, actuator_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(len(actuator_ids), -np.inf, dtype=np.float64)
    hi = np.full(len(actuator_ids),  np.inf, dtype=np.float64)
    for i, aid in enumerate(actuator_ids):
        j_id = model.actuator_trnid[int(aid), 0]
        r = model.jnt_range[j_id] if hasattr(model, "jnt_range") else None
        if r is not None and r[0] < r[1]: lo[i], hi[i] = float(r[0]), float(r[1])
    return lo, hi

def _default_scale(name: str) -> float:
    return 0.5 if ("widow" in (name or "").lower()) else 0.25

# ---------------- Observations (70×10; newest->oldest) ----------------

class ObsHistory:
    def __init__(self, model, qpos_idx, qvel_idx, actuator_ids, hist_cfg, cmd_vel_3, cmd_ee_vec):
        self.model = model; self.qpos_idx = qpos_idx; self.qvel_idx = qvel_idx
        self.actuator_ids = actuator_ids; self.hist = hist_cfg; self.N = len(actuator_ids)
        self.qpos_nominal: Optional[np.ndarray] = None
        self.sens_id_quat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "base_site_quat")
        self.sens_id_ang  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "base_site_angvel")
        self.sens_adr_quat = model.sensor_adr[self.sens_id_quat] if self.sens_id_quat >= 0 else -1
        self.sens_dim_quat = model.sensor_dim[self.sens_id_quat] if self.sens_id_quat >= 0 else 0
        self.sens_adr_ang  = model.sensor_adr[self.sens_id_ang] if self.sens_id_ang >= 0 else -1
        self.sens_dim_ang  = model.sensor_dim[self.sens_id_ang] if self.sens_id_ang >= 0 else 0
        # default base body id
        base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_bid < 0:
            try:
                free_jids = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
                if free_jids.size > 0 and hasattr(model, "jnt_bodyid"):
                    base_bid = int(model.jnt_bodyid[free_jids[0]])
                else:
                    base_bid = 0
            except Exception:
                base_bid = 0
        self.base_body_id = int(base_bid)
        self.cmd_vel_3  = cmd_vel_3.astype(np.float32)
        self.cmd_ee_vec = cmd_ee_vec.astype(np.float32)
        H = hist_cfg.base_ang_vel
        self.buf_base_ori = np.zeros((H, 4), dtype=np.float32)
        self.buf_base_ang = np.zeros((H, 3), dtype=np.float32)
        self.buf_qpos     = np.zeros((H, self.N), dtype=np.float32)
        self.buf_qvel     = np.zeros((H, self.N), dtype=np.float32)
        self.buf_actions  = np.zeros((H, self.N), dtype=np.float32)
        self.buf_cmd_vel  = np.tile(self.cmd_vel_3[None,:3], (H,1)).astype(np.float32)
        self.buf_cmd_ee   = np.tile(self.cmd_ee_vec[None,:], (H,1)).astype(np.float32)
        self.buf_proj_g   = np.zeros((H, 3), dtype=np.float32)

    def _ensure_nominal(self, data):
        if self.qpos_nominal is None:
            self.qpos_nominal = data.qpos[self.qpos_idx].copy()

    def _read_base_ang(self, data):
        if self.sens_adr_ang >= 0 and self.sens_dim_ang >= 3:
            return data.sensordata[self.sens_adr_ang:self.sens_adr_ang+3].astype(np.float32)
        # fallback read angvel from base cvel
        try:
            if getattr(data.cvel, 'ndim', 1) == 2:
                return data.cvel[self.base_body_id, 0:3].astype(np.float32)
            return data.cvel[6*self.base_body_id:6*self.base_body_id+3].astype(np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _read_base_quat(self, data):
        if self.sens_adr_quat >= 0 and self.sens_dim_quat >= 4:
            return data.sensordata[self.sens_adr_quat:self.sens_adr_quat+4].astype(np.float32)
        # fallback get world orientation of base body
        try:
            return data.xquat[self.base_body_id].astype(np.float32)
        except Exception:
            return np.array([1,0,0,0], dtype=np.float32)

    def _project_gravity(self, data):
        g = np.array(self.model.opt.gravity, dtype=np.float64)
        n = np.linalg.norm(g); g = g if n < 1e-9 else g / n
        q = self._read_base_quat(data).astype(np.float64)
        q_inv = q.copy(); q_inv[1:] *= -1.0
        out64 = np.zeros(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(out64, g, q_inv)
        return out64.astype(np.float32)

    def update(self, data, last_action_norm):
        self._ensure_nominal(data)
        base_ang  = self._read_base_ang(data)
        base_quat = self._read_base_quat(data)
        qpos_rel  = (data.qpos[self.qpos_idx].astype(np.float32) - self.qpos_nominal.astype(np.float32))
        qvel      = data.qvel[self.qvel_idx].astype(np.float32)
        proj_g    = self._project_gravity(data)
        for buf, val in ((self.buf_base_ori, base_quat),
                         (self.buf_base_ang, base_ang),
                         (self.buf_qpos, qpos_rel),
                         (self.buf_qvel, qvel),
                         (self.buf_actions, last_action_norm.astype(np.float32)),
                         (self.buf_cmd_vel, self.cmd_vel_3[:3]),
                         (self.buf_cmd_ee, self.cmd_ee_vec),
                         (self.buf_proj_g, proj_g)):
            np.roll(buf, -1, axis=0); buf[-1] = val
        H = self.hist.base_ang_vel
        step_chunks = []
        for t in range(H):
            ti = H - 1 - t
            per_step = [self.buf_base_ori[ti], self.buf_base_ang[ti],
                        self.buf_qpos[ti], self.buf_qvel[ti], self.buf_actions[ti],
                        self.buf_cmd_vel[ti], self.buf_cmd_ee[ti], self.buf_proj_g[ti]]
            step_chunks.append(np.concatenate([p.ravel() for p in per_step]).astype(np.float32))
        return np.concatenate(step_chunks, dtype=np.float32)

# ---------------- Policy IO ----------------

def try_load_torchscript(path: str, device: torch.device) -> Optional[torch.nn.Module]:
    try:
        m = torch.jit.load(path, map_location=device); m.eval(); return m
    except Exception as e:
        print(f"[INFO] TorchScript load failed: {e}"); return None

def extract_policy_from_dict(obj: dict, device: torch.device) -> PolicyBundle:
    m = None
    for k in ("policy","actor","module","model"):
        cand = obj.get(k)
        if isinstance(cand, torch.nn.Module): m = cand; break
    if m is None:
        sd = obj.get("state_dict")
        if isinstance(sd, dict): raise RuntimeError("Checkpoint is a state_dict only. Export TorchScript first.")
        raise RuntimeError("No torch.nn.Module in checkpoint dict")
    m.to(device).eval()
    om, os = obj.get("obs_mean"), obj.get("obs_std")
    if om is not None: om = torch.as_tensor(om, dtype=torch.float32, device=device)
    if os is not None: os = torch.as_tensor(os, dtype=torch.float32, device=device)
    return PolicyBundle(module=m, obs_mean=om, obs_std=os)

def load_policy(path: str, device_str: str) -> PolicyBundle:
    device = torch.device(device_str)
    m = try_load_torchscript(path, device)
    if m is not None: return PolicyBundle(module=m)
    obj = torch.load(path, map_location=device)
    if isinstance(obj, torch.jit.ScriptModule): obj.eval(); return PolicyBundle(module=obj)
    if isinstance(obj, torch.nn.Module): obj.to(device).eval(); return PolicyBundle(module=obj)
    if isinstance(obj, dict): return extract_policy_from_dict(obj, device)
    import os
    sib = os.path.join(os.path.dirname(path), "exported", "policy.pt")
    if os.path.isfile(sib):
        m = try_load_torchscript(sib, device)
        if m is not None:
            print(f"[INFO] Falling back to exported TorchScript at: {sib}")
            return PolicyBundle(module=m)
    raise RuntimeError(f"Unsupported checkpoint format: {type(obj)!r}")

def normalize_obs(bundle: PolicyBundle, obs: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(obs, dtype=torch.float32, device=next(bundle.module.parameters()).device)
    if bundle.obs_mean is not None and bundle.obs_std is not None and bundle.obs_mean.shape == t.shape and bundle.obs_std.shape == t.shape:
        t = (t - bundle.obs_mean) / (bundle.obs_std + 1e-6)
    return t

def apply_policy(bundle: PolicyBundle, obs_tensor: torch.Tensor, num_act: int) -> np.ndarray:
    with torch.no_grad():
        x = obs_tensor.unsqueeze(0) if obs_tensor.ndim == 1 else obs_tensor
        out = bundle.module(x)
    if isinstance(out, (tuple, list)): out = out[0]
    if out.ndim == 2 and out.shape[0] == 1: out = out[0]
    if out.numel() != num_act: raise RuntimeError(f"Policy out {out.numel()} != num_act {num_act}")
    a = out.detach().cpu().numpy().astype(np.float64)
    return np.tanh(a)

# ---------------- Control ----------------

def compute_stand_targets(model, qpos_idx, actuator_names, warmup_map: Dict[str, float] | None = None) -> np.ndarray:
    """
    Prefer warmup.leg_joint_targets from env.yaml if present.
    Else fall back to a symmetric, in‑limit posture.
    """
    warmup_map = warmup_map or {}
    stand = np.zeros_like(qpos_idx, dtype=np.float64)
    for i, _ in enumerate(qpos_idx):
        jn = actuator_names[i]
        if jn in warmup_map:
            stand[i] = warmup_map[jn]
            continue
        n = jn.lower()
        if ("hip_joint" in n) and ("abd" not in n):
            stand[i] = 0.75
        elif "thigh_joint" in n:
            stand[i] = 0.85
        elif ("calf_joint" in n) or ("knee" in n):
            stand[i] = -1.6
        else:
            stand[i] = 0.0
    return stand


def control_loop(model, data, bundle, qpos_idx, qvel_idx, actuator_names, actuator_ids,
                 limits_lo, limits_hi, hist_cfg, cmd_vel_3, cmd_ee_vec,
                 action_scales, q_default, q_warm, control_hz, stand_duration,
                 ramp_s, smooth_alpha) -> None:
    dt = model.opt.timestep
    control_dt = 1.0 / control_hz
    steps_per_action = max(1, int(round(control_dt / dt)))

    paused = False; single_step = False; end_sim = False; viz_enabled = True
    # Keyboard control step sizes
    step_vx = 0.05
    step_vy = 0.05
    step_wz = 0.05
    # EE pose presets (x, y, z, r, p, y)
    ee_presets = [
        np.array([0.52,  0.00, 0.36, 0.00,  0.00,  0.00], dtype=np.float32),
        np.array([0.54,  0.12, 0.34, 0.00,  0.10,  0.10], dtype=np.float32),
        np.array([0.54, -0.12, 0.34, 0.00, -0.10, -0.10], dtype=np.float32),
        np.array([0.50,  0.00, 0.32, 0.00,  0.00,  0.20], dtype=np.float32),
        np.array([0.50,  0.00, 0.28, 0.00,  0.00, -0.20], dtype=np.float32),
        np.array([0.48,  0.10, 0.30, 0.00,  0.10,  0.00], dtype=np.float32),
        np.array([0.48, -0.10, 0.30, 0.00, -0.10,  0.00], dtype=np.float32),
        np.array([0.56,  0.08, 0.26, 0.00,  0.15,  0.10], dtype=np.float32),
        np.array([0.56, -0.08, 0.26, 0.00, -0.15, -0.10], dtype=np.float32),
        np.array([0.52,  0.00, 0.24, 0.00,  0.20,  0.00], dtype=np.float32),
    ]

    def _clamp_cmd_vel(v: np.ndarray) -> np.ndarray:
        v[0] = float(np.clip(v[0], -2.0, 2.0))    # vx
        v[1] = float(np.clip(v[1], -2.0, 2.0))    # vy
        v[2] = float(np.clip(v[2], -2.0, 2.0))    # yaw rate
        return v

    def _apply_preset(idx: int) -> None:
        if 0 <= idx < len(ee_presets):
            obs_builder.cmd_ee_vec[:] = ee_presets[idx]
            print(f"[KEY] EE preset {idx+1}: {obs_builder.cmd_ee_vec.tolist()}")

    def key_callback(key):
        nonlocal paused, end_sim, single_step, viz_enabled
        if key == GLFW_SPACE: paused = not paused
        elif key == GLFW_ESCAPE: end_sim = True
        elif key == GLFW_S and paused: single_step = True
        elif key == GLFW_V:
            viz_enabled = not viz_enabled
            print(f"[KEY] Visualization {'ON' if viz_enabled else 'OFF'}")
        elif key == GLFW_UP:
            obs_builder.cmd_vel_3[0] += step_vx
            _clamp_cmd_vel(obs_builder.cmd_vel_3)
            print(f"[KEY] vx -> {obs_builder.cmd_vel_3[0]:.2f}")
        elif key == GLFW_DOWN:
            obs_builder.cmd_vel_3[0] -= step_vx
            _clamp_cmd_vel(obs_builder.cmd_vel_3)
            print(f"[KEY] vx -> {obs_builder.cmd_vel_3[0]:.2f}")
        elif key == GLFW_RIGHT:
            obs_builder.cmd_vel_3[1] += step_vy
            _clamp_cmd_vel(obs_builder.cmd_vel_3)
            print(f"[KEY] vy -> {obs_builder.cmd_vel_3[1]:.2f}")
        elif key == GLFW_LEFT:
            obs_builder.cmd_vel_3[1] -= step_vy
            _clamp_cmd_vel(obs_builder.cmd_vel_3)
            print(f"[KEY] vy -> {obs_builder.cmd_vel_3[1]:.2f}")
        elif key == GLFW_Q:
            obs_builder.cmd_vel_3[2] += step_wz
            _clamp_cmd_vel(obs_builder.cmd_vel_3)
            print(f"[KEY] wz -> {obs_builder.cmd_vel_3[2]:.2f}")
        elif key == GLFW_E:
            obs_builder.cmd_vel_3[2] -= step_wz
            _clamp_cmd_vel(obs_builder.cmd_vel_3)
            print(f"[KEY] wz -> {obs_builder.cmd_vel_3[2]:.2f}")
        elif key == GLFW_1: _apply_preset(0)
        elif key == GLFW_2: _apply_preset(1)
        elif key == GLFW_3: _apply_preset(2)
        elif key == GLFW_4: _apply_preset(3)
        elif key == GLFW_5: _apply_preset(4)
        elif key == GLFW_6: _apply_preset(5)
        elif key == GLFW_7: _apply_preset(6)
        elif key == GLFW_8: _apply_preset(7)
        elif key == GLFW_9: _apply_preset(8)
        elif key == GLFW_0: _apply_preset(9)

    last_print = time.time(); ctrl_counter = 0; sim_start = time.time()
    obs_builder = ObsHistory(model, qpos_idx, qvel_idx, actuator_ids, hist_cfg, cmd_vel_3, cmd_ee_vec)
    obs_builder.qpos_nominal = data.qpos[qpos_idx].copy()
    last_action_norm = np.zeros(len(actuator_ids), dtype=np.float32)

    # Preset stance to targets to prevent default leg extension at spawn
    data.qpos[qpos_idx] = np.clip(q_warm, limits_lo, limits_hi)
    mujoco.mj_forward(model, data)

    # Settle robot with position control before policy enabled
    full_ctrl_pre = np.zeros(model.nu)
    full_ctrl_pre[actuator_ids] = data.qpos[qpos_idx]
    settle_steps = max(1, int(0.1 / dt))
    for _ in range(settle_steps):
        data.ctrl[:] = full_ctrl_pre
        mujoco.mj_step(model, data)


    with viewer.launch_passive(model, data, key_callback=key_callback) as ui:
        print("Controls: arrows=vx/vy, q/e=yaw rate, 1-9,0=EE pose presets, v=toggle viz, space=pause, s=step, esc=quit")
        # Warmup: hold stance with position control
        if stand_duration > 0:
            steps = int(round(stand_duration / dt))
            full_ctrl = np.zeros(model.nu)
            q_init = data.qpos[qpos_idx].copy()
            for s in range(steps):
                alpha = (s + 1) / steps
                ctrl_t = (1 - alpha) * q_init + alpha * q_warm  # q_warm from env.yaml or symmetric fallback
                ctrl_t = np.minimum(np.maximum(ctrl_t, limits_lo), limits_hi)
                full_ctrl[:] = 0.0
                full_ctrl[actuator_ids] = ctrl_t
                data.ctrl[:] = full_ctrl
                mujoco.mj_step(model, data)
                ui.sync()

        # Align observations and timing to policy start
        obs_builder.qpos_nominal = q_warm.copy()
        sim_start = time.time()
        last_print = sim_start
        next_tick = sim_start + control_dt


        while ui.is_running() and not end_sim:
            if paused and not single_step:
                ui.sync(); continue
            single_step = False

            obs = obs_builder.update(data, last_action_norm)
            if ctrl_counter == 0:
                N = len(actuator_ids); H = hist_cfg.base_ang_vel
                cmd_ee_dim = obs_builder.buf_cmd_ee.shape[1]
                per_step_total = 4 + 3 + N + N + N + 3 + cmd_ee_dim + 3
                print(f"[OBS DEBUG] N={N} H={H} per_step_total={per_step_total} expected_total={H*per_step_total} actual_total={obs.size}")

            obs_tensor = normalize_obs(bundle, obs)
            action = apply_policy(bundle, obs_tensor, len(actuator_ids))
            if ctrl_counter < 5:
                print(f"[ACT DEBUG] min={action.min():.3f} max={action.max():.3f} norm={np.linalg.norm(action):.3f}")

            # smooth policy targets
            if 'smoothed_target' not in locals():
                smoothed_target = data.qpos[qpos_idx].astype(np.float64).copy()

            elapsed = time.time() - sim_start
            scale_ramp = min(1.0, elapsed / max(0.1, float(ramp_s)))
            proposed = q_default + (action_scales * action * scale_ramp)
            proposed = np.minimum(np.maximum(proposed, limits_lo), limits_hi)
            alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
            smoothed_target = (1.0 - alpha) * smoothed_target + alpha * proposed

            # handoff blending to policy control
            handoff = 1.5
            beta = min(1.0, elapsed / handoff)
            blended = (1.0 - beta) * q_warm + beta * smoothed_target

            full_ctrl = np.zeros(model.nu)
            full_ctrl[actuator_ids] = blended
            data.ctrl[:] = full_ctrl
            for _ in range(steps_per_action):
                mujoco.mj_step(model, data)

            last_action_norm = np.clip(action, -1.0, 1.0).astype(np.float32)
            ctrl_counter += 1
            now = time.time()
            if now - last_print >= 1.0:
                sim_hz = ctrl_counter / (now - last_print)
                print(f"ctrl Hz ≈ {sim_hz:.1f}")
                last_print = now; ctrl_counter = 0
            # command and ee pose visualization
            scn = ui.user_scn
            scn.ngeom = 0
            if viz_enabled:
                rgba_cmd = np.array([0.0, 1.0, 0.0, 0.9], dtype=np.float32)     # green
                rgba_act = np.array([0.0, 0.3, 1.0, 0.9], dtype=np.float32)     # blue
                rgba_target = np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32)  # green
                rgba_current = np.array([1.0, 0.8, 0.0, 0.8], dtype=np.float32) # yellow
                rgba_x = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)       # red X
                rgba_y = np.array([0.0, 1.0, 0.0, 0.9], dtype=np.float32)       # green Y
                rgba_z = np.array([0.0, 0.0, 1.0, 0.9], dtype=np.float32)       # blue Z

                def _norm(v):
                    n = float(np.linalg.norm(v))
                    return v / n if n > 1e-9 else v

                def _make_rot_from_z(zdir: np.ndarray) -> np.ndarray:
                    z = _norm(zdir.astype(np.float64))
                    if np.linalg.norm(z) < 1e-9:
                        return np.eye(3, dtype=np.float64)
                    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    if abs(float(np.dot(z, up))) > 0.9:
                        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    x = _norm(np.cross(up, z))
                    y = np.cross(z, x)
                    R = np.column_stack((x, y, z))
                    return R.astype(np.float64)

                def _add_arrow_connector(a_w: np.ndarray, b_w: np.ndarray, width: float, rgba: np.ndarray):
                    geom = scn.geoms[scn.ngeom]
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, float(width),
                                         a_w.astype(np.float64), b_w.astype(np.float64))
                    # Set color after creation
                    geom.rgba[:] = rgba.astype(np.float32)
                    scn.ngeom += 1

                def _add_frame(pos_w: np.ndarray, R: np.ndarray, axis_len: float, radius: float):
                    # Draw RGB axes using connectors (works across MuJoCo versions)
                    pos64 = pos_w.astype(np.float64)
                    for i, col in enumerate((rgba_x, rgba_y, rgba_z)):
                        end = pos64 + axis_len * R[:, i].astype(np.float64)
                        geom = scn.geoms[scn.ngeom]
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, float(radius), pos64, end)
                        geom.rgba[:] = col.astype(np.float32)
                        scn.ngeom += 1

                def _add_sphere(pos_w: np.ndarray, size: float, rgba: np.ndarray):
                    geom = scn.geoms[scn.ngeom]
                    size3 = np.array([float(size), 0.0, 0.0], dtype=np.float64)
                    pos3 = pos_w.astype(np.float64)
                    mat9 = np.array([1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0], dtype=np.float64)
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                        size3, pos3, mat9,
                                        rgba.astype(np.float32))
                    scn.ngeom += 1

                # Base pose
                base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
                if base_bid < 0:
                    base_bid = 0
                base_pos = data.xpos[base_bid].copy()
                base_quat = data.xquat[base_bid].copy()
                Rb9 = np.zeros(9, dtype=np.float64)
                mujoco.mju_quat2Mat(Rb9, base_quat)
                Rb = Rb9.reshape(3, 3, order='F')

                # linvel command @ z+5 above robot base
                v_cmd = obs_builder.cmd_vel_3.copy()
                v_cmd_lin_w = (Rb @ np.array([v_cmd[0], v_cmd[1], 0.0])).astype(np.float32)
                base_viz_pos = (base_pos + np.array([0.0, 0.0, 0.5], dtype=np.float32)).astype(np.float32)
                vlen_cmd = min(1.0, 0.5 * float(np.linalg.norm(v_cmd_lin_w)))
                if vlen_cmd > 1e-6:
                    dir_cmd = v_cmd_lin_w / float(np.linalg.norm(v_cmd_lin_w))
                    _add_arrow_connector(base_viz_pos, base_viz_pos + dir_cmd * vlen_cmd, 0.02, rgba_cmd)

                # real base linvel (taken from cvel)
                try:
                    v_world = data.cvel[base_bid, 3:6] if data.cvel.ndim == 2 else data.cvel[6*base_bid+3:6*base_bid+6]
                except Exception:
                    v_world = np.zeros(3, dtype=np.float32)
                vlen_act = min(1.0, 0.5 * float(np.linalg.norm(v_world)))
                if vlen_act > 1e-6:
                    dir_act = v_world / float(np.linalg.norm(v_world) + 1e-9)
                    _add_arrow_connector(base_viz_pos, base_viz_pos + dir_act.astype(np.float32) * vlen_act, 0.02, rgba_act)

                # ee target pose
                ee_local_pos = obs_builder.cmd_ee_vec[:3].astype(np.float64)
                ee_local_rpy = obs_builder.cmd_ee_vec[3:].astype(np.float64)
                cr, sr = np.cos(ee_local_rpy[0]), np.sin(ee_local_rpy[0])
                cp, sp = np.cos(ee_local_rpy[1]), np.sin(ee_local_rpy[1])
                cy, sy = np.cos(ee_local_rpy[2]), np.sin(ee_local_rpy[2])
                Rr = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
                Rp = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
                Ry = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
                Ree_local = Ry @ Rp @ Rr
                # Use yaw-only rotation so target stays tied to base planform, independent of roll/pitch
                yaw = float(np.arctan2(Rb[1, 0], Rb[0, 0]))
                cy, sy = np.cos(yaw), np.sin(yaw)
                Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                Ree_world = (Rz @ Ree_local).astype(np.float32)
                ee_target_world = (base_pos + (Rz @ ee_local_pos).astype(np.float64)).astype(np.float32)
                _add_sphere(ee_target_world, 0.02, rgba_target)
                # draw axes at target
                _add_frame(ee_target_world, Ree_world, axis_len=0.18, radius=0.015)

                # current ee pose marker
                ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wx250s_gripper_link")
                if ee_bid >= 0:
                    ee_curr_pos = data.xpos[ee_bid].astype(np.float32)
                    ee_curr_quat = data.xquat[ee_bid]
                    Ree_curr9 = np.zeros(9, dtype=np.float64)
                    mujoco.mju_quat2Mat(Ree_curr9, ee_curr_quat)
                    Ree_curr = Ree_curr9.reshape(3, 3, order='F')
                    _add_sphere(ee_curr_pos, 0.02, rgba_current)
                    # draw axes at target
                    _add_frame(ee_curr_pos, Ree_curr.astype(np.float32), axis_len=0.18, radius=0.015)
            # set to real control rate
            sleep_s = next_tick - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            next_tick += control_dt
            ui.sync()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play a Torch policy in MuJoCo")
    p.add_argument("--pt", default="weights/exported/policy.pt", help="Path to TorchScript policy")
    p.add_argument("--xml", default="widowGo1/mjcf/scene.xml", help="MuJoCo XML scene file")
    p.add_argument("--hz", type=float, default=None, help="Control frequency in Hz (if None, parse from env.yaml)")
    p.add_argument("--device", choices=["cpu","cuda"], default="cuda", help="Torch device")
    p.add_argument("--stand_for", type=float, default=0.1, help="Warm-up (s) before policy")
    p.add_argument("--params", default="weights/params/env.yaml", help="Isaac env.yaml (for order/scale)")
    p.add_argument("--cmd_pos", type=float, nargs=3, default=(0.0,0.0,0.0), help="EE pose command (x,y,z) base-frame")
    p.add_argument("--cmd_rpy", type=float, nargs=3, default=(0.0,0.0,0.0), help="EE orientation (r,p,y)")
    p.add_argument("--cmd_linvel", type=float, nargs=3, default=(0.0,0.0,0.0), help="Base cmd (vx,vy,0)")
    p.add_argument("--cmd_angvel", type=float, nargs=3, default=(0.0,0.0,0.0), help="Base cmd (0,0,yaw_rate)")
    p.add_argument("--ramp_s", type=float, default=1.0, help="Policy amplitude ramp seconds")
    p.add_argument("--smooth_alpha", type=float, default=0.12, help="EMA smoothing for position targets [0..1]")
    p.add_argument("--leg_scale_mult", type=float, default=1.0, help="Multiply leg action scales")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    bundle = load_policy(args.pt, args.device)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    # start with base elevated to prevent clipping
    base_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
    if base_jid >= 0 and model.jnt_type[base_jid] == mujoco.mjtJoint.mjJNT_FREE:
        adr = model.jnt_qposadr[base_jid]; data.qpos[adr+2] = 0.37
    mujoco.mj_forward(model, data)

    qpos_idx, qvel_idx, actuator_names, actuator_ids = build_actuator_maps(model, args.params)
    limits_lo, limits_hi = build_joint_limits(model, actuator_ids)

    # derive control_hz from env.yaml dt * decimation, unless user provided
    default_hz = 100.0
    ctrl_hz = args.hz if args.hz is not None else _parse_control_rate(args.params, default_hz)

    # scales from env.yaml, fallback 0.25 legs / 0.5 arm
    scale_map = _parse_action_scales(args.params)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[int(aid), 0]) for aid in actuator_ids]
    action_scales = np.array([scale_map.get(jn, _default_scale(jn)) for jn in joint_names], dtype=np.float64)
    # soften leg amplitude relative to training if desired
    for i, jn in enumerate(joint_names):
        if "widow" not in (jn or "").lower():
            action_scales[i] *= float(args.leg_scale_mult)

    print("[MAP DEBUG] actuator order and scales:")
    for i, (n, s) in enumerate(zip(actuator_names, action_scales)):
        print(f"  {i:02d}: {n} scale={s:.3f}")

    # warmup targets from env.yaml (asymmetric stance matching training)
    warmup_map = _parse_warmup_leg_targets(args.params)
    q_warm = compute_stand_targets(model, qpos_idx, joint_names, warmup_map).astype(np.float64)
    q_default = q_warm.copy()  # policy zero-offset holds same stance

    hist_cfg = _parse_history_cfg(args.params)
    cmd_vel_3 = np.array([args.cmd_linvel[0], args.cmd_linvel[1], args.cmd_angvel[2]], dtype=np.float32)
    cmd_ee_vec = np.array([*args.cmd_pos, *args.cmd_rpy], dtype=np.float32)

    control_loop(
        model=model, data=data, bundle=bundle,
        qpos_idx=qpos_idx, qvel_idx=qvel_idx,
        actuator_names=actuator_names, actuator_ids=actuator_ids,
        limits_lo=limits_lo, limits_hi=limits_hi,
        hist_cfg=hist_cfg, cmd_vel_3=cmd_vel_3, cmd_ee_vec=cmd_ee_vec,
        action_scales=action_scales, q_default=q_default, q_warm=q_warm,
        control_hz=ctrl_hz, stand_duration=args.stand_for,
        ramp_s=args.ramp_s, smooth_alpha=args.smooth_alpha,
    )

if __name__ == "__main__":
    main()
