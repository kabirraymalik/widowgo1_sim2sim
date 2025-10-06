#!/usr/bin/env python3
from __future__ import annotations
import argparse, time, os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import mujoco
from mujoco import viewer

# ---------------- Standing stance (same as pd_control.py) ----------------
LEG_STAND_BASE = {
    "FR_hip_joint":   0.0,  "FR_thigh_joint":   0.9,  "FR_calf_joint":  -1.8,
    "FL_hip_joint":   0.0,  "FL_thigh_joint":   0.9,  "FL_calf_joint":  -1.8,
    "RR_hip_joint":   0.0,  "RR_thigh_joint":   0.9,  "RR_calf_joint":  -1.8,
    "RL_hip_joint":   0.0,  "RL_thigh_joint":   0.9,  "RL_calf_joint":  -1.8,
}

# Arm joints to sample (exclude fingers)
ARM_JOINTS = (
    "widow_waist",
    "widow_shoulder",
    "widow_elbow",
    "widow_forearm_roll",
    "widow_wrist_angle",
    "widow_wrist_rotate",
)

# End-effector body candidates (first found used)
EE_BODY_CANDIDATES = (
    "wx250s/ee_gripper_link",
    "wx250s/gripper_link",
    "wx250s/wrist_link",
)

# Foot geoms (ONLY these may touch the floor); fingers are NOT allowed
FOOT_GEOM_NAMES = (
    "FR_foot_collision",
    "FL_foot_collision",
    "RR_foot_collision",
    "RL_foot_collision",
)

# Keys (viewer mode)
GLFW_SPACE  = 32
GLFW_S      = 83
GLFW_ESCAPE = 256
GLFW_V      = 86

@dataclass
class Cfg:
    xml: str
    frame: str              # 'base' | 'world'
    samples: int            # number of valid samples to collect
    warmup_s: float         # initial warm-up settle (stable launch)
    post_reset_s: float     # settle after each reset
    settle_s: float         # post-target settle seconds before recording
    hold_s: float           # extra hold at target before recording
    rate_hz: float          # viewer loop pacing (slower if smaller); ignored in headless
    headless: bool          # run without viewer or sleeping
    out_prefix: str
    min_base_z: float       # base height floor guard
    drift_xy_max: float     # max allowed XY drift from start [m]
    tilt_deg_max: float     # max allowed |roll| or |pitch| [deg]
    arm_speed_rad_s: float  # per-joint rate limit [rad/s]
    arm_tol_rad: float      # per-joint tolerance to consider target reached [rad]
    verbose: bool

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def build_maps(model: mujoco.MjModel) -> tuple[Dict[str, int], Dict[str, int]]:
    jids: Dict[str, int] = {}
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n:
            jids[n] = j
    aids: Dict[str, int] = {}
    for a in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if n:
            aids[n] = a
    return jids, aids

def actuator_for_joint(aids: Dict[str, int], joint_name: str) -> Optional[int]:
    # position actuators named "<joint>_pos"
    return aids.get(f"{joint_name}_pos")

def find_ee_body_id(model: mujoco.MjModel) -> int:
    for nm in EE_BODY_CANDIDATES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            return int(bid)
    raise RuntimeError(f"No EE body found; tried: {EE_BODY_CANDIDATES}")

def quat_to_R(q: np.ndarray) -> np.ndarray:
    m9 = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(m9, q.astype(np.float64))
    return m9.reshape(3, 3, order="F")

def mat_to_rpy(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    c = np.cos(pitch)
    if abs(c) < 1e-8:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1] / c, R[2, 2] / c)
        yaw = np.arctan2(R[1, 0] / c, R[0, 0] / c)
    return np.array([roll, pitch, yaw], dtype=np.float64)

def measure_ee6(model: mujoco.MjModel, data: mujoco.MjData, cfg: Cfg, ee_bid: int, base_bid: int) -> np.ndarray:
    p_w = data.xpos[ee_bid].astype(np.float64)
    R_w = quat_to_R(data.xquat[ee_bid])
    if cfg.frame == "world":
        return np.concatenate([p_w, mat_to_rpy(R_w)]).astype(np.float64)
    base_p = data.xpos[base_bid].astype(np.float64)
    base_R = quat_to_R(data.xquat[base_bid])
    R_bw = base_R.T
    p_b = R_bw @ (p_w - base_p)
    R_b = R_bw @ R_w
    return np.concatenate([p_b, mat_to_rpy(R_b)]).astype(np.float64)

def ground_contact_invalid(model: mujoco.MjModel, data: mujoco.MjData,
                           floor_gid: Optional[int], allowed_geom_ids: set[int]) -> bool:
    """True if any floor contact with a non-allowed geom (feet allowed; fingers NOT allowed)."""
    if floor_gid is None:
        return False
    ncon = int(data.ncon)
    for i in range(ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 == floor_gid or g2 == floor_gid:
            other = g2 if g1 == floor_gid else g1
            if other not in allowed_geom_ids:
                return True
    return False

def compute_robot_body_mask(model: mujoco.MjModel, base_bid: int) -> np.ndarray:
    """Return boolean mask over bodies that are descendants of 'base' (robot bodies)."""
    nbody = model.nbody
    children = [[] for _ in range(nbody)]
    for b in range(nbody):
        p = int(model.body_parentid[b])
        if p >= 0:
            children[p].append(b)
    mask = np.zeros(nbody, dtype=bool)
    # BFS from base
    q = [int(base_bid)]
    mask[int(base_bid)] = True
    while q:
        u = q.pop()
        for v in children[u]:
            if not mask[v]:
                mask[v] = True
                q.append(v)
    return mask

def _body_name(model: mujoco.MjModel, bid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or ""

def _is_arm_body(name: str) -> bool:
    return name.startswith("wx250s/")

def _leg_prefix(name: str) -> Optional[str]:
    for p in ("FR_", "FL_", "RR_", "RL_"):
        if name.startswith(p):
            return p
    return None

def robot_self_contact_invalid(model: mujoco.MjModel, data: mujoco.MjData, robot_body_mask: np.ndarray) -> bool:
    """
    Return True only for cross-subsystem robot contacts:
    - Arm vs. base/legs (invalid)
    - Left/right legs against each other (invalid)
    Allowed:
    - Arm vs. arm (within-arm contacts)
    - Within-same-leg contacts (FR vs FR, etc.)
    """
    ncon = int(data.ncon)
    for i in range(ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        if not (robot_body_mask[b1] and robot_body_mask[b2]):
            continue  # at least one is not robot -> ignore here

        n1 = _body_name(model, b1)
        n2 = _body_name(model, b2)

        # Allow any within-arm contacts
        if _is_arm_body(n1) and _is_arm_body(n2):
            continue

        # Allow contacts within the same leg group (FR, FL, RR, RL)
        lp1, lp2 = _leg_prefix(n1), _leg_prefix(n2)
        if (lp1 is not None) and (lp1 == lp2):
            continue

        # Otherwise it's cross-subsystem (e.g., arm vs leg/base, or leg-leg) -> invalid
        return True
    return False


def base_state_violated(data: mujoco.MjData, base_bid: int,
                        base_p0: np.ndarray, min_base_z: float,
                        drift_xy_max: float, tilt_deg_max: float) -> bool:
    p = data.xpos[base_bid].astype(np.float64)
    if p[2] < min_base_z:
        return True
    if np.linalg.norm(p[:2] - base_p0[:2]) > drift_xy_max:
        return True
    R = quat_to_R(data.xquat[base_bid])
    rpy = mat_to_rpy(R)
    tilt_deg = np.rad2deg(np.max(np.abs(rpy[:2])))
    return tilt_deg > tilt_deg_max

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual/headless EE 6D sampler (pure standing; slow arm; auto-reset on contacts)")
    p.add_argument("--xml", default="widowGo1/mjcf/scene.xml", help="MuJoCo XML")
    p.add_argument("--frame", choices=["base", "world"], default="base", help="Output pose frame")
    p.add_argument("--samples", type=int, default=600, help="Number of valid samples to collect")
    p.add_argument("--warmup_s", type=float, default=0.25, help="Initial standing warm-up settle (stable launch)")
    p.add_argument("--post_reset_s", type=float, default=0.20, help="Settle seconds after each reset before next target")
    p.add_argument("--settle_s", type=float, default=0.10, help="Settle seconds at target before recording")
    p.add_argument("--hold_s", type=float, default=0.08, help="Hold time once within tolerance before recording")
    p.add_argument("--rate_hz", type=float, default=600.0, help="Viewer pacing Hz (lower is slower; ignored in --headless)")
    p.add_argument("--headless", action="store_true", help="Run without viewer and without sleeps (faster)")
    p.add_argument("--arm_speed_rad_s", type=float, default=0.35, help="Per-joint rate limit [rad/s]")
    p.add_argument("--arm_tol_rad", type=float, default=0.02, help="Per-joint tolerance [rad] for target reached")
    p.add_argument("--min_base_z", type=float, default=0.25, help="Min base z (extra guard)")
    p.add_argument("--drift_xy_max", type=float, default=0.045, help="Max allowed XY drift from start [m]")
    p.add_argument("--tilt_deg_max", type=float, default=10.0, help="Max allowed |roll| or |pitch| [deg]")
    p.add_argument("--out", dest="out_prefix", default="out/ee_samples_stand", help="Output NPZ prefix (timestamp appended)")
    p.add_argument("--verbose", action="store_true", help="Per-sample debug prints")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    cfg = Cfg(
        xml=args.xml, frame=args.frame,
        samples=int(args.samples),
        warmup_s=float(args.warmup_s),
        post_reset_s=float(args.post_reset_s),
        settle_s=float(args.settle_s),
        hold_s=float(args.hold_s),
        rate_hz=float(args.rate_hz),
        headless=bool(args.headless),
        out_prefix=str(args.out_prefix),
        min_base_z=float(args.min_base_z),
        drift_xy_max=float(args.drift_xy_max),
        tilt_deg_max=float(args.tilt_deg_max),
        arm_speed_rad_s=float(args.arm_speed_rad_s),
        arm_tol_rad=float(args.arm_tol_rad),
        verbose=bool(args.verbose),
    )

    model = mujoco.MjModel.from_xml_path(cfg.xml)
    data  = mujoco.MjData(model)

    # Lift base if free joint present (avoid floor clipping)
    base_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
    if base_jid >= 0 and model.jnt_type[base_jid] == mujoco.mjtJoint.mjJNT_FREE:
        adr = model.jnt_qposadr[base_jid]
        data.qpos[adr + 2] = max(float(data.qpos[adr + 2]), 0.37)
    mujoco.mj_forward(model, data)

    jids, aids = build_maps(model)
    ee_bid   = find_ee_body_id(model)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_bid < 0:
        raise RuntimeError("Body 'base' not found.")

    # Arm actuator IDs and ctrl ranges
    arm_act_ids: List[int] = []
    arm_ranges: List[Tuple[float, float]] = []
    for jn in ARM_JOINTS:
        a = actuator_for_joint(aids, jn)
        if a is None:
            raise RuntimeError(f"Missing actuator for arm joint '{jn}'")
        lo, hi = model.actuator_ctrlrange[a, 0], model.actuator_ctrlrange[a, 1]
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise RuntimeError(f"Bad ctrlrange for arm joint '{jn}'")
        arm_act_ids.append(int(a))
        arm_ranges.append((float(lo), float(hi)))

    # Floor + feet for floor-contact safety
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_gid < 0:
        floor_gid = None  # best-effort safety
    allowed_floor_contact: set[int] = set()
    for gn in FOOT_GEOM_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gn)
        if gid >= 0:
            allowed_floor_contact.add(int(gid))

    # Robot body mask for self-contact detection
    robot_body_mask = compute_robot_body_mask(model, base_bid)

    # Helpers
    def clamp_ctrl(a: int, val: float) -> float:
        lo, hi = model.actuator_ctrlrange[a, 0], model.actuator_ctrlrange[a, 1]
        return float(np.clip(val, lo, hi))

    def apply_leg_standing(ctrl: np.ndarray) -> None:
        for jn, q in LEG_STAND_BASE.items():
            a = actuator_for_joint(aids, jn)
            if a is None: continue
            ctrl[a] = clamp_ctrl(a, q)

    rng = np.random.default_rng(12345)
    def sample_arm_target() -> np.ndarray:
        return np.array([rng.uniform(lo, hi) for (lo, hi) in arm_ranges], dtype=np.float64)

    # Control vector and arm home
    ctrl = np.zeros(model.nu, dtype=np.float64)
    apply_leg_standing(ctrl)
    arm_home = np.array([(lo + hi) * 0.5 for (lo, hi) in arm_ranges], dtype=np.float64)
    arm_ctrl = arm_home.copy()
    arm_target = arm_home.copy()

    def write_arm_ctrl(qs: np.ndarray) -> None:
        for q, a in zip(qs, arm_act_ids):
            ctrl[a] = clamp_ctrl(a, float(q))

    # Stable launch: set legs and arm home, then settle
    dt = float(model.opt.timestep)
    warm_steps = max(1, int(round(cfg.warmup_s / max(1e-6, dt))))
    write_arm_ctrl(arm_ctrl)
    for _ in range(warm_steps):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    # Record starting base pose (for drift checks) and standing snapshot for resets
    base_p0 = data.xpos[base_bid].astype(np.float64).copy()
    qpos_standing = data.qpos.copy()
    qvel_zero = np.zeros_like(data.qvel)

    def reset_robot():
        # Reset to standing snapshot and settle for cfg.post_reset_s
        nonlocal arm_ctrl, arm_target
        data.qpos[:] = qpos_standing
        data.qvel[:] = qvel_zero
        mujoco.mj_forward(model, data)
        apply_leg_standing(ctrl)
        arm_ctrl = arm_home.copy()
        arm_target = arm_home.copy()
        write_arm_ctrl(arm_ctrl)
        post_steps = max(1, int(round(cfg.post_reset_s / max(1e-6, dt))))
        for _ in range(post_steps):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)

    # Timing / pacing
    step_dt = 1.0 / max(1.0, float(cfg.rate_hz))
    settle_steps_target = max(1, int(round(cfg.settle_s / max(1e-6, dt))))
    hold_steps_target   = max(1, int(round(cfg.hold_s   / max(1e-6, dt))))
    max_step_per_tick   = float(cfg.arm_speed_rad_s) * dt

    # Sampling buffers & state
    EE: List[np.ndarray] = []
    QQ: List[np.ndarray] = []
    need_valid = int(cfg.samples)
    have_target = False
    settle_left = 0
    hold_left = 0
    paused = False
    single_step = False
    endSim = False
    verbose = cfg.verbose

    def common_step() -> None:
        nonlocal have_target, settle_left, hold_left, endSim, arm_ctrl, arm_target
        # Start new target if needed
        if not have_target:
            arm_target = sample_arm_target()
            settle_left = settle_steps_target
            hold_left = hold_steps_target
            have_target = True
            if verbose:
                print(f"[target] {len(EE)+1}/{need_valid}")

        # Rate-limit arm commands toward target
        err = arm_target - arm_ctrl
        step = np.clip(err, -max_step_per_tick, max_step_per_tick)
        arm_ctrl = arm_ctrl + step
        write_arm_ctrl(arm_ctrl)
        apply_leg_standing(ctrl)

        # Step physics
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        # Invalidity: bad floor contact or any robot-robot contact or base instability
        if ground_contact_invalid(model, data, floor_gid, allowed_floor_contact) or \
           robot_self_contact_invalid(model, data, robot_body_mask) or \
           base_state_violated(data, base_bid, base_p0, cfg.min_base_z, cfg.drift_xy_max, cfg.tilt_deg_max):
            if verbose:
                print("[reset] invalid contact or instability; resetting")
            reset_robot()
            have_target = False
            return

        # Close to target → settle/hold then accept
        if np.all(np.abs(err) <= cfg.arm_tol_rad):
            if settle_left > 0:
                settle_left -= 1
            else:
                if hold_left > 0:
                    hold_left -= 1
                else:
                    ee6 = measure_ee6(model, data, cfg, ee_bid, base_bid)
                    EE.append(ee6)
                    QQ.append(arm_ctrl.copy())
                    have_target = False
                    if verbose:
                        print(f"[accept] {len(EE)}/{need_valid}")
                    if len(EE) >= need_valid:
                        endSim = True

    if cfg.headless:
        # Headless fast loop (no viewer, no sleep)
        while not endSim:
            common_step()
    else:
        # Viewer loop with pacing and controls
        def key_cb(keycode: int):
            nonlocal paused, single_step, endSim, verbose
            if keycode == GLFW_SPACE:
                paused = not paused
                print(f"[KEY] Paused={paused}")
            elif keycode == GLFW_S:
                single_step = True
                print("[KEY] Single step")
            elif keycode == GLFW_ESCAPE:
                endSim = True
                print("[KEY] Exit requested")
            elif keycode == GLFW_V:
                verbose = not verbose
                print(f"[KEY] Verbose={verbose}")

        next_tick = time.time() + step_dt
        with viewer.launch_passive(model, data, key_callback=key_cb) as ui:
            print("Controls: space=pause, s=step, v=verbose, esc=quit")
            while ui.is_running() and not endSim:
                now = time.time()
                if now < next_tick:
                    time.sleep(max(0.0, next_tick - now))
                if paused and not single_step:
                    ui.sync(); next_tick += step_dt; continue
                single_step = False
                common_step()
                ui.sync()
                next_tick += step_dt

    # Save & summary
    if len(EE) == 0:
        print("No valid samples collected.")
        return

    EE_arr = np.vstack(EE).astype(np.float64)
    Q_arr  = np.vstack(QQ).astype(np.float64)

    box_lo = np.percentile(EE_arr, 2.5, axis=0)
    box_hi = np.percentile(EE_arr, 97.5, axis=0)
    box_span = box_hi - box_lo
    center = np.median(EE_arr, axis=0)
    cov = np.cov(EE_arr.T)
    chi2_95 = 12.5915887  # chi2(6, 0.95)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"{cfg.out_prefix}_{ts}.npz"
    _ensure_dir(out_path)
    np.savez_compressed(
        out_path,
        ee_poses_6d=EE_arr,
        arm_q=Q_arr,
        box_lo_6d=box_lo,
        box_hi_6d=box_hi,
        box_span_6d=box_span,
        ellipsoid_center_6d=center,
        ellipsoid_cov_6x6=cov,
        chi2_95=chi2_95,
        meta=dict(frame=cfg.frame, valid_samples=EE_arr.shape[0], headless=cfg.headless),
    )

    fields = ["x", "y", "z", "roll", "pitch", "yaw"]
    print("\n=== 95% Axis-Aligned 6D Range (" + cfg.frame + " frame) ===")
    for i, f in enumerate(fields):
        print(f"  {f:>5}: [{box_lo[i]: .4f}, {box_hi[i]: .4f}]  span={box_span[i]:.4f}")
    print(f"\nSaved {EE_arr.shape[0]} samples -> {out_path}")

if __name__ == "__main__":
    main()
