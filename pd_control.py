# pd_control.py
import os
import mujoco
from mujoco import viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("widowGo1/mjcf/scene.xml")
data  = mujoco.MjData(model)

# legs standing
leg_targets = {
    "FR_hip_joint":   0.0,  "FR_thigh_joint":   0.9,  "FR_calf_joint":  -1.8,
    "FL_hip_joint":   0.0,  "FL_thigh_joint":   0.9,  "FL_calf_joint":  -1.8,
    "RR_hip_joint":   0.0,  "RR_thigh_joint":   0.9,  "RR_calf_joint":  -1.8,
    "RL_hip_joint":   0.0,  "RL_thigh_joint":   0.9,  "RL_calf_joint":  -1.8,
}

# arm at init
arm_joints = [
    "widow_waist",
    "widow_shoulder",
    "widow_elbow",
    "widow_forearm_roll",
    "widow_wrist_angle",
    "widow_wrist_rotate",
    "widow_left_finger",
    "widow_right_finger",
]

# gains
KP_LEG, KD_LEG = 60.0, 2.0
KP_ARM, KD_ARM = 40.0, 1.5

def jid(name):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
def aid(name):  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

j_qadr = {n: model.jnt_qposadr[jid(n)] for n in list(leg_targets.keys()) + arm_joints}
j_dadr = {n: model.jnt_dofadr[jid(n)] for n in list(leg_targets.keys()) + arm_joints}

act = {n: aid(n + "_ctrl") for n in leg_targets.keys()}
act.update({
    "widow_waist":        aid("widow_waist_ctrl"),
    "widow_shoulder":     aid("widow_shoulder_ctrl"),
    "widow_elbow":        aid("widow_elbow_ctrl"),
    "widow_forearm_roll": aid("widow_forearm_roll_ctrl"),
    "widow_wrist_angle":  aid("widow_wrist_angle_ctrl"),
    "widow_wrist_rotate": aid("widow_wrist_rotate_ctrl"),
    "widow_left_finger":  aid("widow_left_finger_ctrl"),
    "widow_right_finger": aid("widow_right_finger_ctrl"),
})

for n, qref in leg_targets.items():
    data.qpos[j_qadr[n]] = qref
mujoco.mj_forward(model, data)

arm_targets = {n: float(data.qpos[j_qadr[n]]) for n in arm_joints}

def pd_hold():
    for n, qref in leg_targets.items():
        q, qd = data.qpos[j_qadr[n]], data.qvel[j_dadr[n]]
        data.ctrl[act[n]] = KP_LEG*(qref - q) - KD_LEG*qd
    for n, qref in arm_targets.items():
        q, qd = data.qpos[j_qadr[n]], data.qvel[j_dadr[n]]
        data.ctrl[act[n]] = KP_ARM*(qref - q) - KD_ARM*qd

# pause/ exit
paused = False
step_once = False
endSim = False

# key codes for teleop
GLFW_SPACE  = 32
GLFW_S      = 83
GLFW_ESCAPE = 256

def key_listener(keycode):
    global paused, step_once, endSim
    if keycode == GLFW_SPACE:
        paused = not paused
        print("Paused:", paused)
    elif keycode == GLFW_S:
        step_once = True
        print("Advanced one step")
    elif keycode == GLFW_ESCAPE:
        endSim = True
        print("Exiting sim...")

# performance readouts
t0 = time.perf_counter()
step_counter = 0
report_every = 1.0  # seconds

with viewer.launch_passive(model, data, key_callback=key_listener) as v:
    while v.is_running() and not endSim:
        if not paused:
            pd_hold()
            mujoco.mj_step(model, data)
            step_counter += 1
        else:
            if step_once:
                pd_hold()
                mujoco.mj_step(model, data)
                step_counter += 1
                step_once = False

        now = time.perf_counter()
        dt = now - t0
        if dt >= report_every:
            hz = step_counter / dt
            print(f"Sim: {hz:.1f} Hz")
            step_counter = 0
            t0 = now

        v.sync()
