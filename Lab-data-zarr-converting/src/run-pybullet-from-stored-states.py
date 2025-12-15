import os
import numpy as np
import pybullet as p
import pybullet_data
import time

# =========================================================
# CONFIG
# =========================================================
STATES_ROOT = "./final-data/states"    
EEF_OUT_ROOT = "./final-data/eef-pos" 
URDF_PATH = "/home/aniruth/Desktop/RRC/XARM7/xArm-Python-SDK/example/wrapper/xarm7/Follow_DROID/Franka_arm/Droid Mask Extraction/Lite-6-data-collection/lite-6-updated-urdf/lite_6_new.urdf"
EEF_INDEX = 6

os.makedirs(EEF_OUT_ROOT, exist_ok=True)

# =========================================================
# SIMULATION
# =========================================================
def simulate_eef(states, urdf_path, eef_index):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(urdf_path, useFixedBase=True)

    eef_traj = []

    for joints in states:
        time.sleep(0.0001)
        for j in range(len(joints)-1):  # exclude gripper
            p.resetJointState(robot, j, joints[j])
        p.stepSimulation()
        pos, quat = p.getLinkState(robot, eef_index)[0:2]
        rpy = p.getEulerFromQuaternion(quat)
        eef_traj.append(list(pos) + list(rpy))

    p.disconnect()
    return np.array(eef_traj, dtype=np.float32)

# =========================================================
# MAIN LOOP
# =========================================================
for file_name in sorted(os.listdir(STATES_ROOT)):
    if not file_name.endswith(".txt"):
        continue

    states_path = os.path.join(STATES_ROOT, file_name)
    states = np.loadtxt(states_path, dtype=np.float32)

    print(f"\nProcessing: {file_name} ({len(states)} frames)")
    eef_traj = simulate_eef(states, URDF_PATH, EEF_INDEX)

    out_path = os.path.join(EEF_OUT_ROOT, file_name)
    np.savetxt(out_path, eef_traj, fmt="%.6f")
    print(f"Saved EEF trajectory: {out_path}")

print("\n✔ All EEF trajectories extracted")
