import pybullet as p
import pybullet_data
import time
import numpy as np

# ----------------- Setup -----------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")

robot_id = p.loadURDF(
    "./lite-6-updated-urdf/lite_6_new.urdf",
    [0, 0, 0],
    useFixedBase=True
)
eef_idx = 6

# Load cube
cube_id = p.loadURDF(
    "./urdf-data/cube/cube.urdf",
    [0.35, 0.0, 0.015]
)
p.changeDynamics(cube_id, -1, lateralFriction=1.2)

# EEF starting pose
eef_pos = np.array([0.35, 0, 0.18])
eef_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

# Gripper open ratio
gripper_ratio = 1.0  # 1 = open, 0 = closed

# ----------------- Helper Functions -----------------
ARM_JOINTS = [0, 1, 2, 3, 4, 5]

def move_ik(target_pos, target_orn):
    ik = p.calculateInverseKinematics(robot_id, eef_idx, target_pos, target_orn)
    for i, j in enumerate(ARM_JOINTS):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, ik[i], force=500)

def control_gripper(open_ratio):
    target = -0.04 * open_ratio
    p.setJointMotorControlArray(
        robot_id, [8, 9], p.POSITION_CONTROL, 
        targetPositions=[target, target], forces=[500, 500]
    )

# ----------------- Teleop Loop -----------------
print("Controls: W/S -> +X/-X, A/D -> +Y/-Y, Q/E -> +Z/-Z, R/F -> Open/Close gripper, Esc -> exit")

while True:
    keys = p.getKeyboardEvents()

    # EEF increments
    delta = 0.01
    for k in keys:
        if keys[k] & p.KEY_IS_DOWN:
            if k == ord('w'): eef_pos[0] += delta
            if k == ord('s'): eef_pos[0] -= delta
            if k == ord('a'): eef_pos[1] += delta
            if k == ord('d'): eef_pos[1] -= delta
            if k == ord('q'): eef_pos[2] += delta
            if k == ord('e'): eef_pos[2] -= delta
            if k == ord('r'): gripper_ratio = min(1.0, gripper_ratio + 0.05)
            if k == ord('f'): gripper_ratio = max(0.0, gripper_ratio - 0.05)
            if k == 27:  # Esc key
                p.disconnect()
                exit()

    move_ik(eef_pos, eef_orn)
    control_gripper(gripper_ratio)
    p.stepSimulation()
    time.sleep(1 / 240)
