import pybullet as p
import pybullet_data
import time
import numpy as np

# ----------------- Helper Functions -----------------

def move_ik_steps(robot_id, eef_idx, start_pos, end_pos, orn, steps=30, force=500):
    """Move end-effector from start_pos to end_pos in fixed steps with enough force."""
    for step in range(steps):
        interp_pos = [
            start + (end - start) * (step + 1) / steps
            for start, end in zip(start_pos, end_pos)
        ]
        ik = p.calculateInverseKinematics(robot_id, eef_idx, interp_pos, orn)
        for j, val in enumerate(ik):
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=val,
                force=force 
            )
        p.stepSimulation()
        time.sleep(1/240)

def control_gripper_slow(robot_id, start_open=1.0, end_open=0.0, steps=30):
    """Close gripper gradually in small increments."""
    for step in range(steps):
        ratio = start_open + (end_open - start_open) * (step + 1)/steps
        target = -0.03 * ratio
        p.setJointMotorControlArray(
            robot_id,
            jointIndices=[8, 9],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[target, target],
            forces=[120, 120]
        )
        p.stepSimulation()
        time.sleep(1/240)

def control_gripper_synchronized(robot_id, start_open=1.0, end_open=0.0, steps=50):
    """
    Close gripper slowly and synchronously so both fingers move together.
    start_open: 1.0 = fully open, 0.0 = fully closed
    """
    for step in range(steps):
        ratio = start_open + (end_open - start_open) * (step + 1) / steps
        target = -0.04 * ratio  # 0.04 is d the max opening for lite-6 gripper . 

        p.setJointMotorControl2(robot_id, 8, p.POSITION_CONTROL, targetPosition=target, force=120)
        p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=target, force=120)
        p.stepSimulation()
        time.sleep(1/240)


# ----------------- PyBullet Setup -----------------

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")

p.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=[0.35, 0, 0.1]
)

# ----------------- Load Robot -----------------

robot_id = p.loadURDF("./lite-6-updated-urdf/lite_6_new.urdf", [0, 0, 0], useFixedBase=True)
eef_idx = 6

# ----------------- Load Cube -----------------

cube_id = p.loadURDF("./urdf-data/cube/cube.urdf", [0.35, 0.0, 0.015])
p.changeDynamics(cube_id, -1, lateralFriction=1.2)


finger_links = ["left_finger", "right_finger"]


num_joints = p.getNumJoints(robot_id)
link_name_to_index = {p.getJointInfo(robot_id, i)[12].decode("utf-8"): i for i in range(num_joints)}

# Get AABB for fingers
for fname in finger_links:
    if fname in link_name_to_index:
        idx = link_name_to_index[fname]
        aabb_min, aabb_max = p.getAABB(robot_id, idx)
        height = aabb_max[2] - aabb_min[2]
        print(f"{fname} AABB min: {aabb_min}, max: {aabb_max}, height: {height:.4f} m")

down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

finger_height = 0.1880
safety_margin = 0.02

start_above = [0.38, 0.0, 0.015 + finger_height + safety_margin]  # slightly above cube
pre_grasp   = [0.38, 0.0, 0.015 + finger_height ]          # just above cube surface

lift = [0.38, 0.0, 0.015 + finger_height + 0.05]  # lift 10cm above cube

control_gripper_slow(robot_id, start_open=0.0, end_open=1.0, steps=10)

move_ik_steps(robot_id, eef_idx, start_above, pre_grasp, down_orn, steps=30, force=500)


control_gripper_synchronized(robot_id, start_open=1.0, end_open=0.02, steps=650)

move_ik_steps(robot_id, eef_idx, pre_grasp, lift, down_orn, steps=1000, force=150)

print("✅ Cube picked slowly and safely")

# ----------------- Keep Simulation Alive -----------------
while True:
    p.stepSimulation()
    time.sleep(1/240)
