import pybullet as p
import pybullet_data
import time
import numpy as np

# ----------------- Helper Functions -----------------

def move_ik_steps(robot_id, eef_idx, target_pos, target_orn,
                  steps=80, force=600):
    """
    Move Panda arm using IK (joints 0–6 only)
    """
    for _ in range(steps):
        ik = p.calculateInverseKinematics(
            robot_id,
            eef_idx,
            target_pos,
            target_orn,
            maxNumIterations=200,
            residualThreshold=1e-4
        )

        for j in range(7):  # panda_joint1 → panda_joint7
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=ik[j],
                force=force
            )

        p.stepSimulation()
        time.sleep(1 / 240)


def control_gripper(robot_id, open_ratio, steps=120, force=150):
    """
    Robotiq 2F-85 control
    open_ratio: 0.0 = open
                1.0 = closed
    """
    max_close = 0.725  # joint limit from your joint dump
    target = open_ratio * max_close

    for _ in range(steps):
        p.setJointMotorControl2(
            robot_id,
            9,  # finger_joint ONLY
            p.POSITION_CONTROL,
            targetPosition=target,
            force=force
        )
        p.stepSimulation()
        time.sleep(1 / 240)


# ----------------- PyBullet Setup -----------------

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")

p.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[0.5, 0.0, 0.2]
)

# ----------------- Load Robot -----------------

robot_id = p.loadURDF(
    "./Exact_Panda/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda_with_2F85_sec.urdf",
    [0, 0, 0],
    useFixedBase=True
)

eef_idx = 7  # panda_link8 (CORRECT EEF)

# ----------------- Load Cube -----------------

cube_id = p.loadURDF(
    "./urdf-data/cube/cube.urdf",
    [0.55, 0.0, 0.02]
)

p.changeDynamics(
    cube_id,
    -1,
    lateralFriction=1.5,
    spinningFriction=0.01,
    rollingFriction=0.01
)

# ----------------- Initial Pose -----------------

home_joints = [0, -0.4, 0, -2.2, 0, 2.0, 0.8]
for j in range(7):
    p.resetJointState(robot_id, j, home_joints[j])

# Open gripper
control_gripper(robot_id, open_ratio=0.0)

# ----------------- Pick Sequence -----------------

down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

above_cube = [0.55, 0.0, 0.25]
grasp_pose = [0.55, 0.0, 0.07]
lift_pose  = [0.55, 0.0, 0.30]

# Move above cube
move_ik_steps(robot_id, eef_idx, above_cube, down_orn)

# # Move down slowly
move_ik_steps(robot_id, eef_idx, grasp_pose, down_orn, steps=120)

# # Close gripper slowly (grasp)
# control_gripper(robot_id, open_ratio=1.0, steps=200)

# # Lift cube
# move_ik_steps(robot_id, eef_idx, lift_pose, down_orn, steps=150)

# print("✅ Panda picked the cube successfully")

# ----------------- Hold -----------------

while True:
    p.stepSimulation()
    time.sleep(1 / 240)
