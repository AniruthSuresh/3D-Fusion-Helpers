import pybullet as p
import pybullet_data
import time
import numpy as np

# ----------------- Constants -----------------

GRIPPER_LEN = 0.1612
CUBE_HALF = 0.02
SAFETY = 0.01
DT = 1 / 240

# ----------------- Helper Functions -----------------

def move_ik(robot_id, eef_idx, target_pos, target_orn,
            steps=120, force=600):
    for _ in range(steps):
        ik = p.calculateInverseKinematics(
            robot_id,
            eef_idx,
            target_pos,
            target_orn,
            maxNumIterations=200,
            residualThreshold=1e-4
        )

        for j in range(7):
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                ik[j],
                force=force
            )

        p.stepSimulation()
        time.sleep(DT)


def control_gripper(robot_id, open_ratio, steps=150, force=200):
    """
    Symmetric gripper control using joint names.
    open_ratio: 0.0 = fully open, 1.0 = fully closed
    """
    max_close = 0.725
    gripper_position = open_ratio * max_close

    # Define which joints control left/right individually
    left_joints = ['left_outer_finger_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint']
    right_joints = ['right_outer_knuckle_joint', 'right_inner_knuckle_joint', 'right_inner_finger_joint']

    # Map joint names to indices
    joint_name_to_index = {p.getJointInfo(robot_id, j)[1].decode('utf-8'): j for j in range(p.getNumJoints(robot_id))}

    left_indices = [joint_name_to_index[name] for name in left_joints]
    right_indices = [joint_name_to_index[name] for name in right_joints]

    for _ in range(steps):
        for j in left_indices:
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=gripper_position,  # left moves positive
                force=force
            )
        for j in right_indices:
            p.setJointMotorControl2(
                robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=gripper_position,  # right moves negative (mirror)
                force=force
            )
        p.stepSimulation()
        time.sleep(DT)

# ----------------- PyBullet Setup -----------------

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.setPhysicsEngineParameter(
    numSolverIterations=200,
    fixedTimeStep=DT,
    contactERP=0.2
)

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

eef_idx = 7  # panda_link8

# Improve finger friction
for link in [11, 12, 13, 14]:
    p.changeDynamics(
        robot_id,
        link,
        lateralFriction=2.0,
        spinningFriction=0.02,
        rollingFriction=0.02
    )

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

home = [0, -0.4, 0, -2.2, 0, 2.0, 0.8]
for j in range(7):
    p.resetJointState(robot_id, j, home[j])

control_gripper(robot_id, open_ratio=0.0)

# ----------------- Pick Sequence -----------------

down_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
cube_z = cube_pos[2]

grasp_z = cube_z + CUBE_HALF + GRIPPER_LEN + SAFETY

above = [cube_pos[0], cube_pos[1], grasp_z + 0.12]
grasp = [cube_pos[0], cube_pos[1], grasp_z]
lift  = [cube_pos[0], cube_pos[1], grasp_z + 0.18]

# 1. Move above cube
move_ik(robot_id, eef_idx, above, down_orn, steps=150)

# 2. Descend slowly
move_ik(robot_id, eef_idx, grasp, down_orn, steps=200)

# 3. Pre-close (touch)
control_gripper(robot_id, open_ratio=0.6, steps=120)

# 4. Firm grasp
control_gripper(robot_id, open_ratio=1.0, steps=180)

# 5. Lift
move_ik(robot_id, eef_idx, lift, down_orn, steps=200)

print("✅ Panda picked the cube successfully")

# ----------------- Hold -----------------

while True:
    p.stepSimulation()
    time.sleep(DT)
