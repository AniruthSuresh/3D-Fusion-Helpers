import pybullet as p
import pybullet_data
import time

# ---------- PyBullet Setup ----------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=60,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0.3]
)

# ---------- Load Robot ----------
urdf_path = "./lite-6-updated-urdf/lite_6_new.urdf"
robot_id = p.loadURDF(
    urdf_path,
    basePosition=[0, 0, 0],
    useFixedBase=True
)

# ---------- Create Sliders ----------
joint_sliders = {}
gripper_joints = set()

num_joints = p.getNumJoints(robot_id)
print(f"Total joints: {num_joints}")

for j in range(num_joints):
    info = p.getJointInfo(robot_id, j)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]

    if joint_type not in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        continue

    # ---- Detect gripper joints ----
    is_gripper = any(k in joint_name.lower() for k in ["finger", "gripper"])

    if is_gripper:
        # Small & safe range for gripper
        lower, upper = -0.04, 0.0
        start = -0.02
        gripper_joints.add(j)
    else:
        lower = info[8]
        upper = info[9]
        if lower > upper:
            lower, upper = -3.14, 3.14
        start = 0.0

    slider = p.addUserDebugParameter(
        paramName=f"{j}:{joint_name}",
        rangeMin=lower,
        rangeMax=upper,
        startValue=start
    )

    joint_sliders[j] = slider

print("Gripper joints:", gripper_joints)

# ---------- Control Loop ----------
while True:
    for joint_id, slider_id in joint_sliders.items():
        target = p.readUserDebugParameter(slider_id)

        force = 100 if joint_id in gripper_joints else 500

        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=force
        )

    p.stepSimulation()
    time.sleep(1.0 / 240.0)
