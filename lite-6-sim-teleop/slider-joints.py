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
    cameraTargetPosition=[0.3, 0, 0.3]
)

p.loadURDF("plane.urdf")

# ---------- Load Panda ----------
urdf_path = "./Exact_Panda/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda_with_2F85_sec.urdf"
robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)

# ---------- Print ALL Joint Details ----------
num_joints = p.getNumJoints(robot_id)
print(f"\n=== Panda Joint Info ({num_joints}) ===\n")

JOINT_TYPES = {
    p.JOINT_REVOLUTE: "REVOLUTE",
    p.JOINT_PRISMATIC: "PRISMATIC",
    p.JOINT_FIXED: "FIXED"
}

for j in range(num_joints):
    info = p.getJointInfo(robot_id, j)
    print(f"""
Joint {j}
  Name      : {info[1].decode()}
  Type      : {JOINT_TYPES.get(info[2])}
  Link name : {info[12].decode()}
  Limits    : [{info[8]}, {info[9]}]
""")

# ---------- Create Sliders ----------
joint_sliders = {}
gripper_joints = []

for j in range(num_joints):
    info = p.getJointInfo(robot_id, j)
    joint_name = info[1].decode("utf-8")
    joint_type = info[2]

    # Only movable joints
    if joint_type not in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        continue

    # ✅ CORRECT Panda gripper detection
    is_gripper = (
        joint_type == p.JOINT_PRISMATIC
        and "finger" in joint_name.lower()
    )

    if is_gripper:
        lower, upper = 0.0, 0.04   # meters
        start = 0.04               # fully open
        gripper_joints.append(j)
    else:
        lower, upper = info[8], info[9]
        if lower > upper:
            lower, upper = -3.14, 3.14
        start = 0.0

    slider = p.addUserDebugParameter(
        f"{j}:{joint_name}",
        lower,
        upper,
        start
    )

    joint_sliders[j] = slider

print("✅ Gripper joints detected:", gripper_joints)
# Expected: [9, 10]

# ---------- Control Loop ----------
while True:
    for joint_id, slider_id in joint_sliders.items():
        target = p.readUserDebugParameter(slider_id)

        force = 100 if joint_id in gripper_joints else 500

        p.setJointMotorControl2(
            robot_id,
            joint_id,
            p.POSITION_CONTROL,
            targetPosition=target,
            force=force
        )

    p.stepSimulation()
    time.sleep(1 / 240)
