import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")

robot_id = p.loadURDF(
    "./Exact_Panda/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda_with_2F85_sec.urdf",
    [0, 0, 0],
    useFixedBase=True
)

time.sleep(0.5)

num_joints = p.getNumJoints(robot_id)

print("\n=== Link Index Map ===")
link_map = {}
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    link_name = info[12].decode("utf-8")
    link_map[link_name] = i
    print(f"{i:2d} : {link_name}")

# -------------------------------
# Define relevant links
# -------------------------------

EEF_LINK = "panda_link8"

FINGER_LINKS = [
    "left_inner_finger",
    "right_inner_finger",
    "left_outer_finger",
    "right_outer_finger"
]

eef_idx = link_map[EEF_LINK]

# -------------------------------
# Get EEF AABB
# -------------------------------

eef_aabb_min, eef_aabb_max = p.getAABB(robot_id, eef_idx)
eef_z = eef_aabb_min[2]

print("\nEEF AABB:")
print(" min:", eef_aabb_min)
print(" max:", eef_aabb_max)

# -------------------------------
# Get lowest fingertip
# -------------------------------

lowest_finger_z = 1e9

print("\nFinger AABBs:")
for fname in FINGER_LINKS:
    if fname not in link_map:
        continue

    idx = link_map[fname]
    aabb_min, aabb_max = p.getAABB(robot_id, idx)

    print(f"{fname}")
    print(" min:", aabb_min)
    print(" max:", aabb_max)

    lowest_finger_z = min(lowest_finger_z, aabb_min[2])

# -------------------------------
# Final gripper length
# -------------------------------

gripper_length = eef_z - lowest_finger_z

print("\n==============================")
print(f"✅ EXACT GRIPPER LENGTH = {gripper_length:.4f} meters")
print("==============================")

while True:
    p.stepSimulation()
    time.sleep(1/240)
