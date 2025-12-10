import os
import numpy as np
import pybullet as p
import pybullet_data
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from bisect import bisect_left
import time

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "../data"            # main folder with trajectory subfolders
OUTPUT_ROOT = "./final-data"

JOINT_TOPIC = "/ufactory/joint_states"
GRIPPER_TOPIC = "/gripper/state"

URDF_PATH = "/home/aniruth/Desktop/RRC/XARM7/xArm-Python-SDK/example/wrapper/xarm7/Follow_DROID/Franka_arm/Droid Mask Extraction/Lite-6-data-collection/lite-6-updated-urdf/lite_6_new.urdf"
EEF_INDEX = 6                                 # end-effector link

# =========================================================
# HELPERS
# =========================================================
def align_to(reference_ts, target_ts):
    """Align target timestamps to reference timestamps."""
    idx = []
    for ts in reference_ts:
        pos = bisect_left(target_ts, ts)
        if pos == 0:
            idx.append(0)
        elif pos == len(target_ts):
            idx.append(len(target_ts)-1)
        else:
            if abs(target_ts[pos]-ts) < abs(target_ts[pos-1]-ts):
                idx.append(pos)
            else:
                idx.append(pos-1)
    return idx

def read_joint_gripper(db3_path):
    """Extract joint and gripper states from a bag file."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )
    topics = reader.get_all_topics_and_types()

    def get_msg_type(name):
        types = [t.type for t in topics if t.name == name]
        return get_message(types[0]) if types else None

    joint_type = get_msg_type(JOINT_TOPIC)
    grip_type = get_msg_type(GRIPPER_TOPIC)

    joint_buf, joint_ts = [], []
    grip_buf, grip_ts = [], []

    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic == JOINT_TOPIC and joint_type:
            msg = deserialize_message(data, joint_type)
            joint_buf.append(np.array(msg.position, dtype=np.float32))
            joint_ts.append(ts)
        elif topic == GRIPPER_TOPIC and grip_type:
            msg = deserialize_message(data, grip_type)
            grip_buf.append(float(msg.data))
            grip_ts.append(ts)

    joint_ts = np.array(joint_ts)
    grip_ts = np.array(grip_ts)

    # Align gripper to joint timestamps
    grip_idx = align_to(joint_ts, grip_ts)
    joint_sync = np.array(joint_buf)
    grip_sync = np.array([grip_buf[i] for i in grip_idx])

    # Combine into full states: [joint1..n, gripper]
    states = np.hstack([joint_sync, grip_sync[:, None]])
    # Compute actions
    actions = np.zeros_like(states)
    if len(states) > 1:
        actions[1:] = states[1:] - states[:-1]

    return states, actions

def simulate_eef(states, urdf_path, eef_index):
    """Simulate PyBullet robot to get EEF pose for each state."""
    p.connect(p.GUI)  # use GUI if visualization needed
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(urdf_path, useFixedBase=True)

    eef_traj = []

    for joints in states:
        time.sleep(0.001)  # adjust as needed for visualization speed
        # Set joint positions (exclude last element = gripper)
        for j in range(len(joints)-1):
            p.resetJointState(robot, j, joints[j])
        p.stepSimulation()

        # Get EEF pose
        state = p.getLinkState(robot, eef_index)
        pos = state[0]  # X,Y,Z
        quat = state[1] # quaternion

        # Convert quaternion to roll, pitch, yaw
        rpy = p.getEulerFromQuaternion(quat)
        eef_traj.append(list(pos) + list(rpy))  # 6 numbers: X,Y,Z,R,P,Y

    p.disconnect()
    return np.array(eef_traj, dtype=np.float32)


# =========================================================
# MAIN LOOP OVER TRAJECTORIES
# =========================================================
eef_out_dir = os.path.join(OUTPUT_ROOT, "eef-pos")
states_out_dir = os.path.join(OUTPUT_ROOT, "states")
actions_out_dir = os.path.join(OUTPUT_ROOT, "actions")
os.makedirs(eef_out_dir, exist_ok=True)
os.makedirs(states_out_dir, exist_ok=True)
os.makedirs(actions_out_dir, exist_ok=True)

for folder in sorted(os.listdir(DATA_ROOT)):
    traj_path = os.path.join(DATA_ROOT, folder)
    if not os.path.isdir(traj_path):
        continue

    # recursively search for .db3 files
    bag_files = []
    for root, dirs, files in os.walk(traj_path):
        for f in files:
            if f.endswith(".db3"):
                bag_files.append(os.path.join(root, f))

    if not bag_files:
        continue

    print(folder, "->", bag_files)
    
    db3 = os.path.join(traj_path, bag_files[0])

    print(f"\nProcessing trajectory: {folder}")
    # extract synchronized joint + gripper states
    states, actions = read_joint_gripper(db3)

    # save states & actions
    np.savetxt(os.path.join(states_out_dir, f"{folder}.txt"), states, fmt="%.6f")
    np.savetxt(os.path.join(actions_out_dir, f"{folder}.txt"), actions, fmt="%.6f")

    # simulate EEF pose
    eef_traj = simulate_eef(states, URDF_PATH, EEF_INDEX)
    np.savetxt(os.path.join(eef_out_dir, f"{folder}.txt"), eef_traj, fmt="%.6f")
    print(f"Saved {len(states)} frames → states, actions, eef")

print("\n✔ All trajectories processed")
