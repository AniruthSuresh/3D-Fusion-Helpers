import os
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from bisect import bisect_left
import time

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "../data"
OUTPUT_ROOT = "./final-data"

RGB_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
JOINT_TOPIC = "/ufactory/joint_states"
GRIPPER_TOPIC = "/gripper/state"

URDF_PATH = "/home/aniruth/Desktop/RRC/XARM7/xArm-Python-SDK/example/wrapper/xarm7/Follow_DROID/Franka_arm/Droid Mask Extraction/Lite-6-data-collection/lite-6-updated-urdf/lite_6_new.urdf"
EEF_INDEX = 6

# Camera intrinsics for point cloud
fx, fy = 325.4990539550781, 325.4990539550781
cx, cy = 319.9093322753906, 180.0956268310547

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

def convert_rgb(msg):
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

def convert_depth(msg):
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0

def depth_to_pc(depth):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

def read_and_sync(db3_path):
    """Read bag file and return synchronized RGB, depth, joint+gripper states"""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )
    topics = reader.get_all_topics_and_types()

    def get_msg_type(name):
        types = [t.type for t in topics if t.name == name]
        return get_message(types[0]) if types else None

    rgb_type = get_msg_type(RGB_TOPIC)
    depth_type = get_msg_type(DEPTH_TOPIC)
    joint_type = get_msg_type(JOINT_TOPIC)
    grip_type = get_msg_type(GRIPPER_TOPIC)

    rgb_buf, rgb_ts = [], []
    depth_buf, depth_ts = [], []
    joint_buf, joint_ts = [], []
    grip_buf, grip_ts = [], []

    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic == RGB_TOPIC and rgb_type:
            msg = deserialize_message(data, rgb_type)
            rgb_buf.append(convert_rgb(msg))
            rgb_ts.append(ts)
        elif topic == DEPTH_TOPIC and depth_type:
            msg = deserialize_message(data, depth_type)
            depth_buf.append(convert_depth(msg))
            depth_ts.append(ts)
        elif topic == JOINT_TOPIC and joint_type:
            msg = deserialize_message(data, joint_type)
            joint_buf.append(np.array(msg.position, dtype=np.float32))
            joint_ts.append(ts)
        elif topic == GRIPPER_TOPIC and grip_type:
            msg = deserialize_message(data, grip_type)
            grip_buf.append(float(msg.data))
            grip_ts.append(ts)

    # convert timestamps to np arrays
    rgb_ts = np.array(rgb_ts)
    depth_ts = np.array(depth_ts)
    joint_ts = np.array(joint_ts)
    grip_ts = np.array(grip_ts)

    # align everything to depth timestamps (or RGB, your choice)
    ref_ts = depth_ts
    rgb_idx = align_to(ref_ts, rgb_ts)
    joint_idx = align_to(ref_ts, joint_ts)
    grip_idx = align_to(ref_ts, grip_ts)

    rgb_sync = [rgb_buf[i] for i in rgb_idx]
    depth_sync = [depth_buf[i] for i in range(len(ref_ts))]
    joint_sync = np.array([joint_buf[i] for i in joint_idx])
    grip_sync = np.array([grip_buf[i] for i in grip_idx])

    states = np.hstack([joint_sync, grip_sync[:, None]])
    actions = np.zeros_like(states)
    actions[1:] = states[1:] - states[:-1]

    return rgb_sync, depth_sync, states, actions

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
        eef_traj.append(list(pos)+list(rpy))

    p.disconnect()
    return np.array(eef_traj, dtype=np.float32)

# =========================================================
# MAIN LOOP
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

    # find .db3 recursively
    bag_files = []
    for root, dirs, files in os.walk(traj_path):
        for f in files:
            if f.endswith(".db3"):
                bag_files.append(os.path.join(root, f))
    if not bag_files:
        continue

    db3 = bag_files[0]  # already full path

    print(f"\nProcessing trajectory: {folder}")
    rgb_sync, depth_sync, states, actions = read_and_sync(db3)

    # save states/actions
    np.savetxt(os.path.join(states_out_dir, f"{folder}.txt"), states, fmt="%.6f")
    np.savetxt(os.path.join(actions_out_dir, f"{folder}.txt"), actions, fmt="%.6f")

    # run PyBullet only on synchronized states
    eef_traj = simulate_eef(states, URDF_PATH, EEF_INDEX)
    np.savetxt(os.path.join(eef_out_dir, f"{folder}.txt"), eef_traj, fmt="%.6f")

    print(f"Saved {len(states)} frames → states, actions, eef")

print("\n✔ All trajectories processed")
