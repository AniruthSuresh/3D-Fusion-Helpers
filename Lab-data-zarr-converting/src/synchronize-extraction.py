import os
import numpy as np
import cv2
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from bisect import bisect_left

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "../data"
OUTPUT_ROOT = "./final-data"

RGB_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
JOINT_TOPIC = "/ufactory/joint_states"
GRIPPER_TOPIC = "/gripper/state"

fx = 325.4990539550781
fy = 325.4990539550781
cx = 319.9093322753906
cy = 180.0956268310547

# =========================================================
# HELPERS
# =========================================================
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

def align_to(reference_ts, target_ts):
    """For each ref timestamp, find closest index in target timestamps"""
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

# =========================================================
# PROCESS ONE TRAJECTORY
# =========================================================
def process_traj(traj_name, db3_path):
    print(f"\n=== Processing trajectory: {traj_name} ===")

    # output folders
    rgb_out = os.path.join(OUTPUT_ROOT, "rgb", traj_name)
    pc_out = os.path.join(OUTPUT_ROOT, "pc", traj_name)
    states_out = os.path.join(OUTPUT_ROOT, "states")
    actions_out = os.path.join(OUTPUT_ROOT, "actions")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(pc_out, exist_ok=True)
    os.makedirs(states_out, exist_ok=True)
    os.makedirs(actions_out, exist_ok=True)

    # reader
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    topics = reader.get_all_topics_and_types()

    def get_msg_type(name):
        types = [t.type for t in topics if t.name == name]
        return get_message(types[0]) if types else None

    rgb_type = get_msg_type(RGB_TOPIC)
    depth_type = get_msg_type(DEPTH_TOPIC)
    joint_type = get_msg_type(JOINT_TOPIC)
    grip_type = get_msg_type(GRIPPER_TOPIC)

    # buffers
    rgb_buf, rgb_ts = [], []
    depth_buf, depth_ts = [], []
    joint_buf, joint_ts = [], []
    grip_buf, grip_ts = [], []

    # read all messages
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

    if not depth_buf:
        print("No depth frames, skipping.")
        return

    rgb_ts = np.array(rgb_ts)
    depth_ts = np.array(depth_ts)
    joint_ts = np.array(joint_ts)
    grip_ts = np.array(grip_ts)

    # Align everything to depth frames
    rgb_idx = align_to(depth_ts, rgb_ts)
    joint_idx = align_to(depth_ts, joint_ts)
    grip_idx = align_to(depth_ts, grip_ts)

    # synchronized arrays
    rgb_sync = [rgb_buf[i] for i in rgb_idx]
    depth_sync = [depth_buf[i] for i in range(len(depth_buf))]
    joint_sync = np.array([joint_buf[i] for i in joint_idx])
    grip_sync = np.array([grip_buf[i] for i in grip_idx])

    # save rgb + point cloud + states
    states = []
    for i, depth in enumerate(depth_sync):
        rgb = rgb_sync[i]
        # save rgb
        cv2.imwrite(os.path.join(rgb_out, f"rgb_{i:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # pc
        pts = depth_to_pc(depth)
        mask = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
        pts = pts[mask]
        colors = (rgb.reshape(-1,3)/255.0)[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
        o3d.io.write_point_cloud(os.path.join(pc_out, f"cloud_{i:05d}.ply"), pcd)
        # states
        state_vec = np.hstack([joint_sync[i], grip_sync[i]])
        states.append(state_vec)

    states = np.array(states, dtype=np.float32)
    actions = np.zeros_like(states)
    actions[1:] = states[1:] - states[:-1]

    # save states + actions
    np.savetxt(os.path.join(states_out, f"{traj_name}.txt"), states, fmt="%.6f")
    np.savetxt(os.path.join(actions_out, f"{traj_name}.txt"), actions, fmt="%.6f")

    print(f"Saved {len(states)} synchronized frames for {traj_name}")

# =========================================================
# MAIN LOOP
# =========================================================
for folder in sorted(os.listdir(DATA_ROOT)):
    traj_path = os.path.join(DATA_ROOT, folder)
    if not os.path.isdir(traj_path):
        continue
    bag_files = [f for f in os.listdir(traj_path) if f.endswith(".db3")]
    if not bag_files:
        continue
    process_traj(folder, os.path.join(traj_path, bag_files[0]))

print("\n✔ All trajectories processed")
