#!/usr/bin/env python3
import os
from bisect import bisect_left
import numpy as np
import cv2
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# -------------------------
# CONFIG
# -------------------------
# DATA_ROOT = "/media/skills/RRC HDD A/cross-emb/Newer Data"
DATA_ROOT = "../data/"
OUTPUT_ROOT = "./final-data"

# CTRL_TOPIC     = "/lite6_traj_controller/controller_state"
RGB_TOPIC      = "/camera/camera/color/image_raw"
DEPTH_TOPIC    = "/camera/camera/depth/image_rect_raw"
JOINT_TOPIC    = "/ufactory/joint_states"
GRIP_STATE     = "/gripper/state"
GRIP_CMD       = "/gripper/command"       
CTRL_TOPIC     = GRIP_CMD


fx = 325.4990539550781
fy = 325.4990539550781
cx = 319.9093322753906
cy = 180.0956268310547

# -------------------------
# HELPERS
# -------------------------
def align_to(reference_ts, target_ts):
    idx = []
    if len(target_ts) == 0:
        return [0]*len(reference_ts)
    for ts in reference_ts:
        pos = bisect_left(target_ts, ts)
        if pos == 0:
            idx.append(0)
        elif pos == len(target_ts):
            idx.append(len(target_ts)-1)
        else:
            if abs(target_ts[pos] - ts) < abs(target_ts[pos-1] - ts):
                idx.append(pos)
            else:
                idx.append(pos-1)
    return idx

def convert_rgb_msg(msg):
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return arr.reshape(h, w, 3)

def convert_depth_msg(msg):
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    return arr.astype(np.float32) / 1000.0

def depth_to_pointcloud(depth):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

def find_first_db3(traj_path):
    for root, dirs, files in os.walk(traj_path):
        for f in files:
            if f.endswith(".db3"):
                return os.path.join(root, f)
    return None

# -------------------------
# PROCESS ONE TRAJ
# -------------------------
def process_traj(traj_name, db3_path):
    print(f"\n=== Traj: {traj_name} ===")
    print("Bag:", db3_path)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )

    topics = reader.get_all_topics_and_types()
    def get_msg_type(topic_name):
        t = [x.type for x in topics if x.name == topic_name]
        return get_message(t[0]) if t else None

    ctrl_type   = get_msg_type(CTRL_TOPIC)
    rgb_type    = get_msg_type(RGB_TOPIC)
    depth_type  = get_msg_type(DEPTH_TOPIC)
    joint_type  = get_msg_type(JOINT_TOPIC)
    grip_s_type = get_msg_type(GRIP_STATE)
    grip_c_type = get_msg_type(GRIP_CMD)         # <--- NEW

    if ctrl_type is None:
        print("  -> No controller_state topic. Skipping.")
        return

    ctrl_ts = []
    rgb_buf,  rgb_ts  = [], []
    depth_buf, depth_ts = [], []
    joint_buf, joint_ts = [], []
    grip_s_buf, grip_s_ts = [], []
    grip_c_buf, grip_c_ts = [], []      # <--- NEW

    # read all messages
    while reader.has_next():
        topic, data, ts = reader.read_next()

        if topic == CTRL_TOPIC:
            ctrl_ts.append(ts)

        elif topic == RGB_TOPIC:
            msg = deserialize_message(data, rgb_type)
            rgb_buf.append(msg)
            rgb_ts.append(ts)

        elif topic == DEPTH_TOPIC:
            msg = deserialize_message(data, depth_type)
            depth_buf.append(msg)
            depth_ts.append(ts)

        elif topic == JOINT_TOPIC:
            msg = deserialize_message(data, joint_type)
            joint_buf.append(np.array(msg.position, dtype=np.float32))
            joint_ts.append(ts)

        elif topic == GRIP_STATE and grip_s_type:
            msg = deserialize_message(data, grip_s_type)
            grip_s_buf.append(float(msg.data))  # Float64
            grip_s_ts.append(ts)

        elif topic == GRIP_CMD and grip_c_type:   # <--- NEW
            msg = deserialize_message(data, grip_c_type)
            grip_c_buf.append(int(msg.data))     # Int32
            grip_c_ts.append(ts)

    ctrl_ts  = np.array(ctrl_ts)
    rgb_ts   = np.array(rgb_ts)
    depth_ts = np.array(depth_ts)
    joint_ts = np.array(joint_ts)
    grip_s_ts = np.array(grip_s_ts)
    grip_c_ts = np.array(grip_c_ts)

    print(f"  Pre-sync counts: ctrl={len(ctrl_ts)}, rgb={len(rgb_buf)}, depth={len(depth_buf)}, joints={len(joint_buf)}, g_state={len(grip_s_buf)}, g_cmd={len(grip_c_buf)}")

    if len(ctrl_ts) == 0:
        return

    rgb_idx   = align_to(ctrl_ts, rgb_ts)
    depth_idx = align_to(ctrl_ts, depth_ts)
    joint_idx = align_to(ctrl_ts, joint_ts)
    grip_s_idx = align_to(ctrl_ts, grip_s_ts) if len(grip_s_ts)>0 else [0]*len(ctrl_ts)
    grip_c_idx = align_to(ctrl_ts, grip_c_ts) if len(grip_c_ts)>0 else [0]*len(ctrl_ts)

    N = len(ctrl_ts)
    print(f"  Synced frames: {N}")

    # output dirs
    rgb_out = os.path.join(OUTPUT_ROOT, "rgb", traj_name)
    pc_out  = os.path.join(OUTPUT_ROOT, "pc", traj_name)
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(pc_out, exist_ok=True)

    for i in range(N):
        rgb_msg   = rgb_buf[rgb_idx[i]]
        depth_msg = depth_buf[depth_idx[i]]
        joints    = joint_buf[joint_idx[i]]
        grip_s    = grip_s_buf[grip_s_idx[i]] if grip_s_buf else 0.0
        grip_c    = grip_c_buf[grip_c_idx[i]] if grip_c_buf else 0

        rgb_arr = convert_rgb_msg(rgb_msg)
        depth_arr = convert_depth_msg(depth_msg)

        # save rgb
        cv2.imwrite(
            os.path.join(rgb_out, f"rgb_{i:05d}.png"),
            cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR),
        )

        # depth → pc
        pts = depth_to_pointcloud(depth_arr)
        colors = (rgb_arr.reshape(-1,3)/255.0)

        mask = np.isfinite(pts).all(1) & (pts[:,2] > 0)
        pts = pts[mask].astype(np.float32)
        cols = colors[mask].astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        o3d.io.write_point_cloud(
            os.path.join(pc_out, f"cloud_{i:05d}.ply"),
            pcd,
            write_ascii=False
        )

    print(f"  Saved {N} frames.")

# -------------------------
# RUN ALL TRAJS
# -------------------------
if __name__ == "__main__":
    count = 0
    for name in sorted(os.listdir(DATA_ROOT)):
        path = os.path.join(DATA_ROOT, name)
        if not os.path.isdir(path):
            continue

        db3 = find_first_db3(path)
        if db3:
            count += 1
            process_traj(name, db3)

    print("Done. Trajs processed:", count)
