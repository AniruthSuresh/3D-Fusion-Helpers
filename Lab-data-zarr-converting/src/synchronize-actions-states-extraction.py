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
DATA_ROOT = "../data"        
OUTPUT_ROOT = "./final-data"  

CTRL_TOPIC    = "/lite6_traj_controller/controller_state"  
RGB_TOPIC     = "/camera/camera/color/image_raw"
DEPTH_TOPIC   = "/camera/camera/depth/image_rect_raw"
JOINT_TOPIC   = "/ufactory/joint_states"
GRIP_STATE    = "/gripper/state"      

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

def extract_positions_from_ctrl_msg(msg):
    if hasattr(msg, "feedback") and hasattr(msg.feedback, "positions"):
        pos = list(msg.feedback.positions)
        if len(pos) >= 6:
            return np.array(pos[:6], dtype=np.float32)

    candidate_fields = [
        ("reference", "positions"),
        ("output", "positions"),
        ("error", "positions"),
        ("actual", "positions"),
        ("desired", "positions"),
    ]
    for obj, field in candidate_fields:
        if hasattr(msg, obj):
            obj_val = getattr(msg, obj)
            if hasattr(obj_val, field):
                arr = getattr(obj_val, field)
                if isinstance(arr, (list, tuple)) and len(arr) >= 6:
                    return np.array(arr[:6], dtype=np.float32)

    print("⚠️ Warning: controller_state has no usable joint position fields → using zeros")
    return np.zeros(6, dtype=np.float32)

# -------------------------
# MAIN processing
# -------------------------
def process_traj(traj_name, db3_path):
    print(f"\n=== Trajectory: {traj_name} ===")
    print("Bag:", db3_path)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )

    topics_info = reader.get_all_topics_and_types()
    def get_type(topic_name):
        tlist = [t.type for t in topics_info if t.name == topic_name]
        return get_message(tlist[0]) if tlist else None

    ctrl_type  = get_type(CTRL_TOPIC)
    rgb_type   = get_type(RGB_TOPIC)
    depth_type = get_type(DEPTH_TOPIC)
    joint_type = get_type(JOINT_TOPIC)
    grip_s_type= get_type(GRIP_STATE)

    # buffers
    ctrl_buf, ctrl_ts = [], []
    rgb_buf,  rgb_ts  = [], []
    depth_buf, depth_ts = [], []
    joint_buf, joint_ts = [], []
    grip_s_buf, grip_s_ts = [], []

    # read messages
    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic == CTRL_TOPIC and ctrl_type:
            ctrl_buf.append(deserialize_message(data, ctrl_type))
            ctrl_ts.append(ts)
        elif topic == RGB_TOPIC and rgb_type:
            rgb_buf.append(deserialize_message(data, rgb_type))
            rgb_ts.append(ts)
        elif topic == DEPTH_TOPIC and depth_type:
            depth_buf.append(deserialize_message(data, depth_type))
            depth_ts.append(ts)
        elif topic == JOINT_TOPIC and joint_type:
            msg = deserialize_message(data, joint_type)
            joint_buf.append(np.array(msg.position, dtype=np.float32))
            joint_ts.append(ts)
        elif topic == GRIP_STATE and grip_s_type:
            msg = deserialize_message(data, grip_s_type)
            grip_s_buf.append(float(msg.data))
            grip_s_ts.append(ts)

    grip_s_ts = np.array(grip_s_ts)
    rgb_ts    = np.array(rgb_ts)
    depth_ts  = np.array(depth_ts)
    joint_ts  = np.array(joint_ts)
    ctrl_ts   = np.array(ctrl_ts)

    print("  counts pre-sync -> ctrl:", len(ctrl_buf),
          "rgb:", len(rgb_buf),
          "depth:", len(depth_buf),
          "joints:", len(joint_buf),
          "grip_state:", len(grip_s_buf))

    if len(grip_s_ts) == 0 or len(ctrl_buf) == 0 or len(joint_buf) == 0 or len(rgb_buf) == 0 or len(depth_buf) == 0:
        print("  -> Missing necessary topics; skipping.")
        return

    # --- Synchronize to gripper_state timestamps ---
    master_ts = grip_s_ts
    N = len(master_ts)

    joint_idx = align_to(master_ts, joint_ts)
    ctrl_idx  = align_to(master_ts, ctrl_ts)

    print(f"  synchronized frames (master /gripper/state): {N}")

    states_out_dir = os.path.join(OUTPUT_ROOT, "states")
    actions_out_dir= os.path.join(OUTPUT_ROOT, "actions")

    os.makedirs(states_out_dir, exist_ok=True)
    os.makedirs(actions_out_dir, exist_ok=True)

    states = []
    ctrl_positions_for_actions = []
    grip_state_arr = []

    saved = 0
    for i in range(N):

        joint_vals = joint_buf[joint_idx[i]]
        ctrl_msg = ctrl_buf[ctrl_idx[i]]
        grip_state = grip_s_buf[i]

        # state = joints + gripper_state
        jvals = np.asarray(joint_vals, dtype=np.float32)[:6]
        state_vec = np.hstack([jvals, np.array([grip_state], dtype=np.float32)])
        states.append(state_vec)

        ctrl_positions = extract_positions_from_ctrl_msg(ctrl_msg)
        ctrl_positions_for_actions.append(ctrl_positions[:6])
        grip_state_arr.append(grip_state)

        saved += 1

    if saved == 0:
        print("  -> No frames saved for this trajectory.")
        return

    states = np.vstack(states).astype(np.float32)
    ctrl_pos = np.vstack(ctrl_positions_for_actions).astype(np.float32)
    grip_state_arr = np.array(grip_state_arr, dtype=np.float32)

    # --- Compute actions ---
    actions = np.zeros_like(states, dtype=np.float32)
    # joint deltas
    joint_deltas = np.vstack([np.zeros((1,6), dtype=np.float32), ctrl_pos[1:] - ctrl_pos[:-1]])
    actions[:, :6] = joint_deltas
    # gripper deltas from gripper_state
    grip_deltas = np.hstack([0.0, grip_state_arr[1:] - grip_state_arr[:-1]])
    actions[:, 6] = grip_deltas

    # --- Remove frames where all actions are zero ---
    nonzero_idx = np.any(actions != 0, axis=1)
    states = states[nonzero_idx]
    actions = actions[nonzero_idx]

    # Print which frames are being removed
    removed_frames = np.where(~nonzero_idx)[0]
    if len(removed_frames) > 0:
        print(f"  Removing frames with all-zero actions: {removed_frames.tolist()}")


    # Save states/actions
    states_file = os.path.join(states_out_dir, f"{traj_name}-bkp.txt")
    actions_file= os.path.join(actions_out_dir, f"{traj_name}-bkp.txt")
    np.savetxt(states_file, states, fmt="%.6f")
    np.savetxt(actions_file, actions, fmt="%.6f")

    print(f"  Saved {len(states)} frames (non-zero actions).")
    print(f"  States -> {states_file}  (shape: {states.shape})")
    print(f"  Actions -> {actions_file} (shape: {actions.shape})")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    total = 0
    for name in sorted(os.listdir(DATA_ROOT)):
        traj_path = os.path.join(DATA_ROOT, name)
        if not os.path.isdir(traj_path):
            continue
        db3 = find_first_db3(traj_path)
        if not db3:
            print(f"Skipping {name}: no .db3 found.")
            continue
        total += 1
        process_traj(name, db3)

    print(f"\nDone. Processed {total} trajectories.")
