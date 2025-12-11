#!/usr/bin/env python3
import os
from bisect import bisect_left
import numpy as np
import cv2
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


DATA_ROOT = "../data"         # top-level folder containing trajectory subfolders
OUTPUT_ROOT = "./final-data"  # output root

# Topics
CTRL_TOPIC    = "/lite6_traj_controller/controller_state"   # controller -> joint positions (for actions)
RGB_TOPIC     = "/camera/camera/color/image_raw"
DEPTH_TOPIC   = "/camera/camera/depth/image_rect_raw"
JOINT_TOPIC   = "/ufactory/joint_states"
GRIP_STATE    = "/gripper/state"      # float64 - gripper state
GRIP_CMD      = "/gripper/command"    # int32  - gripper command (master timestamps)

# Camera intrinsics (your values)
fx = 325.4990539550781
fy = 325.4990539550781
cx = 319.9093322753906
cy = 180.0956268310547

# -------------------------
# HELPERS
# -------------------------
def align_to(reference_ts, target_ts):
    """For each ts in reference_ts find index of closest entry in target_ts."""
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
    """Convert ROS2 Image rgb8 -> HxWx3 uint8 numpy."""
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return arr.reshape(h, w, 3)

def convert_depth_msg(msg):
    """Convert ROS2 Image 16UC1 -> HxW float32 meters."""
    h, w = msg.height, msg.width
    arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    return arr.astype(np.float32) / 1000.0

def depth_to_pointcloud(depth):
    """Backproject depth HxW to (H*W,3) using intrinsics."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

def find_first_db3(traj_path):
    """Return first .db3 under traj_path (recursive), or None."""
    for root, dirs, files in os.walk(traj_path):
        for f in files:
            if f.endswith(".db3"):
                return os.path.join(root, f)
    return None

def extract_positions_from_ctrl_msg(msg):
    """
    Extracts 6 joint positions from controller_state.
    Correct field (based on your bag):
        msg.feedback.positions
    """

    # Use feedback.positions (this is populated with real joint positions)
    if hasattr(msg, "feedback") and hasattr(msg.feedback, "positions"):
        pos = list(msg.feedback.positions)
        if len(pos) >= 6:
            return np.array(pos[:6], dtype=np.float32)

    # Fallbacks (unlikely needed for your case)
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
# MAIN processing for one trajectory
# -------------------------
def process_traj(traj_name, db3_path):
    print(f"\n=== Trajectory: {traj_name} ===")
    print("Bag:", db3_path)

    # Open bag
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "")
    )

    # discover message types
    topics_info = reader.get_all_topics_and_types()
    def get_type(topic_name):
        tlist = [t.type for t in topics_info if t.name == topic_name]
        return get_message(tlist[0]) if tlist else None

    ctrl_type  = get_type(CTRL_TOPIC)
    rgb_type   = get_type(RGB_TOPIC)
    depth_type = get_type(DEPTH_TOPIC)
    joint_type = get_type(JOINT_TOPIC)
    grip_s_type= get_type(GRIP_STATE)
    grip_c_type= get_type(GRIP_CMD)

    # must have master topic
    if grip_c_type is None:
        print("  -> Missing master topic /gripper/command. Skipping.")
        return

    # buffers (store raw msgs for images / depth; numeric arrays for others)
    ctrl_buf, ctrl_ts = [], []
    rgb_buf,  rgb_ts  = [], []
    depth_buf, depth_ts = [], []
    joint_buf, joint_ts = [], []
    grip_s_buf, grip_s_ts = [], []
    grip_c_buf, grip_c_ts = [], []

    # read everything into buffers
    while reader.has_next():
        topic, data, ts = reader.read_next()

        if topic == CTRL_TOPIC and ctrl_type:
            msg = deserialize_message(data, ctrl_type)
            ctrl_buf.append(msg)
            ctrl_ts.append(ts)

        elif topic == RGB_TOPIC and rgb_type:
            msg = deserialize_message(data, rgb_type)
            rgb_buf.append(msg)
            rgb_ts.append(ts)

        elif topic == DEPTH_TOPIC and depth_type:
            msg = deserialize_message(data, depth_type)
            depth_buf.append(msg)
            depth_ts.append(ts)

        elif topic == JOINT_TOPIC and joint_type:
            msg = deserialize_message(data, joint_type)
            # joint positions array
            joint_buf.append(np.array(msg.position, dtype=np.float32))
            joint_ts.append(ts)

        elif topic == GRIP_STATE and grip_s_type:
            msg = deserialize_message(data, grip_s_type)
            grip_s_buf.append(float(msg.data))
            grip_s_ts.append(ts)

        elif topic == GRIP_CMD and grip_c_type:
            msg = deserialize_message(data, grip_c_type)
            # gripper command is Int32; store as float for arithmetic
            grip_c_buf.append(float(msg.data))
            grip_c_ts.append(ts)

    # convert to numpy arrays for aligning
    grip_c_ts = np.array(grip_c_ts)
    rgb_ts    = np.array(rgb_ts)
    depth_ts  = np.array(depth_ts)
    joint_ts  = np.array(joint_ts)
    ctrl_ts   = np.array(ctrl_ts)
    grip_s_ts = np.array(grip_s_ts)

    # quick counts
    print("  counts pre-sync -> ctrl:", len(ctrl_buf),
          "rgb:", len(rgb_buf),
          "depth:", len(depth_buf),
          "joints:", len(joint_buf),
          "grip_state:", len(grip_s_buf),
          "grip_cmd:", len(grip_c_buf))

    if len(grip_c_ts) == 0:
        print("  -> no /gripper/command timestamps; skipping")
        return
    if len(joint_buf) == 0:
        print("  -> no /ufactory/joint_states; skipping")
        return
    if len(ctrl_buf) == 0:
        print("  -> no controller_state messages; skipping")
        return
    if len(depth_buf) == 0 or len(rgb_buf) == 0:
        print("  -> missing rgb/depth; skipping")
        return

    # Align everything to master = grip_cmd timestamps
    master_ts = grip_c_ts
    N = len(master_ts)
    rgb_idx   = align_to(master_ts, rgb_ts)    if len(rgb_ts)>0   else [0]*N
    depth_idx = align_to(master_ts, depth_ts)  if len(depth_ts)>0 else [0]*N
    joint_idx = align_to(master_ts, joint_ts)  if len(joint_ts)>0 else [0]*N
    ctrl_idx  = align_to(master_ts, ctrl_ts)   if len(ctrl_ts)>0  else [0]*N
    grip_s_idx= align_to(master_ts, grip_s_ts) if len(grip_s_ts)>0 else [0]*N
    

    print(f"  synchronized frames (master /gripper/command): {N}")

    # OUTPUT dirs
    rgb_out = os.path.join(OUTPUT_ROOT, "rgb", traj_name)
    pc_out  = os.path.join(OUTPUT_ROOT, "pc", traj_name)
    states_out_dir = os.path.join(OUTPUT_ROOT, "states")
    actions_out_dir= os.path.join(OUTPUT_ROOT, "actions")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(pc_out, exist_ok=True)
    os.makedirs(states_out_dir, exist_ok=True)
    os.makedirs(actions_out_dir, exist_ok=True)

    # build per-frame arrays
    states = []     # Nx7: j1..j6, gripper_state
    ctrl_positions_for_actions = []  # N x 6 from controller_state aligned to master
    master_grip_cmd = []  # N (gripper command aligned)

    saved = 0
    for i in range(N):
        # rgb/depth/joint/grip_state/controller at aligned indices
        rgb_msg = rgb_buf[rgb_idx[i]]
        depth_msg = depth_buf[depth_idx[i]]
        joint_vals = joint_buf[joint_idx[i]]
        ctrl_msg = ctrl_buf[ctrl_idx[i]]
        grip_state = grip_s_buf[grip_s_idx[i]] if len(grip_s_buf)>0 else 0.0
        grip_cmd_val = grip_c_buf[i]  # master aligned by construction

        # save RGB
        try:
            rgb_arr = convert_rgb_msg(rgb_msg)
            rgb_path = os.path.join(rgb_out, f"rgb_{i:05d}.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"    Warning: failed save rgb frame {i}: {e}")
            continue

        # convert depth -> colored pointcloud
        try:
            depth_arr = convert_depth_msg(depth_msg)
            pts = depth_to_pointcloud(depth_arr)
            colors_all = (rgb_arr.reshape(-1,3) / 255.0).astype(np.float32)
            mask = np.isfinite(pts).all(axis=1) & (pts[:,2] > 0)
            pts_valid = pts[mask].astype(np.float32)
            cols_valid = colors_all[mask].astype(np.float32)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_valid)
            pcd.colors = o3d.utility.Vector3dVector(cols_valid)
            ply_path = os.path.join(pc_out, f"cloud_{i:05d}.ply")
            o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
        except Exception as e:
            print(f"    Warning: failed make/save pc frame {i}: {e}")
            # still continue to produce states/actions, do not skip

        # build state (6 joints from joint_vals, plus gripper_state)
        # ensure joint_vals length >=6
        jvals = np.asarray(joint_vals, dtype=np.float32)
        jvals = jvals[:6]
        # if fewer than 6, pad with zeros
        # if jvals.shape[0] < 6:
        #     jvals = np.pad(jvals, (0, 6 - jvals.shape[0]), 'constant', constant_values=0.0)
        state_vec = np.hstack([jvals, np.array([grip_state], dtype=np.float32)])
        states.append(state_vec)

        ctrl_positions = extract_positions_from_ctrl_msg(ctrl_msg)

        ctrl_positions_for_actions.append(ctrl_positions[:6])
        master_grip_cmd.append(grip_cmd_val)

        saved += 1

    if saved == 0:
        print("  -> No frames saved for this traj (all failed).")
        return

    states = np.vstack(states).astype(np.float32)                  # (N,7)
    ctrl_pos = np.vstack(ctrl_positions_for_actions).astype(np.float32)  # (N,6)
    grip_cmd_arr = np.array(master_grip_cmd, dtype=np.float32)     # (N,)

    # compute actions using difference: act(t) = ctrl(t) - ctrl(t-1) 
    actions = np.zeros_like(states, dtype=np.float32)  # (N, 7)

    # if ctrl_pos.shape[0] > 1:
    # joint deltas: act(0) = 0, act(t) = ctrl(t) - ctrl(t-1)
    joint_deltas = np.vstack([
        np.zeros((1, 6), dtype=np.float32),      # first frame has no previous frame
        ctrl_pos[1:] - ctrl_pos[:-1]             # ctrl(t) - ctrl(t-1)
    ])
    actions[:, :6] = joint_deltas

    # gripper deltas using same rule
    grip_deltas = np.hstack([
        0.0,                                     # act(0) = 0
        grip_cmd_arr[1:] - grip_cmd_arr[:-1]     # grip(t) - grip(t-1)
    ])
    actions[:, 6] = grip_deltas


    states_file = os.path.join(states_out_dir, f"{traj_name}.txt")
    actions_file= os.path.join(actions_out_dir, f"{traj_name}.txt")
    np.savetxt(states_file, states, fmt="%.6f")
    np.savetxt(actions_file, actions, fmt="%.6f")

    print(f"  Saved {saved} frames.")
    print(f"  States -> {states_file}  (shape: {states.shape})")
    print(f"  Actions -> {actions_file} (shape: {actions.shape})")
    print(f"  RGBs -> {rgb_out}")
    print(f"  PCs  -> {pc_out}")

# -------------------------
# RUN across all trajectories in DATA_ROOT
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
