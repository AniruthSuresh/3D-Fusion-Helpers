#!/usr/bin/env python3
import os
from bisect import bisect_left
import numpy as np
import cv2
import open3d as o3d
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import pybullet as p
import pybullet_data

# -------------------------
# CONFIG
# -------------------------
DATA_ROOT = "."         # top-level folder containing trajectory subfolders
OUTPUT_ROOT = "./final-data"  # output root

# Topics
CTRL_TOPIC    = "/lite6_traj_controller/controller_state"  
RGB_TOPIC     = "/camera/camera/color/image_raw"
DEPTH_TOPIC   = "/camera/camera/depth/image_rect_raw"
JOINT_TOPIC   = "/ufactory/joint_states"
GRIP_STATE    = "/gripper/state"      

# Camera intrinsics
fx = 325.4990539550781
fy = 325.4990539550781
cx = 319.9093322753906
cy = 180.0956268310547

# Robot FK configuration
URDF_PATH = "/home/varun-edachali/Research/RRC/policy/data/3D-Fusion-Helpers/lite-6-sim-teleop/lite-6-updated-urdf/lite_6_new.urdf"
EEF_INDEX = 6

# EEF to Camera transform (4x4 homogeneous matrix)
# Calibrated Camera-to-End-Effector transformation
EEF_TO_CAMERA = np.array([
    [ 0.00903555,  0.10281995,  0.99465895,  0.07002748],
    [-0.00194421, -0.99469586,  0.10284143, -0.01914177],
    [ 0.99995729, -0.00286305, -0.00878767,  0.03685897],
    [ 0.0,         0.0,         0.0,         1.0]
], dtype=np.float32)

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

def pos_quat_to_matrix(pos, quat):
    """Convert position [x,y,z] and quaternion [x,y,z,w] to 4x4 homogeneous matrix."""
    R = p.getMatrixFromQuaternion(quat)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.array(R, dtype=np.float32).reshape(3, 3)
    T[:3, 3] = pos
    return T

def compute_eef_pose(robot_id, joint_angles):
    """Compute end effector pose from joint angles using FK."""
    for j in range(len(joint_angles)):
        p.resetJointState(robot_id, j, joint_angles[j])
    p.stepSimulation()
    state = p.getLinkState(robot_id, EEF_INDEX)
    pos = np.array(state[0], dtype=np.float32)
    quat = np.array(state[1], dtype=np.float32)
    return pos, quat

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
    rgb_idx   = align_to(master_ts, rgb_ts)
    depth_idx = align_to(master_ts, depth_ts)
    joint_idx = align_to(master_ts, joint_ts)
    ctrl_idx  = align_to(master_ts, ctrl_ts)

    print(f"  synchronized frames (master /gripper/state): {N}")

    # Initialize PyBullet for FK computation
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(URDF_PATH, useFixedBase=True)

    # OUTPUT dirs
    rgb_out = os.path.join(OUTPUT_ROOT, "rgb", traj_name)
    pc_out  = os.path.join(OUTPUT_ROOT, "pc", traj_name)
    states_out_dir = os.path.join(OUTPUT_ROOT, "states")
    actions_out_dir= os.path.join(OUTPUT_ROOT, "actions")
    extrinsics_out_dir = os.path.join(OUTPUT_ROOT, "extrinsics")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(pc_out, exist_ok=True)
    os.makedirs(states_out_dir, exist_ok=True)
    os.makedirs(actions_out_dir, exist_ok=True)
    os.makedirs(extrinsics_out_dir, exist_ok=True)

    states = []
    ctrl_positions_for_actions = []
    grip_state_arr = []
    camera_extrinsics = []

    saved = 0
    for i in range(N):
        rgb_msg = rgb_buf[rgb_idx[i]]
        depth_msg = depth_buf[depth_idx[i]]
        joint_vals = joint_buf[joint_idx[i]]
        ctrl_msg = ctrl_buf[ctrl_idx[i]]
        grip_state = grip_s_buf[i]

        # save RGB
        try:
            rgb_arr = convert_rgb_msg(rgb_msg)
            rgb_path = os.path.join(rgb_out, f"rgb_{i:05d}.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
        except:
            continue

        # save pointcloud
        try:
            depth_arr = convert_depth_msg(depth_msg)
            pts = depth_to_pointcloud(depth_arr)
            colors_all = (rgb_arr.reshape(-1,3)/255.0).astype(np.float32)
            mask = np.isfinite(pts).all(axis=1) & (pts[:,2] > 0)
            pts_valid = pts[mask].astype(np.float32)
            cols_valid = colors_all[mask].astype(np.float32)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_valid)
            pcd.colors = o3d.utility.Vector3dVector(cols_valid)
            ply_path = os.path.join(pc_out, f"cloud_{i:05d}.ply")
            o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
        except:
            pass

        # state = joints + gripper_state
        jvals = np.asarray(joint_vals, dtype=np.float32)[:6]
        state_vec = np.hstack([jvals, np.array([grip_state], dtype=np.float32)])
        states.append(state_vec)

        ctrl_positions = extract_positions_from_ctrl_msg(ctrl_msg)
        ctrl_positions_for_actions.append(ctrl_positions[:6])
        grip_state_arr.append(grip_state)

        # Compute camera extrinsics via FK
        eef_pos, eef_quat = compute_eef_pose(robot, jvals)
        T_base_eef = pos_quat_to_matrix(eef_pos, eef_quat)
        T_base_camera = T_base_eef @ EEF_TO_CAMERA
        camera_extrinsics.append(T_base_camera.flatten())

        saved += 1

    if saved == 0:
        print("  -> No frames saved for this trajectory.")
        p.disconnect()
        return

    states = np.vstack(states).astype(np.float32)
    ctrl_pos = np.vstack(ctrl_positions_for_actions).astype(np.float32)
    grip_state_arr = np.array(grip_state_arr, dtype=np.float32)
    camera_extrinsics = np.vstack(camera_extrinsics).astype(np.float32)

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
    camera_extrinsics = camera_extrinsics[nonzero_idx]

    # Print which frames are being removed
    removed_frames = np.where(~nonzero_idx)[0]
    if len(removed_frames) > 0:
        print(f"  Removing frames with all-zero actions: {removed_frames.tolist()}")

    # remove corresponding RGB/PC files
    for i in removed_frames:
        try:
            os.remove(os.path.join(rgb_out, f"rgb_{i:05d}.png"))
            os.remove(os.path.join(pc_out, f"cloud_{i:05d}.ply"))
        except FileNotFoundError:
            pass


    # Save states/actions/extrinsics
    states_file = os.path.join(states_out_dir, f"{traj_name}.txt")
    actions_file= os.path.join(actions_out_dir, f"{traj_name}.txt")
    extrinsics_file = os.path.join(extrinsics_out_dir, f"{traj_name}.txt")
    np.savetxt(states_file, states, fmt="%.6f")
    np.savetxt(actions_file, actions, fmt="%.6f")
    np.savetxt(extrinsics_file, camera_extrinsics, fmt="%.6f")

    # Disconnect PyBullet
    p.disconnect()

    print(f"  Saved {len(states)} frames (non-zero actions).")
    print(f"  States -> {states_file}  (shape: {states.shape})")
    print(f"  Actions -> {actions_file} (shape: {actions.shape})")
    print(f"  Extrinsics -> {extrinsics_file} (shape: {camera_extrinsics.shape})")
    print(f"  RGBs -> {rgb_out}")
    print(f"  PCs  -> {pc_out}")

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
