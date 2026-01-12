#!/usr/bin/env python3
import os
import json
import numpy as np
import cv2
import open3d as o3d
from shutil import copy2

# -------------------------
# CONFIG
# -------------------------
DATA_ROOT = "../ur5-object-picking/dataset/"  # Input: simulation dataset
OUTPUT_ROOT = "./processed-sim-data-new"          # Output: processed data

# State/Action dimensions for UR5 sim data
STATE_DIM = 13  # [eef_pos(3), eef_orn(3), joints(6), gripper(1)]
ACTION_DIM = 13

# -------------------------
# HELPERS
# -------------------------
def find_iterations(data_root):
    """Find all iter_XXXX directories in the dataset."""
    iterations = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and name.startswith("iter_"):
            iterations.append((name, path))
    return iterations

def load_npy_pointcloud_as_ply(npy_path, rgb_image):
    """
    Load point cloud from .npy file and convert to Open3D PLY format.
    Colors the point cloud using the RGB image (same as original script).

    Args:
        npy_path: Path to .npy file containing Nx3 point cloud
        rgb_image: HxWx3 RGB image for coloring points

    Returns:
        o3d.geometry.PointCloud object
    """
    points = np.load(npy_path)

    # Filter invalid points (same as original script)
    mask = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & (points[:, 2] < 2.5)
    points_valid = points[mask].astype(np.float32)

    # Color using RGB image (same as original script)
    colors_all = (rgb_image.reshape(-1, 3) / 255.0).astype(np.float32)
    cols_valid = colors_all[mask].astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_valid)
    pcd.colors = o3d.utility.Vector3dVector(cols_valid)

    return pcd

# -------------------------
# MAIN PROCESSING
# -------------------------
def process_iteration(iter_name, iter_path):
    """
    Process one iteration of simulation data.
    Loads pre-computed states/actions and processes both camera views.
    """
    print(f"\n=== {iter_name} ===")

    # Load states and actions (already computed in sim)
    agent_pos_file = os.path.join(iter_path, "agent_pos.npy")
    actions_file = os.path.join(iter_path, "actions.npy")

    if not os.path.exists(agent_pos_file) or not os.path.exists(actions_file):
        print(f"  -> Missing agent_pos.npy or actions.npy, skipping.")
        return

    states = np.load(agent_pos_file).astype(np.float32)
    actions = np.load(actions_file).astype(np.float32)

    num_frames = len(states)
    print(f"  Total frames: {num_frames}")

    # Validate dimensions
    if states.shape[1] != STATE_DIM or actions.shape[1] != ACTION_DIM:
        print(f"  -> Unexpected dimensions, skipping.")
        return

    # Create output directories
    output_dirs = create_output_dirs(iter_name)

    # Process both cameras
    for camera in ["third_person", "wrist"]:
        process_camera(iter_path, camera, num_frames, output_dirs)

    # --- Filter zero-action frames (same as original script) ---
    nonzero_idx = np.any(actions != 0, axis=1)
    removed_frames = np.where(~nonzero_idx)[0]

    if len(removed_frames) > 0:
        print(f"  Removing frames with all-zero actions: {removed_frames.tolist()}")

        states = states[nonzero_idx]
        actions = actions[nonzero_idx]

        # Remove corresponding RGB/PC files for both cameras
        for camera in ["third_person", "wrist"]:
            for i in removed_frames:
                try:
                    rgb_file = os.path.join(output_dirs[f"{camera}_rgb"], f"rgb_{i:05d}.png")
                    pc_file = os.path.join(output_dirs[f"{camera}_pc"], f"cloud_{i:05d}.ply")
                    if os.path.exists(rgb_file):
                        os.remove(rgb_file)
                    if os.path.exists(pc_file):
                        os.remove(pc_file)
                except FileNotFoundError:
                    pass

    # Save states and actions
    states_file = os.path.join(output_dirs["states"], f"{iter_name}.txt")
    actions_file_out = os.path.join(output_dirs["actions"], f"{iter_name}.txt")
    np.savetxt(states_file, states, fmt="%.6f")
    np.savetxt(actions_file_out, actions, fmt="%.6f")

    print(f"  Saved {len(states)} frames (non-zero actions).")
    print(f"  States -> {states_file}  (shape: {states.shape})")
    print(f"  Actions -> {actions_file_out} (shape: {actions.shape})")


def process_camera(iter_path, camera, num_frames, output_dirs):
    """Process one camera view: copy RGB, convert PC to PLY, save extrinsics."""
    camera_rgb_dir = os.path.join(iter_path, camera, "rgb")
    camera_pcd_dir = os.path.join(iter_path, camera, "pcd")
    camera_poses_dir = os.path.join(iter_path, "camera_poses")

    if not os.path.exists(camera_rgb_dir) or not os.path.exists(camera_pcd_dir):
        print(f"  -> Missing {camera} data, skipping.")
        return

    print(f"  Processing {camera} camera...")

    # Determine file prefix (tp_ or wr_)
    prefix = "tp" if camera == "third_person" else "wr"

    extrinsics_list = []

    # Process each frame
    for i in range(num_frames):
        rgb_file = os.path.join(camera_rgb_dir, f"{prefix}_rgb_{i:04d}.png")
        pcd_file = os.path.join(camera_pcd_dir, f"{prefix}_pcd_{i:04d}.npy")
        pose_file = os.path.join(camera_poses_dir, f"pose_{i:04d}.json")

        # Copy RGB (same as original script saving RGB)
        if os.path.exists(rgb_file):
            try:
                out_rgb_file = os.path.join(output_dirs[f"{camera}_rgb"], f"rgb_{i:05d}.png")
                copy2(rgb_file, out_rgb_file)
            except Exception as e:
                print(f"    Warning: Failed to copy RGB frame {i}: {e}")
                continue

        # Convert .npy point cloud to PLY (same as original script)
        if os.path.exists(pcd_file) and os.path.exists(rgb_file):
            try:
                rgb_img = cv2.imread(rgb_file)
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

                pcd = load_npy_pointcloud_as_ply(pcd_file, rgb_img)
                ply_file = os.path.join(output_dirs[f"{camera}_pc"], f"cloud_{i:05d}.ply")
                o3d.io.write_point_cloud(ply_file, pcd, write_ascii=False)
            except Exception as e:
                print(f"    Warning: Failed to convert PC frame {i}: {e}")

        # Load camera extrinsics from JSON
        if os.path.exists(pose_file):
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
                camera_key = f"{camera}_camera"
                if camera_key in pose_data:
                    extr_matrix = np.array(pose_data[camera_key]["extrinsics_matrix"], dtype=np.float32)
                    extrinsics_list.append(extr_matrix.flatten())
                else:
                    extrinsics_list.append(np.zeros(16, dtype=np.float32))
        else:
            extrinsics_list.append(np.zeros(16, dtype=np.float32))

    # Save camera extrinsics (same as original script)
    camera_extrinsics = np.vstack(extrinsics_list).astype(np.float32)
    extr_file = os.path.join(output_dirs[f"{camera}_extrinsics"], f"{iter_name}.txt")
    np.savetxt(extr_file, camera_extrinsics, fmt="%.6f")

    print(f"    Extrinsics -> {extr_file} (shape: {camera_extrinsics.shape})")
    print(f"    RGBs -> {output_dirs[f'{camera}_rgb']}")
    print(f"    PCs  -> {output_dirs[f'{camera}_pc']}")


def create_output_dirs(iter_name):
    """Create output directory structure for dual cameras."""
    dirs = {}

    # Third person camera
    dirs["third_person_rgb"] = os.path.join(OUTPUT_ROOT, "third_person_rgb", iter_name)
    dirs["third_person_pc"] = os.path.join(OUTPUT_ROOT, "third_person_pc", iter_name)
    dirs["third_person_extrinsics"] = os.path.join(OUTPUT_ROOT, "third_person_extrinsics")

    # Wrist camera
    dirs["wrist_rgb"] = os.path.join(OUTPUT_ROOT, "wrist_rgb", iter_name)
    dirs["wrist_pc"] = os.path.join(OUTPUT_ROOT, "wrist_pc", iter_name)
    dirs["wrist_extrinsics"] = os.path.join(OUTPUT_ROOT, "wrist_extrinsics")

    # States and actions (shared)
    dirs["states"] = os.path.join(OUTPUT_ROOT, "states")
    dirs["actions"] = os.path.join(OUTPUT_ROOT, "actions")

    # Create all directories
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


# -------------------------
# MAIN ENTRY POINT
# -------------------------
if __name__ == "__main__":
    iterations = find_iterations(DATA_ROOT)

    if len(iterations) == 0:
        print(f"No iterations found in {DATA_ROOT}")
        exit(1)

    total = 0
    for iter_name, iter_path in iterations:
        try:
            process_iteration(iter_name, iter_path)
            total += 1
        except Exception as e:
            print(f"\nError processing {iter_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone. Processed {total} trajectories.")
