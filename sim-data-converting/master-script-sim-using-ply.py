#!/usr/bin/env python3
"""
Process raw simulation data to standardized format.
- Converts point clouds from .npy to .ply
- Extracts camera extrinsics
- Removes zero-action frames
- Saves states, actions, and cube positions as .txt files
"""
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
CUBE_DIM = 7    # [pos(3), quat(4)]

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

def load_ply_pointcloud(ply_path):
    """
    Load a pre-saved .ply file using Open3D.
    """
    if not os.path.exists(ply_path):
        return None
    
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Optional: Apply a safety depth filter if not already filtered in sim
    # points = np.asarray(pcd.points)
    # mask = (points[:, 2] < 2.5)
    # pcd = pcd.select_by_index(np.where(mask)[0])
    
    return pcd

# -------------------------
# MAIN PROCESSING
# -------------------------
def process_iteration(iter_name, iter_path):
    """
    Process one iteration of simulation data.
    Loads pre-computed states/actions/cube_pos and processes both camera views.
    """
    print(f"\n=== {iter_name} ===")

    # Load states, actions, and cube positions (already computed in sim)
    agent_pos_file = os.path.join(iter_path, "agent_pos.npy")
    actions_file = os.path.join(iter_path, "actions.npy")
    cube_pos_file = os.path.join(iter_path, "cube_pos.npy")

    if not os.path.exists(agent_pos_file) or not os.path.exists(actions_file):
        print(f"  -> Missing agent_pos.npy or actions.npy, skipping.")
        return

    states = np.load(agent_pos_file).astype(np.float32)
    actions = np.load(actions_file).astype(np.float32)
    
    # Load cube positions if available
    if os.path.exists(cube_pos_file):
        cube_positions = np.load(cube_pos_file).astype(np.float32)
        print(f"  Loaded cube positions: shape {cube_positions.shape}")
    else:
        print(f"  -> Warning: cube_pos.npy not found, creating zeros")
        cube_positions = np.zeros((len(states), CUBE_DIM), dtype=np.float32)

    num_frames = len(states)
    print(f"  Total frames: {num_frames}")

    # Validate dimensions
    if states.shape[1] != STATE_DIM or actions.shape[1] != ACTION_DIM:
        print(f"  -> Unexpected state/action dimensions, skipping.")
        print(f"     Expected: states={STATE_DIM}, actions={ACTION_DIM}")
        print(f"     Got: states={states.shape[1]}, actions={actions.shape[1]}")
        return
    
    if cube_positions.shape[0] != num_frames:
        print(f"  -> Cube positions length mismatch, skipping.")
        print(f"     Expected: {num_frames}, Got: {cube_positions.shape[0]}")
        return
    
    if cube_positions.shape[1] != CUBE_DIM:
        print(f"  -> Unexpected cube dimension, skipping.")
        print(f"     Expected: {CUBE_DIM}, Got: {cube_positions.shape[1]}")
        return

    # Create output directories
    output_dirs = create_output_dirs(iter_name)

    # Process both cameras
    for camera in ["third_person", "wrist"]:
        process_camera(iter_path, camera, num_frames, output_dirs)

    # --- Filter zero-action frames ---
    nonzero_idx = np.any(actions != 0, axis=1)
    removed_frames = np.where(~nonzero_idx)[0]

    if len(removed_frames) > 0:
        print(f"  Removing {len(removed_frames)} frames with all-zero actions")
        if len(removed_frames) <= 10:
            print(f"    Frame indices: {removed_frames.tolist()}")

        states = states[nonzero_idx]
        actions = actions[nonzero_idx]
        cube_positions = cube_positions[nonzero_idx]

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

    # Save states, actions, and cube positions as text files
    states_file = os.path.join(output_dirs["states"], f"{iter_name}.txt")
    actions_file_out = os.path.join(output_dirs["actions"], f"{iter_name}.txt")
    cube_file_out = os.path.join(output_dirs["cube_pos"], f"{iter_name}.txt")
    
    np.savetxt(states_file, states, fmt="%.6f")
    np.savetxt(actions_file_out, actions, fmt="%.6f")
    np.savetxt(cube_file_out, cube_positions, fmt="%.6f")

    print(f"  ✓ Saved {len(states)} frames (after removing zero-action frames):")
    print(f"    States  -> {states_file} (shape: {states.shape})")
    print(f"    Actions -> {actions_file_out} (shape: {actions.shape})")
    print(f"    Cube    -> {cube_file_out} (shape: {cube_positions.shape})")


def process_camera(iter_path, camera, num_frames, output_dirs):
    """Process one camera view: copy RGB, copy/process PLY, save extrinsics."""
    camera_rgb_dir = os.path.join(iter_path, camera, "rgb")
    camera_pcd_dir = os.path.join(iter_path, camera, "pcd")
    camera_poses_dir = os.path.join(iter_path, "camera_poses")

    if not os.path.exists(camera_rgb_dir) or not os.path.exists(camera_pcd_dir):
        print(f"  -> Missing {camera} data, skipping.")
        return

    print(f"  Processing {camera} camera...")

    prefix = "tp" if camera == "third_person" else "wr"
    extrinsics_list = []
    processed_frames = 0

    for i in range(num_frames):
        rgb_file = os.path.join(camera_rgb_dir, f"{prefix}_rgb_{i:04d}.png")
        # MODIFIED: Target .ply instead of .npy
        ply_src_file = os.path.join(camera_pcd_dir, f"{prefix}_pcd_{i:04d}.ply")
        pose_file = os.path.join(camera_poses_dir, f"pose_{i:04d}.json")

        # Copy RGB
        if os.path.exists(rgb_file):
            out_rgb_file = os.path.join(output_dirs[f"{camera}_rgb"], f"rgb_{i:05d}.png")
            copy2(rgb_file, out_rgb_file)
            processed_frames += 1

        # Process .ply point cloud
        if os.path.exists(ply_src_file):
            try:
                # Load the already saved ply
                pcd = load_ply_pointcloud(ply_src_file)
                if pcd is not None:
                    out_ply_file = os.path.join(output_dirs[f"{camera}_pc"], f"cloud_{i:05d}.ply")
                    # We save as binary for efficiency in training
                    o3d.io.write_point_cloud(out_ply_file, pcd, write_ascii=False)
            except Exception as e:
                print(f"    Warning: Failed to process PLY frame {i}: {e}")

        # Load camera extrinsics
        if os.path.exists(pose_file):
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
                camera_key = f"{camera}_camera"
                extr_matrix = np.array(pose_data[camera_key]["extrinsics_matrix"], dtype=np.float32)
                extrinsics_list.append(extr_matrix.flatten())
        else:
            extrinsics_list.append(np.zeros(16, dtype=np.float32))


    # Save camera extrinsics
    camera_extrinsics = np.vstack(extrinsics_list).astype(np.float32)
    extr_file = os.path.join(output_dirs[f"{camera}_extrinsics"], f"{iter_name}.txt")
    np.savetxt(extr_file, camera_extrinsics, fmt="%.6f")

    print(f"    ✓ Processed {processed_frames} frames")
    print(f"      RGB         -> {output_dirs[f'{camera}_rgb']}")
    print(f"      Point Cloud -> {output_dirs[f'{camera}_pc']}")
    print(f"      Extrinsics  -> {extr_file} (shape: {camera_extrinsics.shape})")


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

    # States, actions, and cube positions (shared)
    dirs["states"] = os.path.join(OUTPUT_ROOT, "states")
    dirs["actions"] = os.path.join(OUTPUT_ROOT, "actions")
    dirs["cube_pos"] = os.path.join(OUTPUT_ROOT, "cube_pos")

    # Create all directories
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


# -------------------------
# MAIN ENTRY POINT
# -------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("UR5 Simulation Data Processing")
    print("=" * 60)
    print(f"Input:  {DATA_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print("=" * 60)
    
    iterations = find_iterations(DATA_ROOT)

    if len(iterations) == 0:
        print(f"\nERROR: No iterations found in {DATA_ROOT}")
        exit(1)

    print(f"\nFound {len(iterations)} trajectories to process\n")

    total_success = 0
    total_failed = 0
    
    for iter_name, iter_path in iterations:
        try:
            process_iteration(iter_name, iter_path)
            total_success += 1
        except Exception as e:
            print(f"\n❌ Error processing {iter_name}: {e}")
            import traceback
            traceback.print_exc()
            total_failed += 1
            continue

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {total_success} trajectories")
    if total_failed > 0:
        print(f"Failed: {total_failed} trajectories")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 60)
    
