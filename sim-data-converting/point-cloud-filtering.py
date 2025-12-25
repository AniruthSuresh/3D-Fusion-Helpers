import os
import numpy as np
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops

# ---------------------------
# FARTHEST POINT SAMPLING
# ---------------------------
def farthest_point_sampling(points, num_points=2500, use_cuda=False):
    K = [num_points]

    pc = torch.from_numpy(points).float()
    if use_cuda:
        pc = pc.cuda()

    sampled, idx = torch3d_ops.sample_farthest_points(
        points=pc.unsqueeze(0), K=K
    )

    sampled = sampled.squeeze(0).cpu().numpy()
    idx = idx.squeeze(0).cpu().numpy()

    return sampled, idx



def compute_workspace_mean_std(pc_xyz, n_std=2):
    """
    Compute workspace bounds using mean ± n_std * std
    pc_xyz: (N,3) numpy array
    n_std: number of standard deviations
    """
    if pc_xyz.shape[0] == 0:
        return None

    mean = pc_xyz.mean(axis=0)
    std = pc_xyz.std(axis=0)

    WORK_SPACE = [
        [mean[0] - n_std * std[0], mean[0] + n_std * std[0]],  # X
        [mean[1] - n_std * std[1], mean[1] + n_std * std[1]],  # Y
        [mean[2] - n_std * std[2], mean[2] + n_std * std[2]]   # Z
    ]

    print(f"Computed workspace (mean ± {n_std}*std): {WORK_SPACE}")
    return WORK_SPACE

# ---------------------------
# WORKSPACE CROP + FPS
# ---------------------------
def process_point_cloud(pc_xyz, pc_rgb, visualize=True):
    """
    pc_xyz: (N, 3)
    pc_rgb: (N, 3)
    """

    # -------- SHOW BEFORE --------
    if visualize:
        pcd_before = o3d.geometry.PointCloud()
        pcd_before.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd_before.colors = o3d.utility.Vector3dVector(pc_rgb)

        print("Showing RAW point cloud (close window to continue)")
        o3d.visualization.draw_geometries([pcd_before])

    WORK_SPACE = compute_workspace_mean_std(pc_xyz, n_std=2)
    
    mask = (
        (pc_xyz[:, 0] > WORK_SPACE[0][0]) & (pc_xyz[:, 0] < WORK_SPACE[0][1]) &
        (pc_xyz[:, 1] > WORK_SPACE[1][0]) & (pc_xyz[:, 1] < WORK_SPACE[1][1]) &
        (pc_xyz[:, 2] > WORK_SPACE[2][0]) & (pc_xyz[:, 2] < WORK_SPACE[2][1])
    )

    pc_xyz = pc_xyz[mask]
    pc_rgb = pc_rgb[mask]

    print(f" → After crop: {pc_xyz.shape[0]} points")

    if pc_xyz.shape[0] == 0:
        print(" Empty after crop — skipping FPS")
        return pc_xyz, pc_rgb

    num_fps = min(2500, pc_xyz.shape[0])
    pc_xyz_fps, idx = farthest_point_sampling(pc_xyz, num_points=num_fps)
    pc_rgb_fps = pc_rgb[idx]

    print(f" → After FPS: {pc_xyz_fps.shape[0]} points")

    # -------- SHOW AFTER --------
    if visualize:
        pcd_after = o3d.geometry.PointCloud()
        pcd_after.points = o3d.utility.Vector3dVector(pc_xyz_fps)
        pcd_after.colors = o3d.utility.Vector3dVector(pc_rgb_fps)

        print("Showing PROCESSED point cloud (close window to continue)")
        o3d.visualization.draw_geometries([pcd_after])

    return pc_xyz_fps, pc_rgb_fps




# ---------------------------
# SAVE PLY
# ---------------------------
def save_ply(path, xyz, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


def print_xyz_distribution(name, pc_xyz):
    if pc_xyz.shape[0] == 0:
        print(f"{name}: EMPTY")
        return

    mins = pc_xyz.min(axis=0)
    maxs = pc_xyz.max(axis=0)
    means = pc_xyz.mean(axis=0)
    stds = pc_xyz.std(axis=0)

    p1 = np.percentile(pc_xyz, 1, axis=0)
    p99 = np.percentile(pc_xyz, 99, axis=0)

    print(f"\n{name} point cloud stats")
    print(f"  Num points: {pc_xyz.shape[0]}")
    print(f"  X: min {mins[0]:.3f}, max {maxs[0]:.3f}, mean {means[0]:.3f}, std {stds[0]:.3f}, p1 {p1[0]:.3f}, p99 {p99[0]:.3f}")
    print(f"  Y: min {mins[1]:.3f}, max {maxs[1]:.3f}, mean {means[1]:.3f}, std {stds[1]:.3f}, p1 {p1[1]:.3f}, p99 {p99[1]:.3f}")
    print(f"  Z: min {mins[2]:.3f}, max {maxs[2]:.3f}, mean {means[2]:.3f}, std {stds[2]:.3f}, p1 {p1[2]:.3f}, p99 {p99[2]:.3f}")




# ---------------------------
# MAIN BATCH PROCESSOR
# ---------------------------
INPUT_ROOT = "./processed-sim-data/"
OUTPUT_ROOT = "./filtered-pc"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

VIZ_ONCE = True

# Process both cameras
for camera in ["third_person_pc", "wrist_pc"]:
    camera_root = os.path.join(INPUT_ROOT, camera)

    if not os.path.exists(camera_root):
        print(f"Skipping {camera} (not found)")
        continue

    camera_name = "third_person" if camera == "third_person_pc" else "wrist"
    camera_out_root = os.path.join(OUTPUT_ROOT, camera_name)
    os.makedirs(camera_out_root, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {camera_name} camera")
    print(f"{'='*60}")

    # Process each iteration
    for iter_folder in sorted(os.listdir(camera_root)):
        in_folder = os.path.join(camera_root, iter_folder)
        if not os.path.isdir(in_folder):
            continue

        out_folder = os.path.join(camera_out_root, iter_folder)
        os.makedirs(out_folder, exist_ok=True)

        ply_files = sorted([f for f in os.listdir(in_folder) if f.endswith(".ply")])

        print(f"\n=== {iter_folder}: {len(ply_files)} clouds ===")

        for fn in ply_files:
            in_path = os.path.join(in_folder, fn)
            out_path = os.path.join(out_folder, fn.replace(".ply", "_filtered.ply"))

            # Load
            pcd = o3d.io.read_point_cloud(in_path)
            pc_xyz = np.asarray(pcd.points)
            pc_rgb = np.asarray(pcd.colors)

            print_xyz_distribution("Before crop", pc_xyz)

            # Filter
            xyz_f, rgb_f = process_point_cloud(
                pc_xyz,
                pc_rgb,
                visualize=VIZ_ONCE
            )

            VIZ_ONCE = False  

            save_ply(out_path, xyz_f, rgb_f)

        print(f"  Saved → {out_folder}/")

print("\n✔ Done. All filtered clouds saved in ./filtered-pc/")
