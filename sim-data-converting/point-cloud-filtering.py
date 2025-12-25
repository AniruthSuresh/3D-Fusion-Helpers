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


# ---------------------------
# WORKSPACE CROP + FPS
# ---------------------------
def process_point_cloud(pc_xyz, pc_rgb):
    """
    pc_xyz: (N, 3)
    pc_rgb: (N, 3)
    """

    WORK_SPACE = [
        [-0.855, 0.855],  # X (radius)
        [-0.855, 0.855],  # Y (radius)
        [-0.360, 1.190]   # Z (height)
    ]

    mask = (
        (pc_xyz[:, 0] > WORK_SPACE[0][0]) & (pc_xyz[:, 0] < WORK_SPACE[0][1]) &
        (pc_xyz[:, 1] > WORK_SPACE[1][0]) & (pc_xyz[:, 1] < WORK_SPACE[1][1]) &
        (pc_xyz[:, 2] > WORK_SPACE[2][0]) & (pc_xyz[:, 2] < WORK_SPACE[2][1])
    )

    pc_xyz = pc_xyz[mask]
    pc_rgb = pc_rgb[mask]

    print(f" → After crop: {pc_xyz.shape[0]} points")

    # FPS
    pc_xyz_fps, idx = farthest_point_sampling(pc_xyz, num_points=2500, use_cuda=False)
    pc_rgb_fps = pc_rgb[idx]

    print(f" → After FPS: {pc_xyz_fps.shape[0]} points")

    return pc_xyz_fps, pc_rgb_fps


# ---------------------------
# SAVE PLY
# ---------------------------
def save_ply(path, xyz, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


# ---------------------------
# MAIN BATCH PROCESSOR
# ---------------------------
INPUT_ROOT = "./processed-sim-data"
OUTPUT_ROOT = "./filtered-pc"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

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

            # Filter
            xyz_f, rgb_f = process_point_cloud(pc_xyz, pc_rgb)

            # Save
            save_ply(out_path, xyz_f, rgb_f)

        print(f"  Saved → {out_folder}/")

print("\n✔ Done. All filtered clouds saved in ./filtered-pc/")
