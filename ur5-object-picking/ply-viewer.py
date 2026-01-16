import open3d as o3d
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops

# ---------------------------
# FARTHEST POINT SAMPLING
# ---------------------------
def farthest_point_sampling(points, num_points=1024, use_cuda=False):
    if points.shape[0] < num_points:
        print(f"[WARN] Requested {num_points}, but only {points.shape[0]} points available")
        num_points = points.shape[0]

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
# WORKSPACE COMPUTATION
# ---------------------------
def compute_workspace_bounds(pc_xyz, n_std=2.5):
    mean = pc_xyz.mean(axis=0)
    std = pc_xyz.std(axis=0)

    workspace = [
        [mean[0] - n_std * std[0], mean[0] + n_std * std[0]],
        [mean[1] - n_std * std[1], mean[1] + n_std * std[1]],
        [mean[2] - n_std * std[2], mean[2] + n_std * std[2]],
    ]

    print(f"Workspace bounds (mean ± {n_std}σ): {workspace}")
    return workspace


# ---------------------------
# WORKSPACE CROP + FPS
# ---------------------------
def process_point_cloud(pc_xyz, pc_rgb, num_fps=2500):
    workspace = compute_workspace_bounds(pc_xyz)

    mask = (
        (pc_xyz[:, 0] >= workspace[0][0]) & (pc_xyz[:, 0] <= workspace[0][1]) &
        (pc_xyz[:, 1] >= workspace[1][0]) & (pc_xyz[:, 1] <= workspace[1][1]) &
        (pc_xyz[:, 2] >= workspace[2][0]) & (pc_xyz[:, 2] <= workspace[2][1])
    )

    pc_xyz = pc_xyz[mask]
    pc_rgb = pc_rgb[mask]

    print(f"After workspace crop: {pc_xyz.shape[0]} points")

    pc_xyz_fps, idx = farthest_point_sampling(
        pc_xyz, num_points=num_fps, use_cuda=False
    )
    pc_rgb_fps = pc_rgb[idx]

    print(f"After FPS: {pc_xyz_fps.shape[0]} points")

    return pc_xyz_fps, pc_rgb_fps


# ---------------------------
# VISUALIZATION HELPERS
# ---------------------------
def visualize_rgb(pc_xyz, pc_rgb, title):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)

    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_depth(pc_xyz, title, dark_bg=True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)

    z = pc_xyz[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors = np.stack([z_norm, z_norm, z_norm], axis=1)
    colors = 1.0 - colors  # invert for nicer contrast

    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    if dark_bg:
        opt.background_color = np.array([0.05, 0.05, 0.05])

    vis.run()
    vis.destroy_window()


# ---------------------------
# MAIN
# ---------------------------
ply_path = "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/ur5-object-picking/dataset/iter_0000/third_person/pcd/tp_pcd_0000.ply"

pcd = o3d.io.read_point_cloud(ply_path)
pc_xyz = np.asarray(pcd.points)
pc_rgb = np.asarray(pcd.colors)

print(pcd)

# 1️⃣ Original RGB
visualize_rgb(pc_xyz, pc_rgb, "Original RGB")

# 2️⃣ Original Depth
visualize_depth(pc_xyz, "Original Depth")

# 3️⃣ Workspace filtering + FPS
pc_xyz_fps, pc_rgb_fps = process_point_cloud(pc_xyz, pc_rgb, num_fps=6000)

# 4️⃣ Filtered + Sampled RGB
visualize_rgb(pc_xyz_fps, pc_rgb_fps, "Workspace + FPS RGB")

# 5️⃣ Filtered + Sampled Depth
visualize_depth(pc_xyz_fps, "Workspace + FPS Depth")
