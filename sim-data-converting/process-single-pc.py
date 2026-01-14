import numpy as np
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops


# ---------------------------
# FARTHEST POINT SAMPLING
# ---------------------------
def farthest_point_sampling(points, num_points=6000, use_cuda=False):
    K = [num_points]

    pc = torch.from_numpy(points).float()
    if use_cuda:
        pc = pc.cuda()

    sampled, idx = torch3d_ops.sample_farthest_points(points=pc.unsqueeze(0), K=K)

    sampled = sampled.squeeze(0).cpu().numpy()
    idx = idx.squeeze(0).cpu().numpy()

    return sampled, idx


# ---------------------------
# WORKSPACE BOUNDS (mean ± n_std)
# ---------------------------
def compute_workspace_bounds(pc_xyz, n_std=2):
    if pc_xyz.shape[0] == 0:
        return None

    mean = pc_xyz.mean(axis=0)
    std = pc_xyz.std(axis=0)

    workspace = [
        [mean[0] - n_std * std[0], mean[0] + n_std * std[0]],  # X
        [mean[1] - n_std * std[1], mean[1] + n_std * std[1]],  # Y
        [mean[2] - n_std * std[2], mean[2] + n_std * std[2]],  # Z
    ]

    print(f"Workspace bounds (mean ± {n_std}*std): {workspace}")
    return workspace


# ---------------------------
# CROP TO WORKSPACE
# ---------------------------
def crop_to_workspace(pc_xyz, pc_rgb, workspace):
    mask = (
        (pc_xyz[:, 0] > workspace[0][0]) & (pc_xyz[:, 0] < workspace[0][1]) &
        (pc_xyz[:, 1] > workspace[1][0]) & (pc_xyz[:, 1] < workspace[1][1]) &
        (pc_xyz[:, 2] > workspace[2][0]) & (pc_xyz[:, 2] < workspace[2][1])
    )
    return pc_xyz[mask], pc_rgb[mask]


# ---------------------------
# PRINT STATS
# ---------------------------
def print_stats(name, pc_xyz):
    if pc_xyz.shape[0] == 0:
        print(f"{name}: EMPTY")
        return

    mins = pc_xyz.min(axis=0)
    maxs = pc_xyz.max(axis=0)
    means = pc_xyz.mean(axis=0)
    stds = pc_xyz.std(axis=0)

    print(f"\n{name} ({pc_xyz.shape[0]} points)")
    print(f"  X: [{mins[0]:.3f}, {maxs[0]:.3f}], mean={means[0]:.3f}, std={stds[0]:.3f}")
    print(f"  Y: [{mins[1]:.3f}, {maxs[1]:.3f}], mean={means[1]:.3f}, std={stds[1]:.3f}")
    print(f"  Z: [{mins[2]:.3f}, {maxs[2]:.3f}], mean={means[2]:.3f}, std={stds[2]:.3f}")


# ---------------------------
# WORKSPACE BOX FOR VISUALIZATION
# ---------------------------
def create_workspace_box(bounds, color=[1, 0, 0]):
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    corners = [
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ]

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


# ---------------------------
# MAIN
# ---------------------------
PLY_PATH = "/home/varun-edachali/Research/RRC/policy/data/3D-Fusion-Helpers/sim-data-converting/tp_pcd_0010.ply"
NUM_FPS_POINTS = 6000

# Load
pcd = o3d.io.read_point_cloud(PLY_PATH)
pc_xyz = np.asarray(pcd.points)
pc_rgb = np.asarray(pcd.colors)

print_stats("Raw", pc_xyz)

# Compute workspace and crop
workspace = compute_workspace_bounds(pc_xyz, n_std=2)
pc_xyz_cropped, pc_rgb_cropped = crop_to_workspace(pc_xyz, pc_rgb, workspace)

print_stats("After crop", pc_xyz_cropped)

# FPS
if pc_xyz_cropped.shape[0] > 0:
    num_fps = min(NUM_FPS_POINTS, pc_xyz_cropped.shape[0])
    pc_xyz_fps, idx = farthest_point_sampling(pc_xyz_cropped, num_points=num_fps)
    pc_rgb_fps = pc_rgb_cropped[idx]
    print_stats("After FPS", pc_xyz_fps)
else:
    print("Empty after crop - skipping FPS")
    pc_xyz_fps, pc_rgb_fps = pc_xyz_cropped, pc_rgb_cropped

# Visualize
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
workspace_box = create_workspace_box(workspace, color=[1, 0, 0])

# Raw point cloud
pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(pc_xyz)
pcd_raw.colors = o3d.utility.Vector3dVector(pc_rgb)

print("\n1. Showing RAW point cloud with workspace box")
o3d.visualization.draw_geometries([pcd_raw, axes, workspace_box], window_name="Raw", width=1024, height=768)

# Processed point cloud
pcd_processed = o3d.geometry.PointCloud()
pcd_processed.points = o3d.utility.Vector3dVector(pc_xyz_fps)
pcd_processed.colors = o3d.utility.Vector3dVector(pc_rgb_fps)

print("2. Showing PROCESSED point cloud (cropped + FPS)")
o3d.visualization.draw_geometries([pcd_processed, axes], window_name="Processed", width=1024, height=768)
