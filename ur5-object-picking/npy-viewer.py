import numpy as np
import open3d as o3d

# Path to .npy point cloud
npy_path = "./dataset/iter_0000/third_person/pcd/tp_pcd_world_0004.npy"

# Load numpy array (N, 3)
points = np.load(npy_path)

print("Loaded points shape:", points.shape)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Visualize
o3d.visualization.draw_geometries([pcd])
