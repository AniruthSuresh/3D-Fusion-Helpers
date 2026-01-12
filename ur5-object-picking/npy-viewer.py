import numpy as np
import open3d as o3d

# Paths to point clouds
npy_path_1 = "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/ur5-object-picking/dataset/iter_0000/third_person/depth/tp_depth_0053.npy"
npy_path_2 = "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/ur5-object-picking/dataset/iter_0000/cube_poses/cube_pcd_0053.npy"

# Load point clouds
points1 = np.load(npy_path_1)
points2 = np.load(npy_path_2)

print("PCD 1 shape:", points1.shape)
print("PCD 2 shape:", points2.shape)

# Create Open3D point clouds
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points1)
pcd1.paint_uniform_color([1, 0, 0])  # Red

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points2)
pcd2.paint_uniform_color([0, 1, 0])  # Green

# Visualize together
o3d.visualization.draw_geometries([pcd1, pcd2])
