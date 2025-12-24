import open3d as o3d

# Path to your .ply file
ply_path = "./dataset/iter_0000/wrist/pcd/wr_pcd_0150.ply"

# Load point cloud
pcd = o3d.io.read_point_cloud(ply_path)

# Print basic info
print(pcd)

# Visualize
o3d.visualization.draw_geometries([pcd])
