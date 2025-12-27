import open3d as o3d

# Path to your .ply file
# ply_path = "./dataset/iter_0000/third_person/pcd/tp_pcd_world_0004.ply"
# ply_path = "./dataset/iter_0000/wrist/pcd/wr_pcd_world_0004.ply"
ply_path = "./dataset/iter_0000/third_person/pcd/"

# Load point cloud
pcd = o3d.io.read_point_cloud(ply_path)

# Print basic info
print(pcd)

# Visualize
o3d.visualization.draw_geometries([pcd])

