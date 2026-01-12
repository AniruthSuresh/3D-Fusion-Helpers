import open3d as o3d

# Path to your .ply file
# ply_path = "./dataset/iter_0000/third_person/pcd/tp_pcd_world_0004.ply"
# ply_path = "./dataset/iter_0000/wrist/pcd/wr_pcd_world_0004.ply"
ply_path = "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/ur5-object-picking/dataset/iter_0000/third_person/overlay_red_cube/tp_overlay_0003.ply"

# Load point cloud
pcd = o3d.io.read_point_cloud(ply_path)

# Print basic info
print(pcd)

# Visualize
o3d.visualization.draw_geometries([pcd])

