# import open3d as o3d

# # Path to your .ply file
# # ply_path = "./dataset/iter_0000/third_person/pcd/tp_pcd_world_0004.ply"
# ply_path = "./dataset/iter_0000/third_person/pcd/tp_pcd_cam_0003.ply"

# # Load point cloud
# pcd = o3d.io.read_point_cloud(ply_path)

# # Print basic info
# print(pcd)

# # Visualize
# o3d.visualization.draw_geometries([pcd])



import open3d as o3d
import numpy as np

pc_cam = o3d.io.read_point_cloud(
    "./dataset/iter_0000/third_person/pcd/tp_pcd_cam_0003.ply"
)

pc_world = o3d.io.read_point_cloud(
    "./dataset/iter_0000/third_person/pcd/tp_pcd_world_0004.ply"
)

# Color point clouds
pc_cam.paint_uniform_color([1, 0, 0])    # RED → camera frame
pc_world.paint_uniform_color([0, 1, 0])  # GREEN → world frame

# ---------- Coordinate frames ----------
# World frame at origin
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.4, origin=[0, 0, 0]
)

# Camera frame at camera cloud center
cam_center = np.asarray(pc_cam.points).mean(axis=0)
cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2, origin=cam_center
)

o3d.visualization.draw_geometries([
    pc_cam,
    pc_world,
    world_frame,
    cam_frame
])
