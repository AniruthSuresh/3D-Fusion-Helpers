import open3d as o3d
import numpy as np

ply_path = "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/ur5-object-picking/dataset/iter_0000/third_person/pcd/tp_pcd_0000.ply"

# Load point cloud
pcd = o3d.io.read_point_cloud(ply_path)
print(pcd)

############################
# 1. RGB visualization
############################
print("Showing RGB point cloud...")
o3d.visualization.draw_geometries(
    [pcd],
    window_name="RGB Point Cloud"
)

############################
# 2. Depth-only visualization
############################
print("Showing depth-colored point cloud...")

# Copy point cloud so RGB is preserved
pcd_depth = o3d.geometry.PointCloud(pcd)

# Extract Z (depth)
points = np.asarray(pcd_depth.points)
z = points[:, 2]

# Normalize depth to [0, 1]
z_min, z_max = z.min(), z.max()
z_norm = (z - z_min) / (z_max - z_min + 1e-8)

# Option A: Grayscale depth (near = bright, far = dark)
colors = np.stack([z_norm, z_norm, z_norm], axis=1)

# Option B (uncomment if you want inverted depth)
colors = 1.0 - colors

pcd_depth.colors = o3d.utility.Vector3dVector(colors)

# Custom visualizer for dark background
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Depth Point Cloud")
vis.add_geometry(pcd_depth)

render_option = vis.get_render_option()
render_option.background_color = np.array([0.05, 0.05, 0.05])  # dark gray
render_option.point_size = 2.0

vis.run()
vis.destroy_window()
