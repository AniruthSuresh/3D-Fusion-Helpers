import open3d as o3d
import os
import cv2
import numpy as np

PCD_DIR = "./dataset/iter_0000/pcd"
OUTPUT_VIDEO = "pointcloud.mp4"

WIDTH, HEIGHT = 1280, 720
FPS = 10

# ---------- Video writer ----------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))

# ---------- Open3D visualizer ----------
vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name="pcd",
    width=WIDTH,
    height=HEIGHT,
    visible=False,   # headless-friendly
)

pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

for ply_file in sorted(os.listdir(PCD_DIR)):
    if not ply_file.endswith(".ply"):
        continue

    # Load point cloud
    pcd = o3d.io.read_point_cloud(os.path.join(PCD_DIR, ply_file))
    vis.clear_geometries()
    vis.add_geometry(pcd)

    # Render
    vis.poll_events()
    vis.update_renderer()

    # Capture frame
    frame = np.asarray(vis.capture_screen_float_buffer())
    frame = (frame * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    video.write(frame)

vis.destroy_window()
video.release()

print(f"Saved video → {OUTPUT_VIDEO}")
