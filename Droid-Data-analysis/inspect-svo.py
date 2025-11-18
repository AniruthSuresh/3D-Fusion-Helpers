import pyzed.sl as sl
import numpy as np
import open3d as o3d
import cv2
import os

# Paths
svo_path = "/home/aniruth/Desktop/RRC/point-cloud-droid-data/droid/svo/28451778.svo"
save_dir = "/home/aniruth/Desktop/RRC/point-cloud-droid-data/droid/results"
os.makedirs(save_dir, exist_ok=True)

# Initialize ZED
zed = sl.Camera()

# Configuration
init_params = sl.InitParameters()
init_params.set_from_svo_file(svo_path)
init_params.svo_real_time_mode = False
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_mode = sl.DEPTH_MODE.ULTRA

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Failed to open SVO:", err)
    exit(1)


print(f"\nOpened {svo_path}")
info = zed.get_camera_information()
print("Model:", info.camera_model)
print("Resolution:", info.camera_configuration.resolution)
print("FPS:", info.camera_configuration.fps)

runtime_params = sl.RuntimeParameters()
point_cloud = sl.Mat()
left_image = sl.Mat()
right_image = sl.Mat()
depth_map = sl.Mat()

if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    # Retrieve images
    zed.retrieve_image(left_image, sl.VIEW.LEFT)
    zed.retrieve_image(right_image, sl.VIEW.RIGHT)
    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # this

    left_img = left_image.get_data()
    right_img = right_image.get_data()
    depth = depth_map.get_data()

    # Save left/right images
    left_path = os.path.join(save_dir, "left.png")
    right_path = os.path.join(save_dir, "right.png")
    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)
    print(f"Saved left image to: {left_path}")
    print(f"Saved right image to: {right_path}")

    # Normalize depth for visualization (in meters)
    depth_vis = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_vis = np.clip(depth_vis, 0, 5)  # Limit to 5 meters for visualization
    depth_vis = (depth_vis / np.max(depth_vis) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_path = os.path.join(save_dir, "depth_colored.png")
    cv2.imwrite(depth_path, depth_color)
    print(f"Saved depth map to: {depth_path}")

    # Retrieve colored point cloud (XYZRGBA)
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
    cloud = point_cloud.get_data()  # shape (H, W, 4)
    xyz = cloud[:, :, :3].reshape(-1, 3)
    rgba = cloud[:, :, 3].reshape(-1)

    # Remove invalid points
    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    rgba = rgba[valid]

    # Convert RGBA float to RGB uint8
    rgba_uint8 = rgba.view(np.uint32)
    r = ((rgba_uint8 >> 0) & 0xFF).astype(np.float32) / 255.0
    g = ((rgba_uint8 >> 8) & 0xFF).astype(np.float32) / 255.0
    b = ((rgba_uint8 >> 16) & 0xFF).astype(np.float32) / 255.0
    colors = np.stack((r, g, b), axis=-1)

    print(f"\nExtracted {xyz.shape[0]} valid 3D points with color")

    # Create Open3D point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize 3D point cloud
    print("Opening colored 3D viewer... (ESC to exit)")
    o3d.visualization.draw_geometries([pcd], window_name="ZED Colored Point Cloud")

    # Save point cloud
    pcd_path = os.path.join(save_dir, "point_cloud_colored.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved colored point cloud to: {pcd_path}")

else:
    print("Failed to grab frame from SVO")

zed.close()
print("\nClosed ZED camera.")
