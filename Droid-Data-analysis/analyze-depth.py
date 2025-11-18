import pyzed.sl as sl
import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt

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
    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Depth in meters

    left_img = left_image.get_data()
    right_img = right_image.get_data()
    depth = depth_map.get_data()

    # Clean NaNs and Infs
    depth_valid = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero_depth = depth_valid[np.nonzero(depth_valid)]

    min_depth = np.min(nonzero_depth)
    max_depth = np.max(nonzero_depth)
    print(f"\nRaw Depth Range: min = {min_depth:.3f} m, max = {max_depth:.3f} m")

    # Convert to millimeters and clip to 16-bit range
    depth_mm = np.clip(depth_valid * 1000.0, 0, 65535).astype(np.uint16)
    min_mm = np.min(depth_mm[np.nonzero(depth_mm)])
    max_mm = np.max(depth_mm)
    print(f"Scaled Depth Range: min = {min_mm} mm, max = {max_mm} mm")

    # Save 16-bit scaled depth map
    depth_raw_path = os.path.join(save_dir, "depth_raw_mm.png")
    cv2.imwrite(depth_raw_path, depth_mm)
    print(f"Saved raw scaled depth (mm) as 16-bit PNG to: {depth_raw_path}")

    # -----------------------------------
    # Visualization: raw depth vs scaled depth
    # -----------------------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(depth_valid, cmap='viridis')
    plt.title("Depth (meters, float32)")
    plt.colorbar(im1, fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(depth_mm, cmap='viridis')
    plt.title("Depth (millimeters, uint16)")
    plt.colorbar(im2, fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # -----------------------------------
    # Save left/right images
    # -----------------------------------
    left_path = os.path.join(save_dir, "left.png")
    right_path = os.path.join(save_dir, "right.png")
    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)
    print(f"Saved left image to: {left_path}")
    print(f"Saved right image to: {right_path}")

    # -----------------------------------
    # Colored visualization of depth (0–5 m)
    # -----------------------------------
    depth_vis = np.clip(depth_valid, 0, 5)
    depth_vis = (depth_vis / np.max(depth_vis) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_color_path = os.path.join(save_dir, "depth_colored.png")
    cv2.imwrite(depth_color_path, depth_color)
    print(f"Saved colored depth map to: {depth_color_path}")

    # -----------------------------------
    # Retrieve and save point cloud
    # -----------------------------------
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
    cloud = point_cloud.get_data()  # shape (H, W, 4)
    xyz = cloud[:, :, :3].reshape(-1, 3)
    rgba = cloud[:, :, 3].reshape(-1)

    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    rgba = rgba[valid]

    rgba_uint8 = rgba.view(np.uint32)
    r = ((rgba_uint8 >> 0) & 0xFF).astype(np.float32) / 255.0
    g = ((rgba_uint8 >> 8) & 0xFF).astype(np.float32) / 255.0
    b = ((rgba_uint8 >> 16) & 0xFF).astype(np.float32) / 255.0
    colors = np.stack((r, g, b), axis=-1)

    print(f"\nExtracted {xyz.shape[0]} valid 3D points with color")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd_path = os.path.join(save_dir, "point_cloud_colored.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved colored point cloud to: {pcd_path}")

else:
    print("Failed to grab frame from SVO")

zed.close()
print("\nClosed ZED camera.")
