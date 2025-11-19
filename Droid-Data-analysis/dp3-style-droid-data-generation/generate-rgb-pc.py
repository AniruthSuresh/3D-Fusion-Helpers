import pyzed.sl as sl
import os
import cv2
import numpy as np
import open3d as o3d

# Paths
root_folder = "/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/dp3-style-droid-data-generation/dp3_style_data"
img_root = "/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/dp3-style-droid-data-generation/final-master-data/img"
pc_root = "/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/dp3-style-droid-data-generation/final-master-data/point_cloud"

# Subfolders to skip
skip_subfolders = {"4", "6"}

# Iterate over all numbered subfolders
for subfolder in sorted(os.listdir(root_folder)):
    if subfolder in skip_subfolders:
        print(f"Skipping folder {subfolder}")
        continue

    subfolder_path = os.path.join(root_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # Find all .svo files in this subfolder
    svo_files = [f for f in os.listdir(subfolder_path) if f.endswith(".svo")]
    if not svo_files:
        print(f"No SVO files in folder {subfolder}")
        continue

    # Create output folders for this subfolder (flattened)
    out_img_folder = os.path.join(img_root, subfolder)
    out_pc_folder = os.path.join(pc_root, subfolder)
    os.makedirs(out_img_folder, exist_ok=True)
    os.makedirs(out_pc_folder, exist_ok=True)

    frame_idx = 0  # single counter across all SVOs in this subfolder

    # Process each SVO
    for svo_file in svo_files:
        svo_path = os.path.join(subfolder_path, svo_file)
        print(f"\n--- Processing {subfolder}/{svo_file} ---")

        # Initialize ZED
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_path)
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Failed to open SVO:", err)
            continue

        runtime_params = sl.RuntimeParameters()
        right_image = sl.Mat()
        point_cloud = sl.Mat()

        # Grab frames until end of SVO
        while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # --- Right Image ---
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            right_rgb = right_image.get_data()
            frame_img_path = os.path.join(out_img_folder, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(frame_img_path, right_rgb)

            # --- Point Cloud ---
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            cloud_data = point_cloud.get_data()  # (H, W, 4)
            xyz = cloud_data[:, :, :3].reshape(-1, 3)
            rgba = cloud_data[:, :, 3].reshape(-1)

            # Remove invalid points
            valid = np.isfinite(xyz).all(axis=1)
            xyz = xyz[valid]
            rgba = rgba[valid]

            # Convert RGBA float to RGB uint8
            rgba_uint32 = rgba.view(np.uint32)
            r = ((rgba_uint32 >> 0) & 0xFF).astype(np.float32) / 255.0
            g = ((rgba_uint32 >> 8) & 0xFF).astype(np.float32) / 255.0
            b = ((rgba_uint32 >> 16) & 0xFF).astype(np.float32) / 255.0
            colors = np.stack((r, g, b), axis=-1)

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            frame_pc_path = os.path.join(out_pc_folder, f"frame_{frame_idx:04d}.ply")
            o3d.io.write_point_cloud(frame_pc_path, pcd)

            print(f"Saved frame {frame_idx} → image + point cloud")
            frame_idx += 1

        zed.close()
        print(f"Finished processing {svo_file}, total frames so far: {frame_idx}")

print("\nAll SVO files processed. Right images and point clouds saved in flattened structure.")
