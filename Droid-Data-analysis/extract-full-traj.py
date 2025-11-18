import pyzed.sl as sl
import numpy as np
import cv2
import os
import json

# Paths
svo_path = "/home/aniruth/Desktop/RRC/point-cloud-droid-data/droid/svo/28813166.svo"
save_dir = "/home/aniruth/Desktop/RRC/point-cloud-droid-data/droid"

# Subfolders
left_dir = os.path.join(save_dir, "images/left")
right_dir = os.path.join(save_dir, "images/right")
depth_dir = os.path.join(save_dir, "images/depth_mm")   # 16-bit depth
depth_vis_dir = os.path.join(save_dir, "images/depth_vis")  # Visualization
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(depth_vis_dir, exist_ok=True)

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

# ------------------------------
# Save Intrinsics
# ------------------------------
calib = info.camera_configuration.calibration_parameters
left_calib = calib.left_cam

K = [
    [left_calib.fx, 0, left_calib.cx],
    [0, left_calib.fy, left_calib.cy],
    [0, 0, 1]
]

dist_coeffs = list(left_calib.disto)
intrinsics_data = {
    "image_width": info.camera_configuration.resolution.width,
    "image_height": info.camera_configuration.resolution.height,
    "camera_matrix": K,
    "distortion_coefficients": dist_coeffs,
    "baseline_meters": calib.get_camera_baseline(),
    "fov": {
        "horizontal": left_calib.h_fov,
        "vertical": left_calib.v_fov,
        "diagonal": left_calib.d_fov
    }
}

json_path = os.path.join(save_dir, "images", "camera_intrinsics.json")
with open(json_path, "w") as f:
    json.dump(intrinsics_data, f, indent=4)
print(f"\nSaved intrinsics to: {json_path}")

# ------------------------------
# Frame Extraction Loop
# ------------------------------
runtime_params = sl.RuntimeParameters()
left_image = sl.Mat()
right_image = sl.Mat()
depth_map = sl.Mat()

frame_count = 0
total_frames = zed.get_svo_number_of_frames()
print(f"\nTotal frames in SVO: {total_frames}\n")

while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # Retrieve frames
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        zed.retrieve_image(right_image, sl.VIEW.RIGHT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        left_img = left_image.get_data()
        right_img = right_image.get_data()
        depth = depth_map.get_data()

        # Replace NaNs/Infs
        depth_valid = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Save depth in millimeters (16-bit PNG) ---
        depth_mm = np.clip(depth_valid * 1000.0, 0, 65535).astype(np.uint16)
        depth_mm_path = os.path.join(depth_dir, f"{frame_count:05d}.png")
        cv2.imwrite(depth_mm_path, depth_mm)

        # --- Visualization for quick check ---
        depth_vis = np.clip(depth_valid, 0, 5)
        depth_vis = (depth_vis / np.max(depth_vis) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        depth_vis_path = os.path.join(depth_vis_dir, f"{frame_count:05d}.png")
        cv2.imwrite(depth_vis_path, depth_color)

        # Save left/right
        left_path = os.path.join(left_dir, f"{frame_count:05d}.png")
        right_path = os.path.join(right_dir, f"{frame_count:05d}.png")
        cv2.imwrite(left_path, left_img)
        cv2.imwrite(right_path, right_img)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Saved {frame_count}/{total_frames} frames...")

    else:
        break

print(f"\nFinished saving {frame_count} frames (Left, Right, and Depth-mm).")
zed.close()
print("Closed ZED camera.")
