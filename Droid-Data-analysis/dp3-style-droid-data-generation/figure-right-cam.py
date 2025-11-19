"""
This script processes all .svo files in numbered subfolders within a specified root directory.

4 & 6 are corresponding to wrist cam weirdly -> so just delete those .. 
"""
import pyzed.sl as sl
import os
import cv2
import numpy as np

# Paths
root_folder = "/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/dp3-style-droid-data-generation/dp3_style_data"

# Iterate over all numbered subfolders
for subfolder in sorted(os.listdir(root_folder)):
    subfolder_path = os.path.join(root_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # Find all .svo files in this subfolder
    svo_files = [f for f in os.listdir(subfolder_path) if f.endswith(".svo")]
    if not svo_files:
        print(f"No SVO files in folder {subfolder}")
        continue

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

        # Enable positional tracking (required to get pose)
        tracking_params = sl.PositionalTrackingParameters()
        zed.enable_positional_tracking(tracking_params)

        runtime_params = sl.RuntimeParameters()
        left_image = sl.Mat()
        right_image = sl.Mat()
        pose = sl.Pose()

        frame_count = 0
        while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS and frame_count < 1:  # only first frame
            # Get extrinsics
            if zed.get_position(pose, sl.REFERENCE_FRAME.WORLD) == sl.ERROR_CODE.SUCCESS:
                extrinsics = pose.get_pose_matrix()
                print("Frame 0 extrinsics (4x4):")
                print(extrinsics)
            else:
                print("Failed to get pose for frame 0")

            # Retrieve left/right images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            left_rgb = left_image.get_data()
            right_rgb = right_image.get_data()

            # Show both images side by side
            combined = np.hstack((left_rgb, right_rgb))
            cv2.imshow(f"{subfolder}/{svo_file} - Left | Right", combined)
            print("Press any key on image window to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            frame_count += 1

        zed.close()
        print(f"Closed {subfolder}/{svo_file}")

print("\nAll SVO files processed.")
