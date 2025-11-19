import pyzed.sl as sl
import os
import cv2
import numpy as np

# Paths
svo_folder = "/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/dp3-style-droid-data-generation/dp3_style_data/2/SVO"

# Find all SVO files
svo_files = [f for f in os.listdir(svo_folder) if f.endswith(".svo")]
if not svo_files:
    print("No SVO files found in folder!")
    exit(1)

# Iterate over each SVO
for svo_file in svo_files:
    svo_path = os.path.join(svo_folder, svo_file)
    print(f"\n--- Processing {svo_file} ---")

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
    left_image = sl.Mat()
    right_image = sl.Mat()
    pose = sl.Pose()

    frame_count = 0
    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS and frame_count < 1:  # only grab first frame for preview
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
        cv2.imshow(f"{svo_file} - Left | Right", combined)
        print("Press any key on image window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        frame_count += 1

    zed.close()
    print(f"Closed {svo_file}")

print("\nAll SVO files processed.")
