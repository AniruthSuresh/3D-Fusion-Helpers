import pyzed.sl as sl

# Path to your SVO file
svo_path = "/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/dp3-style-droid-data-generation/dp3_style_data/7/20521388.svo"

# Initialize ZED
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.set_from_svo_file(svo_path)
init_params.svo_real_time_mode = False

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Failed to open SVO:", err)
    exit(1)

runtime_params = sl.RuntimeParameters()
frame_count = 0

# Grab frames until end of SVO
while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    frame_count += 1

zed.close()
print(f"Total frames in {svo_path}: {frame_count}")
