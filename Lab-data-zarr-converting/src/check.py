import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import numpy as np


def extract_gripper_data(db3_path):
    """
    Extracts gripper state and command timestamps from the bag.
    Topics:
        /gripper/state  -> float64
        /gripper/command -> int32 (master timestamps)
    Returns:
        grip_state_arr: (N_state,) float32
        grip_state_ts : (N_state,) int64
        grip_cmd_arr  : (N_cmd,) float32
        grip_cmd_ts   : (N_cmd,) int64
    """

    GRIP_STATE = "/gripper/state"
    GRIP_CMD   = "/gripper/command"

    # --- Setup ROS2 reader ---
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    # --- Get message types ---
    topics = reader.get_all_topics_and_types()
    grip_state_type = None
    grip_cmd_type = None
    for t in topics:
        if t.name == GRIP_STATE:
            grip_state_type = get_message(t.type)
        if t.name == GRIP_CMD:
            grip_cmd_type = get_message(t.type)

    grip_state_arr = []
    grip_state_ts  = []
    grip_cmd_arr   = []
    grip_cmd_ts    = []

    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic == GRIP_STATE and grip_state_type:
            msg = deserialize_message(data, grip_state_type)
            grip_state_arr.append(float(msg.data))
            grip_state_ts.append(ts)
        elif topic == GRIP_CMD and grip_cmd_type:
            msg = deserialize_message(data, grip_cmd_type)
            grip_cmd_arr.append(float(msg.data))
            grip_cmd_ts.append(ts)

    grip_state_arr = np.array(grip_state_arr, dtype=np.float32)
    grip_state_ts  = np.array(grip_state_ts, dtype=np.int64)
    grip_cmd_arr   = np.array(grip_cmd_arr, dtype=np.float32)
    grip_cmd_ts    = np.array(grip_cmd_ts, dtype=np.int64)

    print(f"✔️ Extracted {len(grip_state_arr)} gripper state messages")
    print(f"✔️ Extracted {len(grip_cmd_arr)} gripper command messages")

    return grip_state_arr, grip_state_ts, grip_cmd_arr, grip_cmd_ts


# Example usage:
grip_state_arr, grip_state_ts, grip_cmd_arr, grip_cmd_ts = extract_gripper_data(
    "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/Lab-data-zarr-converting/data/sync_data_bag_21/sync_data_bag_21_0.db3"
)

print("Sample grip_state:", grip_state_arr[:5])
print("Sample grip_cmd  :", grip_cmd_arr[:5])

unique_grip_cmd = np.unique(grip_cmd_arr)
print("Unique gripper command values:", unique_grip_cmd)
print("Count of unique values:", len(unique_grip_cmd))