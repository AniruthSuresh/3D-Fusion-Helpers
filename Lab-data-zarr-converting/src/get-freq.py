import rosbag2_py
import numpy as np
from rosidl_runtime_py.utilities import get_message

bag_path = "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/Lab-data-zarr-converting/data/sync_data_bag_21/sync_data_bag_21_0.db3"

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
converter_options = rosbag2_py.ConverterOptions("", "")
reader.open(storage_options, converter_options)

# get list of topics
topics = {}
info = reader.get_all_topics_and_types()
for t in info:
    topics[t.name] = []

# read all messages & store timestamps per topic
while reader.has_next():
    topic, data, ts = reader.read_next()
    topics[topic].append(ts)

# compute frequency
print("\n==== Topic Frequencies ====\n")
for topic, ts_list in topics.items():
    if len(ts_list) < 2:
        print(f"{topic}: <not enough data>")
        continue

    ts = np.array(ts_list, dtype=np.int64)
    ts = ts / 1e9  # convert ns → seconds
    dt = np.diff(ts)

    freq = 1.0 / np.mean(dt)
    print(f"{topic}: {freq:.2f} Hz  (samples: {len(ts_list)})")
