import os
import numpy as np
import pybullet as p
import pybullet_data
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import time


# ------------------------------
# CONFIG
# ------------------------------
DB3_PATH = "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/Lab-data-zarr-converting/data/sync_data_bag_21/sync_data_bag_21_0.db3"            # <<< SET THIS
JOINT_TOPIC = "/ufactory/joint_states"
OUTPUT_TXT = "eef_trajectory.txt"

URDF_PATH = "/home/aniruth/Desktop/RRC/XARM7/xArm-Python-SDK/example/wrapper/xarm7/Follow_DROID/Franka_arm/Droid Mask Extraction/Lite-6-data-collection/lite-6-updated-urdf/lite_6_new.urdf"
EEF_INDEX = 6     # end-effector link


# ------------------------------
# READ JOINT ANGLES FROM DB3
# ------------------------------
def read_joint_trajectory(db3_path):
    print("Reading joint states from:", db3_path)

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=db3_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    # find joint_states message type
    joint_type = None
    topics = reader.get_all_topics_and_types()
    for t in topics:
        if t.name == JOINT_TOPIC:
            joint_type = get_message(t.type)

    if joint_type is None:
        raise RuntimeError("No /joint_states topic found in bag!")

    joint_traj = []

    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic == JOINT_TOPIC:
            msg = deserialize_message(data, joint_type)
            positions = list(msg.position)
            joint_traj.append(positions)

    print(f"Total frames read: {len(joint_traj)}")
    return np.array(joint_traj)


# ------------------------------
# SIMULATE AND RECORD EEF POSE
# ------------------------------
def simulate_and_record(joint_traj):

    print("Starting PyBullet simulation...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    robot = p.loadURDF(URDF_PATH, useFixedBase=True)

    results = []

    for idx, joints in enumerate(joint_traj):
        time.sleep(0.05)    
        # time.sleep(0.5) 
        # set joint positions directly
        for j in range(len(joints)):
            p.resetJointState(robot, j, joints[j])

        p.stepSimulation()

        # get EEF pose
        state = p.getLinkState(robot, EEF_INDEX)
        pos = state[0]
        quat = state[1]

        results.append(list(pos) + list(quat))

    p.disconnect()

    return np.array(results)


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    joint_traj = read_joint_trajectory(DB3_PATH)
    eef_traj = simulate_and_record(joint_traj)

    np.savetxt(OUTPUT_TXT, eef_traj, fmt="%.6f")
    print("Saved EEF trajectory →", OUTPUT_TXT)
