"""
This script is used to extract : 

1. "point_cloud": Array of shape (T, Np, 6), Np is the number of point clouds, 6 denotes [x, y, z, r, g, b]. 
2. "image": Array of shape (T, H, W, 3)
3. "depth": Array of shape (T, H, W)
4. "state": Array of shape (T, Nd), Nd is the action dim of the robot agent, i.e. 22 for our dexhand tasks (6d position of end effector + 16d joint position)
5. "action": Array of shape (T, Nd). We use relative end-effector position control for the robot arm and relative joint-angle position control for the dex hand.    
"""

import os
import h5py
import numpy as np

parent_folder = "/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/dp3-style-droid-data-generation/dp3_style_data/1"  # Change this to your parent folder
output_root = "final-master-data"

# Create main folders
state_root = os.path.join(output_root, "state")
action_root = os.path.join(output_root, "action")
os.makedirs(state_root, exist_ok=True)
os.makedirs(action_root, exist_ok=True)

# Counter for numbering
counter = 1

# Walk through all subfolders
for root, dirs, files in os.walk(parent_folder):
    if "trajectory.h5" in files:
        traj_file = os.path.join(root, "trajectory.h5")

        # Load trajectory.h5
        with h5py.File(traj_file, "r") as f:
            cartesian = f["action/cartesian_position"][:]
            joint = f["action/joint_position"][:]

            # Compute state = cartesian + joint
            state = np.concatenate([cartesian, joint], axis=1)

            # Compute action as difference between states
            action = np.zeros_like(state)
            action[1:] = state[1:] - state[:-1]

            # Save state and action as numbered files
            np.savetxt(os.path.join(state_root, f"{counter}.txt"), state, fmt="%.6f")
            np.savetxt(os.path.join(action_root, f"{counter}.txt"), action, fmt="%.6f")

        print(f"Processed {traj_file} → {counter}.txt")
        counter += 1