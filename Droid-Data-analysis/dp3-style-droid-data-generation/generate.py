"""
This script is used to extract : 

1. "point_cloud": Array of shape (T, Np, 6), Np is the number of point clouds, 6 denotes [x, y, z, r, g, b]. 
2. "image": Array of shape (T, H, W, 3)
3. "depth": Array of shape (T, H, W)
4. "state": Array of shape (T, Nd), Nd is the action dim of the robot agent, i.e. 22 for our dexhand tasks (6d position of end effector + 16d joint position)
5. "action": Array of shape (T, Nd). We use relative end-effector position control for the robot arm and relative joint-angle position control for the dex hand.
    
"""


