"""
This file considers Franka Panda and joint position 
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import shutil


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

robot_id = p.loadURDF("../../Exact_Panda/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda_with_2F85_sec.urdf",[0, 0, 0], useFixedBase=True)

end_effector_link_index = 7
positions = []


joint_positions = []
gripper_positions = []

with open("./RLDS_Data/scene_4/joint_ps.txt", "r") as file: 
    for line in file:
        try:
            joint_positions.append(eval(line.strip()))  # Read joint positions as a list
        except SyntaxError as e:
            print(f"Syntax error in line: {line.strip()}")
            print(e)

with open("./RLDS_Data/scene_4/gripper_ps.txt", "r") as file:
    for line in file:
        gripper_positions.append(float(line.strip()))

# print(gripper_positions)

num_joints = p.getNumJoints(robot_id)
print(f"--- Inspecting {num_joints} Joints for Robot ID: {robot_id} ---")

for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_index = info[0]
    joint_name = info[1].decode('utf-8')
    print(f"Index: {joint_index} | Name: {joint_name}")

print("-------------------------------------------")



num_joints = p.getNumJoints(robot_id)
gripper_joint_indices = []
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode('utf-8')
    
    if 'finger_joint' in joint_name:
        gripper_joint_indices.append(i)

print(f"Found gripper finger joints at indices: {gripper_joint_indices}")


def move_to_joint_position_with_feedback(arm_positions, gripper_position):
    """
    Sets the robot's arm and gripper to the specified joint positions.
    The value from  file represents the magnitude of the inward rotation (in radians) for each finger, starting from the fully open position.

    NOTE : The gripper pos is actually the rotation of the finger joints and not the distance between the fingers => can be confirmed from the URDF file.(revolute joint)
    """

    for i in range(len(arm_positions)):
        p.resetJointState(robot_id, i, arm_positions[i])
        
    # 2. Control the GRIPPER joints for opposition
    for joint_index in gripper_joint_indices:

        info = p.getJointInfo(robot_id, joint_index)
        joint_name = info[1].decode('utf-8')

        if 'right' in joint_name:
            p.resetJointState(robot_id, joint_index, -gripper_position) # Rotate right finger INWARD 
        else:
            p.resetJointState(robot_id, joint_index, gripper_position) # Rotate left finger INWARD

    for _ in range(240):
        p.stepSimulation()





def update_intrinsic_matrix(K, old_dims, new_dims):
    """
    Update the intrinsic matrix K based on new image dimensions.
    """
    
    # NOTE :  Mention the site later !

    old_height, old_width = old_dims
    new_height, new_width = new_dims

    scale_w = new_width / old_width
    scale_h = new_height / old_height

    K_updated = K.copy()
    K_updated[0, 0] *= scale_w  # Scale fx
    K_updated[1, 1] *= scale_h  # Scale fy
    K_updated[0, 2] *= scale_w  # Scale cx
    K_updated[1, 2] *= scale_h  # Scale cy

    return K_updated




def cvK2BulletP():
    """
    cvKtoPulletP converst the K interinsic matrix as calibrated using Opencv
    and ROS to the projection matrix used in openGL and Pybullet.

    :param K:  OpenCV 3x3 camera intrinsic matrix
    :param w:  Image width
    :param h:  Image height
    :near:     The nearest objects to be included in the render
    :far:      The furthest objects to be included in the render
    :return:   4x4 projection matrix as used in openGL and pybullet

    # https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12901
    """ 

    near = 0.1
    far = 3.1

    h = 180
    w = 320

    old_dims = (720 , 1280)
    new_dims = (180 , 320)


    """
    NOTE : These are col - no 
    """

    """
    RPL setup 
    """
    # K_old = np.array([
    #     [522.06170654, 0, 661.82672119],
    #     [0, 522.06170654, 355.39929199],
    #     [0, 0, 1]
    # ])

    """
    scene - 12 
    """
    K_old = np.array([
        [522.845, 0, 648.825],
        [0, 522.845, 354.744],
        [0, 0, 1]
    ])



    """
    All AutoLab setup 
    """
    # K_old = np.array([[524.12890625 , 0 , 639.77941895] , 
    # [0,524.12890625 , 370.27819824] ,
    # [0,0,1]] )


    # K_old = np.array([[530.3782959 ,   0.        , 646.08825684],
    #    [  0.        , 530.3782959 , 368.5713501 ],
    #    [  0.        ,   0.        ,   1.        ]])
    
    K = update_intrinsic_matrix(K = K_old , old_dims = old_dims , new_dims = new_dims)

    # print(K)


    f_x = K[0,0]
    f_y = K[1,1]
    c_x = K[0,2]
    c_y = K[1,2]

    A = (near + far)/(near - far)
    B = 2 * near * far / (near - far)

    projection_matrix = [
                        [2/w * f_x,  0,          (w - 2*c_x)/w,  0],
                        [0,          2/h * f_y,  (2*c_y - h)/h,  0],
                        [0,          0,          A,              B],
                        [0,          0,          -1,             0]]

    return np.array(projection_matrix).T.reshape(16).tolist()




def capture_image(camera_position, camera_orientation, file_name):

    if os.path.exists(file_name):
        os.remove(file_name)  

    rot_matrix = R.from_quat(camera_orientation).as_matrix()
    camera_target_position = camera_position + rot_matrix @ np.array([0, 0, 1])

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_position,
        cameraTargetPosition=camera_target_position,
        cameraUpVector=[0, 0, 1]
    )

    """
    # https://reachpranjal19.medium.com/camera-calibration-in-ros-melodic-a0bf4742d636
    # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
    """


    height = 180
    width = 320


    proj_matrix = cvK2BulletP()


    _, _, _, _, seg_img = p.getCameraImage(
        width=width,           
        height=height,           
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL 
    )

    arm_object_ids = [1]


    seg_array = np.reshape(seg_img, (height, width))
    arm_mask = np.isin(seg_array, arm_object_ids).astype(np.uint8) * 255
    seg_mask = cv2.cvtColor(arm_mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(file_name, seg_mask)



filtered_image_dir = "move-cam"

if os.path.exists(filtered_image_dir):
    shutil.rmtree(filtered_image_dir)

os.makedirs(filtered_image_dir)


"""
scene - 4 - GOOD !
"""
# # left
# camera_position = [0.085036	,0.563473	,0.416859]
# camera_orientation = p.getQuaternionFromEuler([-1.95721,	-0.0233935	,-2.11812])


"""
scene - 7
"""
# left cam -> good !
camera_position = [0.167891,	0.447045,	0.488312]

print(camera_position)

camera_orientation = p.getQuaternionFromEuler([-1.75215	,-0.0124033,-2.05865])

visual_shape_id = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[0.05, 0.05, 0.05],
    rgbaColor=[0, 1, 0, 1]  
)

camera_body_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=-1,
    baseVisualShapeIndex=visual_shape_id,
    basePosition=camera_position,
    baseOrientation=camera_orientation
)

red_ball_shape = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=0.01,  # small size
    rgbaColor=[1, 0, 0, 1]  # red
)



# # --- Execute Trajectory ---
# for idx, arm_pos in enumerate(joint_positions):

#     gripper_pos = gripper_positions[idx]

#     move_to_joint_position_with_feedback(arm_pos, gripper_pos)

#     image_name = os.path.join(filtered_image_dir, f"camera_position_{idx}.png")
#     capture_image(camera_position, camera_orientation, image_name)
#     print(f"Set pose {idx} with gripper at {gripper_pos:.4f} and saved image.")

# print("Trajectory finished.")

# p.disconnect()


# Create a reusable visual shape for red spheres
red_ball_shape = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=0.015,  # size
    rgbaColor=[1, 0, 0, 1]  # red
)

# # --- STEP 1: Place all balls first ---
# for idx, arm_pos in enumerate(joint_positions):

#     for j in range(len(arm_pos)):
#         p.resetJointState(robot_id, j, arm_pos[j])

#     ee_state = p.getLinkState(robot_id, end_effector_link_index)
#     ee_pos = ee_state[0]  

#     p.createMultiBody(
#         baseMass=0,
#         baseCollisionShapeIndex=-1,
#         baseVisualShapeIndex=red_ball_shape,
#         basePosition=ee_pos
#     )

print("Placed all markers for trajectory.")

for idx, arm_pos in enumerate(joint_positions):
    gripper_pos = gripper_positions[idx]
    move_to_joint_position_with_feedback(arm_pos, gripper_pos)

    image_name = os.path.join(filtered_image_dir, f"camera_position_{idx}.png")
    capture_image(camera_position, camera_orientation, image_name)

    print(f"Moved to pose {idx} with gripper at {gripper_pos:.4f} and saved image.")

print("Trajectory finished.")