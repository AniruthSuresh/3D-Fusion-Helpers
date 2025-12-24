import os
import pybullet as p
import pybullet_data
import math
import time
import random
from collections import namedtuple
import cv2
import numpy as np

class UR5Robotiq85:
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 3

    def load(self):
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(
                jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            )
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]

    def __setup_mimic_joints__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id,
                                   jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def move_arm_ik(self, target_pos, target_orn):
        joint_poses = p.calculateInverseKinematics(
            self.id, self.eef_id, target_pos, target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i], maxVelocity=self.max_velocity)

    def move_gripper(self, open_length):
        open_length = max(self.gripper_range[0], min(open_length, self.gripper_range[1]))
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle)

    def get_current_ee_position(self):
        eef_state = p.getLinkState(self.id, self.eef_id)
        return eef_state
    
def create_data_folders(iter_folder):
    rgb_dir = os.path.join(iter_folder, "rgb")
    depth_dir = os.path.join(iter_folder, "depth")
    pcd_dir = os.path.join(iter_folder, "pcd")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)

    return rgb_dir, depth_dir, pcd_dir



def depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix, width=224, height=224):
    """Convert depth buffer to 3D point cloud"""
    # Get camera parameters from projection matrix
    fov = 60
    aspect = 1.0
    near = 0.01
    far = 3.0
    
    # Convert depth buffer to actual depth values
    depth_img = far * near / (far - (far - near) * depth_buffer)
    
    # Create pixel grid
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx, cy = width / 2, height / 2
    
    # Get view matrix as numpy array
    view_matrix_np = np.array(view_matrix).reshape(4, 4).T
    
    # Create meshgrid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to 3D coordinates in camera frame
    z = depth_img
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy  # Negative because image y-axis points down
    
    # Stack into point cloud (N x 3)
    points_camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # Transform to world coordinates
    points_camera_homogeneous = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1))], axis=1)
    view_matrix_inv = np.linalg.inv(view_matrix_np)
    points_world = (view_matrix_inv @ points_camera_homogeneous.T).T[:, :3]
    
    return points_world

def update_simulation(steps, sleep_time=0.01, capture_frames=False, iter_folder=None, frame_counter=None):
    """Update simulation and optionally capture frames"""

    rgb_dir, depth_dir, pcd_dir = create_data_folders(iter_folder)

    for _ in range(steps):
        p.stepSimulation()
        # time.sleep(sleep_time)
        
        if capture_frames and iter_folder is not None and frame_counter is not None:
            # Camera parameters
            view_matrix_tp = p.computeViewMatrix(
                cameraEyePosition=[1.1, -0.6, 1.3],   # higher + right
                cameraTargetPosition=[0.5, 0.3, 0.7], # center of scene (tray + robot)
                cameraUpVector=[0, 0, 1]
            )

            proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=3.0)
            
            # Capture RGB and Depth
            width, height, rgb_tp, depth_tp, _ = p.getCameraImage(
                224, 224, 
                viewMatrix=view_matrix_tp, 
                projectionMatrix=proj_matrix
            )
            
            # Process RGB
            rgb_tp = np.array(rgb_tp)[:, :, :3]
            
            # Process Depth
            depth_buffer = np.array(depth_tp)
            
            # Convert depth to point cloud
            point_cloud = depth_to_point_cloud(depth_buffer, view_matrix_tp, proj_matrix)
            
            # ---- Save RGB ----
            frame_path_rgb = os.path.join(rgb_dir, f"tp_rgb_{frame_counter[0]:04d}.png")
            cv2.imwrite(frame_path_rgb, cv2.cvtColor(rgb_tp, cv2.COLOR_RGB2BGR))

            # ---- Save Depth ----
            depth_raw_path = os.path.join(depth_dir, f"tp_depth_{frame_counter[0]:04d}.npy")
            np.save(depth_raw_path, depth_buffer)

            # ---- Save Point Cloud ----
            pcd_path = os.path.join(pcd_dir, f"tp_pcd_{frame_counter[0]:04d}.npy")
            np.save(pcd_path, point_cloud)

            # ---- Save PLY ----
            ply_path = os.path.join(pcd_dir, f"tp_pcd_{frame_counter[0]:04d}.ply")
            save_point_cloud_ply(point_cloud, rgb_tp.reshape(-1, 3), ply_path)
            
            frame_counter[0] += 1

def save_point_cloud_ply(points, colors, filename):
    """Save point cloud in PLY format with colors"""
    # Filter out points that are too far (likely background/invalid)
    valid_mask = points[:, 2] < 2.5  # Keep points closer than 2.5m
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    with open(filename, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write point data
        for point, color in zip(points, colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")

def setup_simulation():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
    tray_pos = [0.5, 0.9, 0.6]
    tray_orn = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("tray/tray.urdf", tray_pos, tray_orn)
    return tray_pos, tray_orn

def random_color_cube(cube_id):
    color = [random.random(), random.random(), random.random(), 1.0]
    p.changeVisualShape(cube_id, -1, rgbaColor=color)

def move_and_grab_cube(robot, tray_pos, base_save_dir="dataset"):
    iteration = 0
    while True:
        # Create subfolder for this iteration
        iter_folder = os.path.join(base_save_dir, f"iter_{iteration:04d}")
        os.makedirs(iter_folder, exist_ok=True)

        # Frame counter for this iteration (using list to pass by reference)
        frame_counter = [0]

        # Reset arm posture
        target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
        update_simulation(200, capture_frames=False, iter_folder=iter_folder, frame_counter=frame_counter)

        # Random cube
        cube_start_pos = [random.uniform(0.3, 0.7), random.uniform(-0.1, 0.1), 0.65]
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orn)
        random_color_cube(cube_id)

        # Get end-effector orientation
        eef_state = robot.get_current_ee_position()
        eef_orientation = eef_state[1]

        # Move above cube
        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 0.83], eef_orientation)
        update_simulation(50, capture_frames=True, iter_folder=iter_folder, frame_counter=frame_counter)
        
        # Move down
        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 0.78], eef_orientation)
        update_simulation(50, capture_frames=True, iter_folder=iter_folder, frame_counter=frame_counter)
        
        # Close gripper
        robot.move_gripper(0.01)
        update_simulation(25, capture_frames=True, iter_folder=iter_folder, frame_counter=frame_counter)
        
        # Lift cube
        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 1.18], eef_orientation)
        update_simulation(50, capture_frames=True, iter_folder=iter_folder, frame_counter=frame_counter)
        
        # Move above tray
        tray_offset = random.uniform(0.1, 0.3)
        robot.move_arm_ik([tray_pos[0]+tray_offset, tray_pos[1]+tray_offset, tray_pos[2]+0.56], eef_orientation)
        update_simulation(150, capture_frames=True, iter_folder=iter_folder, frame_counter=frame_counter)
        
        # Open gripper
        robot.move_gripper(0.085)
        update_simulation(25, capture_frames=True, iter_folder=iter_folder, frame_counter=frame_counter)
        
        # Remove cube
        p.removeBody(cube_id)

        print(f"Completed iteration {iteration} - {frame_counter[0]} frames captured")
        iteration += 1

def main():
    tray_pos, tray_orn = setup_simulation()
    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()
    move_and_grab_cube(robot, tray_pos)

if __name__ == "__main__":
    main()