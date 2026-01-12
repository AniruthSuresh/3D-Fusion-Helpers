import os
import pybullet as p
import pybullet_data
import math
import time
import random
from collections import namedtuple
import cv2
import numpy as np
import json

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
    
    def get_robot_state(self):
        """Get complete robot state: end-effector pose + joint angles"""
        eef_state = p.getLinkState(self.id, self.eef_id)
        eef_pos = np.array(eef_state[0]) 
        eef_orn_quat = np.array(eef_state[1])  
        eef_orn_euler = np.array(p.getEulerFromQuaternion(eef_orn_quat))  
        
        joint_states = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)
            joint_states.append(joint_state[0])
        
        gripper_state = p.getJointState(self.id, self.mimic_parent_id)
        gripper_angle = gripper_state[0]
        
        state = np.concatenate([
            eef_pos,
            eef_orn_euler,
            joint_states,
            [gripper_angle]
        ])
        
        return state

def create_data_folders(iter_folder):
    tp_rgb_dir = os.path.join(iter_folder, "third_person", "rgb")
    tp_depth_dir = os.path.join(iter_folder, "third_person", "depth")
    tp_pcd_dir = os.path.join(iter_folder, "third_person", "pcd")
    
    wr_rgb_dir = os.path.join(iter_folder, "wrist", "rgb")
    wr_depth_dir = os.path.join(iter_folder, "wrist", "depth")
    wr_pcd_dir = os.path.join(iter_folder, "wrist", "pcd")
    
    poses_dir = os.path.join(iter_folder, "camera_poses")

    os.makedirs(tp_rgb_dir, exist_ok=True)
    os.makedirs(tp_depth_dir, exist_ok=True)
    os.makedirs(tp_pcd_dir, exist_ok=True)
    
    os.makedirs(wr_rgb_dir, exist_ok=True)
    os.makedirs(wr_depth_dir, exist_ok=True)
    os.makedirs(wr_pcd_dir, exist_ok=True)
    
    os.makedirs(poses_dir, exist_ok=True)

    return {
        'tp_rgb': tp_rgb_dir,
        'tp_depth': tp_depth_dir,
        'tp_pcd': tp_pcd_dir,
        'wr_rgb': wr_rgb_dir,
        'wr_depth': wr_depth_dir,
        'wr_pcd': wr_pcd_dir,
        'poses': poses_dir
    }

def depth_to_point_cloud(depth_buffer, view_matrix, proj_matrix, width=224, height=224):
    """Convert depth buffer to 3D point cloud"""
    fov = 60
    near = 0.01
    far = 3.0
    
    depth_img = far * near / (far - (far - near) * depth_buffer)
    
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx, cy = width / 2, height / 2
    
    view_matrix_np = np.array(view_matrix).reshape(4, 4).T
    
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    z = depth_img
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    
    points_camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    points_camera_homogeneous = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1))], axis=1)
    view_matrix_inv = np.linalg.inv(view_matrix_np)
    points_world = (view_matrix_inv @ points_camera_homogeneous.T).T[:, :3]
    
    return points_world

def get_cube_point_cloud_from_segmentation(rgb, depth_buffer, seg_mask, view_matrix, proj_matrix, 
                                           cube_id, width=224, height=224):
    """Extract cube point cloud from scene using segmentation mask"""
    fov = 60
    near = 0.01
    far = 3.0
    
    depth_img = far * near / (far - (far - near) * depth_buffer)
    
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx, cy = width / 2, height / 2
    
    view_matrix_np = np.array(view_matrix).reshape(4, 4).T
    
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    z = depth_img
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    
    points_camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    points_camera_homogeneous = np.concatenate([points_camera, np.ones((points_camera.shape[0], 1))], axis=1)
    view_matrix_inv = np.linalg.inv(view_matrix_np)
    points_world = (view_matrix_inv @ points_camera_homogeneous.T).T[:, :3]
    
    colors = rgb.reshape(-1, 3)
    
    seg_flat = seg_mask.flatten()
    cube_mask = (seg_flat == cube_id)
    
    valid_mask = points_world[:, 2] < 2.5
    
    cube_mask_valid = cube_mask & valid_mask
    scene_mask_valid = valid_mask
    
    # Cube points
    cube_points = points_world[cube_mask_valid]
    cube_colors = colors[cube_mask_valid]
    
    # Full scene points (including cube)
    scene_points = points_world[scene_mask_valid]
    scene_colors = colors[scene_mask_valid]
    
    # Scene WITHOUT cube (for clean overlay)
    scene_without_cube_mask = (~cube_mask) & valid_mask
    scene_without_cube_points = points_world[scene_without_cube_mask]
    scene_without_cube_colors = colors[scene_without_cube_mask]
    
    return (cube_points, cube_colors, scene_points, scene_colors, 
            scene_without_cube_points, scene_without_cube_colors, cube_mask)

def save_segmented_point_clouds(cube_points, cube_colors, scene_points, scene_colors, 
                                 output_dir, frame_idx, prefix='tp'):
    """Save both cube-only and full scene point clouds"""
    cube_ply_path = os.path.join(output_dir, f"{prefix}_cube_only_{frame_idx:04d}.ply")
    save_point_cloud_ply(cube_points, cube_colors, cube_ply_path)
    
    cube_npy_path = os.path.join(output_dir, f"{prefix}_cube_only_{frame_idx:04d}.npy")
    np.save(cube_npy_path, cube_points)
    
    scene_ply_path = os.path.join(output_dir, f"{prefix}_scene_full_{frame_idx:04d}.ply")
    save_point_cloud_ply(scene_points, scene_colors, scene_ply_path)
    
    scene_npy_path = os.path.join(output_dir, f"{prefix}_scene_full_{frame_idx:04d}.npy")
    np.save(scene_npy_path, scene_points)
    
    return {
        'cube_ply': cube_ply_path,
        'cube_npy': cube_npy_path,
        'scene_ply': scene_ply_path,
        'scene_npy': scene_npy_path,
        'num_cube_points': len(cube_points),
        'num_scene_points': len(scene_points)
    }

def save_cube_segmentation_mask(seg_mask, cube_id, output_path):
    """Save segmentation mask as image"""
    cube_mask_img = (seg_mask == cube_id).astype(np.uint8) * 255
    cv2.imwrite(output_path, cube_mask_img)

def overlay_cube_and_scene(scene_points, scene_colors, cube_points, output_path, 
                           scene_without_cube_points=None, scene_without_cube_colors=None):
    """Create combined point cloud with cube in red
    
    If scene_without_cube_points is provided, use that as the scene.
    Otherwise, remove cube from scene by finding non-cube points.
    """
    # Create red colors for cube
    cube_colors_red = np.tile([255, 0, 0], (len(cube_points), 1))
    
    # If we have scene without cube, use it directly
    if scene_without_cube_points is not None:
        print("Im here")
        combined_points = np.vstack([scene_without_cube_points, cube_points])
        combined_colors = np.vstack([scene_without_cube_colors, cube_colors_red])
    else:
        # Otherwise just use scene as-is (cube will be duplicated but in red)
        combined_points = np.vstack([scene_points, cube_points])
        combined_colors = np.vstack([scene_colors, cube_colors_red])
    
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(combined_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for point, color in zip(combined_points, combined_colors):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} ")
            f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")

def get_wrist_camera_params(robot):
    """Get wrist camera position and orientation based on end-effector"""
    eef_state = robot.get_current_ee_position()
    eef_pos, eef_orn = eef_state[0], eef_state[1]
    
    rot_matrix = np.array(p.getMatrixFromQuaternion(eef_orn)).reshape(3, 3)
    
    cam_offset_local = np.array([-0.05, 0, 0.12])
    cam_pos = np.array(eef_pos) + rot_matrix @ cam_offset_local
    
    cam_target = cam_pos + rot_matrix[:, 0] * 0.35 + rot_matrix[:, 2] * (-0.15)
    
    cam_up = rot_matrix[:, 2]
    
    return cam_pos, cam_target, cam_up

def save_camera_pose(pose_dict, poses_dir, frame_idx):
    """Save camera pose information"""
    pose_file = os.path.join(poses_dir, f"pose_{frame_idx:04d}.json")
    with open(pose_file, 'w') as f:
        json.dump(pose_dict, f, indent=2)

def compute_extrinsics(cam_pos, cam_target, cam_up):
    """Compute camera extrinsics matrix"""
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    cam_up = np.array(cam_up)
    
    forward = cam_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, cam_up)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    rotation_matrix = np.array([right, up, -forward])
    translation = cam_pos
    
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = -rotation_matrix @ translation
    
    return {
        'rotation_matrix': rotation_matrix.tolist(),
        'translation': translation.tolist(),
        'extrinsics_matrix': extrinsics.tolist()
    }

def update_simulation(steps, sleep_time=0.01, capture_frames=False, iter_folder=None, 
                     frame_counter=None, robot=None, base_pos=None, state_history=None, cube_id=None):
    """Update simulation and capture frames with cube segmentation"""
    
    if capture_frames:
        dirs = create_data_folders(iter_folder)
        tp_cube_dir = os.path.join(iter_folder, "third_person", "cube_segmented")
        wr_cube_dir = os.path.join(iter_folder, "wrist", "cube_segmented")
        tp_mask_dir = os.path.join(iter_folder, "third_person", "segmentation_masks")
        wr_mask_dir = os.path.join(iter_folder, "wrist", "segmentation_masks")
        tp_overlay_dir = os.path.join(iter_folder, "third_person", "overlay_red_cube")
        wr_overlay_dir = os.path.join(iter_folder, "wrist", "overlay_red_cube")
        
        os.makedirs(tp_cube_dir, exist_ok=True)
        os.makedirs(wr_cube_dir, exist_ok=True)
        os.makedirs(tp_mask_dir, exist_ok=True)
        os.makedirs(wr_mask_dir, exist_ok=True)
        os.makedirs(tp_overlay_dir, exist_ok=True)
        os.makedirs(wr_overlay_dir, exist_ok=True)
    
    tp_cam_eye = [1.1, -0.6, 1.3]
    tp_cam_target = [0.5, 0.3, 0.7]
    tp_cam_up = [0, 0, 1]
    
    for _ in range(steps):
        p.stepSimulation()
        
        if capture_frames and iter_folder is not None and frame_counter is not None:
            view_matrix_tp = p.computeViewMatrix(
                cameraEyePosition=tp_cam_eye,
                cameraTargetPosition=tp_cam_target,
                cameraUpVector=tp_cam_up
            )
            proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=3.0)
            
            width, height, rgb_tp, depth_tp, seg_tp = p.getCameraImage(
                224, 224,
                viewMatrix=view_matrix_tp,
                projectionMatrix=proj_matrix,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            )
            
            rgb_tp = np.array(rgb_tp)[:, :, :3]
            depth_buffer_tp = np.array(depth_tp)
            seg_mask_tp = np.array(seg_tp)
            
            cv2.imwrite(
                os.path.join(dirs['tp_rgb'], f"tp_rgb_{frame_counter[0]:04d}.png"),
                cv2.cvtColor(rgb_tp, cv2.COLOR_RGB2BGR)
            )
            np.save(
                os.path.join(dirs['tp_depth'], f"tp_depth_{frame_counter[0]:04d}.npy"),
                depth_buffer_tp
            )
            
            if cube_id is not None:
                result_tp = get_cube_point_cloud_from_segmentation(
                    rgb_tp, depth_buffer_tp, seg_mask_tp, 
                    view_matrix_tp, proj_matrix, cube_id
                )
                cube_pts_tp, cube_cols_tp, scene_pts_tp, scene_cols_tp, \
                scene_no_cube_pts_tp, scene_no_cube_cols_tp, _ = result_tp
                
                save_cube_segmentation_mask(
                    seg_mask_tp, cube_id, 
                    os.path.join(tp_mask_dir, f"tp_mask_{frame_counter[0]:04d}.png")
                )
                
                save_segmented_point_clouds(
                    cube_pts_tp, cube_cols_tp, scene_pts_tp, scene_cols_tp,
                    tp_cube_dir, frame_counter[0], prefix='tp'
                )
                
                overlay_path_tp = os.path.join(tp_overlay_dir, f"tp_overlay_{frame_counter[0]:04d}.ply")
                overlay_cube_and_scene(scene_pts_tp, scene_cols_tp, cube_pts_tp, overlay_path_tp,
                                      scene_no_cube_pts_tp, scene_no_cube_cols_tp)
            else:
                point_cloud_tp = depth_to_point_cloud(depth_buffer_tp, view_matrix_tp, proj_matrix)
                colors_tp = rgb_tp.reshape(-1, 3)
                np.save(
                    os.path.join(dirs['tp_pcd'], f"tp_pcd_{frame_counter[0]:04d}.npy"),
                    point_cloud_tp
                )
                ply_path_tp = os.path.join(dirs['tp_pcd'], f"tp_pcd_{frame_counter[0]:04d}.ply")
                save_point_cloud_ply(point_cloud_tp.reshape(-1, 3), colors_tp, ply_path_tp)

            wr_cam_pos, wr_cam_target, wr_cam_up = get_wrist_camera_params(robot)
            view_matrix_wr = p.computeViewMatrix(
                cameraEyePosition=wr_cam_pos,
                cameraTargetPosition=wr_cam_target,
                cameraUpVector=wr_cam_up
            )
            
            width, height, rgb_wr, depth_wr, seg_wr = p.getCameraImage(
                224, 224,
                viewMatrix=view_matrix_wr,
                projectionMatrix=proj_matrix,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            )
            
            rgb_wr = np.array(rgb_wr)[:, :, :3]
            depth_buffer_wr = np.array(depth_wr)
            seg_mask_wr = np.array(seg_wr)
            
            cv2.imwrite(
                os.path.join(dirs['wr_rgb'], f"wr_rgb_{frame_counter[0]:04d}.png"),
                cv2.cvtColor(rgb_wr, cv2.COLOR_RGB2BGR)
            )
            np.save(
                os.path.join(dirs['wr_depth'], f"wr_depth_{frame_counter[0]:04d}.npy"),
                depth_buffer_wr
            )
            
            if cube_id is not None:
                result_wr = get_cube_point_cloud_from_segmentation(
                    rgb_wr, depth_buffer_wr, seg_mask_wr, 
                    view_matrix_wr, proj_matrix, cube_id
                )
                cube_pts_wr, cube_cols_wr, scene_pts_wr, scene_cols_wr, \
                scene_no_cube_pts_wr, scene_no_cube_cols_wr, _ = result_wr
                
                save_cube_segmentation_mask(
                    seg_mask_wr, cube_id, 
                    os.path.join(wr_mask_dir, f"wr_mask_{frame_counter[0]:04d}.png")
                )
                
                save_segmented_point_clouds(
                    cube_pts_wr, cube_cols_wr, scene_pts_wr, scene_cols_wr,
                    wr_cube_dir, frame_counter[0], prefix='wr'
                )
                
                overlay_path_wr = os.path.join(wr_overlay_dir, f"wr_overlay_{frame_counter[0]:04d}.ply")
                overlay_cube_and_scene(scene_pts_wr, scene_cols_wr, cube_pts_wr, overlay_path_wr,
                                      scene_no_cube_pts_wr, scene_no_cube_cols_wr)
            else:
                point_cloud_wr = depth_to_point_cloud(depth_buffer_wr, view_matrix_wr, proj_matrix)
                colors_wr = rgb_wr.reshape(-1, 3)
                np.save(
                    os.path.join(dirs['wr_pcd'], f"wr_pcd_{frame_counter[0]:04d}.npy"),
                    point_cloud_wr
                )
                ply_path_wr = os.path.join(dirs['wr_pcd'], f"wr_pcd_{frame_counter[0]:04d}.ply")
                save_point_cloud_ply(point_cloud_wr.reshape(-1, 3), colors_wr, ply_path_wr)

            base_pos_array = np.array(base_pos)
            tp_cam_pos_base = np.array(tp_cam_eye) - base_pos_array
            tp_cam_target_base = np.array(tp_cam_target) - base_pos_array
            tp_extrinsics = compute_extrinsics(tp_cam_pos_base, tp_cam_target_base, tp_cam_up)
            
            wr_cam_pos_base = np.array(wr_cam_pos) - base_pos_array
            wr_cam_target_base = np.array(wr_cam_target) - base_pos_array
            wr_extrinsics = compute_extrinsics(wr_cam_pos_base, wr_cam_target_base, wr_cam_up)
            
            pose_dict = {
                'frame': frame_counter[0],
                'third_person_camera': {
                    'rotation_matrix': tp_extrinsics['rotation_matrix'],
                    'translation': tp_extrinsics['translation'],
                    'extrinsics_matrix': tp_extrinsics['extrinsics_matrix']
                },
                'wrist_camera': {
                    'rotation_matrix': wr_extrinsics['rotation_matrix'],
                    'translation': wr_extrinsics['translation'],
                    'extrinsics_matrix': wr_extrinsics['extrinsics_matrix']
                }
            }
            
            save_camera_pose(pose_dict, dirs['poses'], frame_counter[0])
            
            current_state = robot.get_robot_state()
            if state_history is not None:
                state_history.append(current_state)
            
            frame_counter[0] += 1

def save_point_cloud_ply(points, colors, filename):
    """Save point cloud in PLY format with colors"""
    valid_mask = points[:, 2] < 2.5
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    with open(filename, 'w') as f:
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
        iter_folder = os.path.join(base_save_dir, f"iter_{iteration:04d}")
        os.makedirs(iter_folder, exist_ok=True)

        frame_counter = [0]
        state_history = []

        target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
        update_simulation(200, capture_frames=False, iter_folder=iter_folder, 
                         frame_counter=frame_counter, robot=robot, base_pos=robot.base_pos,
                         state_history=state_history, cube_id=None)

        cube_start_pos = [random.uniform(0.3, 0.7), random.uniform(-0.1, 0.1), 0.65]
        cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orn)
        random_color_cube(cube_id)

        eef_state = robot.get_current_ee_position()
        eef_orientation = eef_state[1]

        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 0.83], eef_orientation)
        update_simulation(50, capture_frames=True, iter_folder=iter_folder, 
                         frame_counter=frame_counter, robot=robot, base_pos=robot.base_pos,
                         state_history=state_history, cube_id=cube_id)
        
        robot.move_arm_ik([cube_start_pos[0], cube_start_pos[1], 0.78], eef_orientation)
        update_simulation(50, capture_frames=True, iter_folder=iter_folder, 
                         frame_counter=frame_counter, robot=robot, base_pos=robot.base_pos,
                         state_history=state_history, cube_id=cube_id)
        
        p.removeBody(cube_id)

        agent_pos = np.array(state_history)
        actions = np.diff(agent_pos, axis=0)
        actions = np.vstack([actions, np.zeros(13)])
        
        state_action_data = {
            'agent_pos': agent_pos.tolist(),
            'action': actions.tolist(),
            'num_frames': len(agent_pos),
            'state_dim': 13,
            'state_description': {
                'eef_pos': [0, 1, 2],
                'eef_orn': [3, 4, 5],
                'arm_joints': [6, 7, 8, 9, 10, 11],
                'gripper': [12]
            }
        }
        
        state_action_file = os.path.join(iter_folder, "state_action.json")
        with open(state_action_file, 'w') as f:
            json.dump(state_action_data, f, indent=2)
        
        np.save(os.path.join(iter_folder, "agent_pos.npy"), agent_pos)
        np.save(os.path.join(iter_folder, "actions.npy"), actions)
        
        np.savetxt(os.path.join(iter_folder, "agent_pos.txt"), agent_pos, 
                   fmt='%.6f', delimiter=' ',
                   header='End-effector pose (x,y,z,roll,pitch,yaw) + 6 joint angles + gripper angle (13 dimensions)')
        np.savetxt(os.path.join(iter_folder, "actions.txt"), actions, 
                   fmt='%.6f', delimiter=' ',
                   header='Action deltas: differences between consecutive states (13 dimensions)')

        print(f"Completed iteration {iteration} - {frame_counter[0]} frames captured")
        print(f"  Agent states shape: {agent_pos.shape}")
        print(f"  Actions shape: {actions.shape}")
        iteration += 1

def main():
    tray_pos, tray_orn = setup_simulation()
    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()
    move_and_grab_cube(robot, tray_pos)

if __name__ == "__main__":
    main()