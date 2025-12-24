import os
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import shutil
from scipy.spatial.transform import Rotation as R

def load_shift(folder_path):
    """Read translation.txt and return [shift_x, shift_y, shift_z]"""
    shift_file = os.path.join(folder_path, "translation.txt")
    shift = [0.0, 0.0, 0.0]  # default
    axis_map = {"X":0, "Y":1, "Z":2}
    with open(shift_file, "r") as f:
        for line in f:
            line = line.strip()
            if "shift=" in line:
                axis = line.split(":")[0].strip()
                val_str = line.split("shift=")[-1].strip()
                try:
                    val = float(val_str)
                    shift[axis_map[axis]] = val
                except ValueError:
                    print(f"Warning: could not parse shift in line: {line}")
    return np.array(shift)


def load_cart_positions(folder_path, shift):
    cart_file = os.path.join(folder_path, "cart_pos.txt")
    traj = []
    with open(cart_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) >= 3:
                vals[:3] = [v + s for v, s in zip(vals[:3], shift)]
            traj.append(vals)
    return np.array(traj)

def load_extrinsics(folder_path):
    return np.loadtxt(os.path.join(folder_path, "extrinsics.txt"))

def load_intrinsics(folder_path):
    return np.loadtxt(os.path.join(folder_path, "intrinsics.txt"))

def cvK2BulletP(K, w, h, near=0.1, far=3.1):
    f_x, f_y, c_x, c_y = K[0,0], K[1,1], K[0,2], K[1,2]
    A = (near + far) / (near - far)
    B = 2 * near * far / (near - far)
    proj = [
        [2 / w * f_x, 0, (w - 2 * c_x) / w, 0],
        [0, 2 / h * f_y, (2 * c_y - h) / h, 0],
        [0, 0, A, B],
        [0, 0, -1, 0],
    ]
    return np.array(proj).T.reshape(16).tolist()

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


def execute_and_record(folder_id, base_dir):
    """
    Executes a recorded trajectory in PyBullet and saves both RGB and
    segmentation mask images for each frame.
    """
    folder_path = os.path.join(base_dir, str(folder_id))

    rgb_out_dir = os.path.join(folder_path, "rgb")
    mask_out_dir = os.path.join(folder_path, "mask")

    # Remove if already exists
    if os.path.exists(rgb_out_dir):
        shutil.rmtree(rgb_out_dir)
    if os.path.exists(mask_out_dir):
        shutil.rmtree(mask_out_dir)

    # Recreate directories
    os.makedirs(rgb_out_dir)
    os.makedirs(mask_out_dir)

    # === Load data ===
    shift = load_shift(folder_path)
    traj = load_cart_positions(folder_path, shift)
    extrinsics = load_extrinsics(folder_path)
    intrinsics = load_intrinsics(folder_path)
    width, height = 320, 180  # render resolution
    
    # Original camera resolution for scaling intrinsics
    old_dims = (720, 1280)
    new_dims = (height, width)

    K_resized = update_intrinsic_matrix(K=intrinsics, old_dims=old_dims, new_dims=new_dims)

    print(f"Loaded shift: {shift}")

    # === Start PyBullet ===
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    robot_id = p.loadURDF("./lite-6-updated-urdf/lite_6_new.urdf", [0, 0, 0], useFixedBase=True)
    eef_idx = 6  # End-effector link index

    # === Camera pose setup ===
    # Camera position is defined by extrinsics, adjusted by the world shift
    cam_pos = extrinsics[:3] + shift
    cam_euler = extrinsics[3:]
    cam_quat = p.getQuaternionFromEuler(cam_euler)

    # Convert intrinsic matrix to PyBullet's projection matrix format
    proj_matrix = cvK2BulletP(K_resized, width, height)

    # === Main simulation and recording loop ===
    for idx, pos in enumerate(traj):
        target_pos = pos[:3]
        target_ori = p.getQuaternionFromEuler(pos[3:])
        
        # Calculate inverse kinematics to get joint positions
        ik = p.calculateInverseKinematics(robot_id, eef_idx, target_pos, target_ori)
        
        # Set joints to the calculated positions
        for j, val in enumerate(ik):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=val)
        
        # p.setJointMotorControl2(
            # robot_id,
            # 8,
            # p.POSITION_CONTROL,
            # targetPosition=-0.033,
            # force=500
        # )
        # right finger_joint = index 9
        p.setJointMotorControl2(
            robot_id,
            9,
            p.POSITION_CONTROL,
            targetPosition=-0.033,
            force=500
        )

        # Step simulation to allow the robot to reach the pose
        for _ in range(50):
            p.stepSimulation()

        # === Render and save images ===
        # Define camera view matrix
        rot_matrix = R.from_quat(cam_quat).as_matrix()
        # The camera target is a point forward from the camera position along its z-axis
        cam_target = cam_pos + rot_matrix @ np.array([0, 0, 1])
        # The "up" vector is along the world's z-axis
        view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 0, 1])

        # Get camera image from PyBullet
        _, _, rgb, _, seg = p.getCameraImage(
            width=width,           
            height=height,           
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL 
        )
    

        # # --- Process and save RGB image ---
        rgb_array = np.reshape(rgb, (height, width, 4))
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
        rgb_img_path = os.path.join(rgb_out_dir, f"rgb_{idx:04d}.png")
        cv2.imwrite(rgb_img_path, bgr_array)


        # --- Process and save Mask image ---
        seg_array = np.reshape(seg, (height, width))
        # Create a binary mask where pixels corresponding to the robot_id are white
        mask = np.isin(seg_array, [robot_id]).astype(np.uint8) * 255
        mask_img_path = os.path.join(mask_out_dir, f"mask_{idx:04d}.png")
        cv2.imwrite(mask_img_path, mask)
        
    p.disconnect()


if __name__ == "__main__":
    base_dir = "/home/aniruth/Desktop/RRC/XARM7/xArm-Python-SDK/example/wrapper/xarm7/Follow_DROID/Franka_arm/Droid Mask Extraction/Lite-6-data-collection/lite-6-data"
    
    # Get all subfolders
    # folder_ids = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    folder_ids = ['20']
    # Sort numerically if folder names are numbers
    # folder_ids.sort(key=lambda x: int(x))
    
    for folder_id in folder_ids:
        print(f"Processing folder {folder_id} ...")
        try:
            execute_and_record(folder_id, base_dir)
        except Exception as e:
            print(f"Error in folder {folder_id}: {e}")