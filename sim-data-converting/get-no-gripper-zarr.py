#!/usr/bin/env python3
"""
Restructure zarr file to add episode metadata and organize data for 3D Diffusion Policy.
Specific Version: Removes gripper state/action (Dim 13) to result in 6-DoF joint data.
Supports dual camera setup (third_person + wrist) and cube positions.
"""
import zarr
import numpy as np
import sys
import os
import shutil

def print_usage():
    print("Usage: python get-no-gripper-zarr.py <path_to_old_zarr> [--camera CAMERA]")
    print("\nExamples:")
    print("  python get-no-gripper-zarr.py ./final.zarr")
    print("  python get-no-gripper-zarr.py ./final.zarr --camera both")
    print("  python get-no-gripper-zarr.py ./final.zarr --camera third_person")

    print("\nThis version specifically REMOVES the 13th dimension (gripper).")
    print("It keeps only the 6 joint angles (indices 6-11).")

def restructure_single_camera(old_zarr_path, new_zarr_path, episode_ends, 
                              img_key='img', pc_key='point_cloud', 
                              extrinsics_key=None, camera_name='default'):
    """
    Restructure zarr by reading from data/ group and stripping the 7th dimension.
    """
    # Open old zarr
    print(f"\nOpening old zarr: {old_zarr_path}")
    old_root = zarr.open(old_zarr_path, mode='r')
    
    # Check if the structure is flat or nested
    if 'data' in old_root:
        src = old_root['data']
        print("Detected nested 'data/' structure in source.")
    else:
        src = old_root
        print("Detected flat structure in source.")
    
    # Create new zarr
    print(f"Creating new zarr: {new_zarr_path}")
    if os.path.exists(new_zarr_path):
        shutil.rmtree(new_zarr_path)
    
    new_root = zarr.open(new_zarr_path, mode='w')
    data_group = new_root.create_group('data')
    meta_group = new_root.create_group('meta')
    
    print("\nProcessing and Trimming Arrays...")
    
    # --- PROCESS STATE ---
    # Your source is (13140, 7). We take indices 0-5 to get 6-DoF.
    state_key = 'state' if 'state' in src else 'states'
    old_states = src[state_key]
    states_trimmed = old_states[:, :6] 
    
    data_group.create_dataset(
        'state',
        data=states_trimmed,
        chunks=(old_states.chunks[0], 6) if old_states.chunks else None,
        dtype=old_states.dtype,
        compressor=old_states.compressor
    )
    print(f"  state: {old_states.shape} -> {states_trimmed.shape} (Gripper removed)")
    
    # --- PROCESS ACTION ---
    action_key = 'action' if 'action' in src else 'actions'
    old_actions = src[action_key]
    actions_trimmed = old_actions[:, :6]
    
    data_group.create_dataset(
        'action',
        data=actions_trimmed,
        chunks=(old_actions.chunks[0], 6) if old_actions.chunks else None,
        dtype=old_actions.dtype,
        compressor=old_actions.compressor
    )
    print(f"  action: {old_actions.shape} -> {actions_trimmed.shape} (Gripper removed)")
    
    # --- COPY OTHER DATA ---
    # Point Cloud
    if pc_key in src:
        old_pc = src[pc_key]
        data_group.create_dataset('point_cloud', data=old_pc[:], chunks=old_pc.chunks, 
                                  dtype=old_pc.dtype, compressor=old_pc.compressor)
    
    # Image
    if img_key in src:
        old_img = src[img_key]
        data_group.create_dataset('img', data=old_img[:], chunks=old_img.chunks, 
                                  dtype=old_img.dtype, compressor=old_img.compressor)
    
    # Cube Position
    if 'cube_pos' in src:
        data_group.create_dataset('cube_pos', data=src['cube_pos'][:], chunks=src['cube_pos'].chunks)

    # Extrinsics
    if extrinsics_key and extrinsics_key in src:
        data_group.create_dataset('camera_extrinsics', data=src[extrinsics_key][:])

    # --- METADATA ---
    meta_group.create_dataset('episode_ends', data=episode_ends, dtype=np.int64)
    
    new_root.attrs['camera_view'] = camera_name
    new_root.attrs['num_episodes'] = len(episode_ends)
    new_root.attrs['total_frames'] = int(episode_ends[-1])
    new_root.attrs['state_dim'] = 6
    new_root.attrs['action_dim'] = 6
    
    print("\n" + "=" * 60)
    print("DATASET SUMMARY:")
    print("=" * 60)
    print(f"Final State Shape:  {data_group['state'].shape}")
    print(f"Final Action Shape: {data_group['action'].shape}")
    print(f"Total Episodes:     {len(episode_ends)}")
    print(f"Validation:         {'✓ Match' if states_trimmed.shape[0] == episode_ends[-1] else '✗ Mismatch'}")
    print("=" * 60)

    # Summary Table Output
    print("\n" + "=" * 60)
    print("FINAL 6-DOF ZARR STRUCTURE:")
    print("=" * 60)
    print(new_root.tree())
    print("\n" + "=" * 60)
    print("DATASET SUMMARY:")
    print("=" * 60)
    print(f"Camera view: {camera_name}")
    print(f"Episodes:    {len(episode_ends)}")
    print(f"Total frames:{episode_ends[-1]}")
    print(f"\nData dimensions:")
    print(f"  state:       {data_group['state'].shape}")
    print(f"  action:      {data_group['action'].shape}")
    print(f"  img:         {data_group['img'].shape}")
    print(f"  point_cloud: {data_group['point_cloud'].shape}")
    print("\n" + "=" * 60)
    
    # Validation
    actual_frames = states_trimmed.shape[0]
    total_from_meta = episode_ends[-1]
    print(f"Validation: Meta ({total_from_meta}) vs Actual ({actual_frames})")
    if actual_frames == total_from_meta:
        print("  ✓ Match! Dataset is consistent.")
    else:
        print("  ✗ MISMATCH! Check episode_ends.npy.")

    return new_root

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    old_zarr_path = sys.argv[1]
    camera_mode = "third_person"
    if "--camera" in sys.argv:
        camera_idx = sys.argv.index("--camera")
        if camera_idx + 1 < len(sys.argv):
            camera_mode = sys.argv[camera_idx + 1]

    if not os.path.exists('episode_ends.npy'):
        print("ERROR: episode_ends.npy not found.")
        sys.exit(1)
    
    episode_ends = np.load('episode_ends.npy')
    
    if camera_mode == "both":
        # Third Person
        restructure_single_camera(
            old_zarr_path, old_zarr_path.replace('.zarr', '_6dof_tp.zarr'), episode_ends,
            img_key='third_person_img', pc_key='third_person_point_cloud',
            extrinsics_key='third_person_extrinsics', camera_name='third_person'
        )
        # Wrist
        restructure_single_camera(
            old_zarr_path, old_zarr_path.replace('.zarr', '_6dof_wrist.zarr'), episode_ends,
            img_key='wrist_img', pc_key='wrist_point_cloud',
            extrinsics_key='wrist_extrinsics', camera_name='wrist'
        )
    else:
        suffix = f"_6dof_{camera_mode}"
        new_path = old_zarr_path.replace('.zarr', f'{suffix}.zarr')
        
        # Key determination
        temp = zarr.open(old_zarr_path, mode='r')
        if camera_mode == "third_person":
            img_k = 'third_person_img' if 'third_person_img' in temp else 'img'
            pc_k = 'third_person_point_cloud' if 'third_person_point_cloud' in temp else 'point_cloud'
            ex_k = 'third_person_extrinsics' if 'third_person_extrinsics' in temp else None
        else:
            img_k, pc_k, ex_k = 'wrist_img', 'wrist_point_cloud', 'wrist_extrinsics'
            
        restructure_single_camera(old_zarr_path, new_path, episode_ends, img_k, pc_k, ex_k, camera_mode)

if __name__ == "__main__":
    main()
