#!/usr/bin/env python3
"""
Restructure zarr file to add episode metadata and organize data for 3D Diffusion Policy.
Supports dual camera setup (third_person + wrist) and cube positions.
"""
import zarr
import numpy as np
import sys
import os
import shutil

def print_usage():
    print("Usage: python restructure_zarr.py <path_to_old_zarr> [--camera CAMERA]")
    print("\nExamples:")
    print("  python restructure_zarr.py ./final.zarr")
    print("  python restructure_zarr.py ./final.zarr --camera third_person")
    print("  python restructure_zarr.py ./final.zarr --camera wrist")
    print("  python restructure_zarr.py ./final.zarr --camera both")
    print("\nThis script will:")
    print("  1. Read the old zarr structure (flat arrays)")
    print("  2. Create new zarr with data/ and meta/ groups")
    print("  3. Copy arrays: states→data/state, actions→data/action")
    print("  4. Copy camera data: img→data/img, point_cloud→data/point_cloud")
    print("  5. Copy cube_pos→data/cube_pos (if available)")
    print("  6. Add episode_ends.npy to meta/episode_ends")
    print("  7. Add camera extrinsics to data/camera_extrinsics (if available)")
    print("\nOptions:")
    print("  --camera third_person : Use only third-person camera data (default)")
    print("  --camera wrist        : Use only wrist camera data")
    print("  --camera both         : Create separate zarr files for both cameras")
    print("\nRequires episode_ends.npy in current directory")

def restructure_single_camera(old_zarr_path, new_zarr_path, episode_ends, 
                              img_key='img', pc_key='point_cloud', 
                              extrinsics_key=None, camera_name='default'):
    """
    Restructure zarr for a single camera view.
    
    Args:
        old_zarr_path: Path to input zarr
        new_zarr_path: Path to output zarr
        episode_ends: Episode boundary indices
        img_key: Key for image data in old zarr
        pc_key: Key for point cloud data in old zarr
        extrinsics_key: Key for camera extrinsics in old zarr (optional)
        camera_name: Name of camera for logging
    """
    # Open old zarr
    print(f"\nOpening old zarr: {old_zarr_path}")
    old_root = zarr.open(old_zarr_path, mode='r')
    
    print(f"\nOld structure for {camera_name}:")
    print(old_root.tree())
    
    # Create new zarr
    print(f"\nCreating new zarr: {new_zarr_path}")
    if os.path.exists(new_zarr_path):
        print(f"WARNING: {new_zarr_path} already exists. Removing...")
        shutil.rmtree(new_zarr_path)
    
    new_root = zarr.open(new_zarr_path, mode='w')
    
    # Create groups
    data_group = new_root.create_group('data')
    meta_group = new_root.create_group('meta')
    
    print("\nCopying arrays...")
    
    # Copy states → data/state (remove first 6 entries per row)
    print("  states → data/state (removing first 6 columns)")
    old_states = old_root['states']
    # Remove first 6 columns (eef_pos(3) + eef_orn(3))
    # Keep only: joints(6) + gripper(1) = 7 dimensions
    states_trimmed = old_states[:, 6:]
    
    # Adjust chunk size to match new shape
    new_state_chunks = list(old_states.chunks) if old_states.chunks else [old_states.shape[0]]
    if len(new_state_chunks) > 1:
        new_state_chunks[1] = states_trimmed.shape[1]
    new_state_chunks = tuple(new_state_chunks)
    
    data_group.create_dataset(
        'state',
        data=states_trimmed,
        chunks=new_state_chunks,
        dtype=old_states.dtype,
        compressor=old_states.compressor
    )
    print(f"    original shape={old_states.shape}, new shape={states_trimmed.shape}, dtype={old_states.dtype}")
    print(f"    removed: eef_pos(3) + eef_orn(3), kept: joints(6) + gripper(1)")
    
    # Copy actions → data/action (remove first 6 entries per row)
    print("  actions → data/action (removing first 6 columns)")
    old_actions = old_root['actions']
    # Remove first 6 columns (same as states)
    actions_trimmed = old_actions[:, 6:]
    
    # Adjust chunk size to match new shape
    new_action_chunks = list(old_actions.chunks) if old_actions.chunks else [old_actions.shape[0]]
    if len(new_action_chunks) > 1:
        new_action_chunks[1] = actions_trimmed.shape[1]
    new_action_chunks = tuple(new_action_chunks)
    
    data_group.create_dataset(
        'action',
        data=actions_trimmed,
        chunks=new_action_chunks,
        dtype=old_actions.dtype,
        compressor=old_actions.compressor
    )
    print(f"    original shape={old_actions.shape}, new shape={actions_trimmed.shape}, dtype={old_actions.dtype}")
    
    # Copy point cloud
    print(f"  {pc_key} → data/point_cloud")
    old_pc = old_root[pc_key]
    data_group.create_dataset(
        'point_cloud',
        data=old_pc[:],
        chunks=old_pc.chunks,
        dtype=old_pc.dtype,
        compressor=old_pc.compressor
    )
    print(f"    shape={old_pc.shape}, dtype={old_pc.dtype}")
    
    # Copy image
    print(f"  {img_key} → data/img")
    old_img = old_root[img_key]
    data_group.create_dataset(
        'img',
        data=old_img[:],
        chunks=old_img.chunks,
        dtype=old_img.dtype,
        compressor=old_img.compressor
    )
    print(f"    shape={old_img.shape}, dtype={old_img.dtype}")
    
    # Copy cube positions if available
    if 'cube_pos' in old_root:
        print("  cube_pos → data/cube_pos")
        old_cube_pos = old_root['cube_pos']
        data_group.create_dataset(
            'cube_pos',
            data=old_cube_pos[:],
            chunks=old_cube_pos.chunks,
            dtype=old_cube_pos.dtype,
            compressor=old_cube_pos.compressor
        )
        print(f"    shape={old_cube_pos.shape}, dtype={old_cube_pos.dtype}")
    else:
        print("  WARNING: cube_pos not found in old zarr, skipping")
    
    # Copy camera extrinsics if available
    if extrinsics_key and extrinsics_key in old_root:
        print(f"  {extrinsics_key} → data/camera_extrinsics")
        old_extrinsics = old_root[extrinsics_key]
        data_group.create_dataset(
            'camera_extrinsics',
            data=old_extrinsics[:],
            chunks=old_extrinsics.chunks if hasattr(old_extrinsics, 'chunks') else None,
            dtype=old_extrinsics.dtype,
            compressor=old_extrinsics.compressor if hasattr(old_extrinsics, 'compressor') else None
        )
        print(f"    shape={old_extrinsics.shape}, dtype={old_extrinsics.dtype}")
    elif extrinsics_key:
        print(f"  WARNING: {extrinsics_key} not found in old zarr, skipping")
    
    # Add episode_ends to meta
    print("  episode_ends.npy → meta/episode_ends")
    meta_group.create_dataset(
        'episode_ends',
        data=episode_ends,
        dtype=np.int64
    )
    print(f"    shape={episode_ends.shape}, dtype={episode_ends.dtype}")
    
    # Add metadata attributes
    new_root.attrs['camera_view'] = camera_name
    new_root.attrs['num_episodes'] = len(episode_ends)
    new_root.attrs['total_frames'] = int(episode_ends[-1])
    new_root.attrs['state_dim'] = states_trimmed.shape[1]  # Updated dimension
    new_root.attrs['action_dim'] = actions_trimmed.shape[1]  # Updated dimension
    
    # Copy over old attributes if they exist
    if hasattr(old_root, 'attrs'):
        for key in ['cube_dim', 'state_description', 'action_description', 'cube_description']:
            if key in old_root.attrs:
                new_root.attrs[key] = old_root.attrs[key]
    
    # Update state/action descriptions to reflect trimmed data
    new_root.attrs['state_description'] = 'Trimmed robot state: 6 joint angles + gripper angle (7 dimensions)'
    new_root.attrs['action_description'] = 'Trimmed action deltas: 6 joint deltas + gripper delta (7 dimensions)'
    
    print("\n" + "=" * 60)
    print("FINAL ZARR STRUCTURE:")
    print("=" * 60)
    print(new_root.tree())
    print("\n" + "=" * 60)
    print("DATASET SUMMARY:")
    print("=" * 60)
    print(f"Camera view: {camera_name}")
    print(f"Number of episodes: {len(episode_ends)}")
    print(f"Total frames: {episode_ends[-1]}")
    print(f"\nData dimensions:")
    print(f"  state: {data_group['state'].shape} (dtype: {data_group['state'].dtype})")
    print(f"  action: {data_group['action'].shape} (dtype: {data_group['action'].dtype})")
    print(f"  img: {data_group['img'].shape} (dtype: {data_group['img'].dtype})")
    print(f"  point_cloud: {data_group['point_cloud'].shape} (dtype: {data_group['point_cloud'].dtype})")
    if 'cube_pos' in data_group:
        print(f"  cube_pos: {data_group['cube_pos'].shape} (dtype: {data_group['cube_pos'].dtype})")
    if 'camera_extrinsics' in data_group:
        print(f"  camera_extrinsics: {data_group['camera_extrinsics'].shape} (dtype: {data_group['camera_extrinsics'].dtype})")
    print(f"\nMetadata:")
    print(f"  episode_ends: {meta_group['episode_ends'].shape} (dtype: {meta_group['episode_ends'].dtype})")
    print("\n" + "=" * 60)
    
    # Validate total frames match
    total_frames_from_ends = episode_ends[-1]
    actual_frames = states_trimmed.shape[0]  # Use trimmed states
    
    print(f"\nValidation:")
    print(f"  Total frames from episode_ends: {total_frames_from_ends}")
    print(f"  Actual frames in data: {actual_frames}")
    
    if total_frames_from_ends == actual_frames:
        print("  ✓ Match! Dataset is consistent")
    else:
        print(f"  ✗ MISMATCH! Episode ends don't match data length")
        print(f"    Difference: {abs(total_frames_from_ends - actual_frames)} frames")
        print(f"    WARNING: This may cause issues during training!")
    
    print(f"\n✓ Restructured zarr saved to: {new_zarr_path}")
    return new_root

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    old_zarr_path = sys.argv[1]
    
    # Parse camera option
    camera_mode = "third_person"  # default
    if "--camera" in sys.argv:
        camera_idx = sys.argv.index("--camera")
        if camera_idx + 1 < len(sys.argv):
            camera_mode = sys.argv[camera_idx + 1]
        else:
            print("ERROR: --camera option requires a value")
            print_usage()
            sys.exit(1)
    
    if camera_mode not in ["third_person", "wrist", "both"]:
        print(f"ERROR: Invalid camera mode '{camera_mode}'")
        print("Must be one of: third_person, wrist, both")
        sys.exit(1)
    
    # Check if episode_ends.npy exists
    if not os.path.exists('episode_ends.npy'):
        print("ERROR: episode_ends.npy not found in current directory")
        print("Please run calculate_episode_ends.py first")
        sys.exit(1)
    
    # Load episode_ends
    episode_ends = np.load('episode_ends.npy')
    print(f"Loaded episode_ends: {len(episode_ends)} episodes")
    print(f"  First 5: {episode_ends[:5].tolist()}")
    print(f"  Last 5: {episode_ends[-5:].tolist()}")
    print(f"  Total frames: {episode_ends[-1]}")
    
    # Process based on camera mode
    if camera_mode == "both":
        # Process both cameras
        print("\n" + "=" * 60)
        print("Processing THIRD PERSON camera")
        print("=" * 60)
        
        tp_zarr_path = old_zarr_path.replace('.zarr', '_third_person.zarr')
        restructure_single_camera(
            old_zarr_path, tp_zarr_path, episode_ends,
            img_key='third_person_img',
            pc_key='third_person_point_cloud',
            extrinsics_key='third_person_extrinsics',
            camera_name='third_person'
        )
        
        print("\n" + "=" * 60)
        print("Processing WRIST camera")
        print("=" * 60)
        
        wr_zarr_path = old_zarr_path.replace('.zarr', '_wrist.zarr')
        restructure_single_camera(
            old_zarr_path, wr_zarr_path, episode_ends,
            img_key='wrist_img',
            pc_key='wrist_point_cloud',
            extrinsics_key='wrist_extrinsics',
            camera_name='wrist'
        )
        
        print("\n" + "=" * 60)
        print("DONE - Created two zarr files:")
        print(f"  Third person: {tp_zarr_path}")
        print(f"  Wrist: {wr_zarr_path}")
        print("=" * 60)
        
    else:
        # Process single camera
        suffix = f"_{camera_mode}" if camera_mode != "third_person" else "_restructured"
        new_zarr_path = old_zarr_path.replace('.zarr', f'{suffix}.zarr')
        
        # Determine keys based on camera mode
        if camera_mode == "third_person":
            img_key = 'third_person_img' if 'third_person_img' in zarr.open(old_zarr_path, mode='r') else 'img'
            pc_key = 'third_person_point_cloud' if 'third_person_point_cloud' in zarr.open(old_zarr_path, mode='r') else 'point_cloud'
            extrinsics_key = 'third_person_extrinsics' if 'third_person_extrinsics' in zarr.open(old_zarr_path, mode='r') else None
        else:  # wrist
            img_key = 'wrist_img'
            pc_key = 'wrist_point_cloud'
            extrinsics_key = 'wrist_extrinsics'
        
        restructure_single_camera(
            old_zarr_path, new_zarr_path, episode_ends,
            img_key=img_key,
            pc_key=pc_key,
            extrinsics_key=extrinsics_key,
            camera_name=camera_mode
        )
        
        print(f"\n✓ You can now use {new_zarr_path} for training 3D Diffusion Policy")

if __name__ == "__main__":
    main()