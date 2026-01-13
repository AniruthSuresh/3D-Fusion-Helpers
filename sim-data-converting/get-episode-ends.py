#!/usr/bin/env python3
"""
Calculate episode_ends.npy by counting actual frames in each trajectory folder.
Handles cases where zero-action frames have been removed.
"""
import os
import sys
import numpy as np
from pathlib import Path

def count_frames_in_trajectory(traj_folder, camera="third_person"):
    """
    Count the number of frames in a trajectory by counting RGB files.
    
    Args:
        traj_folder: Path to trajectory folder (e.g., processed-sim-data-new/third_person_rgb/iter_0000)
        camera: Which camera view to count ("third_person" or "wrist")
    
    Returns:
        Number of frames in this trajectory
    """
    rgb_files = sorted([f for f in os.listdir(traj_folder) if f.endswith('.png')])
    return len(rgb_files)

def calculate_episode_ends_from_structure(data_root, camera="third_person"):
    """
    Calculate episode ends by scanning the actual directory structure.
    
    Args:
        data_root: Root directory containing processed data
        camera: Which camera view to use for counting ("third_person" or "wrist")
    
    Returns:
        episode_ends: numpy array of cumulative frame counts
    """
    # Find RGB directory for the specified camera
    rgb_dir = os.path.join(data_root, f"{camera}_rgb")
    
    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    
    # Get all trajectory folders (e.g., iter_0000, iter_0001, ...)
    traj_folders = sorted([
        d for d in os.listdir(rgb_dir) 
        if os.path.isdir(os.path.join(rgb_dir, d)) and d.startswith("iter_")
    ])
    
    if len(traj_folders) == 0:
        raise ValueError(f"No trajectory folders found in {rgb_dir}")
    
    print(f"Found {len(traj_folders)} trajectories in {rgb_dir}")
    
    # Count frames in each trajectory
    episode_lengths = []
    for traj_folder in traj_folders:
        traj_path = os.path.join(rgb_dir, traj_folder)
        num_frames = count_frames_in_trajectory(traj_path, camera)
        episode_lengths.append(num_frames)
        print(f"  {traj_folder}: {num_frames} frames")
    
    # Convert to cumulative episode ends
    episode_lengths = np.array(episode_lengths, dtype=np.int64)
    episode_ends = np.cumsum(episode_lengths)
    
    return episode_ends, episode_lengths

def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_episode_ends.py <data_root> [camera]")
        print("\nExamples:")
        print("  python calculate_episode_ends.py ./processed-sim-data-new")
        print("  python calculate_episode_ends.py ./processed-sim-data-new third_person")
        print("  python calculate_episode_ends.py ./processed-sim-data-new wrist")
        print("\nThis script will:")
        print("  1. Scan the directory structure in <data_root>")
        print("  2. Count frames in each trajectory (after zero-action removal)")
        print("  3. Generate episode_ends.npy for zarr restructuring")
        sys.exit(1)
    
    data_root = sys.argv[1]
    camera = sys.argv[2] if len(sys.argv) > 2 else "third_person"
    
    if camera not in ["third_person", "wrist"]:
        print(f"ERROR: Invalid camera '{camera}'. Must be 'third_person' or 'wrist'")
        sys.exit(1)
    
    print(f"Calculating episode ends from: {data_root}")
    print(f"Using camera view: {camera}")
    print("-" * 60)
    
    try:
        episode_ends, episode_lengths = calculate_episode_ends_from_structure(data_root, camera)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Number of episodes: {len(episode_ends)}")
        print(f"Total frames: {episode_ends[-1]}")
        print(f"\nEpisode lengths (frames per trajectory):")
        print(f"  Min: {episode_lengths.min()}")
        print(f"  Max: {episode_lengths.max()}")
        print(f"  Mean: {episode_lengths.mean():.1f}")
        print(f"  Median: {np.median(episode_lengths):.1f}")
        
        print(f"\nFirst 5 episode ends: {episode_ends[:5].tolist()}")
        print(f"Last 5 episode ends: {episode_ends[-5:].tolist()}")
        
        # Save episode_ends
        output_file = "episode_ends.npy"
        np.save(output_file, episode_ends)
        print(f"\n✓ Saved to {output_file}")
        
        # Also save episode lengths for reference
        lengths_file = "episode_lengths.txt"
        np.savetxt(lengths_file, episode_lengths, fmt='%d')
        print(f"✓ Saved episode lengths to {lengths_file}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()