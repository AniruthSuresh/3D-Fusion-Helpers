#!/usr/bin/env python3
"""
Prune PerAct data to a single task.

This script creates a new directory (new_data) containing only data for a single task,
maintaining the exact same structure as the original data:
- new_data/peract2_test/{task_name}/  (copied from original)
- new_data/Peract2_zarr/train.zarr    (filtered to single task)
- new_data/Peract2_zarr/val.zarr      (filtered to single task)

Usage:
    python prune_to_single_task.py <task_name> [--output-dir OUTPUT_DIR]

Example:
    python prune_to_single_task.py bimanual_lift_ball
    python prune_to_single_task.py bimanual_pick_laptop --output-dir my_pruned_data
"""

import argparse
import os
import shutil
import zarr
import numpy as np
from pathlib import Path


# Task ordering from peract2_to_zarr.py - task_id = index in this list
# Source: https://github.com/Varun0157/3d_flowmatch_actor/blob/master/data_processing/peract2_to_zarr.py
# This ordering is used by the training script to map task_id -> task name
TASKS = [
    'bimanual_push_box',             # task_id 0
    'bimanual_lift_ball',            # task_id 1
    'bimanual_dual_push_buttons',    # task_id 2
    'bimanual_pick_plate',           # task_id 3
    'bimanual_put_item_in_drawer',   # task_id 4
    'bimanual_put_bottle_in_fridge', # task_id 5
    'bimanual_handover_item',        # task_id 6
    'bimanual_pick_laptop',          # task_id 7
    'bimanual_straighten_rope',      # task_id 8
    'bimanual_sweep_to_dustpan',     # task_id 9
    'bimanual_lift_tray',            # task_id 10
    'bimanual_handover_item_easy',   # task_id 11
    'bimanual_take_tray_out_of_oven' # task_id 12
]

TASK_TO_ID = {task: i for i, task in enumerate(TASKS)}


def get_task_id(task_name: str) -> int:
    """Get task_id for a given task name."""
    if task_name not in TASK_TO_ID:
        raise ValueError(f"Task '{task_name}' not found. Available tasks:\n" +
                         "\n".join(f"  {i}: {t}" for i, t in enumerate(TASKS)))
    return TASK_TO_ID[task_name]


def filter_zarr_by_task(src_zarr_path: str, dst_zarr_path: str, target_task_id: int):
    """
    Create a new zarr with only data for the specified task_id.
    The task_id is preserved (not reset) since the training script uses it
    as an index into the task list to get the task name.
    """
    src = zarr.open(src_zarr_path, 'r')

    # Find indices for the target task
    task_ids = src['task_id'][:]
    mask = task_ids == target_task_id
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(f"  Warning: No samples found for task_id {target_task_id}")
        return 0

    print(f"  Found {len(indices)} samples for task_id {target_task_id}")

    # Create destination zarr
    dst = zarr.open(dst_zarr_path, 'w')

    # Copy each array, filtering by the mask
    for key in src.keys():
        src_arr = src[key]
        filtered_data = src_arr[indices]

        # Create array with same dtype and compression
        dst.create_dataset(
            key,
            data=filtered_data,
            dtype=src_arr.dtype,
            chunks=src_arr.chunks,
            compressor=src_arr.compressor
        )
        print(f"    {key}: {src_arr.shape} -> {filtered_data.shape}")

    return len(indices)


def main():
    parser = argparse.ArgumentParser(
        description='Prune PerAct data to a single task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('task_name', type=str, help='Name of the task to extract')
    parser.add_argument('--output-dir', type=str, default='new_data',
                        help='Output directory (default: new_data)')
    parser.add_argument('--source-dir', type=str, default='.',
                        help='Source directory containing peract2_test and Peract2_zarr (default: .)')
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    test_dir = source_dir / 'peract2_test'
    zarr_dir = source_dir / 'Peract2_zarr'

    # Validate source directories exist
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not zarr_dir.exists():
        raise FileNotFoundError(f"Zarr directory not found: {zarr_dir}")

    # Validate task name and get task_id
    task_id = get_task_id(args.task_name)
    print(f"Available tasks:")
    for i, task in enumerate(TASKS):
        marker = " <--" if task == args.task_name else ""
        print(f"  {i}: {task}{marker}")
    print(f"\nSelected task: {args.task_name} (task_id={task_id})\n")

    # Validate task exists in test directory
    src_task_dir = test_dir / args.task_name
    if not src_task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {src_task_dir}")

    # Create output directory structure
    output_test_dir = output_dir / 'peract2_test'
    output_zarr_dir = output_dir / 'Peract2_zarr'

    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_test_dir.mkdir(parents=True, exist_ok=True)
    output_zarr_dir.mkdir(parents=True, exist_ok=True)

    # Copy task directory from peract2_test
    dst_task_dir = output_test_dir / args.task_name
    print(f"Copying {src_task_dir} -> {dst_task_dir}")
    shutil.copytree(src_task_dir, dst_task_dir)
    print(f"  Done copying test data\n")

    # Filter and copy zarr files
    for zarr_name in ['train.zarr', 'val.zarr']:
        src_zarr = zarr_dir / zarr_name
        dst_zarr = output_zarr_dir / zarr_name

        if not src_zarr.exists():
            print(f"Warning: {src_zarr} not found, skipping")
            continue

        print(f"Filtering {zarr_name}:")
        n_samples = filter_zarr_by_task(str(src_zarr), str(dst_zarr), task_id)
        print(f"  Created {dst_zarr} with {n_samples} samples\n")

    print(f"Done! Pruned data saved to: {output_dir}")
    print(f"  - {output_test_dir / args.task_name}")
    print(f"  - {output_zarr_dir / 'train.zarr'}")
    print(f"  - {output_zarr_dir / 'val.zarr'}")


if __name__ == '__main__':
    main()
