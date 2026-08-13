# Raw Sim Data -> Zarr Pipeline

This document explains the 5-script pipeline in:
- sim-data-converting/master-script-sim.py
- sim-data-converting/point-cloud-filtering.py
- sim-data-converting/chunking-data.py
- sim-data-converting/get-episode-ends.py
- sim-data-converting/get-final-zarr.py

It also includes a trajectory-level walkthrough using iter_0080 as an example.

## Quick Sequence

1. master-script-sim.py
2. point-cloud-filtering.py
3. chunking-data.py
4. get-episode-ends.py
5. get-final-zarr.py

## Stage 1: master-script-sim.py

Purpose:
- Convert raw simulation trajectory folders into cleaned, model-ready per-frame assets.
- Keep states/actions/cube poses aligned with image/point-cloud frames.

Input root:
- ../ur5-object-picking/dataset/

Expected per-trajectory structure (example iter_0080):
- iter_0080/agent_pos.npy
- iter_0080/actions.npy
- iter_0080/cube_pos.npy (optional)
- iter_0080/third_person/rgb/tp_rgb_0000.png ...
- iter_0080/third_person/pcd/tp_pcd_0000.npy ...
- iter_0080/wrist/rgb/wr_rgb_0000.png ...
- iter_0080/wrist/pcd/wr_pcd_0000.npy ...
- iter_0080/camera_poses/pose_0000.json ...

What it does:
- Finds all iter_* folders.
- Loads:
  - states from agent_pos.npy (expected dim 13)
  - actions from actions.npy (expected dim 13)
  - cube poses from cube_pos.npy (expected dim 7, otherwise zeros)
- For each camera (third_person and wrist), per frame:
  - Copies RGB png into processed output
  - Loads Nx3 point cloud npy, removes invalid points (non-finite, z <= 0, z >= 2.5)
  - Colors cloud with the RGB frame, writes cloud_XXXXX.ply
  - Reads pose JSON and stores 4x4 extrinsics (flattened to 16 values)
- Removes frames where action is all zeros (across all 13 action dims)
  - Removes those rows from states/actions/cube poses
  - Deletes corresponding RGB and PLY files for both cameras
- Saves cleaned per-trajectory text files.

Outputs under:
- ./processed-sim-data-new/
  - third_person_rgb/iter_0080/rgb_00000.png ...
  - third_person_pc/iter_0080/cloud_00000.ply ...
  - wrist_rgb/iter_0080/rgb_00000.png ...
  - wrist_pc/iter_0080/cloud_00000.ply ...
  - third_person_extrinsics/iter_0080.txt
  - wrist_extrinsics/iter_0080.txt
  - states/iter_0080.txt
  - actions/iter_0080.txt
  - cube_pos/iter_0080.txt

Important note:
- Current code has a bug: process_camera uses iter_name but does not receive it as an argument.
- In sim-data-converting/master-script-sim.py this can raise NameError when writing extrinsics.

## Stage 2: point-cloud-filtering.py

Purpose:
- Post-process PLY clouds from Stage 1 and downsample to fixed size for learning.

Input root:
- ./processed-sim-data-new/
  - third_person_pc/iter_xxxx/*.ply
  - wrist_pc/iter_xxxx/*.ply

What it does for each cloud:
- Loads xyz + rgb from PLY.
- Computes dynamic workspace bounds per cloud using mean +/- n_std * std.
- Crops points outside workspace bounds.
- Runs farthest point sampling (PyTorch3D) to keep up to 6000 points.
- Saves cloud as *_filtered.ply.
- Optional visualization is shown once.

Output root:
- ./filtered-pc/
  - third_person/iter_0080/cloud_00000_filtered.ply ...
  - wrist/iter_0080/cloud_00000_filtered.ply ...

Practical note:
- The code sets n_std=10, which is very permissive. It removes only extreme outliers.

## Stage 3: chunking-data.py

Purpose:
- Aggregate all trajectories into one flat zarr file with chunked arrays.

Input root in current script:
- /media/skills/RRC HDD A/cross-emb/src/final-data

Expected folders inside that root:
- rgb/<iter_folder>/*.png
- filtered-pc/<iter_folder>/*.ply
- final-actions/<iter_folder>.txt
- final-states/<iter_folder>.txt
- final-cube-pos/<iter_folder>.txt

What it does:
- Iterates each trajectory folder.
- Reads all RGB frames into all_imgs.
- Reads filtered PLY into point-cloud arrays [x,y,z,r,g,b].
- Reads action/state/cube text files.
- Stacks all trajectories into flat arrays:
  - img: [N, H, W, 3]
  - point_cloud: [N, P, 6]
  - actions: [N, A]
  - states: [N, S]
  - cube_pos: [N, 7]
- Verifies all modalities have same N.
- Writes final.zarr with compressed, chunked datasets.
- Adds metadata attributes.

Output:
- final.zarr
  - img
  - point_cloud
  - actions
  - states
  - cube_pos

Important alignment note:
- This script expects a merged folder layout (rgb, filtered-pc, final-actions, ...).
- Stage 1/2 output names are different (third_person_rgb, actions, states, cube_pos, filtered-pc/third_person, ...).
- Usually you either:
  - adapt chunking-data.py paths, or
  - reorganize Stage 1/2 outputs into the expected final-data layout.

## Stage 4: get-episode-ends.py

Purpose:
- Build episode boundaries after frame filtering/removal.

Input:
- processed data root (for example ./processed-sim-data-new)
- camera choice (third_person or wrist)

What it does:
- Scans <data_root>/<camera>_rgb/iter_*.
- Counts png frames in each trajectory folder.
- Builds:
  - episode_lengths: per-trajectory frame counts
  - episode_ends: cumulative sums of episode lengths
- Saves:
  - episode_ends.npy
  - episode_lengths.txt

Why this matters:
- Training code needs episode boundaries to split one long flattened sequence into trajectories.

## Stage 5: get-final-zarr.py

Purpose:
- Restructure flat final.zarr into a diffusion-policy-friendly schema.

Input:
- old zarr (for example final.zarr)
- episode_ends.npy from Stage 4
- camera mode: third_person, wrist, or both

What it does:
- Creates new zarr with groups:
  - data/
  - meta/
- Copies and renames arrays:
  - states -> data/state
  - actions -> data/action
  - img key -> data/img
  - point cloud key -> data/point_cloud
  - cube_pos -> data/cube_pos (if present)
  - camera extrinsics -> data/camera_extrinsics (if present)
- Trims first 6 columns from states/actions:
  - removes eef_pos(3) + eef_orn(3)
  - keeps joints(6) + gripper(1) => 7 dims
- Writes meta/episode_ends from episode_ends.npy
- Validates episode_ends[-1] equals number of data rows.

Output examples:
- final_restructured.zarr (single camera default)
- final_wrist.zarr (wrist mode)
- final_third_person.zarr + final_wrist.zarr (both mode)

## iter_0080 Walkthrough Example

Assume raw trajectory:
- dataset/iter_0080 has 150 frames before filtering.

After Stage 1:
- Suppose 20 frames have all-zero actions.
- Remaining frames: 130.
- Files saved:
  - processed-sim-data-new/third_person_rgb/iter_0080 has 130 png files
  - processed-sim-data-new/third_person_pc/iter_0080 has 130 ply files
  - processed-sim-data-new/states/iter_0080.txt has shape [130, 13]
  - processed-sim-data-new/actions/iter_0080.txt has shape [130, 13]
  - processed-sim-data-new/cube_pos/iter_0080.txt has shape [130, 7]

After Stage 2:
- Each PLY becomes filtered/downsampled.
- Example frame cloud shape becomes [6000, 3] xyz (+ rgb colors attached in PLY).

After Stage 3:
- iter_0080 is appended into global flattened arrays.
- If previous trajectories contributed 10450 frames total, then iter_0080 occupies rows:
  - start index = 10450
  - end index = 10450 + 130 - 1 = 10579

After Stage 4:
- episode_ends entry for iter_0080 becomes 10580.

After Stage 5:
- final structured zarr stores:
  - data/state with dim 7 (trimmed)
  - data/action with dim 7 (trimmed)
  - meta/episode_ends containing 10580 at iter_0080 boundary.

## Minimal Run Order (from sim-data-converting)

1. python3 master-script-sim.py
2. python3 point-cloud-filtering.py
3. python3 chunking-data.py
4. python3 get-episode-ends.py ./processed-sim-data-new third_person
5. python3 get-final-zarr.py ./final.zarr --camera third_person

## Sanity Checks Before Training

- Number of rows must match across img, point_cloud, states, actions, cube_pos.
- episode_ends[-1] must equal data/state.shape[0].
- Point clouds should have consistent point count after filtering/FPS.
- Confirm whether training expects 13D state/action or trimmed 7D state/action.
