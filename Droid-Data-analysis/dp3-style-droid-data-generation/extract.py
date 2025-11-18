"""
Script to extract the svo recordings and trajectory files for top-K quality metric data samples.
"""

import json
import os
import subprocess


JSON_PATH = "/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/filtered-iou-sorted-cleaned.json"
OUT_DIR = "dp3_style_data"
TOP_N = 1   # Change to get top-K
# ==============================

with open(JSON_PATH, "r") as f:
    data = json.load(f)

items = []
for k, v in data.items():
    metric = v.get("max_quality_metric", 0)
    rel = v.get("relative_path", "")
    items.append((metric, rel))

items.sort(reverse=True, key=lambda x: x[0])
top_items = items[:TOP_N]

print("Top items:")
for i, (metric, path) in enumerate(top_items, 1):
    print(i, metric, path)

os.makedirs(OUT_DIR, exist_ok=True)


for idx, (metric, rel_path) in enumerate(top_items, 1):
    target_dir = os.path.join(OUT_DIR, str(idx))
    os.makedirs(target_dir, exist_ok=True)

    gcs_prefix = f"gs://gresearch/robotics/droid_raw/1.0.1/{rel_path}"

    # Folder to copy (recursive)
    svo_folder = f"{gcs_prefix}/recordings/SVO"

    # File to copy
    traj_file = f"{gcs_prefix}/trajectory.h5"

    # ---- Download SVO folder ----
    print(f"\nDownloading folder recursively: {svo_folder}")
    subprocess.run(["gsutil", "-m", "cp", "-r", svo_folder, target_dir], check=True)

    # ---- Download trajectory file ----
    print(f"\nDownloading trajectory file: {traj_file}")
    subprocess.run(["gsutil", "-m", "cp", traj_file, target_dir], check=True)


print("\nAll downloads complete.")
print(f"Saved in: {OUT_DIR}")