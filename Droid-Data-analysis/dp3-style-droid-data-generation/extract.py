"""
Script to extract the middle(cause right cam we want) SVO recording and trajectory files for top-K quality metric data samples.
"""

import json
import os
import subprocess

JSON_PATH = "/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/dp3-style-droid-data-generation/final.json"
OUT_DIR = "dp3_style_data"
TOP_N = 4   # Change to get top-K
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

    # ---- List SVO files ----
    print(f"\nListing SVO files in: {gcs_prefix}/recordings/SVO")
    result = subprocess.run(
        ["gsutil", "ls", f"{gcs_prefix}/recordings/SVO/"],
        capture_output=True, text=True, check=True
    )
    svo_files = [f.strip() for f in result.stdout.splitlines() if f.endswith(".svo")]
    if not svo_files:
        print("No SVO files found!")
        continue

    # Sort and select middle SVO
    svo_files.sort()
    middle_idx = len(svo_files) // 2
    middle_svo = svo_files[middle_idx]
    print(f"Selected middle SVO: {middle_svo}")

    # ---- Download only the middle SVO ----
    subprocess.run(["gsutil", "-m", "cp", middle_svo, target_dir], check=True)

    # ---- Download trajectory file ----
    traj_file = f"{gcs_prefix}/trajectory.h5"
    print(f"Downloading trajectory file: {traj_file}")
    subprocess.run(["gsutil", "-m", "cp", traj_file, target_dir], check=True)

print("\nAll downloads complete.")
print(f"Saved in: {OUT_DIR}")
