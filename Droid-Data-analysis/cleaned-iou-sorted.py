"""
Cleans the filtered_iou_sorted.json by removing entries with WEIRD or FAILURE in their paths.
"""
import json

# Load the original JSON
with open("/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/filtered_iou_sorted.json", "r") as f:
    data = json.load(f)

filtered = {}

# Filter out entries containing WEIRD or FAILURE in their relative_path
for k, v in data.items():
    path = v.get("relative_path", "")
    if "WEIRD" not in path and "FAILURE" not in path and "failure" not in path:
        filtered[k] = v

# Dump to new cleaned JSON file
output_path = "/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/filtered-iou-sorted-cleaned.json"

with open(output_path, "w") as f:
    json.dump(filtered, f, indent=4)

print(f"Saved cleaned JSON to: {output_path}")
