import json

# Load your JSON file
with open("/home/aniruth/Desktop/RRC/point-cloud-droid-data/droid/cam2base_extrinsic_superset.json", "r") as f:
    data = json.load(f)

filtered = {}

for session, session_data in data.items():
    new_entry = {"relative_path": session_data.get("relative_path", "")}
    keep = False
    max_quality = -1  # keep track of highest IoU quality per session

    # find all metric ids (like 22008760, 24400334)
    ids = {k.split('_')[0] for k in session_data.keys() if k.endswith("_metric_type")}

    for obj_id in ids:
        metric_type = session_data.get(f"{obj_id}_metric_type")
        quality = session_data.get(f"{obj_id}_quality_metric")

        # check conditions
        if metric_type == "IoU" and quality and quality > 0.95:
            # copy only the relevant data for this id
            new_entry[obj_id] = session_data[obj_id]
            new_entry[f"{obj_id}_metric_type"] = metric_type
            new_entry[f"{obj_id}_quality_metric"] = quality
            new_entry[f"{obj_id}_source"] = session_data.get(f"{obj_id}_source", "")
            keep = True
            max_quality = max(max_quality, quality)

    if keep:
        new_entry["max_quality_metric"] = max_quality  # store for sorting
        filtered[session] = new_entry

# Sort sessions by their highest IoU quality (descending)
sorted_filtered = dict(sorted(filtered.items(), key=lambda x: x[1]["max_quality_metric"], reverse=True))

# Print sorted result
print(json.dumps(sorted_filtered, indent=4))

# Save sorted data to file
with open("filtered_iou_sorted.json", "w") as f:
    json.dump(sorted_filtered, f, indent=4)
