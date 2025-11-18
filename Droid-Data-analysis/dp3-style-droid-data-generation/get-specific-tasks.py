import json

# User-defined keywords (both must appear)
KEYWORDS = ["put", "marker"]

# Load the main JSON
with open("/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/filtered-iou-sorted-cleaned.json", "r") as f:
    clean_data = json.load(f)

# Load the annotation JSON
with open("/home/aniruth/Desktop/3D-Fusion-Helpers/Droid-Data-analysis/droid_language_annotations.json", "r") as f:
    anno_data = json.load(f)

# Filtered dictionary
final_data = {}

for key, val in clean_data.items():
    # Get the corresponding annotation
    if key in anno_data:
        instructions = anno_data[key]
        # Check all language_instruction fields
        found = False
        for k2, instr in instructions.items():
            instr_lower = instr.lower()
            if all(keyword.lower() in instr_lower for keyword in KEYWORDS):
                found = True
                break
        if found:
            final_data[key] = val

# Save to final JSON
with open("final.json", "w") as f:
    json.dump(final_data, f, indent=4)

print(f"Filtered {len(final_data)} entries into final.json")
