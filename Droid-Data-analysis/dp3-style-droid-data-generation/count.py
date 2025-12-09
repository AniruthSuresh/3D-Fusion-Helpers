import os

root = "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/Droid-Data-analysis/master_data/point_cloud"

total_pcs = 0

# List all subfolders
subfolders = sorted(os.listdir(root), key=lambda x: int(x))

for subf in subfolders:
    subf_path = os.path.join(root, subf)
    if os.path.isdir(subf_path):
        # Count files ending with .ply
        pc_files = [f for f in os.listdir(subf_path) if f.lower().endswith(".ply")]
        count = len(pc_files)
        total_pcs += count
        print(f"Subfolder {subf}: {count} point clouds")

print(f"\nTotal point clouds across all subfolders: {total_pcs}")
