import os

# Ask user for the root folder
root = "/home/aniruth/Desktop/RRC/3D-Fusion-Helpers/Droid-Data-analysis/master_data"

folders = ["action", "state"]  # folders to process

for folder in folders:
    folder_path = os.path.join(root, folder)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
            # Remove first line
            updated_lines = lines[1:]
            # Save updated file
            updated_file_path = os.path.join(folder_path, file.replace(".txt", "_updated.txt"))
            with open(updated_file_path, "w") as f:
                f.writelines(updated_lines)
            print(f"Processed {file} -> {updated_file_path}")
