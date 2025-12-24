import rerun as rr
import os
import trimesh

PCD_DIR = "./dataset/iter_0001/pcd"

# 1. Initialize the recording with a name
rr.init("ur5_pointclouds", spawn=True)

# 2. Iterate through your files
for t, ply_file in enumerate(sorted(os.listdir(PCD_DIR))):
    if not ply_file.endswith(".ply"):
        continue

    # Set the time for this log entry
    rr.set_time_sequence("step", t)

    mesh = trimesh.load(os.path.join(PCD_DIR, ply_file), process=False)

    # 3. Log directly using rr.log
    rr.log(
        "world/pointcloud",
        rr.Points3D(
            mesh.vertices,
            colors=mesh.visual.vertex_colors[:, :3]
            if hasattr(mesh.visual, "vertex_colors") else None,
        )
    )