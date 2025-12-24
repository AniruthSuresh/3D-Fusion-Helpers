import pybullet as p
import pybullet_data
import time
import signal
import sys

# Connect to PyBullet (GUI)
physicsClient = p.connect(p.GUI)

# Optional: set search path for built-in URDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Basic simulation settings
p.setGravity(0, 0, -9.81)

# Load ground plane (optional but useful)
plane_id = p.loadURDF("plane.urdf")

# Load your URDF
# Replace this path with your own URDF file
robot_id = p.loadURDF(
    "./urdf-data/cube/cube.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)

print("URDF loaded. Press Ctrl+C to exit.")

# Graceful exit on Ctrl+C
def signal_handler(sig, frame):
    print("\nExiting simulation...")
    p.disconnect()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Keep simulation alive
while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
