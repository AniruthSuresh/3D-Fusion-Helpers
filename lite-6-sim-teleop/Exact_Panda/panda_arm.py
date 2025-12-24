import pybullet as p
import pybullet_data

# Initialize PyBullet in GUI mode
p.connect(p.GUI)

# Add PyBullet's default search path for built-in URDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane and Panda robot URDF
plane_id = p.loadURDF("plane.urdf")
# panda_id = p.loadURDF("/home/aniruth/Desktop/RRC/XARM7/xArm-Python-SDK/example/wrapper/xarm7/Follow_DROID/Exact_Panda/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf", useFixedBase=True)

panda_id = p.loadURDF("/home/aniruth/Desktop/RRC/XARM7/xArm-Python-SDK/example/wrapper/xarm7/Follow_DROID/Exact_Panda/bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda_with_2F85_sec.urdf", useFixedBase=True)

# Set simulation parameters
p.setGravity(0, 0, -9.8)
p.setTimeStep(1.0 / 240.0)

# Keep the simulation running
while True:
    p.stepSimulation()
