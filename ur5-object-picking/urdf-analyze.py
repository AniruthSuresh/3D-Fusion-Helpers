import pybullet as p
import pybullet_data
from collections import namedtuple

class UR5Robotiq85:
    def __init__(self, base_pos=[0,0,0], base_ori=[0,0,0]):
        self.base_pos = base_pos
        self.base_ori = p.getQuaternionFromEuler(base_ori)
        self.arm_num_dofs = 6
        self.eef_id = 7  # End-effector link index

    def load(self, urdf_path='./urdf/ur5_robotiq_85.urdf'):
        """Load URDF and parse joint info"""
        self.id = p.loadURDF(urdf_path, self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
        """Parse all joint info including limits"""
        JointInfo = namedtuple('JointInfo',
                               ['id','name','type','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode()
            jointType = info[2]
            lowerLimit = info[8]
            upperLimit = info[9]
            maxForce = info[10]
            maxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED
            if controllable:
                self.controllable_joints.append(jointID)
            self.joints.append(JointInfo(jointID, jointName, jointType,
                                         lowerLimit, upperLimit, maxForce, maxVelocity, controllable))

    def __setup_mimic_joints__(self):
        """Setup mimic joints for the Robotiq gripper"""
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        # Parent joint
        self.mimic_parent_id = [j.id for j in self.joints if j.name == mimic_parent_name][0]
        # Child joints with multiplier
        self.mimic_child_multiplier = {j.id: mimic_children_names[j.name] for j in self.joints if j.name in mimic_children_names}
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id, self.id, joint_id,
                                   jointType=p.JOINT_GEAR, jointAxis=[0,1,0],
                                   parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def print_joint_info(self):
        """Print all joint info"""
        for j in self.joints:
            print(f"ID: {j.id}, Name: {j.name}, Type: {j.type}, Lower: {j.lowerLimit}, Upper: {j.upperLimit}, Controllable: {j.controllable}")

# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    robot = UR5Robotiq85([0,0,0.62])
    robot.load()  # Load URDF and parse joints
    robot.print_joint_info()
