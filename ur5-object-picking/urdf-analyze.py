import pybullet as p
import pybullet_data
from collections import namedtuple
import time

class UR5Robotiq85:
    def __init__(self, base_pos=[0,0,0], base_ori=[0,0,0]):
        self.base_pos = base_pos
        self.base_ori = p.getQuaternionFromEuler(base_ori)
        self.arm_num_dofs = 6
        self.eef_id = 7

    def load(self, urdf_path='./urdf/ur5_robotiq_85.urdf'):
        self.id = p.loadURDF(
            urdf_path,
            self.base_pos,
            self.base_ori,
            useFixedBase=True
        )
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
        JointInfo = namedtuple(
            'JointInfo',
            ['id','name','type','lowerLimit','upperLimit','maxForce','maxVelocity','controllable']
        )
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

            self.joints.append(
                JointInfo(jointID, jointName, jointType,
                          lowerLimit, upperLimit,
                          maxForce, maxVelocity,
                          controllable)
            )

    def __setup_mimic_joints__(self):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }

        self.mimic_parent_id = [
            j.id for j in self.joints if j.name == mimic_parent_name
        ][0]

        for j in self.joints:
            if j.name in mimic_children_names:
                multiplier = mimic_children_names[j.name]
                c = p.createConstraint(
                    self.id,
                    self.mimic_parent_id,
                    self.id,
                    j.id,
                    jointType=p.JOINT_GEAR,
                    jointAxis=[0,1,0],
                    parentFramePosition=[0,0,0],
                    childFramePosition=[0,0,0]
                )
                p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def create_joint_sliders(self):
        self.joint_sliders = {}

        for j in self.joints:
            if not j.controllable:
                continue

            lower = j.lowerLimit
            upper = j.upperLimit

            # Continuous joints
            if lower > upper:
                lower, upper = -3.14, 3.14

            slider = p.addUserDebugParameter(
                j.name,
                lower,
                upper,
                0.0
            )
            self.joint_sliders[j.id] = slider

    def apply_slider_controls(self):
        for joint_id, slider_id in self.joint_sliders.items():
            target = p.readUserDebugParameter(slider_id)
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=200
            )

    def print_joint_info(self):
        for j in self.joints:
            print(f"ID {j.id:2d} | {j.name:30s} | "
                  f"lower {j.lowerLimit:.2f} | upper {j.upperLimit:.2f} | "
                  f"controllable {j.controllable}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")

    robot = UR5Robotiq85([0,0,0.62])
    robot.load()
    robot.print_joint_info()
    robot.create_joint_sliders()

    while True:
        robot.apply_slider_controls()
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
