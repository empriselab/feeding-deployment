import numpy as np
from scipy.spatial.transform import Rotation
import threading

# ros imports
try:
    import rospy
    import tf2_ros
    from geometry_msgs.msg import Point
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String, Float64, Bool
    ROSPY_IMPORTED = True
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

from feeding_deployment.utils.pixel_selector import PixelSelector
from feeding_deployment.utils.camera_utils import angle_between_pixels, pixel2World, world2Pixel
from feeding_deployment.utils.tf_utils import TFUtils
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentWorldState

from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.wrist_controller.wrist_controller import WristInterface
from feeding_deployment.robot_controller.command_interface import (
    CartesianCommand,
    CloseGripperCommand,
    JointCommand,
    KinovaCommand,
    OpenGripperCommand,
)

from pybullet_helpers.geometry import Pose

# PLATE_HEIGHT = 0.16 # 0.192 for scooping, 0.2 for skewering, 0.198 for pushing, twirling
PLATE_HEIGHT = 0.12 # 0.192 for scooping, 0.2 for skewering, 0.198 for pushing, twirling

class FoodManipulationSkillLibrary:
    def __init__(self, sim : FeedingDeploymentPyBulletSimulator, robot_interface: ArmInterfaceClient, wrist_interface: WristInterface, perception_interface: PerceptionInterface, rviz_interface: RVizInterface, no_waits=False):
        
        self.sim = sim
        self.robot_interface = robot_interface
        self.wrist_interface = wrist_interface
        self.perception_interface = perception_interface
        self.rviz_interface = rviz_interface
        self.no_waits = no_waits

        self.pixel_selector = PixelSelector()
        if self.robot_interface is not None:
            self.tf_utils = TFUtils()

        print("Skill library initialized")

    def move_to_joint_positions(self, joint_positions):

        plan = self.sim.plan_to_joint_positions(joint_positions)
        print("Plan has length", len(plan))
        if self.robot_interface is None:
            self.sim.visualize_plan(plan)
        else:
            self.robot_interface.execute_command(JointCommand(pos=self.sim.scene_description.retract_pos[:7]))

    def move_to_ee_pose(self, pose, plan_override=False):

        if not plan_override:
            plan = self.sim.plan_to_ee_pose(pose)
        if self.robot_interface is None:
            self.sim.visualize_plan(plan)
        else:
            self.robot_interface.execute_command(CartesianCommand(pos=pose.position, quat=pose.orientation))

    def set_wrist_state(self, pitch_angle, roll_angle):
        if self.robot_interface is None:
            raise NotImplementedError("Wrist state setting not implemented for simulation")
        else:
            self.wrist_interface.set_wrist_state(pitch_angle, roll_angle)

    def reset(self):

        self.move_to_joint_positions(self.sim.scene_description.above_plate_pos)
        self.set_wrist_state(0, 0)

    def move_utensil_to_pose(self, tip_pose, tip_to_wrist = None):

        if self.robot_interface is not None:

            self.tf_utils.publishTransformationToTF('base_link', 'fork_tip_target', tip_pose)

            if tip_to_wrist is None:
                tip_to_wrist = self.tf_utils.getTransformationFromTF('fork_tip', 'tool_frame')
            tool_frame_target = tip_pose @ tip_to_wrist

            self.rviz_interface.visualize_fork(tip_pose)
            self.tf_utils.publishTransformationToTF('base_link', 'tool_frame_target', tool_frame_target)
            
            if not self.no_waits:
                input("Execute command?")

            pose = Pose.from_matrix(tool_frame_target)
            self.move_to_ee_pose(pose)
        else:
            raise NotImplementedError("Move utensil to pose not implemented for simulation")
    
    def get_transform(self, from_frame, to_frame):
        if self.robot_interface is not None:
            return self.tf_utils.getTransformationFromTF(from_frame, to_frame)
        else:
            raise NotImplementedError("Get transform not implemented for simulation")

    def skewering_skill(self, color_image, depth_image, camera_info, keypoint=None, major_axis=None, action_index=0):
        if keypoint is not None:
            (center_x, center_y) = keypoint
        else:
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            major_axis = -np.pi/2
        
        print(f"Center x {center_x}, Center y {center_y}, Action index {action_index}")

        # get 3D point from depth image
        validity, point = pixel2World(camera_info, center_x, center_y, depth_image)
        # breakpoint()
        if not validity:
            print("Invalid point")
            return

        print("Getting transformation from base_link to camera_color_optical_frame")
        food_transform = np.eye(4)
        food_transform[:3,3] = point.reshape(1,3)

        food_base = self.get_transform("base_link", "camera_color_optical_frame") @ food_transform
        print("---- Height of skewer point: ", food_base[2,3])

        print("Food detection height: ", food_base[2,3])
        if not self.no_waits:
            input("Press enter to continue")
        food_base[2,3] = max(food_base[2,3] - 0.01, PLATE_HEIGHT) 
        print("---- Height of skewer point (after max): ", food_base[2,3]) 

        food_base[:3,:3] = Rotation.from_euler('xyz', [0,0,0], degrees=True).as_matrix()

        # magic number for skewering offset
        food_base[0,3] += 0.012

        if self.robot_interface is not None:
            self.tf_utils.publishTransformationToTF('base_link', 'food_frame', food_base)
            self.rviz_interface.visualize_food(food_base)

        base_to_tip = self.get_transform('base_link', 'fork_tip')
        food_base[:3,:3] = food_base[:3,:3] @ base_to_tip[:3,:3]

        if major_axis < np.pi/2:
            major_axis = major_axis + np.pi/2

        # caching this so that the robot doesn't rotate the wrist again
        tip_to_wrist = self.get_transform('fork_tip', 'tool_frame')
        
        # Action 0: Rotate twirl DoF to skewer angle
        self.set_wrist_state(0, -major_axis)

        # Action 1: Move to action start position
        waypoint_1_tip = np.copy(food_base)
        waypoint_1_tip[2,3] += 0.05
        self.move_utensil_to_pose(waypoint_1_tip, tip_to_wrist)

        # Action 2: Move inside food item
        waypoint_2_tip = np.copy(food_base)
        self.move_utensil_to_pose(waypoint_2_tip, tip_to_wrist)

        # Rajat ToDo: Switch to scooping pick up
        self.scooping_pickup()
        # self.move_utensil_to_pose(waypoint_1_tip)

    def scooping_pickup(self, hack = True):

        forkpitch_to_tip = self.get_transform('forkpitch', 'fork_tip')
        print("Forkpitch to tip: ", forkpitch_to_tip)
        distance = forkpitch_to_tip[0,3]

        print("Distance: ", distance)

        tool_frame = self.get_transform('base_link', 'tool_frame')

        tool_frame_displacement = np.eye(4)
        tool_frame_displacement[0,3] = distance/8 # move down
        tool_frame_displacement[1,3] = -distance*3/4 # move back

        tool_frame_target = tool_frame @ tool_frame_displacement

        if self.robot_interface is not None:
            self.tf_utils.publishTransformationToTF('base_link', 'tool_frame_target', tool_frame_target)
        
        # input("Press enter to start scooping pickup")

        if self.robot_interface is not None:
            scoop_thread = threading.Thread(target=self.wrist_interface.scoop_wrist)
            scoop_thread.start()
        else:
            raise NotImplementedError("Scooping pickup not implemented for simulation")

        # input("Press enter to also move the robot...")
        self.move_to_ee_pose(Pose.from_matrix(tool_frame_target), plan_override=True) # Necessary so that robot doesn't spend time planning in simulation

        # wait for scoop thread to finish
        if self.robot_interface is not None:
            scoop_thread.join()