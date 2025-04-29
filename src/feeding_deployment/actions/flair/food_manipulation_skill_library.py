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

from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.control.wrist_controller.wrist_controller import WristInterface
from feeding_deployment.control.robot_controller.command_interface import (
    CartesianCommand,
    CloseGripperCommand,
    JointCommand,
    KinovaCommand,
    OpenGripperCommand,
)

from pybullet_helpers.geometry import Pose

class FoodManipulationSkillLibrary:
    def __init__(self, sim : FeedingDeploymentPyBulletSimulator, robot_interface: ArmInterfaceClient, wrist_interface: WristInterface, perception_interface: PerceptionInterface, rviz_interface: RVizInterface, no_waits=False):
        
        self.sim = sim
        self.robot_interface = robot_interface
        self.wrist_interface = wrist_interface
        self.perception_interface = perception_interface
        self.rviz_interface = rviz_interface
        self.no_waits = no_waits

        if self.sim.scene_description.scene_label == "wheelchair":
            self.plate_height = 0.12
        elif self.sim.scene_description.scene_label == "vention":
            # self.plate_height = 0.155 # for silicone fork
            # self.plate_height = 0.158 # for metal fork
            # self.plate_height = 0.16 # for metal fork
            # self.plate_height = 0.185
            # self.plate_height = 0.197 # green table
            # self.plate_height = 0.221
            self.plate_height = 0.166
        else:
            raise NotImplementedError("Scene label not recognized; plate height required for bite acquisition")

        self.pixel_selector = PixelSelector()
        if self.robot_interface is not None:
            self.tf_utils = TFUtils()

        self.cached_reset_tip_to_wrist =  np.array(
            [[ 4.97225726e-05, -1.53284719e-03, -9.99998824e-01, -1.79554958e-02],
            [-3.22083141e-02,  9.99480001e-01, -1.53365339e-03,  5.15432893e-04],
            [ 9.99481176e-01,  3.22083525e-02,  3.26293322e-07, -2.55042925e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
        )

        print("Skill library initialized")

    def move_to_joint_positions(self, joint_positions):

        if self.robot_interface is None:
            plan = self.sim.plan_to_joint_positions(joint_positions)
            print("Plan has length", len(plan))
            self.sim.visualize_plan(plan)
        else:
            self.robot_interface.execute_command(JointCommand(pos=joint_positions))

    def move_to_ee_pose(self, pose, plan_override=False):

        if not plan_override and not self.no_waits:
            plan = self.sim.plan_to_ee_pose(pose)
        if self.robot_interface is None:
            self.sim.visualize_plan(plan)
        else:
            self.robot_interface.execute_command(CartesianCommand(pos=pose.position, quat=pose.orientation))

    def set_wrist_state(self, pitch_angle, roll_angle):
        if self.robot_interface is None:
            self.sim.set_wrist_state(pitch_angle, roll_angle)
        else:
            self.wrist_interface.set_wrist_state(pitch_angle, roll_angle)

    def robot_reset(self):
        self.move_to_joint_positions(self.sim.scene_description.above_plate_pos)

    def reset(self):

        print("Moving to above plate position: ", self.sim.scene_description.above_plate_pos)
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
            if tip_to_wrist is None:
                raise ValueError("tip_to_wrist must be provided in simulation")
            
            tool_frame_target = tip_pose @ tip_to_wrist
            plan = self.sim.plan_to_ee_pose(Pose.from_matrix(tool_frame_target))
            self.sim.visualize_plan(plan)
    
    def get_transform(self, from_frame, to_frame):
        if self.robot_interface is not None:
            return self.tf_utils.getTransformationFromTF(from_frame, to_frame)
        else:
            if from_frame == "fork_tip" and to_frame == "tool_frame":
                tip_to_wrist = np.array([[0, 0, -1, -1.79500833e-02],
                                        [0, 1, 0, -2.66243553e-03],
                                        [1, 0, 0, -2.55099477e-01],
                                        [0, 0, 0, 1]])
                return tip_to_wrist

            pose_transform = self.sim.get_transform(from_frame, to_frame)
            return pose_transform.to_matrix()

    def skewering_skill(self, color_image, depth_image, camera_info, keypoint=None, major_axis=None, skewering_depth=0.015, action_index=0):
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
            return False

        print("Getting transformation from base_link to camera_color_optical_frame")
        base_to_camera_transform = self.get_transform('base_link', 'camera_color_optical_frame')
        print("Base to camera transform: ", base_to_camera_transform)

        food_base = np.eye(4)
        food_base[:3,3] = point.reshape(1,3)
        food_base = base_to_camera_transform @ food_base
        print("Depth to skewer: ", food_base[2,3] - skewering_depth)
        print("Plate height: ", self.plate_height)
        food_base[2,3] = max(food_base[2,3] - skewering_depth, self.plate_height) 
        # magic number for skewering offset
        # food_base[0,3] += 0.012 # positive moves away from the robot
        # keep the orientation of the food base fixed
        food_base[:3,:3] = Rotation.from_quat([-0.7071068, 0.7071068, 0, 0]).as_matrix()

        if self.robot_interface is not None:
            self.tf_utils.publishTransformationToTF('base_link', 'food_frame', food_base)
            self.rviz_interface.visualize_food(food_base)

        if major_axis < np.pi/2:
            major_axis = major_axis + np.pi/2

        # caching this so that the robot doesn't rotate the wrist again
        tip_to_wrist = self.get_transform('fork_tip', 'tool_frame')
        print("Tip to wrist: ", tip_to_wrist)
        
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
        if self.robot_interface is not None:
            self.scooping_pickup()
        else:
            # Rajat ToDo: Implement scooping pick up for simulation
            self.move_utensil_to_pose(waypoint_1_tip, tip_to_wrist)

        return True

    def dipping_skill(self, color_image, depth_image, camera_info, keypoint=None, dipping_depth=0.02):
        """ Dipping amount must be between 0.02 and 0.05"""

        if keypoint is not None:
            (center_x, center_y) = keypoint
            major_axis = -np.pi/2
        else:
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            major_axis = -np.pi/2

        # # visualize keypoint
        # import cv2
        # cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
        # cv2.imshow("Color image", color_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # input("Press enter to continue...")

        # get 3D point from depth image
        validity, point = pixel2World(camera_info, center_x, center_y, depth_image)
        # breakpoint()
        if not validity:
            print("Invalid point")
            return False
        
        print("Getting transformation from base_link to camera_color_optical_frame")
        base_to_camera_transform = self.get_transform('base_link', 'camera_color_optical_frame')
        print("Base to camera transform: ", base_to_camera_transform)

        food_base = np.eye(4)
        food_base[:3,3] = point.reshape(1,3)
        food_base = base_to_camera_transform @ food_base
        print("Food height detected: ", food_base[2,3])
        print("Plate height: ", self.plate_height)
        food_base[2,3] = self.plate_height + 0.08 - dipping_depth
        print("Food height after plate update: ", food_base[2,3])
        # food_base[2,3] = max(food_base[2,3] - dipping_depth, self.plate_height) 
        # magic number for skewering offset
        # food_base[0,3] += 0.012 # positive moves away from the robot
        # keep the orientation of the food base fixed
        food_base[:3,:3] = Rotation.from_quat([-0.7071068, 0.7071068, 0, 0]).as_matrix()

        if self.robot_interface is not None:
            self.tf_utils.publishTransformationToTF('base_link', 'food_frame', food_base)
            self.rviz_interface.visualize_food(food_base)

        if major_axis < np.pi/2:
            major_axis = major_axis + np.pi/2

        # caching this so that the robot doesn't rotate the wrist again
        # tip_to_wrist = self.get_transform('fork_tip', 'tool_frame')
        # print("Tip to wrist: ", tip_to_wrist)
        
        # action 0: Rotate scooping DoF to dip angle
        self.wrist_interface.set_to_dip_pos()

        # Action 1: Move above food
        waypoint_1_tip = np.copy(food_base)
        waypoint_1_tip[2,3] -= 0.07
        waypoint_1_tip[2,3] += 0.13
        waypoint_1_tip[0,3] += 0.11
        self.move_utensil_to_pose(waypoint_1_tip, self.cached_reset_tip_to_wrist)

        # Action 2: Dip
        waypoint_2_tip = np.copy(food_base)
        waypoint_2_tip[2,3] -= 0.07
        waypoint_2_tip[0,3] += 0.11
        self.move_utensil_to_pose(waypoint_2_tip, self.cached_reset_tip_to_wrist)

        # Action 3: Move above food
        waypoint_3_tip = np.copy(food_base)
        waypoint_3_tip[2,3] -= 0.07
        waypoint_3_tip[2,3] += 0.13
        waypoint_3_tip[0,3] += 0.11
        self.move_utensil_to_pose(waypoint_3_tip, self.cached_reset_tip_to_wrist)

        # Action 4: Set scooping state
        self.wrist_interface.scoop_wrist()

        return True
        
    def scooping_pickup(self, hack = True):

        forkpitch_to_tip = self.get_transform('forkpitch', 'fork_tip')
        print("Forkpitch to tip: ", forkpitch_to_tip)
        distance = forkpitch_to_tip[0,3]

        print("Distance: ", distance)

        tool_frame = self.get_transform('base_link', 'tool_frame')

        tool_frame_displacement = np.eye(4)
        tool_frame_displacement[0,3] = distance/10 # move down
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