
import numpy as np
from scipy.spatial.transform import Rotation

from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.robot_controller.command_interface import CartesianCommand
from feeding_deployment.actions.feel_the_bite.base import Transfer

from pybullet_helpers.geometry import Pose

DISTANCE_INFRONT_MOUTH = 0.20

class OutsideMouthTransfer(Transfer):

    def move_to_transfer_state(self, maintain_position_at_goal = False):

        # move to infront of mouth
        forque_target_base = self.perception_interface.get_head_perception_tool_tip_target_pose()
        servo_point_forque_target = np.identity(4)
        servo_point_forque_target[:3,3] = np.array([0, 0, -DISTANCE_INFRONT_MOUTH]).reshape(1,3)
        infront_mouth_target = forque_target_base @ servo_point_forque_target

        # mouth is assumed to be facing away from the wheelchair
        infront_mouth_target[:3, :3] = Rotation.from_quat([0.478, -0.505, -0.515, 0.502]).as_matrix()
        if self.tool == "fork":
            wrist_to_tip = self.sim.scene_description.tool_frame_to_utensil_tip
        elif self.tool == "drink":
            wrist_to_tip = self.sim.scene_description.tool_frame_to_drink_tip
        elif self.tool == "wipe":
            wrist_to_tip = self.sim.scene_description.tool_frame_to_wipe_tip
        else:
            raise ValueError("Tool not recognized")
        
        tip_to_wrist = np.linalg.inv(wrist_to_tip.to_matrix())
        tool_frame_target = infront_mouth_target @ tip_to_wrist

        target_pose = Pose.from_matrix(tool_frame_target)

        self.move_to_ee_pose(target_pose, plan_override=True)

    def move_to_before_transfer_state(self):
        self.move_to_ee_pose(self.sim.scene_description.before_transfer_pose, plan_override=True)