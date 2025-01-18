
import abc
import numpy as np

from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.control.robot_controller.command_interface import CartesianCommand

try:
    import rospy
    from std_msgs.msg import Bool
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

class Transfer(abc.ABC):
    """ Base class for transfer actions. """

    def __init__(self, sim : FeedingDeploymentPyBulletSimulator, robot_interface: ArmInterfaceClient, perception_interface: PerceptionInterface, rviz_interface: RVizInterface, no_waits=False):
            
        self.sim = sim
        self.robot_interface = robot_interface
        self.perception_interface = perception_interface
        self.rviz_interface = rviz_interface
        self.no_waits = no_waits

        if self.robot_interface is not None:
            self.set_filter_noisy_readings_pub = rospy.Publisher('/head_perception/set_filter_noisy_readings', Bool, queue_size=1)

    def set_tool(self, tool):
        self.tool = tool

    def get_tip_wrist_transform(self):

        if self.tool == "fork":
            wrist_to_tip = self.sim.scene_description.tool_frame_to_utensil_tip
        elif self.tool == "drink":
            wrist_to_tip = self.sim.scene_description.tool_frame_to_drink_tip
        elif self.tool == "wipe":
            wrist_to_tip = self.sim.scene_description.tool_frame_to_wipe_tip
        else:
            raise ValueError("Tool not recognized")
        
        tip_to_wrist = np.linalg.inv(wrist_to_tip.to_matrix())
        return tip_to_wrist

    def move_to_ee_pose(self, pose):

        if self.robot_interface is None:
            plan = self.sim.plan_to_ee_pose(pose)
            self.sim.visualize_plan(plan)
        else:
            self.robot_interface.execute_command(CartesianCommand(pos=pose.position, quat=pose.orientation))

    @abc.abstractmethod
    def move_to_transfer_state(self, outside_mouth_distance, maintain_position_at_goal = False):
        """Move robot to the transfer state."""

    @abc.abstractmethod
    def move_to_before_transfer_state(self):
        """Move robot to the state before transfer."""

    