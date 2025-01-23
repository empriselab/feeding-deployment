
import numpy as np
from scipy.spatial.transform import Rotation
import pickle

try:
    import rospy
    from std_msgs.msg import Bool
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.control.robot_controller.command_interface import CartesianCommand
from feeding_deployment.actions.feel_the_bite.base import Transfer

from pybullet_helpers.geometry import Pose

DISTANCE_INFRONT_MOUTH = 0.10

class OutsideMouthTransfer(Transfer):

    def __init__(self, sim : FeedingDeploymentPyBulletSimulator, robot_interface: ArmInterfaceClient, perception_interface: PerceptionInterface, rviz_interface: RVizInterface, no_waits=False, log_dir=None):
            
        super().__init__(sim, robot_interface, perception_interface, rviz_interface, no_waits)

        self.log_dir = log_dir

    def move_to_transfer_state(self, outside_mouth_distance, maintain_position_at_goal = False):

        if self.robot_interface is not None:
            self.set_filter_noisy_readings_pub.publish(Bool(data=True))

        # move to infront of mouth
        head_perception_data = self.perception_interface.get_head_perception_data()
        forque_target_base = head_perception_data["tool_tip_target_pose"]
        head_pose = head_perception_data["head_pose"]

        file_name = "head_perception_data"
        id = 0
        while (self.log_dir / f"{file_name}_{id}.pkl").exists():
            id += 1
        with open(self.log_dir / f"{file_name}_{id}.pkl", "wb") as f:
            pickle.dump(head_perception_data, f)
        self.sim.set_head_pose(Pose(position=head_pose[:3], orientation=Rotation.from_euler('yxz', head_pose[3:], degrees=True).as_quat()))

        # set mouth pose to be facing away from the wheelchair
        forque_target_base[:3, :3] = Rotation.from_quat([0.523, -0.503, -0.469, 0.503]).as_matrix()
        
        servo_point_forque_target = np.identity(4)
        servo_point_forque_target[:3,3] = np.array([0, 0, -outside_mouth_distance]).reshape(1,3)
        infront_mouth_target = forque_target_base @ servo_point_forque_target

        # # mouth is assumed to be facing away from the wheelchair
        # infront_mouth_target[:3, :3] = Rotation.from_quat([0.478, -0.505, -0.515, 0.502]).as_matrix()
        tool_frame_target = infront_mouth_target @ self.get_tip_wrist_transform()

        target_pose = Pose.from_matrix(tool_frame_target)

        self.move_to_ee_pose(target_pose)

        if self.robot_interface is not None:
            self.set_filter_noisy_readings_pub.publish(Bool(data=False))

    def move_to_before_transfer_state(self):
        self.move_to_ee_pose(self.sim.scene_description.before_transfer_pose)