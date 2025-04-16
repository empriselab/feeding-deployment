'''
Entrypoint for controlling the robot arm on compute machine. Additionally runs two important threads:
1. A thread that checks no safety anomalies have occurred using the watchdog
2. A thread that publishes joint states to ROS
'''

import threading
import time
import numpy as np

try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
    from geometry_msgs.msg import Pose
    # from netft_rdt_driver.srv import String_cmd
    ROSPY_IMPORTED = True
except ModuleNotFoundError as e:
    # print(f"ROS not imported: {e}")
    ROSPY_IMPORTED = False

from feeding_deployment.control.robot_controller.arm_interface import ArmInterface, ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY
from feeding_deployment.control.robot_controller.command_interface import KinovaCommand, JointTrajectoryCommand, JointCommand, CartesianCommand, OpenGripperCommand, CloseGripperCommand
# from feeding_deployment.safety.watchdog import WATCHDOG_MONITOR_FREQUENCY, PeekableQueue

class ArmInterfaceClient:
    def __init__(self):

        assert ROSPY_IMPORTED, "ROS is required to run on the real robot"

        # make sure watchdog is running
        print("Waiting for Watchdog status...")
        rospy.wait_for_message("/watchdog_status", Bool)
        print("Watchdog is running, continuing...")

        # Register ArmInterface (no lambda needed on the client-side)
        ArmManager.register("ArmInterface")

        # Client setup
        self.manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        self.manager.connect()

        # This will now use the single, shared instance of ArmInterface
        self._arm_interface = self.manager.ArmInterface()
        self.in_compliant_mode = False

    def switch_to_task_compliant_mode(self):
        assert not self.in_compliant_mode, "Already in compliant mode"
        self._arm_interface.switch_to_task_compliant_mode()
        self.in_compliant_mode = True

    def switch_to_joint_compliant_mode(self):
        assert not self.in_compliant_mode, "Already in compliant mode"
        self._arm_interface.switch_to_joint_compliant_mode()
        self.in_compliant_mode = True

    def switch_out_of_compliant_mode(self):
        assert self.in_compliant_mode, "Not in compliant mode"
        # time.sleep(2.0) # Wait for the arm to settle
        self._arm_interface.switch_out_of_compliant_mode()
        self.in_compliant_mode = False

    def get_state(self):
        return self._arm_interface.get_state()
    
    def get_speed(self):
        return self._arm_interface.get_speed()
    
    def set_speed(self, speed: str):
        assert not self.in_compliant_mode, "Cannot set speed in compliant mode"
        assert speed in ["low", "medium", "high"], "Speed must be one of 'low', 'medium', 'high'"
        self._arm_interface.set_speed(speed)
        time.sleep(1.0) # Make sure the arm has time to change speed

    def set_tool(self, tool: str):
        assert not self.in_compliant_mode, "Cannot set tool in compliant mode"
        self._arm_interface.set_tool(tool)

    def execute_command(self, cmd: KinovaCommand) -> None:

        # if not self.in_compliant_mode:
            # input("Press enter to execute command...")

        if cmd.__class__.__name__ == "JointTrajectoryCommand":
            return self._arm_interface.set_joint_trajectory(cmd.traj)

        if cmd.__class__.__name__ == "JointCommand":
            if self.in_compliant_mode:
                return self._arm_interface.compliant_set_joint_position(cmd.pos)
            else:
                joint_command_pos = cmd.pos
                if isinstance(joint_command_pos, np.ndarray):
                    joint_command_pos = joint_command_pos.tolist()  # Convert to a list if it's a NumPy array
                return self._arm_interface.set_joint_position(joint_command_pos)

        if cmd.__class__.__name__ == "CartesianCommand":
            if self.in_compliant_mode:
                return self._arm_interface.compliant_set_ee_pose(cmd.pos, cmd.quat)
            else:
                return self._arm_interface.set_ee_pose(cmd.pos, cmd.quat)

        if cmd.__class__.__name__ == "OpenGripperCommand":
            return self._arm_interface.open_gripper()

        if cmd.__class__.__name__ == "CloseGripperCommand":
            return self._arm_interface.close_gripper()

        raise NotImplementedError(f"Unrecognized command: {cmd}")

if __name__ == "__main__":

    rospy.init_node("arm_interface_client", anonymous=True)
    arm_client_interface = ArmInterfaceClient()

    run_commands = input("Press 'y' to run commands")

    if run_commands != "y":
        exit()

    inside_mount_pose = Pose(
        position=[0.2385, 0.08, 0.169],
        orientation=[-1.0, 0.0, 0.0, 0.0],
    )

    arm_client_interface.execute_command(CartesianCommand(inside_mount_pose.position, inside_mount_pose.orientation))