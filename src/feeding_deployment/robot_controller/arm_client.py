'''
Entrypoint for controlling the robot arm on compute machine. Additionally runs two important threads:
1. A thread that checks no safety anomalies have occurred using the watchdog
2. A thread that publishes joint states to ROS
'''

import threading
import time
import numpy as np

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
# from netft_rdt_driver.srv import String_cmd

from feeding_deployment.robot_controller.arm_interface import ArmInterface, ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY
from feeding_deployment.robot_controller.command_interface import KinovaCommand, JointTrajectoryCommand, JointCommand, CartesianCommand, OpenGripperCommand, CloseGripperCommand
# from feeding_deployment.safety.watchdog import WATCHDOG_MONITOR_FREQUENCY, PeekableQueue

class ArmInterfaceClient:
    def __init__(self):

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

    def switch_to_joint_compliant_mode(self):
        assert not self.in_compliant_mode, "Already in compliant mode"
        self._arm_interface.switch_to_joint_compliant_mode()

    def switch_out_of_joint_compliant_mode(self):
        assert self.in_compliant_mode, "Not in compliant mode"
        self._arm_interface.switch_out_of_joint_compliant_mode()

    def execute_command(self, cmd: KinovaCommand) -> None:

        if cmd.__class__.__name__ == "JointTrajectoryCommand":
            if self.in_compliant_mode:
                return self._arm_interface.compliant_set_joint_trajectory(cmd.traj)
            else:
                return self._arm_interface.set_joint_trajectory(cmd.traj)

        if cmd.__class__.__name__ == "JointCommand":
            return self._arm_interface.set_joint_position(cmd.pos)

        if cmd.__class__.__name__ == "CartesianCommand":
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

    cup_inside_mount = (
        np.array([0.55, 0.52, 0.305]),
        np.array([-0.2126311, -0.6743797, -0.6743797, 0.2126311]),
    )

    cup_outside_mount = (
        cup_inside_mount[0].copy(),
        cup_inside_mount[1].copy(),
    )
    cup_outside_mount[0][1] -= 0.06

    cup_above_mount = (
        cup_inside_mount[0].copy(),
        cup_inside_mount[1].copy(),
    )
    cup_above_mount[0][2] += 0.1

    cup_inside_mount_pos = [-3.0706449768856463, -1.233024942579057, -1.0107718990709298, -1.3064307169693468, -1.0801286033398636, -0.6790118020676168, -3.0605814237584545]

    # depend on the offsets set above
    cup_above_mount_pos = [-3.0822113518159693, -1.0554986243143745, -0.9943061066831875, -1.2514305815048123, -1.0634620086059297, -0.7145069457084112, 3.0023889570952424]
    cup_outside_mount_pos = [-2.9457621628368873, -1.206488672845289, -1.0073524002312677, -1.3997867637176382, -1.0606635589324744, -0.7359768177844117, -3.048252585808042]

    before_transfer_pos = [
        -2.86554642,
        -1.61951779,
        -2.60986085,
        -1.37302839,
        1.11779249,
        -1.18028264,
        2.05515862,
    ]

    input("Press enter to move to outside cup pose...")
    arm_client_interface.execute_command(JointCommand(cup_outside_mount_pos))

    # input("Press enter to move to outside cup mount pose...")
    # arm_client_interface.set_ee_pose(outside_cup_pose[0], outside_cup_pose[1])

    input("Press enter to move to inside cup mount pose...")
    arm_client_interface.execute_command(CartesianCommand(cup_inside_mount[0], cup_inside_mount[1]))

    input("Press enter to grasp the cup...")
    arm_client_interface.execute_command(OpenGripperCommand())

    input("Press enter to pickup the cup...")
    arm_client_interface.execute_command(CartesianCommand(cup_above_mount[0], cup_above_mount[1]))

    input("Press enter to move to before transfer pose...")
    arm_client_interface.execute_command(JointCommand(before_transfer_pos))

    input("Press enter to move above the mount...")
    arm_client_interface.execute_command(CartesianCommand(cup_above_mount[0], cup_above_mount[1]))

    input("Press enter to move to inside cup mount pose...")
    arm_client_interface.execute_command(CartesianCommand(cup_inside_mount[0], cup_inside_mount[1]))

    input("Press enter to release the cup...")
    arm_client_interface.execute_command(CloseGripperCommand())

    input("Press enter to move to outside cup pose...")
    arm_client_interface.execute_command(CartesianCommand(cup_outside_mount[0], cup_outside_mount[1]))
