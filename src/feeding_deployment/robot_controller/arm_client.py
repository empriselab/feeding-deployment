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

    utensil_inside_mount = (
        np.array([0.242, -0.077, 0.07]),
        np.array([-1, 0, 0, 0]),
    )

    utensil_outside_mount = (
        utensil_inside_mount[0].copy(),
        utensil_inside_mount[1].copy(),
    )
    utensil_outside_mount[0][0] += 0.13

    utensil_outside_above_mount = (
        utensil_outside_mount[0].copy(),
        utensil_outside_mount[1].copy(),
    )
    utensil_outside_above_mount[0][2] += 0.1

    utensil_above_mount = (
        utensil_inside_mount[0].copy(),
        utensil_inside_mount[1].copy(),
    )
    utensil_above_mount[0][2] += 0.1

    utensil_inside_mount_pos = [0.03610898146611135, 0.46058565446690675, -2.7959276599029583, -2.4802831496040416, -0.5912420020218754, -0.27659484555119374, 0.9346522303717946]

    # depend on the offsets set above
    utensil_above_mount_pos = [-0.3081224117999879, 0.1449308244187662, -2.4515079603418446, -2.3539334664268674, -0.14376009880356744, -0.6872590793313744, 0.5028097739444904]
    utensil_outside_mount_pos = [-0.13616795916796942, 0.6152003983994736, -2.673508269862523, -2.09060888627143, -0.5157179493005364, -0.5540355110645967, 0.7160963868259976]
    utensil_outside_above_mount_pos = [-0.2692035082617874, 0.4127082432063301, -2.513398492494741, -1.9930522355357558, -0.31928105676741936, -0.8392446174777604, 0.5472652562309106]

    before_transfer_pos = [
        -2.86554642,
        -1.61951779,
        -2.60986085,
        -1.37302839,
        1.11779249,
        -1.18028264,
        2.05515862,
    ]

    above_plate_pos = [
        -2.86495014,
        -1.61460533,
        -2.6115943,
        -1.37673391,
        1.11842806,
        -1.17904586,
        -2.6957422,
    ]

    # input("Press enter to move to above utensil mount pose...")
    # arm_client_interface.execute_command(CartesianCommand(utensil_above_mount[0], utensil_above_mount[1]))

    input("Press enter to move to above utensil pos...")
    arm_client_interface.execute_command(JointCommand(utensil_above_mount_pos))

    input("Press enter to move to inside utensil mount pose...")
    arm_client_interface.execute_command(CartesianCommand(utensil_inside_mount[0], utensil_inside_mount[1]))

    input("Press enter to grasp the utensil...")
    arm_client_interface.execute_command(OpenGripperCommand())

    input("Press enter to move outside the mount...")
    arm_client_interface.execute_command(CartesianCommand(utensil_outside_mount[0], utensil_outside_mount[1]))

    input("Press enter to move to above outside utensil pose...")
    arm_client_interface.execute_command(CartesianCommand(utensil_outside_above_mount[0], utensil_outside_above_mount[1]))

    # input("Press enter to move to before transfer pose...")
    # arm_client_interface.execute_command(JointCommand(before_transfer_pos))

    input("Press enter to move to above plate pose...")
    arm_client_interface.execute_command(JointCommand(above_plate_pos))

    input("Press enter to move to above outside utensil mount pos...")
    arm_client_interface.execute_command(JointCommand(utensil_outside_above_mount_pos))

    input("Press enter to move outside the mount...")
    arm_client_interface.execute_command(JointCommand(utensil_outside_mount_pos))

    input("Press enter to move to inside utensil mount pose...")
    arm_client_interface.execute_command(CartesianCommand(utensil_inside_mount[0], utensil_inside_mount[1]))

    input("Press enter to ungrasp the utensil...")
    arm_client_interface.execute_command(CloseGripperCommand())

    input("Press enter to move to above utensil pose...")
    arm_client_interface.execute_command(CartesianCommand(utensil_above_mount[0], utensil_above_mount[1]))
