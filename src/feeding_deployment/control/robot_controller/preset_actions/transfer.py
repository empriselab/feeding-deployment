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


if __name__ == "__main__":

    assert ROSPY_IMPORTED, "ROS is required to run on the real robot"
    rospy.init_node("before_transfer_action")

    # make sure watchdog is running
    print("Waiting for Watchdog status...")
    rospy.wait_for_message("/watchdog_status", Bool)
    print("Watchdog is running, moving to before transfer configuration...")

    # Register ArmInterface (no lambda needed on the client-side)
    ArmManager.register("ArmInterface")

    # Client setup
    manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    manager.connect()

    # This will now use the single, shared instance of ArmInterface
    arm_interface = manager.ArmInterface()

    before_transfer_pos = [-2.86554642, -1.61951779, -2.60986085, -1.37302839, 1.11779249, -1.18028264, 2.05515862]
    arm_interface.set_joint_position(before_transfer_pos)