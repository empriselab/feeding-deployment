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
    rospy.init_node("retract_action")

    # make sure watchdog is running
    print("Waiting for Watchdog status...")
    rospy.wait_for_message("/watchdog_status", Bool)
    print("Watchdog is running, moving to retract configuration...")

    # Register ArmInterface (no lambda needed on the client-side)
    ArmManager.register("ArmInterface")

    # Client setup
    manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    manager.connect()

    # This will now use the single, shared instance of ArmInterface
    arm_interface = manager.ArmInterface()

    retract_pos = [0.0, -0.34903602299465675, -3.141591055693139, -2.5482592711638783, 0.0, -0.872688061814757, 1.57075917569769]
    arm_interface.set_joint_position(retract_pos)