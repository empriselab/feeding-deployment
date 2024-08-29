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
from netft_rdt_driver.srv import String_cmd

from feeding_deployment.robot_controller.arm_interface import ArmInterface, ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY
from feeding_deployment.robot_controller.command_interface import KinovaCommand, JointTrajectoryCommand, JointCommand, CartesianCommand, OpenGripperCommand, CloseGripperCommand
from feeding_deployment.safety.watchdog import WatchDog, AnomalyStatus, WATCHDOG_MONITOR_FREQUENCY

class ArmInterfaceClient:
    def __init__(self):

        rospy.init_node("arm_interface_client", anonymous=True)

        self.manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        self.manager.connect()
        self._arm_interface = self.manager.ArmInterface()
        self.in_compliant_mode = False

        # create joint/cartesian states publishers
        self.joint_states_pub = rospy.Publisher("/robot_joint_states", JointState, queue_size=10)
        self.cartesian_states_pub = rospy.Publisher("/robot_cartesian_states", Pose, queue_size=10)

        # spin joint states thread
        self.joint_state_thread = threading.Thread(target=self.publish_joint_states)
        self.joint_state_thread.start()

        # bias FT sensor
        bias = rospy.ServiceProxy('/forque/bias_cmd', String_cmd)
        bias('bias')

        # create watchdog
        self.watchdog = WatchDog()

        # spin watchdog monitor thread
        self.watchdog_thread = threading.Thread(target=self.monitor_watchdog)
        self.watchdog_thread.start()

    def publish_joint_states(self):

        while not rospy.is_shutdown():
            arm_pos, ee_pose, gripper_pos = self._arm_interface.get_state()
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "joint_7",
                "finger_joint",
            ]
            joint_state_msg.position = arm_pos.tolist() + [gripper_pos]
            joint_state_msg.velocity = [0.0] * 8
            joint_state_msg.effort = [0.0] * 8
            self.joint_states_pub.publish(joint_state_msg)

            cartesian_state_msg = Pose()
            cartesian_state_msg.position.x = ee_pose[0]
            cartesian_state_msg.position.y = ee_pose[1]
            cartesian_state_msg.position.z = ee_pose[2]
            cartesian_state_msg.orientation.x = ee_pose[3]
            cartesian_state_msg.orientation.y = ee_pose[4]
            cartesian_state_msg.orientation.z = ee_pose[5]
            cartesian_state_msg.orientation.w = ee_pose[6]
            self.cartesian_states_pub.publish(cartesian_state_msg) 

    def monitor_watchdog(self):
        while True:
            start_time = time.time()
            anomaly_status = self.watchdog.run()

            if anomaly_status != AnomalyStatus.NO_ANOMALY:
                self._arm_interface.stop()
                self._arm_interface.close()
                raise Exception(f"Anomaly detected: {anomaly_status}")
            
            end_time = time.time()
            time.sleep(max(0, 1.0/WATCHDOG_MONITOR_FREQUENCY - (end_time - start_time)))

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
