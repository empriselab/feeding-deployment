'''
Publishes joint states and cartesian states of the robot arm to ROS.
'''

import threading
import time
import numpy as np

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

from feeding_deployment.control.robot_controller.arm_interface import ArmInterface, ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY

class JointStatesPublisher:
    def __init__(self):

        # Register ArmInterface (no lambda needed on the client-side)
        ArmManager.register("ArmInterface")

        # Client setup
        self.manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        self.manager.connect()

        # This will now use the single, shared instance of ArmInterface
        self._arm_interface = self.manager.ArmInterface()

        # create joint/cartesian states publishers
        self.joint_states_pub = rospy.Publisher("/robot_joint_states", JointState, queue_size=10)
        self.cartesian_states_pub = rospy.Publisher("/robot_cartesian_state", Pose, queue_size=10)

    def publish_joint_states(self):

        try:
            current_state = self._arm_interface.get_state()
        except Exception as e:
            raise Exception(f"Error getting state: {e}")
        
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

        gripper_pos = current_state["gripper_pos"]
        if gripper_pos < 0.0:
            gripper_pos = 0.0 # sometimes the gripper position is slightly negative

        joint_state_msg.position = current_state["position"].tolist() + [gripper_pos*0.8] # sim/rviz gripper is 0.8x real gripper
        joint_state_msg.velocity = current_state["velocity"].tolist() + [0.0]
        joint_state_msg.effort = current_state["effort"].tolist() + [0.0]
        self.joint_states_pub.publish(joint_state_msg)

        ee_pose = current_state["ee_pos"]
        cartesian_state_msg = Pose()
        cartesian_state_msg.position.x = ee_pose[0]
        cartesian_state_msg.position.y = ee_pose[1]
        cartesian_state_msg.position.z = ee_pose[2]
        cartesian_state_msg.orientation.x = ee_pose[3]
        cartesian_state_msg.orientation.y = ee_pose[4]
        cartesian_state_msg.orientation.z = ee_pose[5]
        cartesian_state_msg.orientation.w = ee_pose[6]
        self.cartesian_states_pub.publish(cartesian_state_msg) 

    def run(self):
        while not rospy.is_shutdown():
            self.publish_joint_states()

if __name__ == "__main__":

    rospy.init_node("joint_states_publisher", anonymous=True)
    joint_states_publisher = JointStatesPublisher()
    joint_states_publisher.run()