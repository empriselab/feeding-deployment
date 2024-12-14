#! /usr/bin/env python

import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import threading
import time
import signal

try:
    import rospy
    from std_msgs.msg import Int64, String
    from std_msgs.msg import Bool
    from geometry_msgs.msg import WrenchStamped, Point
    from netft_rdt_driver.srv import String_cmd
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

# Parameters
DISTANCE_INFRONT_MOUTH = 0.20

from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.robot_controller.command_interface import CartesianCommand
from feeding_deployment.actions.tf_utils import TFUtils

class OutsideMouthTransfer:
    def __init__(self, perception_interface: PerceptionInterface, robot_interface: ArmInterfaceClient, rviz_interface: RVizInterface):    

        if perception_interface is None:
            raise ValueError("Perception interface is required")
        if robot_interface is None:
            raise ValueError("Robot interface is required")
        
        self.tf_utils = TFUtils()
        
        # Rajat ToDo: Removed just for isolated testing
        # if rviz_interface is None:
        #     raise ValueError("RViz interface is required")

        self.perception_interface = perception_interface
        self.robot_interface = robot_interface
        self.rviz_interface = rviz_interface

        self.speak_pub = rospy.Publisher('/speak', String, queue_size=1)

        self.set_filter_noisy_readings_pub = rospy.Publisher('/head_perception/set_filter_noisy_readings', Bool, queue_size=1)

        self.transfer_button = False
        self.transfer_button_sub = rospy.Subscriber('/transfer_button', Bool, self.transfer_button_callback)

        self.mouth_open = False
        self.mouth_state_sub = rospy.Subscriber('/head_perception/mouth_state', Bool, self.mouth_state_callback)
        
        self.ft_threshold_exceeded = False
        self.ft_sensor_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, self.ft_callback)

        self.head_shake_detected = False
        self.head_still_detected = False
        self.neck_rotation_sub = rospy.Subscriber('/head_perception/neck_rotation', Point, self.neck_rotation_callback)

        self.ready_for_transfer_interaction = "silent" # "silent", "voice" or "led"
        self.initiate_transfer_interaction = "open_mouth" # "button", "open_mouth" or "auto_timeout"
        self.transfer_complete_interaction = "sense" # "button", "sense" or "auto_timeout"

        self.tool = None

    def set_tool(self, tool):
        self.tool = tool

    def relay_ready_for_transfer(self):
        if self.ready_for_transfer_interaction == "silent":
            pass
        elif self.ready_for_transfer_interaction == "voice":
            print("Speaking: Please open your mouth when ready")
            self.speak_pub.publish(String(data="Please oopen your mouth when ready"))

    def neck_rotation_callback(self, msg):

        neck_rotation = np.array([msg.x, msg.y, msg.z])
        if neck_rotation[1] > 5: # in degrees
            self.head_shake_detected = True
        if np.abs(neck_rotation[0]) < 5 and np.abs(neck_rotation[1]) < 5 and np.abs(neck_rotation[2]) < 5:
            self.head_still_detected = True
        else:
            self.head_still_detected = False

    def detect_initiate_transfer(self):
        if self.initiate_transfer_interaction == "button":
            self.transfer_button = False
            # wait for button press
            print("Waiting for button press to initiate transfer")
            while not rospy.is_shutdown() and not self.transfer_button:
                time.sleep(0.05)
            self.transfer_button = False
        elif self.initiate_transfer_interaction == "open_mouth":
            print("Waiting for mouth to open to initiate transfer")
            self.mouth_open = False
            # wait for mouth to open
            while not rospy.is_shutdown() and not self.mouth_open:
                time.sleep(0.05)
            self.mouth_open = False
        elif self.initiate_transfer_interaction == "auto_timeout":
            print("Waiting for auto timeout to initiate transfer")
            time.sleep(5.0)

        print("Initiating transfer")

    def detect_transfer_complete(self):
        if self.transfer_complete_interaction == "button":
            self.transfer_button = False
            # wait for button press
            print("Waiting for button press to complete transfer")
            while not rospy.is_shutdown() and not self.transfer_button:
                time.sleep(0.05)
            self.transfer_button = False
        elif self.transfer_complete_interaction == "sense":
            if self.tool == "fork":
                self.ft_threshold_exceeded = False
                # wait for force torque threshold to be exceeded
                print("Waiting for force torque threshold to be exceeded")
                while not rospy.is_shutdown() and not self.ft_threshold_exceeded:
                    time.sleep(0.05)
                self.ft_threshold_exceeded = False
            elif self.tool == "drink":
                self.set_filter_noisy_readings_pub.publish(Bool(data=False))
                self.head_shake_detected = False
                # wait for head shake
                print("Waiting for head shake to complete transfer")
                while not rospy.is_shutdown() and not self.head_shake_detected:
                    time.sleep(0.05)
                self.set_filter_noisy_readings_pub.publish(Bool(data=True))
                self.head_shake_detected = False
            elif self.tool == "wipe":
                self.head_shake_detected = True
                self.set_filter_noisy_readings_pub.publish(Bool(data=False))
                print("Waiting for head still to be detected for 4 seconds")
                while not rospy.is_shutdown():
                    head_still_start_time = time.time()
                    while not rospy.is_shutdown():
                        print("Head still detected: ",self.head_still_detected)
                        if not self.head_still_detected or time.time() - head_still_start_time > 4.0:
                            break
                        if time.time() - head_still_start_time > 3.0:
                            print("Waiting for head still to be detected for 1 more second")
                        elif time.time() - head_still_start_time > 2.0:
                            print("Waiting for head still to be detected for 2 more seconds")
                        elif time.time() - head_still_start_time > 1.0:
                            print("Waiting for head still to be detected for 3 more seconds")
                        time.sleep(0.05)
                    if time.time() - head_still_start_time > 4.0:
                        break
                self.set_filter_noisy_readings_pub.publish(Bool(data=True))
                self.head_still_detected = True
        elif self.transfer_complete_interaction == "auto_timeout":
            time.sleep(5.0)
        print("Detected transfer completion")

    def ft_callback(self, msg):

        ft_reading = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

        down_torque = ft_reading[3]
        if np.abs(down_torque) > 0.1:
            self.ft_threshold_exceeded = True

    def mouth_state_callback(self, msg):
        self.mouth_open = msg.data

    def transfer_button_callback(self, msg):
        print("Transfer button pressed")
        self.transfer_button = True

    def head_shake_callback(self, msg):
        with self.head_shake_lock:
            self.head_shake_detected = msg.data

    def publishTaskCommand(self, tip_pose):

        tip_pose[:3, :3] = Rotation.from_quat([0.478, -0.505, -0.515, 0.502]).as_matrix()

        if self.tool == "fork":
            tip_to_wrist = self.tf_utils.getTransformationFromTF('fork_tip', 'tool_frame')
        elif self.tool == "drink":
            tip_to_wrist = self.tf_utils.getTransformationFromTF('drink_tip', 'tool_frame')
        elif self.tool == "wipe":
            tip_to_wrist = self.tf_utils.getTransformationFromTF('wipe_tip', 'tool_frame')
        else:
            raise ValueError("Tool not recognized")
        tool_frame_target = tip_pose @ tip_to_wrist

        # self.visualizer.visualize_fork(tip_pose)
        self.tf_utils.publishTransformationToTF('base_link', 'tool_frame_target_viz', tool_frame_target)
      
        tool_frame_pos = tool_frame_target[:3,3].reshape(1,3).tolist()[0] # one dimensional list
        tool_frame_quat = Rotation.from_matrix(tool_frame_target[:3,:3]).as_quat()
        self.robot_interface.execute_command(CartesianCommand(tool_frame_pos, tool_frame_quat))

    def start_head_perception_thread(self):
        self.perception_interface.start_head_perception_thread()

    def execute_transfer_loop(self, maintain_position_at_goal = False):
        
        assert self.perception_interface.head_perception_thread_is_running(), "Head perception thread is not running"
        assert self.tool is not None, "Tool is not set"

        # just in case it was not set before
        self.set_filter_noisy_readings_pub.publish(Bool(data=True))
        
        # bias the force torque sensor
        # bias FT sensor
        bias = rospy.ServiceProxy('/forque/bias_cmd', String_cmd)
        bias('bias')

        closed_loop = True
        run_once = True
        paused_once = False

        self.relay_ready_for_transfer()
        self.detect_initiate_transfer()

        print("Setting state to 1")

        # move to infront of mouth
        forque_target_base = self.perception_interface.get_head_perception_tool_tip_target_pose()
        servo_point_forque_target = np.identity(4)
        servo_point_forque_target[:3,3] = np.array([0, 0, -DISTANCE_INFRONT_MOUTH]).reshape(1,3)
        infront_mouth_target = forque_target_base @ servo_point_forque_target
        self.publishTaskCommand(infront_mouth_target)

        self.detect_transfer_complete()
        # shutdown the head perception thread
        self.perception_interface.stop_head_perception_thread()

        # move to before transfer position
        final_target = self.perception_interface.get_tool_tip_pose_at_staging()
        self.publishTaskCommand(final_target)

        # incase for some reason the head perception thread is still running
        self.perception_interface.stop_head_perception_thread()            
        print("Exiting transfer loop")

    def signal_handler(self, signal, frame):

        print("\nprogram exiting gracefully")
        sys.exit(0)

if __name__ == '__main__':

    rospy.init_node('outside_mouth_transfer', anonymous=True)
    robot_interface = ArmInterfaceClient()
    perception_interface = PerceptionInterface(robot_interface=robot_interface, simulate_head_perception=False)

    perception_interface.set_head_perception_tool("fork")
    perception_interface.start_head_perception_thread()

    outside_mouth_transfer = OutsideMouthTransfer(perception_interface, robot_interface, None)
    outside_mouth_transfer.set_tool("wipe")
    outside_mouth_transfer.set_filter_noisy_readings_pub.publish(Bool(data=False))

    print("Starting inside mouth transfer interaction loop")
    while not rospy.is_shutdown():
        input("Press enter to start transfer")
        outside_mouth_transfer.relay_ready_for_transfer()
        outside_mouth_transfer.detect_initiate_transfer()
        outside_mouth_transfer.detect_transfer_complete()