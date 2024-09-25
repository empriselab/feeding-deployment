#! /usr/bin/env python

import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from std_msgs.msg import Int64, String
import threading
import time
import signal

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import WrenchStamped, Point
from netft_rdt_driver.srv import String_cmd

# Parameters
OPEN_LOOP_RADIUS = 0.02
# OPEN_LOOP_RADIUS = 0.0
INTERMEDIATE_THRESHOLD_RELAXED = 0.02
INTERMEDIATE_ANGULAR_THRESHOLD_RELAXED = 5*np.pi/180
INTERMEDIATE_THRESHOLD = 0.014
INFRONT_DISTANCE_LOOKAHEAD = 0.04
INSIDE_DISTANCE_LOOKAHEAD_Z = 0.04 # for slower inside mouth movement
# INSIDE_DISTANCE_LOOKAHEAD_Z = 0.045
INSIDE_DISTANCE_LOOKAHEAD_XY = 0.025
ANGULAR_LOOKAHEAD = 5*np.pi/180
MOVE_OUTSIDE_DISTANCE = 0.12
DISTANCE_INFRONT_MOUTH = 0.12

from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.robot_controller.command_interface import CartesianCommand

class InsideMouthTransfer:
    def __init__(self, perception_interface: PerceptionInterface, robot_interface: ArmInterfaceClient, rviz_interface: RVizInterface):    

        if perception_interface is None:
            raise ValueError("Perception interface is required")
        if robot_interface is None:
            raise ValueError("Robot interface is required")
        
        # Rajat ToDo: Removed just for isolated testing
        # if rviz_interface is None:
        #     raise ValueError("RViz interface is required")

        self.perception_interface = perception_interface
        self.robot_interface = robot_interface
        self.rviz_interface = rviz_interface

        self.control_rate = rospy.Rate(100.0)

        self.speak_pub = rospy.Publisher('/speak', String, queue_size=1)

        self.set_filter_noisy_readings_pub = rospy.Publisher('/head_perception/set_filter_noisy_readings', Bool, queue_size=1)

        self.state = 0
        self.state_lock = threading.Lock()
        self.state_sub = rospy.Subscriber('/state', Int64, self.state_callback)

        self.transfer_button = False
        self.transfer_button_sub = rospy.Subscriber('/transfer_button', Bool, self.transfer_button_callback)

        self.mouth_open = False
        self.mouth_state_sub = rospy.Subscriber('/head_perception/mouth_state', Bool, self.mouth_state_callback)
        
        self.ft_threshold_exceeded = False
        self.ft_sensor_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, self.ft_callback)

        self.head_shake_detected = False
        self.head_still_detected = False
        self.neck_rotation_sub = rospy.Subscriber('/head_perception/neck_rotation', Point, self.neck_rotation_callback)

        self.ready_for_transfer_interaction = "voice" # "silent", "voice" or "led"
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
    
    def state_callback(self, msg):

        with self.state_lock:
            self.state = msg.data

    def getAngularDistance(self, rotation_a, rotation_b):
        return np.linalg.norm(Rotation.from_matrix(np.dot(rotation_a, rotation_b.T)).as_rotvec())

    def getNextWaypoint(self, source, target, distance_lookahead, angular_lookahead = ANGULAR_LOOKAHEAD):

        position_error = np.linalg.norm(source[:3,3] - target[:3,3])
        orientation_error = self.getAngularDistance(source[:3,:3], target[:3,:3])

        next_waypoint = np.zeros((4,4))
        next_waypoint[3,3] = 1

        if position_error <= distance_lookahead:
            next_waypoint[:3,3] = target[:3,3]
        else:    
            next_waypoint[:3,3] = source[:3,3].reshape(1,3) + distance_lookahead*(target[:3,3] - source[:3,3]).reshape(1,3)/position_error

        if orientation_error <= angular_lookahead:
            next_waypoint[:3,:3] = target[:3,:3]
        else:
            key_times = [0, 1]
            key_rots = Rotation.concatenate((Rotation.from_matrix(source[:3,:3]), Rotation.from_matrix(target[:3,:3])))
            slerp = Slerp(key_times, key_rots)

            interp_rotations = slerp([angular_lookahead/orientation_error]) #second last is also aligned
            next_waypoint[:3,:3] = interp_rotations[0].as_matrix()
        
        return next_waypoint

    # def publishTaskMode(self, mode):
    #     mode_command = String()
    #     mode_command.data = mode
    #     self.task_mode_publisher.publish(mode_command)
    #     time.sleep(0.1)

    def publishTaskCommand(self, target):

        # input("Press enter to move to next target")
        command = CartesianCommand(pos=target[:3,3], quat=Rotation.from_matrix(target[:3,:3]).as_quat())
        self.robot_interface.execute_command(command)

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

        # start at state 1
        with self.state_lock:
            self.state = 1

        # Assumption: No one will be updating state when this runs
        previous_state = self.state
        current_state = self.state

        last_time = time.time()
        while not rospy.is_shutdown():

            self.control_rate.sleep()

            # print("Frequency: ",1.0/(time.time() - last_time))
            last_time = time.time()
            
            with self.state_lock:
                current_state = self.state

            if current_state != previous_state:
                print("Switching to self.state: ",current_state)
                closed_loop = True
                run_once = True
                previous_state = current_state

            if current_state == 1: # move to infront of mouth
                
                # trajectory positions
                if closed_loop:

                    if run_once:
                        # for i in range(0,10):

                        #     forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                        #     self.publishTaskCommand(forque_base)

                        time.sleep(0.1)
                        print("Need to set default stiffness for moving infront of mouth")

                        # self.publishTaskMode("none")
                        # self.publishTaskMode("default_stiffness")
                        run_once = False

                    forque_target_base = self.perception_interface.get_head_perception_tool_tip_target_pose()

                    closed_loop = False
                    servo_point_forque_target = np.identity(4)
                    servo_point_forque_target[:3,3] = np.array([0, 0, -DISTANCE_INFRONT_MOUTH]).reshape(1,3)

                    servo_point_base = forque_target_base @ servo_point_forque_target

                # current position
                forque_base = self.perception_interface.get_tool_tip_pose()

                distance = np.linalg.norm(forque_base[:3,3] - servo_point_base[:3,3])
                angular_distance = self.getAngularDistance(forque_base[:3,:3], servo_point_base[:3,:3])

                # print("Distance: {} Angular Distance: {}".format(distance, angular_distance))
                # print("Threshold: {} Angular Threshold: {}".format(INTERMEDIATE_THRESHOLD_RELAXED, INTERMEDIATE_ANGULAR_THRESHOLD_RELAXED))

                if distance < INTERMEDIATE_THRESHOLD_RELAXED and angular_distance < INTERMEDIATE_ANGULAR_THRESHOLD_RELAXED:
                    with self.state_lock:
                        self.state = 2

                target = self.getNextWaypoint(forque_base, servo_point_base, distance_lookahead=INFRONT_DISTANCE_LOOKAHEAD)
                # target = servo_point_base

                self.publishTaskCommand(target)
                self.rviz_interface.visualizeTransform("base_link", "next_target", target)
                self.rviz_interface.visualizeTransform("base_link", "final_target", servo_point_base)

            elif current_state == 2: # move inside mouth

                if closed_loop:

                    if run_once:

                        # for i in range(0,10):

                        #     forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                        #     self.publishTaskCommand(forque_base)

                        time.sleep(0.1)
                        print("Need to set move inside mouth stiffness and pose integral for moving inside mouth")

                        # self.publishTaskMode("use_pose_integral")
                        # self.publishTaskMode("move_inside_mouth_stiffness")
                        run_once = False

                        if not paused_once:
                            paused_once = True
                            time.sleep(0.5)
                            print("PAUSING")
                        
                    forque_target_base = self.perception_interface.get_head_perception_tool_tip_target_pose()

                    # closed_loop = False

                forque_source = self.perception_interface.get_tool_tip_pose()

                distance = np.linalg.norm(forque_source[:3,3] - forque_target_base[:3,3])

                if distance < OPEN_LOOP_RADIUS and closed_loop:
                    closed_loop = False
                    self.detect_transfer_complete()
                    # shutdown the head perception thread
                    self.perception_interface.stop_head_perception_thread()
                    with self.state_lock:
                        self.state = 3

                intermediate_forque_target = np.zeros((4,4))
                intermediate_forque_target[0, 0] = 1
                intermediate_forque_target[1, 1] = 1
                intermediate_forque_target[2, 2] = 1
                intermediate_forque_target[:3,3] = np.array([0, 0, -distance]).reshape(1,3)
                intermediate_forque_target[3,3] = 1

                intermediate_forque_target = forque_target_base @ intermediate_forque_target

                intermediate_position_error = np.linalg.norm(forque_source[:3,3] - intermediate_forque_target[:3,3])
                intermediate_angular_error = self.getAngularDistance(forque_source[:3,:3], intermediate_forque_target[:3,:3])

                # print("closed_loop: ",closed_loop)
                # print("intermediate_position_error: ", forque_source[:3,3] - intermediate_forque_target[:3,3])
                # print("intermediate_position_error mag: ", intermediate_position_error)
                # print("INTERMEDIATE_THRESHOLD mag: ", INTERMEDIATE_THRESHOLD)
                # print("intermediate_angular_error: ", intermediate_angular_error)

                ipe_forque_frame = np.linalg.inv(forque_source[:3,:3]) @ (forque_source[:3,3] - intermediate_forque_target[:3,3]).reshape(3,1)
                # print("Error in forque frame: ",ipe_forque_frame)
                error_mag = np.linalg.norm(np.array([ipe_forque_frame[0], ipe_forque_frame[1]]))
                # print("Error mag:",error_mag)
 
                if intermediate_position_error > INTERMEDIATE_THRESHOLD: # The thresholds here should be ideally larger than the thresholds for tracking trajectories
                    # print("Tracking intermediate position... ")
                    target = self.getNextWaypoint(forque_source, intermediate_forque_target, distance_lookahead = INSIDE_DISTANCE_LOOKAHEAD_XY)
                else:
                    # print("Tracking target position... ")
                    distance_lookahead_update = INSIDE_DISTANCE_LOOKAHEAD_Z - intermediate_position_error
                    orientation_lookahead_update = ANGULAR_LOOKAHEAD - intermediate_angular_error
                    target = self.getNextWaypoint(intermediate_forque_target, forque_target_base, distance_lookahead = distance_lookahead_update)
                
                # if (not closed_loop) and maintain_position_at_goal: 
                #     pass # maintain position at goal, do not update target
                #     self.detect_transfer_complete()
                #     with self.state_lock:
                #         self.state = 3
                # else:
                self.publishTaskCommand(target)
                self.rviz_interface.visualizeTransform("base_link", "next_target", target)
                self.rviz_interface.visualizeTransform("base_link", "final_target", forque_target_base)
                self.rviz_interface.visualizeTransform("base_link", "intermediate_target", intermediate_forque_target)

            elif current_state == 3: # move outside mouth

                if closed_loop:
                    
                    # for i in range(0,10):

                    #     forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                    #     self.publishTaskCommand(forque_base)

                    time.sleep(0.1)
                    print("Need to set move outside mouth stiffness for moving outside mouth")

                    # self.publishTaskMode("none")
                    # self.publishTaskMode("move_outside_mouth_stiffness")

                    source_base = self.perception_interface.get_tool_tip_pose()

                    forque_target_source = np.identity(4)
                    forque_target_source[:3,3] = np.array([0, 0, -MOVE_OUTSIDE_DISTANCE]).reshape(1,3)

                    forque_target_base = source_base @ forque_target_source

                    closed_loop = False

                forque_base = self.perception_interface.get_tool_tip_pose()

                distance = np.linalg.norm(forque_base[:3,3] - forque_target_base[:3,3])

                if distance < INTERMEDIATE_THRESHOLD_RELAXED:
                    with self.state_lock:
                        self.state = 4

                target = self.getNextWaypoint(forque_base, forque_target_base, distance_lookahead = INSIDE_DISTANCE_LOOKAHEAD_Z)
                
                self.publishTaskCommand(target)
                self.rviz_interface.visualizeTransform("base_link", "next_target", target)
                self.rviz_interface.visualizeTransform("base_link", "final_target", forque_target_base)

            if current_state == 4: # move to fixed position and exit

                if run_once:

                    # for i in range(0,10):

                    #         forque_base = self.getTransformationFromTF("base_link", "forque_end_effector")
                    #         self.publishTaskCommand(forque_base)

                    time.sleep(0.1)
                    print("Need to set default stiffness for moving to fixed position")

                    # self.publishTaskMode("none")
                    # self.publishTaskMode("default_stiffness")
                    run_once = False
                
                final_target = self.perception_interface.get_tool_tip_pose_at_staging()
                
                # current position
                forque_base = self.perception_interface.get_tool_tip_pose()

                distance = np.linalg.norm(forque_base[:3,3] - final_target[:3,3])

                if distance < INTERMEDIATE_THRESHOLD_RELAXED:
                    with self.state_lock:
                        self.state = 0
                    break
                
                target = self.getNextWaypoint(forque_base, final_target, distance_lookahead=INFRONT_DISTANCE_LOOKAHEAD)

                self.publishTaskCommand(target)
                self.rviz_interface.visualizeTransform("base_link", "next_target", target)
                self.rviz_interface.visualizeTransform("base_link", "final_target", final_target)

        # incase for some reason the head perception thread is still running
        self.perception_interface.stop_head_perception_thread()
            
        print("Exiting transfer loop")

    def signal_handler(self, signal, frame):

        print("\nprogram exiting gracefully")
        sys.exit(0)

if __name__ == '__main__':

    rospy.init_node('inside_mouth_transfer', anonymous=True)
    robot_interface = ArmInterfaceClient()
    perception_interface = PerceptionInterface(robot_interface=robot_interface, simulate_head_perception=False)

    perception_interface.set_head_perception_tool("fork")
    perception_interface.start_head_perception_thread()

    inside_mouth_transfer = InsideMouthTransfer(perception_interface, robot_interface, None)
    inside_mouth_transfer.set_tool("wipe")
    inside_mouth_transfer.set_filter_noisy_readings_pub.publish(Bool(data=False))

    print("Starting inside mouth transfer interaction loop")
    while not rospy.is_shutdown():
        input("Press enter to start transfer")
        inside_mouth_transfer.relay_ready_for_transfer()
        inside_mouth_transfer.detect_initiate_transfer()
        inside_mouth_transfer.detect_transfer_complete()