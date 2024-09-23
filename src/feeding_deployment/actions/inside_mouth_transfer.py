#! /usr/bin/env python

import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from std_msgs.msg import Int64
import threading
import time
import signal

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import WrenchStamped
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
        if rviz_interface is None:
            raise ValueError("RViz interface is required")

        self.perception_interface = perception_interface
        self.robot_interface = robot_interface
        self.rviz_interface = rviz_interface

        self.control_rate = rospy.Rate(100.0)

        self.state = 0
        self.state_lock = threading.Lock()
        self.state_sub = rospy.Subscriber('/state', Int64, self.state_callback)

        self.transfer_completed_sub = rospy.Subscriber('/transfer_complete', Bool, self.transfer_completed_callback)

        self.mouth_open = False
        self.mouth_open_lock = threading.Lock()
        self.mouth_state_sub = rospy.Subscriber('/head_perception/mouth_state', Bool, self.mouth_state_callback)
        
        self.ft_sensor_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, self.ft_callback)

    def ft_callback(self, msg):

        ft_reading = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

        # down_torque = ft_reading[3]
        # if np.abs(down_torque) > 0.05:
        #     with self.state_lock:
        #         # if inside mouth, move outside
        #         if self.state == 2:
        #             print(f"Bite detected with down torque: {down_torque}. Moving outside mouth")
        #             self.state = 3

    def mouth_state_callback(self, msg):
        with self.mouth_open_lock:
            self.mouth_open = msg.data

    def transfer_completed_callback(self, msg):
        print("Complete transfer button pressed")
        if self.state == 2:
            with self.state_lock:
                self.state = 3
    
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
        
        # bias the force torque sensor
        # bias FT sensor
        bias = rospy.ServiceProxy('/forque/bias_cmd', String_cmd)
        bias('bias')

        closed_loop = True
        run_once = True
        paused_once = False

        # wait until the mouth is open
        while not rospy.is_shutdown():
            with self.mouth_open_lock:
                if self.mouth_open:
                    print("Detected mouth to be open")
                    # set it to false once (for the next time)
                    self.mouth_open = False
                    print("Setting mouth open to false")
                    break
            self.control_rate.sleep()

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
                    # shutdown the head perception thread
                    self.perception_interface.stop_head_perception_thread()

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
                
                if (not closed_loop) and maintain_position_at_goal: 
                    pass # maintain position at goal, do not update target
                else:
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

# if __name__ == '__main__':

#     bite_transfer_trajectory_tracker = InsideMouthTransfer()
#     signal.signal(signal.SIGINT, bite_transfer_trajectory_tracker.signal_handler) # ctrl+c
    
#     bite_transfer_trajectory_tracker.execute_transfer_loop()

#     rospy.spin()