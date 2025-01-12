#! /usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import time

try:
    import rospy
    from std_msgs.msg import Bool
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

# Parameters
OPEN_LOOP_RADIUS = 0.02
# OPEN_LOOP_RADIUS = 0.0
INTERMEDIATE_THRESHOLD_RELAXED = 0.03
INTERMEDIATE_ANGULAR_THRESHOLD_RELAXED = 7*np.pi/180
INTERMEDIATE_THRESHOLD = 0.014
INFRONT_DISTANCE_LOOKAHEAD = 0.04
INSIDE_DISTANCE_LOOKAHEAD_Z = 0.04 # for slower inside mouth movement
# INSIDE_DISTANCE_LOOKAHEAD_Z = 0.045
INSIDE_DISTANCE_LOOKAHEAD_XY = 0.025
ANGULAR_LOOKAHEAD = 5*np.pi/180
MOVE_OUTSIDE_DISTANCE = 0.12
DISTANCE_INFRONT_MOUTH = 0.12

from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.control.robot_controller.command_interface import CartesianCommand
from feeding_deployment.actions.feel_the_bite.base import Transfer

from pybullet_helpers.geometry import Pose

class InsideMouthTransfer(Transfer):
    def __init__(self, sim : FeedingDeploymentPyBulletSimulator, robot_interface: ArmInterfaceClient, perception_interface: PerceptionInterface, rviz_interface: RVizInterface, no_waits=False):
            
        super().__init__(sim, robot_interface, perception_interface, rviz_interface, no_waits)

        self.control_time = 0.01

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

    def move_to_transfer_state(self, maintain_position_at_goal = False):

        if self.robot_interface is not None:
            self.set_filter_noisy_readings_pub.publish(Bool(data=True))

        head_perception_data = self.perception_interface.get_head_perception_data()
        forque_target_base = head_perception_data["tool_tip_target_pose"]
        head_pose = head_perception_data["head_pose"]
        self.sim.set_head_pose(Pose(position=head_pose[:3], orientation=Rotation.from_euler('yxz', head_pose[3:], degrees=True).as_quat()))

        servo_point_forque_target = np.identity(4)
        servo_point_forque_target[:3,3] = np.array([0, 0, -DISTANCE_INFRONT_MOUTH]).reshape(1,3)

        servo_point_base = forque_target_base @ servo_point_forque_target

        if self.robot_interface is None:
            # In simulation, directly move to infront of mouth
            tool_frame_target = servo_point_base @ self.get_tip_wrist_transform()
            target_pose = Pose.from_matrix(tool_frame_target)
            self.move_to_ee_pose(target_pose)
        else:
            # For real robot, move infront of mouth by setting intermediate waypoints for compliant controller
            while True:
                time.sleep(self.control_time)

                forque_base = self.perception_interface.get_tool_tip_pose()
                distance = np.linalg.norm(forque_base[:3,3] - servo_point_base[:3,3])
                angular_distance = self.getAngularDistance(forque_base[:3,:3], servo_point_base[:3,:3])

                if distance < INTERMEDIATE_THRESHOLD_RELAXED and angular_distance < INTERMEDIATE_ANGULAR_THRESHOLD_RELAXED:
                    break

                target = self.getNextWaypoint(forque_base, servo_point_base, distance_lookahead=INFRONT_DISTANCE_LOOKAHEAD)

                self.move_to_ee_pose(Pose.from_matrix(target))
                self.rviz_interface.visualizeTransform("base_link", "next_target", target)
                self.rviz_interface.visualizeTransform("base_link", "final_target", servo_point_base)
            
            # Pause so that new head pose can be re-perceived
            time.sleep(0.2)

        if self.robot_interface is None:
            head_perception_data = self.perception_interface.get_head_perception_data()
            forque_target_base = head_perception_data["tool_tip_target_pose"]
            tool_frame_target = forque_target_base @ self.get_tip_wrist_transform()
            target_pose = Pose.from_matrix(tool_frame_target)
            self.move_to_ee_pose(target_pose)
        else:
            # For real robot, move inside the mouth by setting intermediate waypoints for compliant controller
            while True:
                time.sleep(self.control_time)

                head_perception_data = self.perception_interface.get_head_perception_data()
                forque_target_base = head_perception_data["tool_tip_target_pose"]

                forque_source = self.perception_interface.get_tool_tip_pose()
                distance = np.linalg.norm(forque_source[:3,3] - forque_target_base[:3,3])

                if distance < OPEN_LOOP_RADIUS:
                    break

                intermediate_forque_target = np.eye(4)
                intermediate_forque_target[:3,3] = np.array([0, 0, -distance]).reshape(1,3)
                intermediate_forque_target = forque_target_base @ intermediate_forque_target

                intermediate_position_error = np.linalg.norm(forque_source[:3,3] - intermediate_forque_target[:3,3])
                intermediate_angular_error = self.getAngularDistance(forque_source[:3,:3], intermediate_forque_target[:3,:3])

                ipe_forque_frame = np.linalg.inv(forque_source[:3,:3]) @ (forque_source[:3,3] - intermediate_forque_target[:3,3]).reshape(3,1)

                if intermediate_position_error > INTERMEDIATE_THRESHOLD: # The thresholds here should be ideally larger than the thresholds for tracking trajectories
                    target = self.getNextWaypoint(forque_source, intermediate_forque_target, distance_lookahead = INSIDE_DISTANCE_LOOKAHEAD_XY)
                else:
                    distance_lookahead_update = INSIDE_DISTANCE_LOOKAHEAD_Z - intermediate_position_error
                    orientation_lookahead_update = ANGULAR_LOOKAHEAD - intermediate_angular_error
                    target = self.getNextWaypoint(intermediate_forque_target, forque_target_base, distance_lookahead = distance_lookahead_update)
                
                self.move_to_ee_pose(Pose.from_matrix(target))
                self.rviz_interface.visualizeTransform("base_link", "next_target", target)
                self.rviz_interface.visualizeTransform("base_link", "final_target", forque_target_base)
                self.rviz_interface.visualizeTransform("base_link", "intermediate_target", intermediate_forque_target)

            self.set_filter_noisy_readings_pub.publish(Bool(data=False))

    def move_to_before_transfer_state(self):

        if self.robot_interface is None:
            # In simulation, directly move to outside the mouth
            head_perception_data = self.perception_interface.get_head_perception_data()
            source_base = head_perception_data["tool_tip_target_pose"]
            forque_target_source = np.identity(4)
            forque_target_source[:3,3] = np.array([0, 0, -MOVE_OUTSIDE_DISTANCE]).reshape(1,3)
            forque_target_base = source_base @ forque_target_source
            tool_frame_target = forque_target_base @ self.get_tip_wrist_transform()
            target_pose = Pose.from_matrix(tool_frame_target)
            self.move_to_ee_pose(target_pose)
        else:
            # For real robot, move outside the mouth by setting intermediate waypoints for compliant controller
            source_base = self.perception_interface.get_tool_tip_pose()
            forque_target_source = np.identity(4)
            forque_target_source[:3,3] = np.array([0, 0, -MOVE_OUTSIDE_DISTANCE]).reshape(1,3)
            forque_target_base = source_base @ forque_target_source
            
            while True:
                time.sleep(self.control_time)
                forque_base = self.perception_interface.get_tool_tip_pose()
                distance = np.linalg.norm(forque_base[:3,3] - forque_target_base[:3,3])

                if distance < INTERMEDIATE_THRESHOLD_RELAXED:
                    break   

                target = self.getNextWaypoint(forque_base, forque_target_base, distance_lookahead = INSIDE_DISTANCE_LOOKAHEAD_Z)
                
                self.move_to_ee_pose(Pose.from_matrix(target))
                self.rviz_interface.visualizeTransform("base_link", "next_target", target)
                self.rviz_interface.visualizeTransform("base_link", "final_target", forque_target_base)

        final_target = self.perception_interface.get_tool_tip_pose_at_staging()

        if self.robot_interface is None:
            # In simulation, directly move to staging configuration
            tool_frame_target = final_target @ self.get_tip_wrist_transform()
            target_pose = Pose.from_matrix(tool_frame_target)
            self.move_to_ee_pose(target_pose)
        else:
            # For real robot, move to staging configuration by setting intermediate waypoints for compliant controller
            while True:
                time.sleep(self.control_time)
                forque_base = self.perception_interface.get_tool_tip_pose()
                distance = np.linalg.norm(forque_base[:3,3] - final_target[:3,3])

                if distance < INTERMEDIATE_THRESHOLD_RELAXED:
                    break
                
                target = self.getNextWaypoint(forque_base, final_target, distance_lookahead=INFRONT_DISTANCE_LOOKAHEAD)

                self.move_to_ee_pose(Pose.from_matrix(target))
                self.rviz_interface.visualizeTransform("base_link", "next_target", target)
                self.rviz_interface.visualizeTransform("base_link", "final_target", final_target)