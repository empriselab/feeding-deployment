import rospy

import sys
sys.path.append('../../../FLAIR/bite_acquisition/scripts')
sys.path.append('../../pybullet-cup-manipulation')

from robot_controller.kinova_controller import KinovaRobotController
from cup_manipulation import generate_trajectory
from geometry_msgs.msg import Pose, Point, Quaternion
from pybullet_helpers.geometry import Pose as PHPose
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
import numpy as np
import utils


if __name__ == "__main__":
    rospy.init_node('robot_controller', anonymous=True)
    robot_controller = KinovaRobotController()
    tf_utils = utils.TFUtils()

    # Move to a default start pose.
    x, y, z = 0.3, 0.25, 0.25
    default_tool_rot = Rotation.from_euler('xyz', [90, 0, 90], degrees=True)

    # Set by looking at the robot.
    quat = tuple(Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_quat())
    robot_base_pose = PHPose((0.0, 0.0, 0.0), quat)

    start_pose = Pose(position=Point(x, y, z), orientation=Quaternion(*default_tool_rot.as_quat()))
    robot_controller.move_to_pose(start_pose)

    # Get the initial joints.
    initial_joint_msg = rospy.wait_for_message('/robot_joint_states', JointState)
    assert initial_joint_msg.name == ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "finger_joint"]
    finger_val = initial_joint_msg.position[-1]
    initial_joints = tuple(initial_joint_msg.position[:7]) + (finger_val, finger_val)
    joint_space_plan = generate_trajectory(initial_joints, robot_base_pose=robot_base_pose)

    for joint_positions in joint_space_plan:
        robot_controller.set_joint_position(joint_positions[:8])
        rospy.sleep(0.1)

    # Try to close the grippers.
    # joint_state_msg = rospy.wait_for_message('/robot_joint_states', JointState)
    # closed_finger_joints = np.copy(joint_state_msg.position)
    # finger_idx = joint_state_msg.name.index("finger_joint")
    # closed_finger_joints[finger_idx] = 0.5  # TODO figure out how to get gripper joint limits
    # robot_controller.set_joint_position(closed_finger_joints)
   
   
    # # Detect the aruco marker and transform it into the tool frame.
    # aruco_center_in_camera_msg = rospy.wait_for_message('/aruco_center', Point)
    # aruco_center_in_camera = np.array([aruco_center_in_camera_msg.x, aruco_center_in_camera_msg.y, aruco_center_in_camera_msg.z])

    # base_to_camera = tf_utils.getTransformationFromTF("base_link", "camera_color_optical_frame")
    # base_to_tip = tf_utils.getTransformationFromTF('base_link', 'finger_tip')

    # aruco_transform = np.eye(4)
    # aruco_transform[:3,3] = aruco_center_in_camera.reshape(1,3)
    # aruco_base = base_to_camera @ aruco_transform

    # tf_utils.publishTransformationToTF('base_link', 'finger_tip_target', aruco_base)
    # tip_to_tool = tf_utils.getTransformationFromTF('finger_tip', 'tool_frame')
    # tool_frame_target = aruco_base @ tip_to_tool

    # # Prevent any rotations.
    # tool_frame_target[:3,:3] = default_tool_rot.as_matrix()
    # tf_utils.publishTransformationToTF('base_link', 'tool_frame_target', tool_frame_target)

    # # Execute movement.
    # end_pose = tf_utils.get_pose_msg_from_transform(tool_frame_target)
    # print("END POSE: ")
    # print(end_pose)
    # # robot_controller.move_to_pose(end_pose)

