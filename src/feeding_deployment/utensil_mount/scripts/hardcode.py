import rospy

import sys

sys.path.append("../../../FLAIR/bite_acquisition/scripts")

from robot_controller.kinova_controller import KinovaRobotController
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
import numpy as np
import utils


if __name__ == "__main__":
    rospy.init_node("robot_controller", anonymous=True)
    robot_controller = KinovaRobotController()
    tf_utils = utils.TFUtils()

    home_pos = [
        2.2912759438800285,
        0.7308686750765581,
        2.082994642398784,
        4.109475142253324,
        0.2853091081120964,
        5.818345985240578,
        5.988186420599291,
    ]

    inside_mount_pose = Pose(
        position=Point(-0.147, -0.17, 0.07),
        orientation=Quaternion(0.7071068, -0.7071068, 0, 0),
    )

    outside_mount_pose = Pose(
        position=Point(-0.147, -0.29, 0.07),
        orientation=Quaternion(0.7071068, -0.7071068, 0, 0),
    )
    outside_mount_joint_states = [
        2.6266411620509817,
        0.6992626121546339,
        2.306749708761716,
        4.053362604401464,
        0.9559379448584164,
        5.655628973165609,
        5.80065247559031,
    ]

    above_mount_pose = Pose(
        position=Point(-0.147, -0.17, 0.15),
        orientation=Quaternion(0.7071068, -0.7071068, 0, 0),
    )
    above_mount_joint_states = [
        3.300153003835367,
        0.39120874346320217,
        1.8613410764520344,
        3.862447510072517,
        0.6143839397882825,
        5.583536137192727,
        6.276739392077158,
    ]

    input("Press enter to move to home joint pos...")
    joint_state_msg = rospy.wait_for_message("/robot_joint_states", JointState)
    finger_idx = joint_state_msg.name.index("finger_joint")
    next_pos = home_pos.copy()
    next_pos.append(joint_state_msg.position[finger_idx])
    robot_controller.set_joint_position(next_pos)

    input("Press enter to move to outside mount joint pos...")
    joint_state_msg = rospy.wait_for_message("/robot_joint_states", JointState)
    finger_idx = joint_state_msg.name.index("finger_joint")
    next_pos = outside_mount_joint_states.copy()
    next_pos.append(joint_state_msg.position[finger_idx])
    robot_controller.set_joint_position(next_pos)

    input("Press enter to move to inside mount pose...")
    robot_controller.move_to_pose(inside_mount_pose)

    input("Press enter to release the utensil...")
    joint_state_msg = rospy.wait_for_message("/robot_joint_states", JointState)
    closed_finger_joints = np.copy(joint_state_msg.position)
    finger_idx = joint_state_msg.name.index("finger_joint")
    closed_finger_joints[finger_idx] = 1.0
    robot_controller.set_joint_position(closed_finger_joints)

    input("Press enter to move up...")
    robot_controller.move_to_pose(above_mount_pose)

    input("Press enter to move to home joint pos...")
    joint_state_msg = rospy.wait_for_message("/robot_joint_states", JointState)
    finger_idx = joint_state_msg.name.index("finger_joint")
    next_pos = home_pos.copy()
    next_pos.append(joint_state_msg.position[finger_idx])
    robot_controller.set_joint_position(next_pos)

    input("Press enter to move to above mount joint states  with closed gripper...")
    joint_state_msg = rospy.wait_for_message("/robot_joint_states", JointState)
    finger_idx = joint_state_msg.name.index("finger_joint")
    next_pos = above_mount_joint_states.copy()
    next_pos.append(1.0)
    robot_controller.set_joint_position(next_pos)
    robot_controller.move_to_pose(above_mount_pose)

    input("Press enter to inside mount pose...")
    robot_controller.move_to_pose(inside_mount_pose)

    input("Press enter to grab the utensil...")
    joint_state_msg = rospy.wait_for_message("/robot_joint_states", JointState)
    closed_finger_joints = np.copy(joint_state_msg.position)
    finger_idx = joint_state_msg.name.index("finger_joint")
    closed_finger_joints[finger_idx] = 0.5
    robot_controller.set_joint_position(closed_finger_joints)

    input("Press enter to move to outside mount pose...")
    robot_controller.move_to_pose(outside_mount_pose)

    input("Press enter to move to home joint pos...")
    joint_state_msg = rospy.wait_for_message("/robot_joint_states", JointState)
    finger_idx = joint_state_msg.name.index("finger_joint")
    next_pos = home_pos.copy()
    next_pos.append(joint_state_msg.position[finger_idx])
    robot_controller.set_joint_position(next_pos)

    # # Try to close the grippers.
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
