import rospy

import sys

sys.path.append("../../../FLAIR/bite_acquisition/scripts")
sys.path.append("../../pybullet-cup-manipulation")

from robot_controller.kinova_controller import KinovaRobotController
from cup_manipulation import generate_trajectory
from scene import (
    create_cup_manipulation_scene,
    CupManipulationSceneDescription,
)
from cup_manipulation_utils import make_cup_manipulation_video
import pybullet as p
from pybullet_helpers.gui import create_gui_connection
from geometry_msgs.msg import Pose, Point, Quaternion
from pybullet_helpers.geometry import Pose as PHPose
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
import numpy as np
import utils
import pickle


if __name__ == "__main__":
    rospy.init_node("robot_controller", anonymous=True)
    robot_controller = KinovaRobotController()
    tf_utils = utils.TFUtils()

    stow_joints = (3.810379416009879e-05, -0.3490871556875863, -3.1415676198755462, -2.5481961542460434, 1.2354585983175708e-05, -0.8726976491946816, 1.5708019194331857, 0.006986920833587647, 0.006986920833587647)
    robot_controller.set_joint_position(stow_joints)

    # Get the initial joints.
    initial_joint_msg = rospy.wait_for_message("/robot_joint_states", JointState)
    assert initial_joint_msg.name == [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
        "finger_joint",
    ]
    finger_val = initial_joint_msg.position[-1]
    initial_joints = tuple(initial_joint_msg.position[:7]) + (finger_val, finger_val)
    print("Initial joints:", initial_joints)

    # Create the scene description.
    scene_description = CupManipulationSceneDescription(initial_joints=initial_joints)
    
    # with open("debug.traj", "rb") as f:
    #     saved_scene_description, saved_traj = pickle.load(f)
    # assert scene_description.allclose(saved_scene_description, atol=1e-4)
    # traj = saved_traj

    # Visualize the scene.
    physics_client_id = create_gui_connection(camera_yaw=180)
    scene = create_cup_manipulation_scene(physics_client_id, scene_description)
    traj = generate_trajectory(
        scene,
        scene_description,
        seed=0,
        max_motion_plan_time=10,
    )
    with open("debug.traj", "wb") as f:
        pickle.dump((scene_description, traj), f)
    print(f"Dumped trajectory to debug.traj")

    video_outfile = "generated_trajectory.mp4"
    make_cup_manipulation_video(
        scene,
        scene_description,
        traj,
        video_outfile,
    )

    input("Press enter to execute motion plan")
    for joint_state in traj.joint_states:
        robot_controller.set_joint_position(joint_state)
