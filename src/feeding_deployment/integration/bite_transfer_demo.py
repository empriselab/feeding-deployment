"""Demonstrate the full drinking pipeline."""

import time
import threading
from scipy.spatial.transform import Rotation as R
import pickle

import pybullet as p
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.geometry import Pose, Pose3D, Quaternion, multiply_poses

from feeding_deployment.drinking.planning import generate_bite_transfer_trajectory
from feeding_deployment.drinking.scene import (
    CupManipulationSceneDescription,
    create_cup_manipulation_scene,
)
from feeding_deployment.drinking.utils import (
    make_cup_manipulation_video,
)
from feeding_deployment.integration.utils import (
    cup_manipulation_trajectory_to_kinova_commands,
)
from feeding_deployment.robot_controller.arm_client import (
    ARM_RPC_PORT,
    NUC_HOSTNAME,
    RPC_AUTHKEY,
    ArmManager,
)

from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper

import rospy
from sensor_msgs.msg import JointState
OFFLINE = True

def publish_joint_states(arm):

    # publish joint states
    joint_states_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)

    while not rospy.is_shutdown():
        arm_pos, gripper_pos = arm.get_state()
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
        joint_states_pub.publish(joint_state_msg)
        time.sleep(0.01)

def _main():

    if not OFFLINE:

        rospy.init_node("arm_client", anonymous=True)

        # Initialize the robot arm and get the initial joint pose.
        manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        manager.connect()
        arm = manager.Arm()

        # publish joint states in separate thread
        joint_state_thread = threading.Thread(target=publish_joint_states, args=(arm,))
        joint_state_thread.start()


        # before_transfer_pose = [-2.7611776687351686, -1.1868025860674232, -1.701402540342344, -1.8118757886810117, 0.2697561974117632, -0.09096026703165627, 2.4944170781413106]
        # arm.set_joint_position(before_transfer_pose)
        
        # arm.retract()
        time.sleep(1.0)  # make sure arm stabilizes
        q, gripper_position = arm.get_state()
        print("gripper_position: ", gripper_position)
        joint_state = q.tolist() + [gripper_position, gripper_position]

        # run head perception
        head_perception = HeadPerceptionROSWrapper()

        # warm start head perception
        for _ in range(10):
            head_perception.run_head_perception()

        input("Press enter to take a snapshot of the user's head pose.")
        forque_target_transform = head_perception.run_head_perception()

        pickle.dump(forque_target_transform, open("forque_target_transform.pkl", "wb"))
        pickle.dump(joint_state, open("joint_state.pkl", "wb"))
    else:
        forque_target_transform = pickle.load(open("forque_target_transform.pkl", "rb"))
        joint_state = pickle.load(open("joint_state.pkl", "rb"))

    forque_target_pose: Pose = Pose(
        (forque_target_transform[0, 3], forque_target_transform[1, 3], forque_target_transform[2, 3]),
        R.from_matrix(forque_target_transform[:3, :3]).as_quat(),
    )

    # Create the scene and generate a trajectory.
    scene_description = CupManipulationSceneDescription(initial_joints=joint_state)
    physics_client_id = create_gui_connection(camera_yaw=180)
    scene = create_cup_manipulation_scene(physics_client_id, scene_description)

    # premultiply forque_target_pose by the robot base pose to get the target pose in the pybullet frame
    forque_target_pose = multiply_poses(scene_description.robot_base_pose, forque_target_pose)
    traj = generate_bite_transfer_trajectory(forque_target_pose, scene, scene_description)

    video_outfile = "last.mp4"
    make_cup_manipulation_video(scene, scene_description, traj, video_outfile)
    print(f"Video saved to {video_outfile}")
    p.disconnect(physics_client_id)

    # Execute the trajectory.
    cmds = cup_manipulation_trajectory_to_kinova_commands(traj)

    if not OFFLINE:
        input("Press enter to execute the plan.")
        for cmd in cmds:
            arm.execute_command(cmd)
        print("Plan executed!")

        # wait for joint state thread to finish
        joint_state_thread.join()

        arm.close()


if __name__ == "__main__":
    _main()
