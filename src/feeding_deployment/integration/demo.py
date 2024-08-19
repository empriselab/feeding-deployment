"""Demonstrate the full drinking pipeline."""

import time

import pybullet as p
from pybullet_helpers.gui import create_gui_connection

from feeding_deployment.drinking.planning import generate_trajectory
from feeding_deployment.drinking.scene import (
    CupManipulationSceneDescription,
    create_cup_manipulation_scene,
)
from feeding_deployment.drinking.utils import (
    get_kinova_controller_trajectory,
    make_cup_manipulation_video,
)
from feeding_deployment.robot_controller.kinova import KinovaArm


def _main() -> None:
    # Initialize the robot arm and get the initial joint pose.
    arm = KinovaArm()
    arm.retract()
    time.sleep(1.0)  # make sure arm stabilizes
    q, gripper_pos = arm.get_state()
    joint_state = q.tolist() + [gripper_pos, gripper_pos]

    # Create the scene and generate a trajectory.
    scene_description = CupManipulationSceneDescription(initial_joints=joint_state)
    physics_client_id = create_gui_connection(camera_yaw=180)
    scene = create_cup_manipulation_scene(physics_client_id, scene_description)
    traj = generate_trajectory(scene, scene_description)

    video_outfile = "last.mp4"
    make_cup_manipulation_video(scene, scene_description, traj, video_outfile)
    print(f"Video saved to {video_outfile}")
    p.disconnect(physics_client_id)

    # Execute the trajectory.
    cmds = get_kinova_controller_trajectory(traj)
    joint_cmds = [j for j, _ in cmds]
    input("Press enter to execute the plan.")
    arm.move_angular_trajectory(joint_cmds)


if __name__ == "__main__":
    _main()
