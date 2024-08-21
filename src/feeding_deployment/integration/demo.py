"""Demonstrate the full drinking pipeline."""

import time
import pickle

import pybullet as p
from pybullet_helpers.gui import create_gui_connection

from feeding_deployment.drinking.planning import generate_trajectory
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
OFFLINE = True


def _main():

    if not OFFLINE:
        # Initialize the robot arm and get the initial joint pose.
        manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        manager.connect()
        arm = manager.Arm()

        arm.retract()
        time.sleep(1.0)  # make sure arm stabilizes
        q, _ = arm.get_state()
        joint_state = q.tolist() + [0.0, 0.0]
        pickle.dump(joint_state, open("joint_state.pkl", "wb"))
    else:
        joint_state = pickle.load(open("joint_state.pkl", "rb"))

    # Create the scene and generate a trajectory.
    scene_description = CupManipulationSceneDescription(initial_joints=joint_state)
    physics_client_id = create_gui_connection(camera_yaw=180)
    scene = create_cup_manipulation_scene(physics_client_id, scene_description)
    traj = generate_trajectory(scene, scene_description)

    video_outfile = "last.mp4"
    make_cup_manipulation_video(scene, scene_description, traj, video_outfile)
    print(f"Video saved to {video_outfile}")
    p.disconnect(physics_client_id)

    if not OFFLINE:
        # Execute the trajectory.
        cmds = cup_manipulation_trajectory_to_kinova_commands(traj)
        input("Press enter to execute the plan.")
        for cmd in cmds:
            arm.execute_command(cmd)
        print("Plan executed!")

        arm.close()


if __name__ == "__main__":
    _main()
