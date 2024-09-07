"""The main entry point for running the integrated system."""

from pathlib import Path
from typing import Any

try:
    import rospy

    ROSPY_IMPORTED = True
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

# Rajat ToDo: Remove this hacky addition
FLAIR_PATH = "/home/isacc/deployment_ws/src/FLAIR/bite_acquisition/scripts"
import sys

sys.path.append(FLAIR_PATH)
try:
    # raise ModuleNotFoundError  # Just to skip this block
    from wrist_controller import WristController
    from flair import FLAIR

    FLAIR_IMPORTED = True
    print("FLAIR imported successfully")
except ModuleNotFoundError:
    FLAIR_IMPORTED = False
    pass

from relational_structs import (
    GroundAtom,
    LiftedAtom,
    Object,
    PDDLDomain,
    PDDLProblem,
    Predicate,
)
from relational_structs.utils import parse_pddl_plan
from tomsutils.pddl_planning import run_pyperplan_planning
from pybullet_helpers.geometry import Pose
from pybullet_helpers.link import get_link_pose, get_relative_link_pose

from feeding_deployment.actions.high_level_actions import (
    TransferToolHLA,
    tool_type,
)
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.interfaces.web_interface import WebInterface
from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.simulation.scene_description import (
    SceneDescription,
    create_scene_description_from_config,
)
from feeding_deployment.simulation.simulator import (
    FeedingDeploymentPyBulletSimulator,
    FeedingDeploymentSimulatorState,
)
from feeding_deployment.simulation.video import make_simulation_video

import numpy as np
from scipy.spatial.transform import Rotation

def _main(
    tool: str, test: bool, record_rom: bool, max_motion_planning_time: float = 10
) -> None:
    """Testing components of the system."""

    if ROSPY_IMPORTED:
        rospy.init_node("test_actions")
    else:
        raise RuntimeError("ROS not imported. Please run this script in a ROS environment and with the real robot.")

    # Initialize the interface to the robot.
    robot_interface = ArmInterfaceClient()  # type: ignore  # pylint: disable=no-member

    if ROSPY_IMPORTED:
        web_interface = WebInterface()
    else:
        web_interface = None

    # Initialize the perceiver (e.g., get joint states or human head poses).
    perception_interface = PerceptionInterface(robot_interface, not test)
    
    wrist_controller = WristController()

    if not test: # calibrate utensil tip
        if not record_rom:
            # set tool transform - only required once globally
            perception_interface._head_perception.save_tool_tip_transform(args.tool)
            
            perception_interface._head_perception.set_tool(args.tool)
            while not rospy.is_shutdown():
                perception_interface._head_perception.run_head_perception()
        else:
            perception_interface._head_perception.set_tool(args.tool)
            while not rospy.is_shutdown():
                perception_interface._head_perception.run_head_perception()
    else: # actually do the transfer
        run_on_robot = True

        # Initialize the simulator.
        kwargs: dict[str, Any] = {}
        kwargs["initial_joints"] = perception_interface.get_robot_joints()
        scene_description = SceneDescription(**kwargs)
        sim = FeedingDeploymentPyBulletSimulator(scene_description, use_gui=False)

        rviz_interface = RVizInterface(scene_description)

        # Create skills for high-level planning.
        hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}

        high_level_action = TransferToolHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_controller, None)

        if tool == "fork":
            object = Object("utensil", tool_type)
            sim.held_object_name = "utensil"
            sim.held_object_id = sim.utensil_id
        elif tool == "drink":
            object = Object("drink", tool_type)
            sim.held_object_name = "drink"
            sim.held_object_id = sim.drink_id
        elif tool == "wipe":
            object = Object("wipe", tool_type)
            sim.held_object_name = "wipe"
            sim.held_object_id = sim.wipe_id
        else:
            raise ValueError(f"Invalid tool: {tool}")
        sim.robot.set_finger_state(sim.scene_description.tool_grasp_fingers_value)
        finger_frame_id = sim.robot.link_from_name("finger_tip")
        end_effector_link_id = sim.robot.link_from_name(sim.robot.tool_link_name)
        utensil_from_end_effector = get_relative_link_pose(
            sim.robot.robot_id, finger_frame_id, end_effector_link_id, sim.physics_client_id
        )
        sim.held_object_tf = utensil_from_end_effector
        print(f"utensil_from_end_effector: {utensil_from_end_effector}")

        rviz_interface.tool_update(True, sim.held_object_name, Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the tool in rviz

        perception_interface._head_perception.set_tool(args.tool)
        sim_traj = high_level_action.execute_action(objects=[object], params={})

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=str, default="fork")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--record_rom", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    args = parser.parse_args()

    if args.tool not in ["fork", "drink", "wipe"]:
        raise ValueError(f"Invalid tool: {args.tool}, must be one of fork, drink, wipe")

    _main(args.tool, args.test, args.record_rom, args.max_motion_planning_time)
