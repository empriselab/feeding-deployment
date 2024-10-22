"""The main entry point for running the integrated system."""

from pathlib import Path
from typing import Any

import json

try:
    import rospy
    from std_msgs.msg import String

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
    LookAtPlateHLA,
    AcquireBiteHLA,
    tool_type,
)
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.web_interface import WebInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
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

def test_TransferToolHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_controller, flair, make_videos, tool):

    high_level_action = TransferToolHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_controller, flair)

    if tool == "fork":
        utensil = Object("utensil", tool_type)
        sim.held_object_name = "utensil"
        sim.held_object_id = sim.utensil_id
        perception_interface.set_head_perception_tool("fork") # set the tool in the perception interface
    elif tool == "drink":
        utensil = Object("drink", tool_type)
        sim.held_object_name = "drink"
        sim.held_object_id = sim.drink_id
        perception_interface.set_head_perception_tool("drink") # set the tool in the perception interface
    elif tool == "wipe":
        utensil = Object("wipe", tool_type)
        sim.held_object_name = "wipe"
        sim.held_object_id = sim.wipe_id
        perception_interface.set_head_perception_tool("wipe") # set the tool in the perception interface
    else:
        raise ValueError(f"Tool {tool} not recognized")

    sim.robot.set_finger_state(sim.scene_description.tool_grasp_fingers_value)
    finger_frame_id = sim.robot.link_from_name("finger_tip")
    end_effector_link_id = sim.robot.link_from_name(sim.robot.tool_link_name)
    utensil_from_end_effector = get_relative_link_pose(
        sim.robot.robot_id, finger_frame_id, end_effector_link_id, sim.physics_client_id
    )
    sim.held_object_tf = utensil_from_end_effector

    rviz_interface.tool_update(True, sim.held_object_name, Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the tool in rviz

    sim_traj = high_level_action.execute_action(objects=[utensil], params={})

    if make_videos:
        outfile = Path(__file__).parent / "single_action.mp4"
        make_simulation_video(sim, sim_traj, outfile)
        print(f"Saved video to {outfile}")

def test_LookAtPlateHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_controller, flair, make_videos):

    high_level_action = LookAtPlateHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_controller, flair)
    utensil = Object("utensil", tool_type)

    sim.held_object_name = "utensil"
    sim.held_object_id = sim.utensil_id
    sim.robot.set_finger_state(sim.scene_description.tool_grasp_fingers_value)
    finger_frame_id = sim.robot.link_from_name("finger_tip")
    end_effector_link_id = sim.robot.link_from_name(sim.robot.tool_link_name)
    utensil_from_end_effector = get_relative_link_pose(
        sim.robot.robot_id, finger_frame_id, end_effector_link_id, sim.physics_client_id
    )
    sim.held_object_tf = utensil_from_end_effector
    print(f"utensil_from_end_effector: {utensil_from_end_effector}")

    rviz_interface.tool_update(True, sim.held_object_name, Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the tool in rviz

    sim_traj = high_level_action.execute_action(objects=[utensil], params={})

def test_AcquireBiteHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_controller, flair, make_videos):

    print("WAITING FOR MESSAGE on /WebAppComm")
    msg = rospy.wait_for_message("/WebAppComm", String)
    msg_dict = json.loads(msg.data)   
    print(f"Received message: {msg_dict}")

    high_level_action = AcquireBiteHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_controller, flair)
    utensil = Object("utensil", tool_type)

    sim.held_object_name = "utensil"
    sim.held_object_id = sim.utensil_id
    sim.robot.set_finger_state(sim.scene_description.tool_grasp_fingers_value)
    finger_frame_id = sim.robot.link_from_name("finger_tip")
    end_effector_link_id = sim.robot.link_from_name(sim.robot.tool_link_name)
    utensil_from_end_effector = get_relative_link_pose(
        sim.robot.robot_id, finger_frame_id, end_effector_link_id, sim.physics_client_id
    )
    sim.held_object_tf = utensil_from_end_effector
    print(f"utensil_from_end_effector: {utensil_from_end_effector}")

    rviz_interface.tool_update(True, sim.held_object_name, Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the tool in rviz

    sim_traj = high_level_action.execute_action(objects=[utensil], params=msg_dict)
    
def _main(
    run_on_robot: bool, simulate_head_perception: bool, make_videos: bool, max_motion_planning_time: float = 10, tool: str = "fork"
) -> None:
    """Testing components of the system."""

    if ROSPY_IMPORTED:
        rospy.init_node("test_actions")
    else:
        assert not args.run_on_robot, "Need ROS to run on robot"

    # Initialize the interface to the robot.
    if run_on_robot:
        robot_interface = ArmInterfaceClient()  # type: ignore  # pylint: disable=no-member
    else:
        robot_interface = None

    if ROSPY_IMPORTED:
        web_interface = WebInterface()
    else:
        web_interface = None

    # Initialize the perceiver (e.g., get joint states or human head poses).
    perception_interface = PerceptionInterface(robot_interface=robot_interface, simulate_head_perception=simulate_head_perception)
    print("Perception Interface Loaded")

    # Initialize the FLAIR interface.
    if FLAIR_IMPORTED:
        wrist_controller = WristController()
        print("wrist controller init")
        flair = FLAIR(robot_interface, wrist_controller)
        print("FLAIR init")
    else:
        wrist_controller = None
        flair = None

    print("FLAIR loaded")

    # Initialize the simulator.
    kwargs: dict[str, Any] = {}
    if run_on_robot:
        kwargs["initial_joints"] = perception_interface.get_robot_joints()
        print(f"Initial joint state: {kwargs['initial_joints']}")
    else:
        print("Running in simulation mode.")
    scene_description = SceneDescription(**kwargs)

    print("Scene Description loaded")
    sim = FeedingDeploymentPyBulletSimulator(scene_description, use_gui=False)

    print("Feeding Deployment Simulator loaded")

    if ROSPY_IMPORTED:
        # Initialize the interface to RViz.
        rviz_interface = RVizInterface(scene_description)
    else:
        rviz_interface = None

    # Create skills for high-level planning.
    hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}

    # test_LookAtPlateHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_controller, flair, make_videos)
    # test_AcquireBiteHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_controller, flair, make_videos)
    test_TransferToolHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_controller, flair, make_videos, tool)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--simulate_head_perception", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    parser.add_argument("--tool", type=str, default="fork")
    args = parser.parse_args()

    _main(args.run_on_robot, args.simulate_head_perception, args.make_videos, args.max_motion_planning_time, args.tool)
