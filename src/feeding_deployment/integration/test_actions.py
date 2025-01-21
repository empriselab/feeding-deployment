"""The main entry point for running the integrated system."""

from pathlib import Path
from typing import Any
import shutil
import json
import sys

try:
    import rospy
    from std_msgs.msg import String

    ROSPY_IMPORTED = True
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

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

from feeding_deployment.actions.base import tool_type
from feeding_deployment.actions.transfer_tool import TransferToolHLA
from feeding_deployment.actions.acquisition import AcquireBiteHLA
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.web_interface import WebInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.control.wrist_controller.wrist_controller import WristInterface
from feeding_deployment.simulation.scene_description import (
    SceneDescription,
    create_scene_description_from_config,
)
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.actions.flair.flair import FLAIR
from feeding_deployment.actions.flair.food_manipulation_skill_library import FoodManipulationSkillLibrary

def test_FoodManipulationSkillLibrary(sim, robot_interface, wrist_interface, perception_interface, rviz_interface, no_waits):

    wrist_interface.set_velocity_mode()    
    food_manipulation_skill_library = FoodManipulationSkillLibrary(sim, robot_interface, wrist_interface, perception_interface, rviz_interface, no_waits)
    food_manipulation_skill_library.reset()
    camera_color_data, camera_info_data, camera_depth_data = perception_interface.get_camera_data()
    food_manipulation_skill_library.dipping_skill(camera_color_data, camera_depth_data, camera_info_data)

def test_TransferToolHLA(tool, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, flair, run_behavior_tree_dir, no_waits, log_path=None):

    high_level_action = TransferToolHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, flair=None, behavior_tree_dir=run_behavior_tree_dir , no_waits=no_waits, log_path=None)    

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

    if robot_interface is not None:
        rviz_interface.tool_update(True, sim.held_object_name, Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the tool in rviz

    high_level_action.execute_action(objects=[utensil], params={})

def test_AcquireBiteHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, run_on_robot, wrist_interface, flair, no_waits):

    print("WAITING FOR MESSAGE on /WebAppComm")
    msg = rospy.wait_for_message("/WebAppComm", String)
    msg_dict = json.loads(msg.data)   
    print(f"Received message: {msg_dict}")

    high_level_action = AcquireBiteHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, flair, no_waits, log_path=None)
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

    if robot_interface is not None:
        rviz_interface.tool_update(True, sim.held_object_name, Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the tool in rviz

    high_level_action.execute_action(objects=[utensil], params=msg_dict)
    
def _main(
    scene_config: str, transfer_type: str, run_on_robot: bool, use_interface: bool, simulate_head_perception: bool, use_gui: bool, make_videos: bool, max_motion_planning_time: float = 10, tool: str = "fork", no_waits: bool = False
) -> None:
    """Testing components of the system."""

    if ROSPY_IMPORTED:
        rospy.init_node("test_actions")
    else:
        assert not args.run_on_robot, "Need ROS to run on robot"

    # logs are saved in user/scenario directory
    log_dir = Path(__file__).parent / "log" / "test_actions_log"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(exist_ok=True)

    execution_log = Path(__file__).parent / "log" / "execution_log.txt" # in root log directory
    run_behavior_tree_dir = log_dir / "behavior_trees"
    gesture_detectors_dir = log_dir / "gesture_detectors"
            
    # Copy the initial behavior trees into a directory for this run, where
    # they will be modified based on user feedback.
    run_behavior_tree_dir.mkdir(exist_ok=True)
    original_behavior_tree_dir = Path(__file__).parents[1] / "actions" / "behavior_trees"
    assert original_behavior_tree_dir.exists()
    for original_bt_filename in original_behavior_tree_dir.glob("*.yaml"):
        shutil.copy(original_bt_filename, run_behavior_tree_dir)

    # Copy the initial gesture detection file into a directory for this run,
    # where it will be updated from LLM-based few-shot learning.
    gesture_detectors_dir.mkdir(exist_ok=True)
    original_gesture_detection_filepath = Path(__file__).parents[1] / "perception" / "gestures_perception" / "synthesized_gesture_detectors.py"
    assert original_gesture_detection_filepath.exists()
    shutil.copy(original_gesture_detection_filepath, gesture_detectors_dir)


    log_dir = Path(__file__).parent / "log" / "test_actions_log"
    log_dir.mkdir(exist_ok=True)

    # Initialize the interface to the robot.
    if run_on_robot:
        robot_interface = ArmInterfaceClient()  # type: ignore  # pylint: disable=no-member
        wrist_interface = WristInterface()
    else:
        robot_interface = None
        wrist_interface = None

    flair = FLAIR(log_dir)

    if use_interface:
        web_interface = WebInterface()
    else:
        web_interface = None

    # Initialize the perceiver (e.g., get joint states or human head poses).
    perception_interface = PerceptionInterface(robot_interface=robot_interface, simulate_head_perception=simulate_head_perception, log_dir=log_dir)

    scene_config_path = Path(__file__).parent.parent / "simulation" / "configs" / f"{scene_config}.yaml"
    scene_description = create_scene_description_from_config(str(scene_config_path), transfer_type)
    sim = FeedingDeploymentPyBulletSimulator(scene_description, use_gui=use_gui)

    if robot_interface is not None:
        # Initialize the interface to RViz.
        rviz_interface = RVizInterface(scene_description)
    else:
        rviz_interface = None

    # # Create skills for high-level planning.
    # hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}

    # # Copy the initial behavior trees into a directory for this run, where
    # # they will be modified based on user feedback.
    # run_behavior_tree_dir = Path(__file__).parent / "log" / "test_actions" / "behavior_trees"
    # run_behavior_tree_dir.mkdir(exist_ok=True)
    # original_behavior_tree_dir = Path(__file__).parents[1] / "actions" / "behavior_trees"
    # assert original_behavior_tree_dir.exists()
    # for original_bt_filename in original_behavior_tree_dir.glob("*.yaml"):
    #     shutil.copy(original_bt_filename, run_behavior_tree_dir)

    test_FoodManipulationSkillLibrary(sim, robot_interface, wrist_interface, perception_interface, rviz_interface, no_waits)
    # test_AcquireBiteHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, flair, tool, no_waits)
    # for i in range(25):
    # test_TransferToolHLA(tool, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, flair, run_behavior_tree_dir, no_waits, log_path=None)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_config", type=str, default="vention")
    parser.add_argument("--transfer_type", type=str, default="inside")
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--use_interface", action="store_true")
    parser.add_argument("--simulate_head_perception", action="store_true")
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    parser.add_argument("--tool", type=str, default="fork")
    parser.add_argument("--no_waits", action="store_true")
    args = parser.parse_args()

    _main(args.scene_config, args.transfer_type, args.run_on_robot, args.use_interface, args.simulate_head_perception, args.use_gui, args.make_videos, args.max_motion_planning_time, args.tool, args.no_waits)
