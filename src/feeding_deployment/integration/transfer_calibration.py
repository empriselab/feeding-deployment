"""The main entry point for running the integrated system."""

from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
import shutil

try:
    import rospy
    from std_msgs.msg import Bool

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
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.interfaces.web_interface import WebInterface
from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.control.wrist_controller.wrist_controller import WristInterface
from feeding_deployment.simulation.scene_description import create_scene_description_from_config
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator

def _main(
    scene_config: str, transfer_type: str, tool: str, test: bool, record_rom: bool, max_motion_planning_time: float = 10
) -> None:
    """Testing components of the system."""

    if ROSPY_IMPORTED:
        rospy.init_node("transfer_calibration")
        disable_collision_sensor_pub = rospy.Publisher("/disable_collision_sensor", Bool, queue_size=1)
    else:
        raise RuntimeError("ROS not imported. Please run this script in a ROS environment and with the real robot.")

    # Initialize the interface to the robot.
    robot_interface = ArmInterfaceClient()  # type: ignore  # pylint: disable=no-member
    web_interface = WebInterface()

    # Initialize the perceiver (e.g., get joint states or human head poses).
    if record_rom:
        test = True

    perception_interface = PerceptionInterface(robot_interface = robot_interface, record_goal_pose = not test)

    if not test: # calibrate utensil tip
        
        # turn off collision sensor
        disable_collision_sensor_pub.publish(Bool(data=True))
        print("Sent message to turn off collision sensor")
        
        # set tool transform - only required once globally
        perception_interface._head_perception.save_tool_tip_transform(args.tool)
        
        perception_interface._head_perception.set_tool(args.tool)
        while not rospy.is_shutdown():
            head_perception_data = perception_interface._head_perception.run_head_perception()
            if head_perception_data is None:
                break
        
        input("Can I turn the collision sensor back on? Press Enter to continue...")
        # turn on collision sensor
        disable_collision_sensor_pub.publish(Bool(data=False))
        print("Sent message to turn on collision sensor")

    elif record_rom:
        # turn off collision sensor
        disable_collision_sensor_pub.publish(Bool(data=True))
        print("Sent message to turn off collision sensor")

        perception_interface._head_perception.set_tool(args.tool)
        while not rospy.is_shutdown():
            head_perception_data = perception_interface._head_perception.run_head_perception()
            if head_perception_data is None:
                print("No head perception data")

        input("Can I turn the collision sensor back on? Press Enter to continue...")
        # turn on collision sensor
        disable_collision_sensor_pub.publish(Bool(data=False))
        print("Sent message to turn on collision sensor")
        
    else: # actually do the transfer
        run_on_robot = True

        scene_config_path = Path(__file__).parent.parent / "simulation" / "configs" / f"{scene_config}.yaml"
        scene_description = create_scene_description_from_config(str(scene_config_path), transfer_type)
        sim = FeedingDeploymentPyBulletSimulator(scene_description, use_gui=False)

        rviz_interface = RVizInterface(scene_description)
        wrist_controller = WristInterface()

        # Create skills for high-level planning.
        hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}

        # Copy the initial behavior trees into a directory for this run, where
        # they will be modified based on user feedback.
        run_behavior_tree_dir = Path(__file__).parent / "log" / "behavior_trees"
        run_behavior_tree_dir.mkdir(exist_ok=True)
        original_behavior_tree_dir = Path(__file__).parents[1] / "actions" / "behavior_trees"
        assert original_behavior_tree_dir.exists()
        for original_bt_filename in original_behavior_tree_dir.glob("*.yaml"):
            shutil.copy(original_bt_filename, run_behavior_tree_dir)

        high_level_action = TransferToolHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_controller, flair=None, behavior_tree_dir=run_behavior_tree_dir , no_waits=False, log_path=None)

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
        high_level_action.execute_action(objects=[object], params={})

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_config", type=str, default="vention")
    parser.add_argument("--transfer_type", type=str, default="inside")
    parser.add_argument("--tool", type=str, default="fork")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--record_rom", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    args = parser.parse_args()

    if args.tool not in ["fork", "drink", "wipe"]:
        raise ValueError(f"Invalid tool: {args.tool}, must be one of fork, drink, wipe")

    _main(args.scene_config, args.transfer_type, args.tool, args.test, args.record_rom, args.max_motion_planning_time)
