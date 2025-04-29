"""The main entry point for running the integrated system."""

import json
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, List
import pickle
import queue
import os
import sys
import signal
import shutil
import numpy as np
from tomsutils.llm import OpenAILLM
import time
import types
import inspect
import base64
import itertools
from dataclasses import dataclass
from PIL import Image

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
from relational_structs.utils import parse_pddl_plan, get_object_combinations
from tomsutils.pddl_planning import run_pddl_planner
from tomsutils.spaces import EnumSpace
from pybullet_helpers.geometry import Pose

from feeding_deployment.actions.base import (
    GripperFree,
    Holding,
    IsUtensil,
    PlateInView,
    ToolPrepared,
    ToolTransferDone,
    EmulateTransferDone,
    ResetPos,
    tool_type,
    GroundHighLevelAction,
    ResetHLA,
    pddl_plan_to_hla_plan,
    interpret_user_update_request,
    load_behavior_tree,
    save_behavior_tree,
    NodeModificationUserUpdateRequest,
    NodeAdditionUserRequest,
    UserUpdateRequest,
    ParameterizedActionBehaviorTreeNode
)
from feeding_deployment.actions.pick_tool import PickToolHLA
from feeding_deployment.actions.stow_tool import StowToolHLA
from feeding_deployment.actions.transfer_tool import TransferToolHLA
from feeding_deployment.actions.emulate_transfer import EmulateTransferHLA
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
from feeding_deployment.simulation.simulator import (
    FeedingDeploymentPyBulletSimulator,
    FeedingDeploymentWorldState,
)
from feeding_deployment.actions.flair.flair import FLAIR
from feeding_deployment.transparency.query_llm import TransparencyQuery


# All the high level actions we want to consider.
HLAS = {PickToolHLA, StowToolHLA, AcquireBiteHLA, TransferToolHLA, EmulateTransferHLA, ResetHLA}

assert os.environ.get("PYTHONHASHSEED") == "0", \
        "Please add `export PYTHONHASHSEED=0` to your bash profile!"

class _Runner:
    """A class for running the integrated system."""

    def __init__(self, scene_config: str, user: str, scenario:str, transfer_type: str, run_on_robot: bool, use_interface: bool, use_gui: bool, simulate_head_perception: bool, max_motion_planning_time: float,
                 resume_from_state: str = "", no_waits: bool = False) -> None:
        self.run_on_robot = run_on_robot
        self.use_interface = use_interface  
        self.simulate_head_perception = simulate_head_perception
        self.max_motion_planning_time = max_motion_planning_time
        self.no_waits = no_waits

        # logs are saved in user/scenario directory
        self.log_dir = Path(__file__).parent / "log" / user / scenario
        self.execution_log = Path(__file__).parent / "log" / "execution_log.txt" # in root log directory
        self.run_behavior_tree_dir = self.log_dir / "behavior_trees"
        self.gesture_detectors_dir = self.log_dir / "gesture_detectors"
        self._gesture_detection_filepath = self.gesture_detectors_dir / "synthesized_gesture_detectors.py"
        
        if not self.log_dir.exists():
            if not (self.log_dir.parent).exists(): # new user
                assert scenario == "default", "First run with a new user must be in default scenario."
                os.makedirs(self.log_dir, exist_ok=True)
                
                # Copy the initial behavior trees into a directory for this run, where
                # they will be modified based on user feedback.
                self.run_behavior_tree_dir.mkdir(exist_ok=True)
                original_behavior_tree_dir = Path(__file__).parents[1] / "actions" / "behavior_trees"
                assert original_behavior_tree_dir.exists()
                for original_bt_filename in original_behavior_tree_dir.glob("*.yaml"):
                    shutil.copy(original_bt_filename, self.run_behavior_tree_dir)

                # Copy the initial gesture detection file into a directory for this run,
                # where it will be updated from LLM-based few-shot learning.
                self.gesture_detectors_dir.mkdir(exist_ok=True)
                original_gesture_detection_filepath = Path(__file__).parents[1] / "perception" / "gestures_perception" / "synthesized_gesture_detectors.py"
                assert original_gesture_detection_filepath.exists()
                shutil.copy(original_gesture_detection_filepath, self.gesture_detectors_dir)

            elif not (self.log_dir).exists(): # new scenario
                assert (Path(__file__).parent / "log" / user / "default").exists(), "Do not have default scenario for this user."
                os.makedirs(self.log_dir, exist_ok=True)
                shutil.copytree(Path(__file__).parent / "log" / user / "default", self.log_dir, dirs_exist_ok=True)

        if resume_from_state == "":
            # clear behavior tree execution log
            with open(self.execution_log, "w") as f:
                f.write("")
            self._saved_state_infile = None
        else:
            self._saved_state_infile = Path(__file__).parent / "saved_states" / (resume_from_state + ".p")
        self._saved_state_outfile = Path(__file__).parent / "saved_states" / "last_state.p"

        # Initialize the interface to the robot.
        if run_on_robot:
            self.robot_interface = ArmInterfaceClient()  # type: ignore  # pylint: disable=no-member
            self.wrist_interface = WristInterface()
        else:
            self.robot_interface = None
            self.wrist_interface = None

        self.llm = OpenAILLM(
            model_name="gpt-4.5-preview-2025-02-27",
            cache_dir=self.log_dir / "llm_cache",
        )

        # Initialize the perceiver (e.g., get joint states or human head poses).
        self.perception_interface = PerceptionInterface(robot_interface=self.robot_interface, simulate_head_perception=self.simulate_head_perception, log_dir=self.log_dir)

        # Initialize the simulator.
        scene_config_path = Path(__file__).parent.parent / "simulation" / "configs" / f"{scene_config}.yaml"
        self.scene_description = create_scene_description_from_config(str(scene_config_path), transfer_type)

        if run_on_robot:
            if not np.allclose(self.scene_description.initial_joints, self.perception_interface.get_robot_joints(), atol=0.2):
                print("Initial joint state in scene description does not match the actual robot joint state.")
                print("Initial Robot Joints:", self.perception_interface.get_robot_joints())
                print("Initial Joints in Scene Description:", self.scene_description.initial_joints)
                
        else:
            print("Running in simulation mode.")

        self.flair = FLAIR(self.log_dir)

        if self.run_on_robot:
            self.rviz_interface = RVizInterface(self.scene_description)
        else:
            self.rviz_interface = None

        self.sim = FeedingDeploymentPyBulletSimulator(self.scene_description, use_gui=use_gui, ignore_user=True)

        if self.use_interface:
            # Initialize the web interface.
            self.task_selection_queue = queue.Queue()
            self.web_interface = WebInterface(self.task_selection_queue, self.log_dir)
        else:
            self.web_interface = None

        # Create skills for high-level planning.
        hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}
        print("Creating HLAs...")
        self.hlas = {
            cls(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.web_interface, hla_hyperparams,
                self.wrist_interface, self.flair, self.no_waits, self.log_dir, self.run_behavior_tree_dir, self.execution_log, self.gesture_detectors_dir,
                self.register_gesture_detector, self.load_synthesized_gestures) for cls in HLAS  # type: ignore
        }
        print("HLAs created.")
        self.hla_name_to_hla = {hla.get_name(): hla for hla in self.hlas}
        self.operators = {hla.get_operator() for hla in self.hlas}
        self.predicates: set[Predicate] = {
            ToolPrepared,
            GripperFree,
            Holding,
            ToolTransferDone,
            EmulateTransferDone,
            IsUtensil,
            PlateInView,
            ResetPos,
        }
        self.types = {tool_type}
        self.domain = PDDLDomain(
            "AssistedFeeding", self.operators, self.predicates, self.types
        )
        self.drink = Object("drink", tool_type)
        self.wipe = Object("wipe", tool_type)
        self.utensil = Object("utensil", tool_type)
        self.plate = Object("plate", tool_type)
        self.all_objects = {self.drink, self.wipe, self.utensil, self.plate}
        self.object_name_to_object = {
            "drink": self.drink,
            "wipe": self.wipe,
            "utensil": self.utensil,
            "plate": self.plate,
        }
        # Create all ground HLAs that will be used.
        self._all_ground_hlas = []
        for hla_name, hla in sorted(self.hla_name_to_hla.items()):
            types = [p.type for p in hla.get_operator().parameters]
            for obj_combo in get_object_combinations(sorted(self.all_objects), types):
                # Major hack. The proper way to do this would be to define subtypes
                # but I am too scared to make any change like that at this point.
                if "AcquireBite" in hla_name:
                    assert len(obj_combo) == 1
                    if obj_combo[0].name != "utensil":
                        continue
                ground_hla = (hla, obj_combo)
                self._all_ground_hlas.append(ground_hla)
        # Rewrite the behavior trees to avoid any inconsistencies.
        for hla, objs in self._all_ground_hlas:
            # Super Hack: skip the plate transfer behavior tree.
            if hla == self.hla_name_to_hla["TransferTool"] and objs[0].name == "plate":
                continue
            try:
                bt_filepath = hla.behavior_tree_dir / hla.get_behavior_tree_filename(objs, {})
            except NotImplementedError:
                continue
            bt = load_behavior_tree(bt_filepath, hla)
            save_behavior_tree(bt, bt_filepath, hla)

        # Track the current high-level state.
        self.current_atoms = {
            LiftedAtom(GripperFree, []),
            ToolPrepared([self.wipe]),
            ToolPrepared([self.drink]),
            ToolPrepared([self.plate]),
            IsUtensil([self.utensil]),
        }

        self.transparency_query = TransparencyQuery(self.log_dir)
        print("Initialized transparency query.")

        if self._saved_state_infile:
            self._load_from_state()
            print("WARNING: The system state has been restored to:")
            print(" ", sorted(self.current_atoms))
            resp = input("Are you sure you want to continue from here? [y/n] ")
            while resp not in ["y", "n"]:
                resp = input("Please enter 'y' or 'n': ")
                if resp == "n":
                    self.stop_all_threads()
                    sys.exit(0)

        print("Runner is ready.")
        self.active = True

    def run(self) -> None:

        assert self.web_interface is not None, "Run takes user commands from the web interface which is None."
        
        self.web_interface.ready_for_task_selection()
        last_task_type = None
        while self.active:
            try:
                task_selection_command = self.task_selection_queue.get(timeout=1)
                self.web_interface.clear_received_messages() # So that only the latest message is processed
                task, task_type = task_selection_command["task"], task_selection_command["type"]
                if task == "reset":
                    self.process_user_command(GroundHighLevelAction(self.hla_name_to_hla["Reset"], ()))
                    last_task_type = None
                elif task == "meal_assistance":
                    if task_type == "bite":
                        self.process_user_command(GroundHighLevelAction(self.hla_name_to_hla["TransferTool"], (self.utensil,)))
                    elif task_type == "sip":
                        self.process_user_command(GroundHighLevelAction(self.hla_name_to_hla["TransferTool"], (self.drink,)))
                    elif task_type == "wipe":
                        self.process_user_command(GroundHighLevelAction(self.hla_name_to_hla["TransferTool"], (self.wipe,)))
                    last_task_type = task_type
                elif task == "personalization":
                    if task_type == "transparency":
                        while self.active:
                            query = self.web_interface.get_transparency_request()
                            if query:
                                response = self.transparency_query.answer_query(query)
                                self.web_interface.update_transparency_response(response)
                            else:
                                break
                    elif task_type == "adaptability":
                        while self.active:
                            adaptation_request = self.web_interface.get_adaptability_request()
                            if adaptation_request:
                                try:
                                    print("Processing user update request:", adaptation_request)
                                    update_summary = self.process_user_update_request(adaptation_request)
                                    print('Processed user update request.')
                                    self.web_interface.update_adaptability_response(update_summary)
                                except Exception as e:
                                    print(f"Update failed: {str(e)}")
                                    self.web_interface.update_adaptability_response("Adaptation failed. Please try rephrasing the request.")
                            else:
                                break
                    elif task_type == "gesture":
                        print("Triggered gesture")
                        gesture_task_type = self.web_interface.get_gesture_type()
                        print(f"Gesture task type: {gesture_task_type}")
                        if gesture_task_type == "add":
                            gesture_label, gesture_description = self.web_interface.get_new_gesture_details()
                            self.process_user_command(GroundHighLevelAction(self.hla_name_to_hla["EmulateTransfer"], (), {"test_mode": False, "gesture_label":gesture_label, "gesture_description": gesture_description} ))
                        else: # test
                            self.process_user_command(GroundHighLevelAction(self.hla_name_to_hla["EmulateTransfer"], (), {"test_mode": True} ))
                    last_task_type = task_type
                else:
                    print(f"Invalid task selection: {task_selection_command}")
                    last_task_type = None
                # self.web_interface.clear_received_messages() # So that only the latest message is processed
                # time.sleep(1.0)
                self.web_interface.ready_for_task_selection(last_task_type=last_task_type)
                print("Ready for next user command.")
                print("Current web interface page:", self.web_interface.current_page)
            except queue.Empty:
                # Wait for user commands.
                time.sleep(0.1) 
                continue

    def stop_all_threads(self) -> None:
        self.active = False
        if self.web_interface is not None:
            self.web_interface.stop_all_threads()

    def signal_handler(self, signal, frame):
        print("\nReceived SIGINT.")
        self.stop_all_threads()
        print("\nprogram exiting gracefully")
        sys.exit(0)

    def process_user_command(
        self, user_command: GroundHighLevelAction | set[GroundAtom]
    ) -> None:
        """Process a user command."""

        print(f"Working towards user command: {user_command}")

        # Plan to the preconditions of the HLA.
        if isinstance(user_command, GroundHighLevelAction):
            goal_atoms = user_command.get_preconditions()
        else:
            goal_atoms = user_command
        problem = PDDLProblem(
            self.domain.name,
            "AssistedFeeding",
            self.all_objects,
            self.current_atoms,
            goal_atoms,
        )
        plan_strs = run_pddl_planner(
            str(self.domain), str(problem), planner="fd-opt",
        )
        assert plan_strs is not None
        plan_ops = parse_pddl_plan(plan_strs, self.domain, problem)
        print("Found plan to the preconditions of the command:")
        for i, op in enumerate(plan_ops):
            print(f"{i}. {op.short_str}")
        plan_hlas = pddl_plan_to_hla_plan(plan_ops, self.hlas)
        # Append the user command to the plan if it's an action.
        if isinstance(user_command, GroundHighLevelAction):
            plan_hlas.append(user_command)

        for ground_hla in plan_hlas:
            print(f"Refining {ground_hla}")
            operator = ground_hla.get_operator()

            # import ipdb; ipdb.set_trace()
            assert operator.preconditions.issubset(self.current_atoms)

            # Execute the high-level plan in simulation
            ground_hla.execute_action()

            sim_state = self.sim.get_current_state()

            self.current_atoms -= operator.delete_effects
            self.current_atoms |= operator.add_effects

            # Super hack: the drink, wipe and plate are always prepared.
            self.current_atoms.add(ToolPrepared([self.wipe]))
            self.current_atoms.add(ToolPrepared([self.drink]))
            self.current_atoms.add(ToolPrepared([self.plate]))

            # Save the latest state in case we want to resume execution
            # after a crash.
            self._save_state(sim_state, self.current_atoms)

    def make_video(self, outfile: Path) -> None:
        """Create a video of the simulated trajectory."""
        self.sim.make_simulation_video(outfile)
        print(f"Saved video to {outfile}")

    def _save_state(self, sim_state: FeedingDeploymentWorldState, atoms: set[GroundAtom]) -> None:
        with open(self._saved_state_outfile, "wb") as f:
            pickle.dump((sim_state, atoms), f)
        print(f"Saved system state to {self._saved_state_outfile}")

    def _load_from_state(self) -> None:
        with open(self._saved_state_infile, "rb") as f:
            sim_state, self.current_atoms = pickle.load(f)
        if sim_state is not None:
            assert isinstance(sim_state, FeedingDeploymentWorldState)
            self.sim.sync(sim_state)
            if self.rviz_interface is not None:
                self.rviz_interface.joint_state_update(sim_state.robot_joints)
                if sim_state.held_object:
                    self.rviz_interface.tool_update(True, sim_state.held_object, Pose((0, 0, 0), (0, 0, 0, 1)))
                
        print(f"Loaded system state from {self._saved_state_infile}")

    def process_user_update_request(self, request_text: str) -> str:
        """Validate and update behavior trees."""
        self.perception_interface.sync_rviz()
        available_hla_object_names = []
        for hla, obj_combo in self._all_ground_hlas:
            hla_name = hla.get_name()
            object_strs = [obj.name for obj in obj_combo]
            objects_str = ", ".join(object_strs)
            available_hla_object_name = f"hla_name={hla_name}, hla_object_names=({objects_str},)"
            available_hla_object_names.append(available_hla_object_name)
        requested_updates = interpret_user_update_request(request_text, self.llm, available_hla_object_names, self.run_behavior_tree_dir)
        if len(requested_updates) == 0:
            raise ValueError("No valid updates requested.")
        all_update_messages = []
        for update in requested_updates:
            assert isinstance(update, UserUpdateRequest)
            if update.hla_name not in self.hla_name_to_hla:
                print(f"BT UPDATE FAILED: Unknown HLA name {update.hla_name}")
                raise ValueError(f"BT UPDATE FAILED: Unknown HLA name {update.hla_name}")
            hla = self.hla_name_to_hla[update.hla_name]
            hla_object_list = []
            failed_object_name = None
            for obj_name in update.hla_object_names:
                if obj_name not in self.object_name_to_object:
                    failed_object_name = obj_name
                    break
                hla_object_list.append(self.object_name_to_object[obj_name])
            if failed_object_name is not None:
                print(f"BT UPDATE FAILED: Unknown object name {failed_object_name}")
                raise ValueError(f"BT UPDATE FAILED: Unknown object name {failed_object_name}")
            ground_hla = GroundHighLevelAction(hla, tuple(hla_object_list))            
            if isinstance(update, NodeModificationUserUpdateRequest):
                message = ground_hla.process_behavior_tree_parameter_update(update.node_name, update.parameter_name, update.new_value)
                print(message)
                all_update_messages.append((update, message))
            elif isinstance(update, NodeAdditionUserRequest):
                message = ground_hla.process_behavior_tree_node_addition(update.new_node_type, update.new_node_parameters,
                                                               update.anchor_node_name, update.before_or_after)
                print(message)
                all_update_messages.append((update, message))
            else:
                print("Not implemented")
                raise NotImplementedError
        # TODO query LLM to summarize all_update_messages
        all_update_str = ""
        for request, message in all_update_messages:
            all_update_str += f"\nRequest: {request}"
            all_update_str += f"\nResult: {message}"
        prompt = f"""A user requested the following change to a robot assisted feeding system:

"{request_text}"
                
Here is a log of changes that were requested to behavior trees and the results:

{all_update_str}

Write a VERY BRIEF summary of all the changes for a non-technical end user. Make sure not to use technical terms like "behavior tree".
"""
        summary = self.llm.sample_completions(prompt, imgs=None, temperature=0.0, seed=0)[0]
        print("SUMMARY:", summary)
        return summary

    def register_gesture_detector(self, gesture_fn_name: str, gesture_fn_text: str) -> bool:
        """Add the gesture function to this run's python file."""
        with open(self._gesture_detection_filepath, "r", encoding="utf-8") as f:
            gesture_file_text = f.read()
        assert f"def {gesture_fn_name}(" not in gesture_file_text
        gesture_file_text += "\n" + gesture_fn_text + "\n"
        with open(self._gesture_detection_filepath, "w", encoding="utf-8") as f:
            f.write(gesture_file_text)
        # Immediately add the new gesture to specific BT nodes.
        gesture_interaction_parameters = [
            "InitiateTransferInteraction",
            "TransferCompleteInteraction",
        ]
        for hla, objs in self._all_ground_hlas:
            try:
                bt_filepath = hla.behavior_tree_dir / hla.get_behavior_tree_filename(objs, {})
            except NotImplementedError:
                continue
            bt = load_behavior_tree(bt_filepath, hla)
            for node in bt.walk():
                if isinstance(node, ParameterizedActionBehaviorTreeNode):
                    for parameter_name in gesture_interaction_parameters:
                        parameter = node.get_parameter(parameter_name)
                        if parameter is None:
                            continue
                        assert parameter.is_user_editable
                        assert isinstance(parameter.space, EnumSpace)
                        current_choices = list(parameter.space.elements)
                        new_choices = current_choices + [gesture_fn_name]
                        new_parameter_space = EnumSpace(new_choices)
                        parameter.space = new_parameter_space
            save_behavior_tree(bt, bt_filepath, hla)
            
        print(f"Registered new gesture detection function: {gesture_fn_name}")    

    def load_synthesized_gestures(self) -> list[tuple[str, Callable]]:
        """Returns a list of function names and functions."""
        with open(self._gesture_detection_filepath, "r", encoding="utf-8") as f:
            gesture_file_text = f.read()
        synthesized_gesture_module = types.ModuleType('synthesized_gestures')
        exec(gesture_file_text, synthesized_gesture_module.__dict__)
        return inspect.getmembers(synthesized_gesture_module, inspect.isfunction)
    
    def get_plate_pose(self) -> Pose | None:
        skill = self.hla_name_to_hla["PickTool"]
        skill.move_to_joint_positions(self.sim.scene_description.retract_pos)
        skill.close_gripper()
        skill.move_to_joint_positions(self.sim.scene_description.plate_gaze_pos)
        self.perception_interface.perceive_plate_pickup_poses()
        skill.move_to_joint_positions(self.sim.scene_description.retract_pos)
        if self.perception_interface.last_plate_poses:
            plate_pose = self.perception_interface.last_plate_poses["plate_pose"]
            return plate_pose
        else:
            print("No plate pose detected.")
            return None
        
    def get_drink_pose(self) -> Pose | None:
        skill = self.hla_name_to_hla["PickTool"]
        skill.move_to_joint_positions(self.sim.scene_description.retract_pos)
        skill.close_gripper()
        skill.move_to_joint_positions(self.sim.scene_description.drink_gaze_pos)
        self.perception_interface.perceive_drink_pickup_poses()
        skill.move_to_joint_positions(self.sim.scene_description.retract_pos)
        if self.perception_interface.last_drink_poses:
            drink_pose = self.perception_interface.last_drink_poses["drink_pose"]
            return drink_pose
        else:
            print("No drink pose detected.")
            return None
    
    def get_multitask_personalization_state(self, user_request: str, occluded: bool = False,
                                            actively_detect_plate: bool = False,
                                            actively_detect_drink: bool = False) -> dict[str, Any]:
        """Get a sufficient state for multitask personalization."""
        mp_state = {"user_request": user_request}

        if actively_detect_plate:
            mp_state["plate_pose"] = self.get_plate_pose()

        if actively_detect_drink:
            mp_state["drink_pose"] = self.get_drink_pose()

        mp_state["robot_joints"] = self.perception_interface.get_robot_joints()

        if occluded:
            mp_state["occluded"] = True

        self.perception_interface.sync_rviz()

        return mp_state
    
    def update_scene_spec(self, scene_spec_updates: dict[str, Any]) -> None:
        """Update the scene spec with the given updates."""
        for key, value in scene_spec_updates.items():
            if hasattr(self.scene_description, key):
                setattr(self.scene_description, key, value)
            else:
                raise ValueError(f"Invalid scene spec update: {key}")
        print("Updated scene spec:", scene_spec_updates)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_config", type=str, default="vention") # name of the scene config (rough head-plate-robot setup)
    parser.add_argument("--user", type=str, default="") # name of the user
    parser.add_argument("--scenario", type=str, default="default") # name of the scenario
    # parser.add_argument("--transfer_type", type=str, default="inside")
    parser.add_argument("--transfer_type", type=str, default="outside")
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--use_interface", action="store_true")
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--simulate_head_perception", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    parser.add_argument("--resume_from_state", type=str, default="")
    parser.add_argument("--no_waits", action="store_true")
    parser.add_argument("--cbtl", action="store_true")
    parser.add_argument("--meal_id", type=int, default=1)
    parser.add_argument("--results_dir", type=Path, default=Path("feast_default_user"), help="Directory for saving and loading results and user responses. Make one of these directories per user.")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    if args.user == "":
        raise ValueError("Please provide a user name.")

    if args.run_on_robot or args.use_interface:
        if not ROSPY_IMPORTED:
            raise ModuleNotFoundError("Need ROS to run on robot or use interface")
        else:
            rospy.init_node("feeding_deployment", anonymous=True)

    runner = _Runner(args.scene_config,
                     args.user,
                     args.scenario,
                     args.transfer_type,
                     args.run_on_robot, 
                     args.use_interface,
                     args.use_gui,
                     args.simulate_head_perception,
                     args.max_motion_planning_time,
                     args.resume_from_state,
                     args.no_waits)
    
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, runner.signal_handler)

    # Uncomment to test commands.
    # drink_pickup_msg = {"status": "drink_pickup"}
    # runner.hla_command_queue.put(drink_pickup_msg)

    # drink_transfer_msg = {"status": "drink_transfer"}
    # runner.hla_command_queue.put(drink_transfer_msg)

    if args.cbtl:
        # sends encoded messages to multitask personalization on ROS topic /mp_request, and expects to receive a response on /mp_response

        field_to_choice = {}
        results = {}  # field name -> {"options": ..., "prediction": ..., "choice": ...}
        results_dir: Path = args.results_dir
        results_dir.mkdir(exist_ok=True)
        field_to_choice_file = results_dir / f"field_to_choice_meal{args.meal_id}.json"
        results_file = results_dir / f"results_meal{args.meal_id}.json"
        if args.load and field_to_choice_file.exists():
            with open(field_to_choice_file, "r") as f:
                field_to_choice = json.load(f)
        user_description_file = results_dir / "user_description.txt"
        if not user_description_file.exists():
            user_description = input("Write any kind of description for this user that will be helpful for us to refer back to later: ")
            with open(user_description_file, "w") as f:
                f.write(user_description)

        # Helper function to send a request to the multitask personalization module.
        def _send_mp_request(data):
            # Encode the message.
            global _mp_response
            s = pickle.dumps(data)
            s = base64.b64encode(s).decode('ascii')
            msg = String()
            msg.data = s
            _mp_response = None
            mp_request_pub.publish(msg)
            print("Sent MP request: ", data)  
            while _mp_response is None:
                time.sleep(0.1)  # Wait for the response
            print("Received MP response: ", _mp_response)
            return _mp_response

        # Response callback function
        def mp_response_callback(msg):
            # Decode the message.
            s = base64.b64decode(msg.data.encode('ascii'))
            data = pickle.loads(s)
            global _mp_response
            _mp_response = data

        mp_response_sub = rospy.Subscriber("/mp_response", String, mp_response_callback)
        mp_request_pub = rospy.Publisher("/mp_request", String, queue_size=10)
        time.sleep(1)  # Wait for the subscriber and publisher to be ready
        
        # Helper function to verify predictions with the user.
        def verify_predictions(field_name, prediction, options):
            print("Field name:", field_name)
            if prediction not in options:
                raise ValueError(f"Invalid prediction: {prediction}. Expected one of {options}.")
            if args.load and field_name in field_to_choice:
                choice = field_to_choice[field_name]
                print(f"Loaded choice {choice} for {field_name}")
                results[field_name] = {"options": options, "prediction": prediction, "choice": choice}
                with open(results_file, "w") as f:
                    json.dump(results, f)
                return choice
            print("From the following options:")
            for i in range(len(options)):
                print(f"{i+1}. {options[i]}")
            print("The robot predicted the following preference: ", prediction)
            user_input = input("Do you agree with the robot's prediction? (y/n): ")
            while user_input not in ["y", "n"]:
                user_input = input("Please enter 'y' or 'n': ")
            if user_input == "y":
                print("User agreed with the robot's prediction.")
                choice = prediction
            else:
                # get user's preference
                preferred_id = input("Please enter the number of your preferred option: ")
                while not preferred_id.isdigit() or int(preferred_id) < 1 or int(preferred_id) > len(options):
                    preferred_id = input("Please enter a valid number: ")
                preferred_id = int(preferred_id) - 1
                print(f"User preferred option: {options[preferred_id]}")
                choice = options[preferred_id]
            field_to_choice[field_name] = choice
            with open(field_to_choice_file, "w") as f:
                json.dump(field_to_choice, f)
            results[field_name] = {"options": options, "prediction": prediction, "choice": choice}
            with open(results_file, "w") as f:
                json.dump(results, f)
            return choice
            
        # Helper function to generate all possible bite orderings.
        def generate_bite_orderings(food_items: List[str], dips: List[str]) -> List[str]:
            orderings = []

            # All permutations of food items with all dipping combinations
            for food_perm in itertools.permutations(food_items):
                food_dip_variants = []
                for food in food_perm:
                    variants = [f"{food} without any dipping"]
                    variants += [f"{food} dipped in {dip}" for dip in dips]
                    food_dip_variants.append(variants)

                for combo in itertools.product(*food_dip_variants):
                    orderings.append(" then ".join(combo))

            if len(food_items) > 1:
                # One alternating pattern across all food items
                alt_variants = []
                for food in food_items:
                    if dips:
                        alt_variants.append(f"{food} dipped in {dips[0]}")
                    else:
                        alt_variants.append(food)
                if len(alt_variants) == 1:
                    alt_pattern = f"alternating bites of {alt_variants[0]}"
                else:
                    alt_pattern = "alternating bites of " + " and ".join(alt_variants)
                orderings.append(alt_pattern)

            return orderings
        

        # Make sure web interface is running:
        assert runner.web_interface is not None, "Run takes user commands from the web interface which is None."
        runner.web_interface.ready_for_task_selection()
        time.sleep(2)
        
        @dataclass
        class Meal:
            meal_id: int
            context: str
            table_type: str
            food_items: List[str]
            dips: List[str]

        MEALS = [
            Meal(1, "personal", "rectangular table", ["french fries"], ["ketchup", "BBQ sauce"]),
            Meal(2, "social with friend on left", "circular table", ["raw vegetables"], ["ranch dressing", "hummus"]),
            Meal(3, "watching TV in front", "circular table", ["potato wedges"], ["ketchup", "BBQ sauce"]),
            Meal(4, "personal", "circular table", ["carrot sticks"], ["ranch dressing", "hummus"]),
            Meal(5, "social TV-watching (with TV in front) and with friend on left side", "rectangular table", ["tater tot"], ["ketchup", "BBQ sauce"])
        ]

        current_meal = MEALS[args.meal_id-1]
        assert current_meal.meal_id == args.meal_id
        bite_ordering_options = generate_bite_orderings(current_meal.food_items, current_meal.dips)

        # send mealContext, table_type, food_items and bite_ordering_options to multitask personalization
        mp_response = _send_mp_request({"request_type": "initialization_query",
                    "meal_id": current_meal.meal_id,
                    "context": current_meal.context, 
                    "table_type": current_meal.table_type,
                    "food_items": current_meal.food_items,
                    "dips": current_meal.dips,
                    "bite_ordering_options": bite_ordering_options})
        assert mp_response["response_type"] == "initialization_query"
        feeding_side = mp_response["feeding_side"]
        bite_ordering = mp_response["bite_ordering"]
        ready_signal = mp_response["ready_signal"]
        be_verbal = mp_response["be_verbal"]

        # verify predictions with the user (using the terminal)
        feeding_side = verify_predictions("feeding_side", feeding_side, ["left", "right"])
        bite_ordering = verify_predictions("bite_ordering", bite_ordering, bite_ordering_options)
        ready_signal = verify_predictions("ready_signal", ready_signal, ["mouth_open", "button", "auto_continue"])
        be_verbal = verify_predictions("be_verbal", be_verbal, [True, False])

        # send the verified predictions to multitask personalization
        mp_response = _send_mp_request({"request_type": "initialization_dataset",
                          "feeding_side": feeding_side, 
                          "bite_ordering": bite_ordering, 
                          "ready_signal": ready_signal,
                          "be_verbal": be_verbal})
        assert mp_response["response_type"] == "initialization_dataset"


        # initialize FEAST with the predicted parameters

        # ready signal
        if ready_signal == "auto_continue":
            ready_signal = "auto_timeout"
        runner.process_user_update_request(f"For all transfer actions, set the initiate transfer interaction to {ready_signal}.")
        # be verbal
        if not be_verbal:
            runner.process_user_update_request("For all actions, be quiet.")
        # set params for bite acquisition
        runner.flair.inference_server.FOOD_CLASSES = []
        runner.flair.inference_server.FOOD_CATEGORIES = []
        for food_item in current_meal.food_items:
            runner.flair.inference_server.FOOD_CLASSES.append(food_item)
            runner.flair.inference_server.FOOD_CATEGORIES.append("solid")
        for dip in current_meal.dips:
            runner.flair.inference_server.FOOD_CLASSES.append(dip)
            runner.flair.inference_server.FOOD_CATEGORIES.append("dip")
        runner.flair.user_preference = bite_ordering

        # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["PickTool"], (runner.utensil,)))
        # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,)))
        # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.utensil,)))

        pick_tool = runner.hla_name_to_hla["PickTool"]
        stow_tool = runner.hla_name_to_hla["StowTool"]

        # # SEQUENCE 1: Feeding with only plate

        plate_pose = runner.get_plate_pose()
        while plate_pose is None:
            print("No plate pose detected.")
            input("Please adjust the plate and press Enter to continue...")
            plate_pose = runner.get_plate_pose()
        plate_pose = Pose((plate_pose.position[0], plate_pose.position[1], 0.17))

        mp_response = _send_mp_request({"request_type": "occlusion_query",
                              "plate_pose": plate_pose, 
                              "drink_pose": None})
        assert mp_response["response_type"] == "occlusion_query"
        disoriented_plate_delta_xy = mp_response["plate_delta_xy"]
        plate_delta_xy = (-1 * disoriented_plate_delta_xy[1], disoriented_plate_delta_xy[0])
        before_transfer_pose = mp_response["before_transfer_pose"]
        before_transfer_pos = mp_response["before_transfer_pos"]
        above_plate_pos = mp_response["above_plate_pos"]
        occlusion_poi_relevance = mp_response["occlusion_poi_relevance"]
        bite_occlusion_image = mp_response["bite_occlusion_image"]

        Image.fromarray(bite_occlusion_image).show()
        print("Visualizing bite occlusion image.")
        input("Press Enter to continue...")

        runner.update_scene_spec({"plate_delta_xy": plate_delta_xy})
        runner.update_scene_spec({"before_transfer_pose": before_transfer_pose})
        runner.update_scene_spec({"before_transfer_pos": before_transfer_pos})
        runner.update_scene_spec({"above_plate_pos": above_plate_pos})

        if not np.allclose(plate_delta_xy, [0, 0], atol=1e-3):
            # pick and stow the plate
            input("Press Enter to move the plate...")
            runner.process_user_command(GroundHighLevelAction(pick_tool, (runner.plate,)))
            runner.process_user_command(GroundHighLevelAction(stow_tool, (runner.plate,)))

        runner.process_user_command(GroundHighLevelAction(pick_tool, (runner.utensil,)))
        # pick_tool.move_to_joint_positions(runner.sim.scene_description.above_plate_pos)
        # pick_tool.move_to_joint_positions(runner.sim.scene_description.before_transfer_pos)
        # pick_tool.move_to_joint_positions(runner.sim.scene_description.absolute_before_transfer_pos)
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,)))
        runner.process_user_command(GroundHighLevelAction(stow_tool, (runner.utensil,)))

        # # SEQUENCE 2: Feeding with plate and drink

        plate_pose = runner.get_plate_pose()
        drink_pose = runner.get_drink_pose()
        
        while plate_pose is None or drink_pose is None:
            print("No plate or drink pose detected.")
            input("Please adjust the plate and drink and press Enter to continue...")
            plate_pose = runner.get_plate_pose()
            drink_pose = runner.get_drink_pose()

        # adjust poses because table height is different in simulation
        plate_pose = Pose((plate_pose.position[0], plate_pose.position[1], 0.17)) 
        drink_pose = Pose((drink_pose.position[0], drink_pose.position[1], 0.35), (0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0)) 

        print("Plate pose:", plate_pose)
        print("Drink pose:", drink_pose)
        input("Press Enter to continue...")

        # send plate_pose and drink_pose to multitask personalization
        mp_response = _send_mp_request({"request_type": "occlusion_query",
                            "plate_pose": plate_pose, 
                            "drink_pose": drink_pose})
        assert mp_response["response_type"] == "occlusion_query"
        disoriented_plate_delta_xy = mp_response["plate_delta_xy"]
        disoriented_drink_delta_xy = mp_response["drink_delta_xy"]
        plate_delta_xy = (-1 * disoriented_plate_delta_xy[1], disoriented_plate_delta_xy[0])
        drink_delta_xy = (-1 * disoriented_drink_delta_xy[1], disoriented_drink_delta_xy[0])
        before_transfer_pose = mp_response["before_transfer_pose"]
        before_transfer_pos = mp_response["before_transfer_pos"]
        above_plate_pos = mp_response["above_plate_pos"]
        occlusion_poi_relevance = mp_response["occlusion_poi_relevance"]
        bite_occlusion_image = mp_response["bite_occlusion_image"]
        drink_occlusion_image = mp_response["drink_occlusion_image"]

        Image.fromarray(bite_occlusion_image).show()
        print("Visualizing bite occlusion image.")
        input("Press Enter to continue...")

        Image.fromarray(drink_occlusion_image).show()
        print("Visualizing drink occlusion image.")
        input("Press Enter to continue...")

        runner.update_scene_spec({"plate_delta_xy": plate_delta_xy})
        runner.update_scene_spec({"before_transfer_pose": before_transfer_pose})
        runner.update_scene_spec({"before_transfer_pos": before_transfer_pos})
        runner.update_scene_spec({"above_plate_pos": above_plate_pos})
        runner.update_scene_spec({"drink_delta_xy": drink_delta_xy})

        if not np.allclose(plate_delta_xy, [0, 0], atol=1e-3):
            # pick and stow the plate
            input("Press Enter to move the plate...")
            runner.process_user_command(GroundHighLevelAction(pick_tool, (runner.plate,)))
            runner.process_user_command(GroundHighLevelAction(stow_tool, (runner.plate,)))

        if not np.allclose(drink_delta_xy, [0, 0], atol=1e-3):
            # pick and stow the drink
            input("Press Enter to move the drink...")
            runner.process_user_command(GroundHighLevelAction(pick_tool, (runner.drink,)))
            runner.process_user_command(GroundHighLevelAction(stow_tool, (runner.drink,)))

        runner.process_user_command(GroundHighLevelAction(pick_tool, (runner.drink,)))
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.drink,)))
        runner.process_user_command(GroundHighLevelAction(stow_tool, (runner.drink,)))

        runner.process_user_command(GroundHighLevelAction(pick_tool, (runner.utensil,)))
        # pick_tool.move_to_joint_positions(runner.sim.scene_description.above_plate_pos)
        # pick_tool.move_to_joint_positions(runner.sim.scene_description.before_transfer_pos)
        # pick_tool.move_to_joint_positions(runner.sim.scene_description.absolute_before_transfer_pos)
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,)))
        runner.process_user_command(GroundHighLevelAction(stow_tool, (runner.utensil,)))

        occlusion_dataset_dict = {
            "request_type": "occlusion_dataset",
            "plate_pose": plate_pose,
            "drink_pose": drink_pose,
            "occlusion": {}
        }
        for poi, prediction in occlusion_poi_relevance.items():
            print(f"Verifying the RELEVANCE of POI={poi} for this meal")
            relevance = verify_predictions(prediction, [True, False])
            if relevance:
                print(f"Verifying whether view was occluded for POI={poi} during FEEDING")
                plate_occlusion = verify_predictions(False, [True, False])
                print(f"Verifying whether view was occluded for POI={poi} during DRINKING")
                drink_occlusion = verify_predictions(False, [True, False])
            else:
                plate_occlusion = False
                drink_occlusion = False
            occlusion_dataset_dict["occlusion"][poi] = {
                "relevance": relevance,
                "plate_occlusion": plate_occlusion,
                "drink_occlusion": drink_occlusion,
            }

            if plate_occlusion or drink_occlusion:
                occlusion = True
        mp_response = _send_mp_request(occlusion_dataset_dict)
        assert mp_response["response_type"] == "occlusion_dataset" 
        
        # start meal
        if not args.use_interface:
            # some testing stuff
            raise NotImplementedError
        else:
            runner.run()

    else:
        if not args.use_interface:

            for i in range(3):
                input("Press Enter to pick up the drink...")
                runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["PickTool"], (runner.drink,)))
                runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.drink,)))

            # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.drink,)))
            # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.drink,)))
            # for i in range(3):
                # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["PickTool"], (runner.drink,)))
                # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.drink,)))
        else:
            runner.web_interface.ready_for_task_selection()
            time.sleep(2)

            @dataclass
            class Meal:
                meal_id: int
                context: str
                table_type: str
                food_items: List[str]
                dips: List[str]

            current_meal = Meal(5, "social TV-watching (with TV in front) and with friend on left side", "rectangular table", ["tater tot"], ["ketchup", "BBQ sauce"])

            runner.flair.inference_server.FOOD_CLASSES = []
            runner.flair.inference_server.FOOD_CATEGORIES = []
            for food_item in current_meal.food_items:
                runner.flair.inference_server.FOOD_CLASSES.append(food_item)
                runner.flair.inference_server.FOOD_CATEGORIES.append("solid")
            for dip in current_meal.dips:
                runner.flair.inference_server.FOOD_CLASSES.append(dip)
                runner.flair.inference_server.FOOD_CATEGORIES.append("dip")
            runner.flair.user_preference = "tater tot dipped in ketchup"


            runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,)))
            runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.utensil,)))
            # runner.run()

    if args.make_videos:
        output_path = Path(__file__).parent / "videos" / "full.mp4"
        runner.make_video(output_path)

    if args.run_on_robot:
        rospy.spin()
