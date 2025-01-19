"""The main entry point for running the integrated system."""

import json
from collections import namedtuple
from pathlib import Path
from typing import Any
import pickle
import queue
import os
import sys
import signal
import shutil
import numpy as np
from tomsutils.llm import OpenAILLM
import time

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

    def __init__(self, scene_config: str, transfer_type: str, run_on_robot: bool, use_interface: bool, use_gui: bool, simulate_head_perception: bool, max_motion_planning_time: float,
                 resume_from_state: str = "", no_waits: bool = False) -> None:
        self.run_on_robot = run_on_robot
        self.use_interface = use_interface  
        self.simulate_head_perception = simulate_head_perception
        self.max_motion_planning_time = max_motion_planning_time
        self.no_waits = no_waits

        if resume_from_state == "":
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

        self.log_dir = Path(__file__).parent / "log"
        self.log_dir.mkdir(exist_ok=True)

        self.llm = OpenAILLM(
            model_name="gpt-4o",
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

        self.flair = FLAIR()

        if self.run_on_robot:
            self.rviz_interface = RVizInterface(self.scene_description)
        else:
            self.rviz_interface = None

        self.sim = FeedingDeploymentPyBulletSimulator(self.scene_description, use_gui=use_gui, ignore_user=True)

        # Copy the initial behavior trees into a directory for this run, where
        # they will be modified based on user feedback.
        self.run_behavior_tree_dir = self.log_dir / "behavior_trees"
        self.run_behavior_tree_dir.mkdir(exist_ok=True)
        original_behavior_tree_dir = Path(__file__).parents[1] / "actions" / "behavior_trees"
        assert original_behavior_tree_dir.exists()
        for original_bt_filename in original_behavior_tree_dir.glob("*.yaml"):
            shutil.copy(original_bt_filename, self.run_behavior_tree_dir)

        # Copy the initial gesture detction file into a directory for this run,
        # where it will be updated from LLM-based few-shot learning.
        original_gesture_detection_filepath = Path(__file__).parents[1] / "perception" / "gestures_perception" / "synthesized_gesture_detectors.py"
        assert original_gesture_detection_filepath.exists()
        shutil.copy(original_gesture_detection_filepath, self.run_behavior_tree_dir)
        self._gesture_detection_filepath = self.run_behavior_tree_dir / original_gesture_detection_filepath.name

        if self.use_interface:
            # Initialize the web interface.
            self.task_selection_queue = queue.Queue()
            self.web_interface = WebInterface(self.task_selection_queue)
        else:
            self.web_interface = None

        # Create skills for high-level planning.
        hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}
        print("Creating HLAs...")
        self.hlas = {
            cls(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.web_interface, hla_hyperparams,
                self.wrist_interface, self.flair, self.run_behavior_tree_dir, self.no_waits, self.log_dir) for cls in HLAS  # type: ignore
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
        self.all_objects = {self.drink, self.wipe, self.utensil}
        self.object_name_to_object = {
            "drink": self.drink,
            "wipe": self.wipe,
            "utensil": self.utensil,
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
            IsUtensil([self.utensil]),
        }

        self.transparency_query = TransparencyQuery()
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
                if task == "meal_assistance":
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
                                    print(f"Adaptation failed. Please try rephrasing the request.")
                                    self.web_interface.update_adaptability_response(f"Update failed: {str(e)}")
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
                self.web_interface.ready_for_task_selection(last_task_type=last_task_type)
                print("Ready for next user command.")
            except queue.Empty:
                # Wait for user command
                print("Current web interface page:", self.web_interface.current_page)
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

            # Super hack: the drink and wipe are always prepared.
            self.current_atoms.add(ToolPrepared([self.wipe]))
            self.current_atoms.add(ToolPrepared([self.drink]))

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
        print(f"Registered new gesture detection function: {gesture_fn_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_config", type=str, default="vention")
    parser.add_argument("--transfer_type", type=str, default="outside")
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--use_interface", action="store_true")
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--simulate_head_perception", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    parser.add_argument("--resume_from_state", type=str, default="")
    parser.add_argument("--no_waits", action="store_true")
    args = parser.parse_args()

    if args.run_on_robot or args.use_interface:
        if not ROSPY_IMPORTED:
            raise ModuleNotFoundError("Need ROS to run on robot or use interface")
        else:
            rospy.init_node("feeding_deployment", anonymous=True)

    runner = _Runner(args.scene_config,
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

    if not args.use_interface:

        ## Variations on modifying the speed of the robot.

        # All fast.
        # runner.process_user_update_request("Set the speed of the robot to high.") 
        # runner.process_user_update_request("Make the robot move fast.") 
        # runner.process_user_update_request("Can the robot move faster.") 
        # runner.process_user_update_request("The robot is too slow right now.") 
        # runner.process_user_update_request("Go faster.") 

        # All slow.
        # runner.process_user_update_request("Set the speed of the robot to low.") 
        # runner.process_user_update_request("Make the robot move slow.") 
        # runner.process_user_update_request("Can the robot move slower.") 
        # runner.process_user_update_request("The robot is too fast right now.") 
        # runner.process_user_update_request("Go slower.")

        # All medium.
        # runner.process_user_update_request("Can the robot go not too fast but also not too slow?") 

        # Selective speeds.
        # runner.process_user_update_request("When the robot is coming close to my mouth can it go more slowly") 
        # runner.process_user_update_request("When the robot is bringing food into my mouth can it not go so fast")  # currently updates all transfers, but that's okay
        # runner.process_user_update_request("I only wanted to update the speed for the food, not for the drink or the wipe. Can you set the drink and wipe back to normal") 
        # runner.process_user_update_request("When the robot is stabbing the food it is really slow right now") 

        # Autocontinue times for transfer.
        # runner.process_user_update_request("Stop waiting so long in between things") 
        # runner.process_user_update_request("Can you wait just a little longer to let me decide if I want to continue") 
        # runner.process_user_update_request("I need some more time to think")
        # runner.process_user_update_request("I need some more time to think after taking a bite")  # updates more times than it should
        # runner.process_user_update_request("I don't need to wait so long after drinking")  # updates more times than it should

        # Outside mouth distance.
        # runner.process_user_update_request("Set the outside mouth distance for transfer to 12 cms.")
        # runner.process_user_update_request("Can you come closer to my mouth with the food?")
        # runner.process_user_update_request("Please stay farther away from me")
        # runner.process_user_update_request("Set the outside mouth distance for transfer to 0 cm.")  # should fail
        # runner.process_user_update_request("Come just a tiny bit closer to my mouth.")

        # NOTE: this is not working perfectly -- it updates the transfer distance for all 3 transfer skills.
        # runner.process_user_update_request("Stay just a little farther away from my mouth when you are bringing the drink.")

        # Removing the "Confirm Ready for Transfer" page on the web app.
        # runner.process_user_update_request("Remove all transfer confirmations from the web app.")
        # runner.process_user_update_request("Remove all transfer confirmations.")

        # NOTE: this is not working perfectly -- it updates "silent for ReadyForTransferInteraction" instead of the web app confirmations.
        runner.process_user_update_request("On the iPad, don't ask me to confirm when I'm ready.")


        input("Press Enter to continue...")

        # Run some commands.
        # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,)))
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.drink,)))
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.wipe,)))
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.drink,)))
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.wipe,)))
        # for i in range(5):
        #     runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["PickTool"], (runner.drink,)))
        #     runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.drink,)))
        # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["PickTool"], (runner.drink,)))
    else:
        runner.run()

    if args.make_videos:
        output_path = Path(__file__).parent / "videos" / "full.mp4"
        runner.make_video(output_path)

    if args.run_on_robot:
        rospy.spin()
