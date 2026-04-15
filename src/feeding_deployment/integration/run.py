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
    object_type,
    tool_type,
    plate_type,
    plate_location_type,
    nav_target_type,
    table_type,
    sink_type,
    appliance_type,
    fridge_type,
    microwave_type,
    holder_type,
    GripperFree,
    Holding,
    IsUtensil,
    PlateInView,
    ToolPrepared,
    ToolTransferDone,
    EmulateTransferDone,
    ResetPos,
    InFrontOf,
    DoorOpen,
    DoorClosed,
    PlateAt,
    FoodHeated,
    SafeToNavigate,
    GroundHighLevelAction,
    ResetHLA,
    pddl_plan_to_hla_plan,
    interpret_user_update_request,
    load_behavior_tree,
    save_behavior_tree,
    NodeModificationUserUpdateRequest,
    NodeAdditionUserRequest,
    UserUpdateRequest,
    ParameterizedActionBehaviorTreeNode,
)
from feeding_deployment.actions.navigate import NavigateHLA
from feeding_deployment.actions.open_door import OpenDoorHLA
from feeding_deployment.actions.close_door import CloseDoorHLA
from feeding_deployment.actions.pick_plate import (
    PickPlateFromApplianceHLA,
    PickPlateFromHolderHLA,
    PickPlateFromTableHLA,
)
from feeding_deployment.actions.place_plate import (
    PlacePlateOnHolderHLA,
    PlacePlateInApplianceHLA,
    PlacePlateOnTableHLA,
    PlacePlateInSinkHLA,
)
from feeding_deployment.actions.press_microwave_button import PressMicrowaveButtonHLA
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
from feeding_deployment.integration.preference_context import build_preference_context
from feeding_deployment.preference_learning.methods.prediction_model import PredictionModel, PREF_OPTIONS
from feeding_deployment.preference_learning.config import MEALS, SETTINGS, TIMES_OF_DAY
from feeding_deployment.preference_learning.config.physical_capabilities import (
    PHYSICAL_CAPABILITY_PROFILES,
)

# All the high level actions we want to consider.
HLAS = {
    NavigateHLA,
    OpenDoorHLA,
    CloseDoorHLA,
    PickToolHLA,
    StowToolHLA,
    PickPlateFromApplianceHLA,
    PickPlateFromHolderHLA,
    PickPlateFromTableHLA,
    PlacePlateOnHolderHLA,
    PlacePlateInApplianceHLA,
    PlacePlateOnTableHLA,
    PlacePlateInSinkHLA,
    PressMicrowaveButtonHLA,
    AcquireBiteHLA,
    TransferToolHLA,
    EmulateTransferHLA,
    ResetHLA,
}

assert os.environ.get("PYTHONHASHSEED") == "0", \
        "Please add `export PYTHONHASHSEED=0` to your bash profile!"

class _Runner:
    """A class for running the integrated system."""

    def __init__(self, scene_config: str, user: str, scenario:str, transfer_type: str, run_on_robot: bool, use_interface: bool, use_gui: bool, simulate_head_perception: bool, max_motion_planning_time: float,
                 resume_from_state: str = "", no_waits: bool = False,
                 physical_profile_label: str | None = None,
                 pref_day: int | None = None,
                 pref_mode: str = "none") -> None:
        self.run_on_robot = run_on_robot
        self.use_interface = use_interface  
        self.simulate_head_perception = simulate_head_perception
        self.max_motion_planning_time = max_motion_planning_time
        self.no_waits = no_waits
        self.deployment_user = user
        self.physical_profile_label = physical_profile_label.strip() if physical_profile_label else None
        self._pref_day = pref_day
        self._pref_mode = pref_mode
        self._prediction_model: PredictionModel | None = None
        self.predicted_bundle: dict[str, str] | None = None

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
            model_name="gpt-4.1-2025-04-14",
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
            InFrontOf,
            DoorOpen,
            DoorClosed,
            PlateAt,
            FoodHeated,
            SafeToNavigate,
        }
        self.types = {
            object_type,
            plate_type,
            tool_type,
            plate_location_type,
            nav_target_type,
            sink_type,
            table_type,
            appliance_type,
            fridge_type,
            microwave_type,
            holder_type,
        }
        self.domain = PDDLDomain(
            "AssistedFeeding", self.operators, self.predicates, self.types
        )

        self.drink = Object("drink", tool_type)
        self.wipe = Object("wipe", tool_type)
        self.utensil = Object("utensil", tool_type)
        self.plate = Object("plate", plate_type)

        self.fridge = Object("fridge", fridge_type)
        self.microwave = Object("microwave", microwave_type)

        self.holder = Object("holder", holder_type)

        self.sink = Object("sink", sink_type)
        self.table = Object("table", table_type)

        self.all_objects = {
            self.drink,
            self.wipe,
            self.utensil,
            self.plate,
            self.fridge,
            self.microwave,
            self.sink,
            self.table,
            self.holder,
        }
        self.object_name_to_object = {
            "drink": self.drink,
            "wipe": self.wipe,
            "utensil": self.utensil,
            "plate": self.plate,
            "fridge": self.fridge,
            "microwave": self.microwave,
            "sink": self.sink,
            "table": self.table,
            "holder": self.holder,
        }
        # Create all ground HLAs that will be used.
        self._all_ground_hlas = []
        for hla_name, hla in sorted(self.hla_name_to_hla.items()):
            types = [p.type for p in hla.get_operator().parameters]
            for obj_combo in get_object_combinations(sorted(self.all_objects), types):
                # Major hack. The proper way to do this would be to define subtypes
                if "AcquireBite" in hla_name:
                    assert len(obj_combo) == 2
                    if obj_combo[0].name != "utensil" or obj_combo[1].name != "table":
                        # print(f"Skipping invalid HLA grounding: {hla_name} with object {obj_combo[0].name}")
                        continue
                if hla_name == "Navigate":
                    assert len(obj_combo) == 2
                    if obj_combo[0] == obj_combo[1]:
                        # print(f"Skipping invalid HLA grounding: {hla_name} with identical nav targets {obj_combo[0].name}")
                        continue
                if hla_name == "PressMicrowaveButton":
                    assert len(obj_combo) == 1
                    if obj_combo[0].name != "microwave":
                        # print(f"Skipping invalid HLA grounding: {hla_name} with object {obj_combo[0].name}")
                        continue
                if hla_name == "PickPlateFromTable" or hla_name == "PlacePlateOnTable":
                    assert len(obj_combo) == 2
                    if obj_combo[0].name != "plate" or obj_combo[1].name != "table":
                        # print(f"Skipping invalid HLA grounding: {hla_name} with objects {obj_combo[0].name}, {obj_combo[1].name}")
                        continue
                if hla_name == "PickPlateFromHolder" or hla_name == "PlacePlateOnHolder":
                    assert len(obj_combo) == 2
                    if obj_combo[0].name != "plate" or obj_combo[1].name != "holder":
                        # print(f"Skipping invalid HLA grounding: {hla_name} with objects {obj_combo[0].name}, {obj_combo[1].name}")
                        continue
                if hla_name == "PlacePlateInSinkHLA":
                    assert len(obj_combo) == 2
                    if obj_combo[0].name != "plate" or obj_combo[1].name != "sink":
                        # print(f"Skipping invalid HLA grounding: {hla_name} with objects {obj_combo[0].name}, {obj_combo[1].name}")
                        continue
                if hla_name in {"PlacePlateInAppliance", "PickPlateFromAppliance"}:
                    assert len(obj_combo) == 2
                    if obj_combo[0].name != "plate" or obj_combo[1].name not in {"fridge", "microwave"}:
                        # print(f"Skipping invalid HLA grounding: {hla_name} with objects {obj_combo[0].name}, {obj_combo[1].name}")
                        continue
                if hla_name in {"PickTool", "StowTool", "TransferTool"}:
                    assert len(obj_combo) == 2
                    if obj_combo[0].name == "plate" or obj_combo[1].name != "table":
                        # print(f"Skipping invalid HLA grounding: {hla_name} with object {obj_combo[0].name}")
                        continue
                # print(f"Adding ground HLA: {hla_name} with objects {[obj.name for obj in obj_combo]}")
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
            GroundAtom(GripperFree, []),
            GroundAtom(ToolPrepared, [self.wipe]),
            GroundAtom(ToolPrepared, [self.drink]),
            GroundAtom(IsUtensil, [self.utensil]),
            GroundAtom(DoorClosed, [self.fridge]),
            GroundAtom(DoorClosed, [self.microwave]),
            GroundAtom(InFrontOf, [self.microwave]),
            GroundAtom(PlateAt, [self.holder]),
            GroundAtom(SafeToNavigate, []),
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
        self.preference_context: dict[str, str] | None = None

    def ensure_preference_context(self) -> dict[str, str]:
        """Require a valid preference context before the web session; no implicit defaults."""
        if self.preference_context is None:
            raise RuntimeError(
                "preference_context is required but unset. Call "
                "set_meal_preference_context(meal, setting, time_of_day) before run()."
            )
        return self.preference_context

    def set_meal_preference_context(
        self,
        meal: str,
        setting: str,
        time_of_day: str,
    ) -> dict[str, str]:
        """Validated observable context for this run (meal / setting / time); in-memory only."""
        self.preference_context = build_preference_context(
            meal=meal,
            setting=setting,
            time_of_day=time_of_day,
        )
        return self.preference_context

    def run(self, continuous = True) -> None:

        assert self.web_interface is not None, "Run takes user commands from the web interface which is None."

        if self._pref_mode == "none":
            self.web_interface.ready_for_task_selection()
        else:
            # --- Step 1: Collect context ---
            if self._pref_mode == "terminal":
                from feeding_deployment.integration.terminal_preferences import (
                    terminal_collect_context,
                    terminal_correct_preferences,
                )
                ctx_dict = terminal_collect_context()
                self.set_meal_preference_context(
                    meal=ctx_dict["meal"],
                    setting=ctx_dict["setting"],
                    time_of_day=ctx_dict["time_of_day"],
                )
            elif self._pref_mode == "interface" and self.preference_context is None:
                ctx_defaults = {
                    "meal": MEALS[0],
                    "setting": SETTINGS[0],
                    "time_of_day": TIMES_OF_DAY[0],
                }
                ctx_dict = self.web_interface.get_preference_context(
                    list(MEALS),
                    list(SETTINGS),
                    list(TIMES_OF_DAY),
                    ctx_defaults,
                )
                self.set_meal_preference_context(
                    meal=ctx_dict["meal"],
                    setting=ctx_dict["setting"],
                    time_of_day=ctx_dict["time_of_day"],
                )

            ctx = self.ensure_preference_context()
            print("Preference context (meal / setting / time_of_day):", ctx)
            # --- Step 2: Predict ---
            assert self.physical_profile_label is not None, (
                "physical_profile_label is required for preference prediction "
                "(pass --physical_profile_file)."
            )
            pref_logs = self.log_dir / "preference_learning"
            self._prediction_model = PredictionModel(
                user=self.deployment_user,
                physical_profile_label="deployment_physical_profile",
                logs_dir=pref_logs,
                physical_profile_description=self.physical_profile_label,
            )
            self.predicted_bundle = self._prediction_model.predict_bundle(dict(ctx), {})
            print("Predicted preference bundle (initial):", json.dumps(self.predicted_bundle, indent=2))

            # --- Step 3: Correct ---
            if self._pref_mode == "terminal":
                user_bundle = terminal_correct_preferences(
                    self.predicted_bundle, dict(PREF_OPTIONS),
                )
            else:
                user_bundle = self.web_interface.get_preference_corrections(
                    self.predicted_bundle, dict(PREF_OPTIONS),
                )
            self.ground_truth_bundle = user_bundle
            self.corrected = {
                k: v for k, v in user_bundle.items()
                if v != self.predicted_bundle.get(k)
            }
            print("Ground truth bundle:", json.dumps(self.ground_truth_bundle, indent=2))
            print("Corrected fields:", json.dumps(self.corrected, indent=2))

            # --- Step 4: Apply ---
            from feeding_deployment.integration.apply_preferences import (
                apply_bundle_to_behavior_trees,
                apply_transfer_mode,
                apply_microwave_preference,
                apply_dip_preference,
                apply_occlusion_preference,
            )
            bt_warnings = apply_bundle_to_behavior_trees(
                self.ground_truth_bundle, self.run_behavior_tree_dir,
            )
            for w in bt_warnings:
                print(f"[preference-apply] WARNING: {w}")
            apply_transfer_mode(
                self.ground_truth_bundle,
                self.sim.scene_description,
                self.hla_name_to_hla,
            )
            microwave_duration = apply_microwave_preference(
                self.ground_truth_bundle,
                self.current_atoms,
                GroundAtom(FoodHeated, []),
            )
            if microwave_duration is None:
                print("Microwave preference: no microwave (FoodHeated added to planner state).")
            else:
                print(f"Microwave preference: {microwave_duration}s (planner will include microwave steps).")
            apply_dip_preference(self.ground_truth_bundle, self.flair)
            apply_occlusion_preference(self.ground_truth_bundle, self.flair)
            print("Applied ground-truth bundle to behavior trees and scene config.")

            # --- Step 5: Learn ---
            day = self._prediction_model.next_day() if self._pref_day is None else self._pref_day
            print(f"[learn] Updating memory models (day {day}) ...")
            self._prediction_model.update(
                day=day,
                context=dict(ctx),
                corrected=self.corrected,
                ground_truth_bundle=self.ground_truth_bundle,
            )
            print(f"[learn] Memory update complete (day {day}).")
            if self._pref_mode == "interface":
                self.web_interface.notify_preference_corrections_applied(
                    "Preferences were applied successfully."
                )

            self.web_interface.ready_for_task_selection()
        last_task_type = None
        while self.active:
            if not continuous:
                resp = input("Press 'y' to continue RUNNER, 'n' to stop: ")
                while resp not in ["y", "n"]:
                    resp = input("Press 'y' to continue RUNNER, 'n' to stop: ")
                if resp == "n":
                    break
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
                                user_input = input("Do you want to manually process this? (y/n): ")
                                while user_input not in ["y", "n"]:
                                    user_input = input("Please enter 'y' or 'n': ")
                                if user_input == "y":
                                    update_summary = input("Please enter the update summary to show on the web interface: ")
                                    self.web_interface.update_adaptability_response(update_summary)
                                else:
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
        
        if goal_atoms.issubset(self.current_atoms):
            print("Preconditions already satisfied; no planning needed.")
            plan_ops = []
        else:
            problem = PDDLProblem(
                self.domain.name,
                "AssistedFeeding",
                self.all_objects,
                self.current_atoms,
                goal_atoms,
            )

            # print("Current atoms:", sorted(self.current_atoms))
            # print("Goal atoms:", sorted(goal_atoms))
            # print("Operators:", [op.name for op in self.operators])
            # print("DOMAIN:")
            # print(str(self.domain))
            # print("PROBLEM:")
            # print(str(problem))

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

            # Hack: if the action is navigation, we want to update the InFrontOf predicate for the target and remove it for all other navigation targets, since we assume the robot can only be in front of one navigation target at a time. For other actions, we just apply the add and delete effects as normal.
            self.current_atoms -= operator.delete_effects
            self.current_atoms |= operator.add_effects

            # if ground_hla.hla.get_name() == "Navigate":
            #     known_nav_targets = {self.fridge, self.microwave, self.sink, self.table}
            #     target = ground_hla.objects[0]
            #     for obj in known_nav_targets:
            #         if obj != target:
            #             self.current_atoms.discard(InFrontOf([obj]))

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
    parser.add_argument(
        "--pref_mode",
        type=str,
        choices=["none", "terminal", "interface"],
        default="none",
        help="Preference interaction mode. "
             "'none': no personalization (default). "
             "'terminal': predict + correct via terminal prompts. "
             "'interface': predict + correct via web interface (requires frontend).",
    )
    parser.add_argument(
        "--physical_profile_file",
        type=str,
        default="",
        help="UTF-8 text file describing the user's physical capabilities. "
             "Required with --pref_mode=terminal or --pref_mode=interface.",
    )
    parser.add_argument(
        "--pref_day", type=int, default=None,
        help="Override the deployment day number for preference learning. "
             "If omitted, auto-detected from existing log files (next unused day).",
    )
    args = parser.parse_args()

    if args.user == "":
        raise ValueError("Please provide a user name.")

    if args.run_on_robot or args.use_interface:
        if not ROSPY_IMPORTED:
            raise ModuleNotFoundError("Need ROS to run on robot or use interface")
        else:
            rospy.init_node("feeding_deployment", anonymous=True)

    physical_profile_label: str | None = None
    if args.pref_mode in ("terminal", "interface"):
        if not args.physical_profile_file.strip():
            raise ValueError(
                f"With --pref_mode={args.pref_mode}, pass --physical_profile_file "
                "pointing to a UTF-8 .txt file with freeform physical-capability text."
            )
        profile_path = Path(args.physical_profile_file.strip())
        if not profile_path.is_file():
            raise ValueError(f"physical profile file not found: {profile_path}")
        physical_profile_label = profile_path.read_text(encoding="utf-8").strip()
        if not physical_profile_label:
            raise ValueError(f"physical profile file is empty: {profile_path}")

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
                     args.no_waits,
                     physical_profile_label=physical_profile_label,
                     pref_day=args.pref_day,
                     pref_mode=args.pref_mode)

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, runner.signal_handler)

    if not args.use_interface:
        # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["OpenDoor"], (runner.fridge,)))
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,runner.table)))
        # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["PlacePlateInSink"], (runner.plate, runner.sink)))
    else:
        runner.run()

    if args.make_videos:
        output_path = Path(__file__).parent / "videos" / "full.mp4"
        runner.make_video(output_path)

    if args.run_on_robot:
        rospy.spin()
