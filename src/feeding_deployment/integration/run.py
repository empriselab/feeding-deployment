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

from feeding_deployment.actions.base import (
    GripperFree,
    Holding,
    IsUtensil,
    PlateInView,
    ToolPrepared,
    ToolTransferDone,
    ResetPos,
    tool_type,
    GroundHighLevelAction,
    ResetHLA,
    pddl_plan_to_hla_plan,
)
from feeding_deployment.actions.pick_tool import PickToolHLA
from feeding_deployment.actions.stow_tool import StowToolHLA
from feeding_deployment.actions.transfer_tool import TransferToolHLA
from feeding_deployment.actions.acquisition import LookAtPlateHLA, AcquireBiteHLA
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


# All the high level actions we want to consider.
HLAS = {PickToolHLA, StowToolHLA, LookAtPlateHLA, AcquireBiteHLA, TransferToolHLA, ResetHLA}

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

        if self.use_interface:
            # Initialize the web interface.
            self.hla_command_queue = queue.Queue()
            self.web_interface = WebInterface(self.hla_command_queue)
        else:
            self.web_interface = None

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

        # Track the current high-level state.
        self.current_atoms = {
            LiftedAtom(GripperFree, []),
            ToolPrepared([self.wipe]),
            ToolPrepared([self.drink]),
            IsUtensil([self.utensil]),
        }

        if self._saved_state_infile:
            self._load_from_state()
            print("WARNING: The system state has been restored to:")
            print(" ", sorted(self.current_atoms))
            resp = input("Are you sure you want to continue from here? [y/n] ")
            while resp not in ["y", "n"]:
                resp = input("Please enter 'y' or 'n': ")
                if resp == "n":
                    sys.exit(0)

        print("Runner is ready.")
        self.active = True

    def run(self) -> None:
        
        while self.active:
            try:
                hla_interface_msg = self.hla_command_queue.get(timeout=1)
                self.parse_interface_msg(hla_interface_msg)
                print("Ready for next user command.")
            except queue.Empty:
                continue

    def signal_handler(self, signal, frame):
        self.active = False
        print("\nprogram exiting gracefully")
        sys.exit(0)

    def parse_interface_msg(self, msg_dict: dict[str, Any]) -> None:
        """Pass high level action message from the web interface."""
        if msg_dict["status"] == "finish_feeding":
            user_cmd = GroundHighLevelAction(
                self.hla_name_to_hla["Reset"], ()
            )
        elif msg_dict["status"] == "drink_pickup":
            user_cmd = GroundHighLevelAction(
                self.hla_name_to_hla["PickTool"], (self.drink,)
            )
        elif msg_dict["status"] == "drink_transfer":
            user_cmd = GroundHighLevelAction(
                self.hla_name_to_hla["TransferTool"], (self.drink,)
            )
        elif msg_dict["status"] == "move_to_above_plate" \
            or (msg_dict["status"] == "return_to_main" and msg_dict["state"] == "post_bite_pickup") \
            or (msg_dict["status"] == "return_to_main" and msg_dict["state"] == "post_bite_transfer") \
            or (msg_dict["status"] == "return_to_main" and msg_dict["state"] == "post_drink_transfer") \
            or (msg_dict["status"] == "back" and msg_dict["state"] == "bite_selection"): 
            user_cmd = GroundHighLevelAction(
                self.hla_name_to_hla["LookAtPlate"], (self.utensil,)
            )
        elif msg_dict["status"] == "aquire_food" or msg_dict["status"] == 0: # manual acquire food
            user_cmd = GroundHighLevelAction(
                self.hla_name_to_hla["AcquireBite"], (self.utensil,), params=msg_dict
            )
        elif msg_dict["status"] == "bite_transfer":
            user_cmd = GroundHighLevelAction(
                self.hla_name_to_hla["TransferTool"], (self.utensil,)
            )
        elif msg_dict["status"] == "mouth_wiping" and msg_dict["state"] == "bite_selection":
            user_cmd = GroundHighLevelAction(
                self.hla_name_to_hla["PickTool"], (self.wipe,)
            )
        elif msg_dict["status"] == "move_to_wiping_position" and msg_dict["state"] == "prepared_mouth_wiping":
            user_cmd = GroundHighLevelAction(
                self.hla_name_to_hla["TransferTool"], (self.wipe,)
            )
        else:
            print("WARNING: Unrecognized high level action message from web interface.")
            return
        self.process_user_command(user_cmd)

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
        plan_strs = run_pyperplan_planning(
            str(self.domain), str(problem), heuristic="lmcut", search="astar"
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

    def process_user_update_request(self, request_text: str) -> None:
        """Validate and update behavior trees."""
        import ipdb; ipdb.set_trace()


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
        # Example of using LLM to generate updates to behavior trees.
        runner.process_user_update_request("Pause for 1 second before getting the food with the fork")
        runner.process_user_update_request("Stop for half a second after picking up the food")
        runner.process_user_update_request("Move a little bit faster while picking up the food")

        # bite_acquisition = GroundHighLevelAction(runner.hla_name_to_hla["AcquireBite"], (runner.utensil,))
        # bite_acquisition.process_behavior_tree_node_addition("Pause", {"duration": 1.0}, "AcquireBite", "before")
        # bite_acquisition.process_behavior_tree_node_addition("Pause", {"duration": 0.5}, "AcquireBite", "after")
        # bite_acquisition.process_behavior_tree_parameter_update("AcquireBite", "Speed", 1.25)

        # Run some commands.
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,)))
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.drink,)))
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.wipe,)))
        runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.wipe,)))
    else:
        runner.run()

    if args.make_videos:
        output_path = Path(__file__).parent / "videos" / "full.mp4"
        runner.make_video(output_path)

    if args.run_on_robot:
        rospy.spin()
