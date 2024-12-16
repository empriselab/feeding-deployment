"""The main entry point for running the integrated system."""

import json
from collections import namedtuple
from pathlib import Path
from typing import Any
import pickle
import queue
import os
import sys

try:
    import rospy
    from std_msgs.msg import String

    ROSPY_IMPORTED = True
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

try:
    FLAIR_PATH = "/home/isacc/deployment_ws/src/FLAIR/bite_acquisition/scripts"
    sys.path.append(FLAIR_PATH)

    # raise ModuleNotFoundError  # Just to skip this block
    from wrist_controller import WristController
    from flair import FLAIR

    FLAIR_IMPORTED = True
except ModuleNotFoundError:
    FLAIR_IMPORTED = False

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

from feeding_deployment.actions.high_level_actions import (
    GripperFree,
    GroundHighLevelAction,
    Holding,
    IsUtensil,
    PlateInView,
    PickToolHLA,
    LookAtPlateHLA,
    AcquireBiteHLA,
    StowToolHLA,
    ToolPrepared,
    ToolTransferDone,
    TransferToolHLA,
    ResetPos,
    ResetHLA,
    pddl_plan_to_hla_plan,
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
from pybullet_helpers.geometry import Pose

# All the high level actions we want to consider.
HLAS = {PickToolHLA, StowToolHLA, LookAtPlateHLA, AcquireBiteHLA, TransferToolHLA, ResetHLA}

assert os.environ.get("PYTHONHASHSEED") == "0", \
        "Please add `export PYTHONHASHSEED=0` to your bash profile!"

class _Runner:
    """A class for running the integrated system."""

    def __init__(self, run_on_robot: bool, use_interface: bool, simulate_head_perception: bool, max_motion_planning_time: float,
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
        else:
            self.robot_interface = None

        # Initialize the perceiver (e.g., get joint states or human head poses).
        self.perception_interface = PerceptionInterface(robot_interface=self.robot_interface, simulate_head_perception=self.simulate_head_perception)

        # Initialize the simulator.
        kwargs: dict[str, Any] = {}
        if run_on_robot:
            kwargs["initial_joints"] = self.perception_interface.get_robot_joints()
            print(f"Initial joint state: {kwargs['initial_joints']}")
        else:
            print("Running in simulation mode.")

        self.scene_description = SceneDescription(**kwargs)


        if self.use_interface:
            # Initialize the web interface.
            self.hla_command_queue = queue.Queue()
            self.web_interface = WebInterface(self.hla_command_queue)
        else:
            self.web_interface = None

        if self.run_on_robot:
            self.rviz_interface = RVizInterface(self.scene_description)
            wrist_controller = WristController()
            flair = FLAIR(self.robot_interface, wrist_controller, self.no_waits)
        else:
            self.rviz_interface = None
            wrist_controller = None
            flair = None

        # self.sim = FeedingDeploymentPyBulletSimulator(self.scene_description)
        self.sim = FeedingDeploymentPyBulletSimulator(self.scene_description, use_gui=False)

        # Create skills for high-level planning.
        hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}
        print("Creating HLAs...")
        self.hlas = {
            cls(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.web_interface, hla_hyperparams,
                wrist_controller, flair, self.no_waits) for cls in HLAS  # type: ignore
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

        # Record the full simulated trajectory for viz and debug.
        self.full_simulated_traj: list[FeedingDeploymentSimulatorState] = []

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

        # for i in range(2):
        #     # # Uncomment to test commands.
        #     drink_pickup_msg = {"status": "drink_pickup"}
        #     self.hla_command_queue.put(drink_pickup_msg)

        # wipe_transfer_msg = {"status": "move_to_wiping_position", "state": "prepared_mouth_wiping"}
        # self.hla_command_queue.put(wipe_transfer_msg)

        # drink_pickup_msg = {"status": "drink_pickup", "state": "pre_bite_pickup"}
        # self.hla_command_queue.put(drink_pickup_msg)

        while not rospy.is_shutdown():
            try:
                hla_interface_msg = self.hla_command_queue.get(timeout=1)
                self.parse_interface_msg(hla_interface_msg)
                print("Ready for next user command.")
            except queue.Empty:
                continue

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
            sim_traj = ground_hla.execute_action()

            if sim_traj:
                self.full_simulated_traj.extend(sim_traj)

            # Make sure the states are in sync.
            if sim_traj:
                self.sim.sync(sim_traj[-1])
            self.current_atoms -= operator.delete_effects
            self.current_atoms |= operator.add_effects

            # Save the latest state in case we want to resume execution
            # after a crash.
            sim_state = self.full_simulated_traj[-1] if self.full_simulated_traj else None
            self._save_state(sim_state, self.current_atoms)

    def make_video(self, outfile: Path) -> None:
        """Create a video of the simulated trajectory."""
        make_simulation_video(self.sim, self.full_simulated_traj, outfile)
        print(f"Saved video to {outfile}")

    def _save_state(self, sim_state: FeedingDeploymentSimulatorState, atoms: set[GroundAtom]) -> None:
        with open(self._saved_state_outfile, "wb") as f:
            pickle.dump((sim_state, atoms), f)
        print(f"Saved system state to {self._saved_state_outfile}")

    def _load_from_state(self) -> None:
        with open(self._saved_state_infile, "rb") as f:
            sim_state, self.current_atoms = pickle.load(f)
        if sim_state is not None:
            assert isinstance(sim_state, FeedingDeploymentSimulatorState)
            self.sim.sync(sim_state)
            self.rviz_interface.joint_state_update(sim_state.robot_joints)
            if sim_state.held_object:
                self.rviz_interface.tool_update(True, sim_state.held_object, Pose((0, 0, 0), (0, 0, 0, 1)))
                
        print(f"Loaded system state from {self._saved_state_infile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--use_interface", action="store_true")
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
    
    if args.run_on_robot:
        if not FLAIR_IMPORTED:
            raise ModuleNotFoundError("Need FLAIR to run on robot")
        
    # Rajat ToDo: have run on robot without interface functionality
    if args.run_on_robot:
        args.use_interface = True

    runner = _Runner(args.run_on_robot, 
                     args.use_interface,
                     args.simulate_head_perception,
                     args.max_motion_planning_time,
                     args.resume_from_state,
                     args.no_waits)

    # Uncomment to test commands.
    # drink_pickup_msg = {"status": "drink_pickup"}
    # runner.hla_command_queue.put(drink_pickup_msg)

    # drink_transfer_msg = {"status": "drink_transfer"}
    # runner.hla_command_queue.put(drink_transfer_msg)

    # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,)))
    # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.utensil,)))
    # for _ in range(10):
        # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.drink,)))
        # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.drink,)))
    # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.wipe,)))
    # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["StowTool"], (runner.wipe,)))

    if args.make_videos:
        runner.make_video(Path("full.mp4"))

    if ROSPY_IMPORTED:
        rospy.spin()
