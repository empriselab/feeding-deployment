"""The main entry point for running the integrated system."""

import json
from collections import namedtuple
from pathlib import Path
from typing import Any

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

from feeding_deployment.integration.high_level_actions import (
    GripperFree,
    GroundHighLevelAction,
    Holding,
    PickToolHLA,
    PrepareToolHLA,
    StowToolHLA,
    ToolPrepared,
    ToolTransferDone,
    TransferToolHLA,
    pddl_plan_to_hla_plan,
    tool_type,
)
from feeding_deployment.integration.perception_interface import PerceptionInterface
from feeding_deployment.integration.rviz_interface import RVizInterface
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

# All the high level actions we want to consider.
HLAS = {PickToolHLA, StowToolHLA, PrepareToolHLA, TransferToolHLA}


class _Runner:
    """A class for running the integrated system."""

    def __init__(self, run_on_robot: bool, max_motion_planning_time: float):
        self.run_on_robot = run_on_robot
        self.max_motion_planning_time = max_motion_planning_time

        # Subscribe to the web interface topics.
        if ROSPY_IMPORTED:
            self.web_interface_sub = rospy.Subscriber(
                "WebAppComm", String, self.web_interface_callback
            )

        # Initialize the interface to the robot.
        if run_on_robot:
            self.robot_interface = ArmInterfaceClient()  # type: ignore  # pylint: disable=no-member
        else:
            self.robot_interface = None

        # Initialize the perceiver (e.g., get joint states or human head poses).
        self.perception_interface = PerceptionInterface(self.robot_interface)

        # Initialize the simulator.
        kwargs: dict[str, Any] = {}
        if run_on_robot:
            kwargs["initial_joints"] = self.perception_interface.get_robot_joints()
            print(f"Initial joint state: {kwargs['initial_joints']}")
        else:
            print("Running in simulation mode.")

        input("Press enter to populate the scene description.")
        self.scene_description = SceneDescription(**kwargs)

        self.rviz_interface = RVizInterface(self.scene_description)

        input("Press enter to create the simulator.")
        self.sim = FeedingDeploymentPyBulletSimulator(self.scene_description)
        # self.sim = FeedingDeploymentPyBulletSimulator(self.scene_description, use_gui=False)

        input("Press enter to create the high-level actions.")
        # Create skills for high-level planning.
        hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}
        self.hlas = {
            cls(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, hla_hyperparams, run_on_robot) for cls in HLAS  # type: ignore
        }
        self.hla_name_to_hla = {hla.get_name(): hla for hla in self.hlas}
        self.operators = {hla.get_operator() for hla in self.hlas}
        self.predicates: set[Predicate] = {
            ToolPrepared,
            GripperFree,
            Holding,
            ToolTransferDone,
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
        }

        # Record the full simulated trajectory for viz and debug.
        self.full_simulated_traj: list[FeedingDeploymentSimulatorState] = []

    def web_interface_callback(self, msg: "String") -> None:
        """Callback for the web interface."""
        msg_dict = json.loads(msg.data)
        print("RECEIVED MESSAGE FROM WEB INTERFACE:")
        print(msg_dict)
        if msg_dict["status"] == "drink_pickup":
            user_cmd = GroundHighLevelAction(
                self.hla_name_to_hla["PickTool"], (self.drink,)
            )
        elif msg_dict["status"] == "drink_transfer":
            user_cmd = GroundHighLevelAction(
                self.hla_name_to_hla["TransferTool"], (self.drink,)
            )
        else:
            print("WARNING: Unrecognized message from web interface.")
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

        # TODO: send a message back to the web interface upon completion!

    def make_video(self, outfile: Path) -> None:
        """Create a video of the simulated trajectory."""
        make_simulation_video(self.sim, self.full_simulated_traj, outfile)
        print(f"Saved video to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    args = parser.parse_args()

    if ROSPY_IMPORTED:
        rospy.init_node("feeding_deployment_integration")
    else:
        assert not args.run_on_robot, "Need ROS to run on robot"

    runner = _Runner(args.run_on_robot, args.max_motion_planning_time)

    # # Uncomment to test commands.
    # msg = namedtuple("String", ["data"])
    # runner.web_interface_callback(msg(json.dumps({"status": "drink_pickup"})))
    # runner.web_interface_callback(msg(json.dumps({"status": "drink_transfer"})))
    runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,)))
    runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.drink,)))
    # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,)))
    # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.drink,)))
    # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.utensil,)))
    # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.drink,)))
    # runner.process_user_command(GroundHighLevelAction(runner.hla_name_to_hla["TransferTool"], (runner.wipe,)))

    if args.make_videos:
        runner.make_video(Path("full.mp4"))

    if ROSPY_IMPORTED:
        rospy.spin()
