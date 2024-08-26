"""The main entry point for running the integrated system."""

from pathlib import Path
from typing import Any

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
from feeding_deployment.robot_controller.arm_client import (
    ARM_RPC_PORT,
    NUC_HOSTNAME,
    RPC_AUTHKEY,
    ArmManager,
)
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


def _main(
    run_on_robot: bool, make_videos: bool, max_motion_planning_time: float = 10
) -> None:
    """The main entry point for running the integrated system."""

    # Initialize the interface to the robot.
    if run_on_robot:
        manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        manager.connect()
        robot_interface = manager.Arm()  # type: ignore  # pylint: disable=no-member
    else:
        robot_interface = None

    # Initialize the perceiver (e.g., get joint states or human head poses).
    perception_interface = PerceptionInterface(robot_interface)

    # Initialize the simulator.
    kwargs: dict[str, Any] = {}
    if run_on_robot:
        kwargs["initial_joints"] = perception_interface.get_robot_joints()
        print(f"Initial joint state: {kwargs['initial_joints']}")
    else:
        kwargs["initial_joints"] = [0.0] * 9
        print("Running in simulation mode.")
    scene_description = SceneDescription(**kwargs)
    sim = FeedingDeploymentPyBulletSimulator(scene_description)

    # Create skills for high-level planning.
    hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}
    hlas = {
        cls(sim, robot_interface, perception_interface, hla_hyperparams, run_on_robot) for cls in HLAS  # type: ignore
    }
    hla_name_to_hla = {hla.get_name(): hla for hla in hlas}
    operators = {hla.get_operator() for hla in hlas}
    predicates: set[Predicate] = {ToolPrepared, GripperFree, Holding, ToolTransferDone}
    types = {tool_type}
    domain = PDDLDomain("AssistedFeeding", operators, predicates, types)
    cup = Object("cup", tool_type)
    wiper = Object("wiper", tool_type)
    utensil = Object("utensil", tool_type)
    all_objects = {cup, wiper, utensil}
    current_atoms = {
        LiftedAtom(GripperFree, []),
        ToolPrepared([wiper]),
        ToolPrepared([cup]),
    }

    # TODO update this once the user interface is ready.
    TransferTool = hla_name_to_hla["TransferTool"]
    user_command_queue: list[GroundHighLevelAction | set[GroundAtom]] = [
        GroundHighLevelAction(TransferTool, (utensil,), {"mask": "TODO"}),
        GroundHighLevelAction(TransferTool, (cup,)),
        GroundHighLevelAction(TransferTool, (wiper,)),
        {GroundAtom(GripperFree, [])},  # reset at the end
    ]

    full_simulated_traj: list[FeedingDeploymentSimulatorState] = []

    while user_command_queue:
        user_command = user_command_queue.pop(0)
        print(f"Working towards new command: {user_command}")

        # Plan to the preconditions of the HLA.
        if isinstance(user_command, GroundHighLevelAction):
            goal_atoms = user_command.get_preconditions()
        else:
            goal_atoms = user_command
        problem = PDDLProblem(
            domain.name, "AssistedFeeding", all_objects, current_atoms, goal_atoms
        )
        plan_strs = run_pyperplan_planning(
            str(domain), str(problem), heuristic="lmcut", search="astar"
        )
        assert plan_strs is not None
        plan_ops = parse_pddl_plan(plan_strs, domain, problem)
        print("Found plan to the preconditions of the command:")
        for i, op in enumerate(plan_ops):
            print(f"{i}. {op.short_str}")
        plan_hlas = pddl_plan_to_hla_plan(plan_ops, hlas)
        # Append the user command to the plan if it's an action.
        if isinstance(user_command, GroundHighLevelAction):
            plan_hlas.append(user_command)

        for ground_hla in plan_hlas:
            print(f"Refining {ground_hla}")
            operator = ground_hla.get_operator()

            assert operator.preconditions.issubset(current_atoms)

            # Execute the high-level plan in simulation
            sim_traj = ground_hla.execute_action()

            if sim_traj:
                full_simulated_traj.extend(sim_traj)

            # Make sure the states are in sync.
            if sim_traj:
                sim.sync(sim_traj[-1])
            current_atoms -= operator.delete_effects
            current_atoms |= operator.add_effects

    if make_videos:
        outfile = Path(__file__).parent / "full.mp4"
        make_simulation_video(sim, full_simulated_traj, outfile)
        print(f"Saved video to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    args = parser.parse_args()

    _main(args.run_on_robot, args.make_videos, args.max_motion_planning_time)
