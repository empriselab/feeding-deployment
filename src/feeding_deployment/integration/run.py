"""The main entry point for running the integrated system."""

from pathlib import Path

from relational_structs import LiftedAtom, Object, PDDLDomain, PDDLProblem, Predicate
from relational_structs.utils import parse_pddl_plan
from tomsutils.pddl_planning import run_pyperplan_planning

from feeding_deployment.integration.high_level_actions import (
    GripperFree,
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
from feeding_deployment.robot_controller.arm_client import Arm
from feeding_deployment.simulation.scene_description import SceneDescription
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.video import make_simulation_video

# All the high level actions we want to consider.
HLAS = {PickToolHLA, StowToolHLA, PrepareToolHLA, TransferToolHLA}


def _main(run_on_robot: bool, make_videos: bool) -> None:
    """The main entry point for running the integrated system."""

    # Initialize the simulator.
    scene_description = SceneDescription()
    sim = FeedingDeploymentPyBulletSimulator(scene_description)

    # Initialize the interface to the robot.
    robot_interface = Arm() if run_on_robot else None

    # Initialize the perceiver (e.g., get joint states or human head poses).
    perception_interface = None  # TODO

    # Create a domain for high-level planning.
    hlas = {cls(sim, robot_interface, perception_interface) for cls in HLAS}
    operators = {hla.get_operator() for hla in hlas}
    predicates: set[Predicate] = {ToolPrepared, GripperFree, Holding, ToolTransferDone}
    types = {tool_type}
    domain = PDDLDomain("AssistedFeeding", operators, predicates, types)

    # TODO automate?
    cup = Object("cup", tool_type)
    wiper = Object("wiper", tool_type)
    utensil = Object("utensil", tool_type)
    all_objects = {cup, wiper, utensil}
    current_atoms = {
        LiftedAtom(GripperFree, []),
        ToolPrepared([wiper]),
        ToolPrepared([cup]),
    }

    # TODO update this once the interface is ready.
    goal_queue = [
        ("Assist Drinking", {ToolTransferDone([cup])}),
        ("Assist Feeding", {ToolTransferDone([utensil])}),
        ("Assist Wiping", {ToolTransferDone([wiper])}),
    ]

    while goal_queue:
        goal_str, goal_atoms = goal_queue.pop(0)
        print("Working towards new goal:", goal_str)

        # Plan a sequence of high-level actions to execute.
        problem = PDDLProblem(
            domain.name, "AssistedFeeding", all_objects, current_atoms, goal_atoms
        )
        plan_strs = run_pyperplan_planning(
            str(domain), str(problem), heuristic="lmcut", search="astar"
        )
        assert plan_strs is not None
        plan_ops = parse_pddl_plan(plan_strs, domain, problem)
        print("Found plan:")
        for i, op in enumerate(plan_ops):
            print(f"{i}. {op.short_str}")
        plan_hlas = pddl_plan_to_hla_plan(plan_ops, hlas)

        for (hla, object_params), operator in zip(plan_hlas, plan_ops, strict=True):
            print(f"Refining {operator.short_str}")

            assert operator.preconditions.issubset(current_atoms)
            # Turn into a low-level plan that can be simulated.
            sim_traj = hla.get_simulated_trajectory(object_params)

            # Optionally make a video of the simulated trajectory.
            if make_videos:
                outfile = Path(__file__).parent / "last.mp4"
                make_simulation_video(sim, sim_traj, outfile)

            # Get commands to execute on the robot.
            robot_commands = hla.get_robot_commands(object_params, sim_traj)

            # Execute the commands.
            if run_on_robot:
                for robot_command in robot_commands:
                    robot_interface.execute_command(robot_command)

            # Make sure the states are in sync.
            if sim_traj:
                sim.sync(sim_traj[-1])
            current_atoms -= operator.delete_effects
            current_atoms |= operator.add_effects


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    args = parser.parse_args()

    _main(args.run_on_robot, args.make_videos)
