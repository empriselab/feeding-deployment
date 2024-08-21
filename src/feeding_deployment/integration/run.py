"""The main entry point for running the integrated system."""

from feeding_deployment.integration.high_level_actions import tool_type, ToolPrepared, GripperFree, Holding, ToolTransferDone, PickToolHLA, StowToolHLA, PrepareToolHLA, TransferToolHLA, pddl_plan_to_hla_plan
from relational_structs import PDDLDomain, PDDLProblem, Predicate, LiftedAtom
from relational_structs.utils import parse_pddl_plan
from tomsutils.pddl_planning import run_pyperplan_planning
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.scene_description import SceneDescription
from feeding_deployment.simulation.video import make_simulation_video
from feeding_deployment.robot_controller.arm_client import Arm
from pathlib import Path
import logging


def _main(run_on_robot: bool, make_videos: bool) -> None:
    """The main entry point for running the integrated system."""

    # Initialize the simulator.
    scene_description = SceneDescription()
    sim = FeedingDeploymentPyBulletSimulator(scene_description)
    
    if run_on_robot:
        arm = Arm()

    # Create a domain for high-level planning.
    hlas = {PickToolHLA(), StowToolHLA(), PrepareToolHLA(), TransferToolHLA()}
    operators = {hla.get_operator() for hla in hlas}
    predicates: set[Predicate] = {ToolPrepared, GripperFree, Holding, ToolTransferDone}
    types = {tool_type}
    domain = PDDLDomain("AssistedFeeding", operators, predicates, types)

    # TODO automate?
    cup = tool_type("cup")
    wiper = tool_type("wiper")
    utensil = tool_type("utensil")
    all_objects = {cup, wiper, utensil}
    current_atoms = {LiftedAtom(GripperFree, []), ToolPrepared([wiper]), ToolPrepared([cup])}

    # TODO update this once the interface is ready.
    goal_queue = [
        {ToolTransferDone([cup])},  # drinking
        {ToolTransferDone([utensil])},  # feeding
        {ToolTransferDone([wiper])},  # wiping
    ]

    while goal_queue:
        goal_atoms = goal_queue.pop(0)
        print("Working towards new goal:", goal_atoms)

        # Plan a sequence of high-level actions to execute.
        problem = PDDLProblem(domain.name, "AssistedFeeding", all_objects, current_atoms, goal_atoms)
        plan_strs = run_pyperplan_planning(str(domain), str(problem), heuristic="lmcut", search="astar")
        print("Found plan:", plan_strs)

        plan_ops = parse_pddl_plan(plan_strs, domain, problem)
        plan_hlas = pddl_plan_to_hla_plan(plan_ops, hlas)

        for (hla, object_params), operator in zip(plan_hlas, plan_ops, strict=True):
            print(f"Refining {operator.short_str}")

            assert operator.preconditions.issubset(current_atoms)
            # Turn into a low-level plan that can be simulated.
            sim_traj = hla.get_simulated_trajectory(object_params, sim)

            # Optionally make a video of the simulated trajectory.
            if make_videos:
                outfile = Path(__file__).parent / "last.mp4"
                make_simulation_video(sim, sim_traj, outfile)

            # Get commands to execute on the robot.
            robot_commands = hla.get_robot_commands(object_params, sim, sim_traj)

            # Execute the commands.
            if run_on_robot:
                for robot_command in robot_commands:
                    arm.execute_command(robot_command)

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
