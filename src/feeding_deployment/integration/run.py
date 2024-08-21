"""The main entry point for running the integrated system."""

from feeding_deployment.integration.high_level_actions import tool_type, ToolPrepared, GripperFree, Holding, ToolTransferDone, PickToolHLA, StowToolHLA, PrepareToolHLA, TransferToolHLA, pddl_plan_to_hla_plan
from relational_structs import PDDLDomain, PDDLProblem, Predicate, LiftedAtom
from relational_structs.utils import parse_pddl_plan
from tomsutils.pddl_planning import run_pyperplan_planning


def _main() -> None:
    """The main entry point for running the integrated system."""

    # Initialize the simulator.
    sim = None  # TODO
    arm = None  # TODO

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
    objects = {cup, wiper, utensil}
    init_atoms = {LiftedAtom(GripperFree, []), ToolPrepared([wiper]), ToolPrepared([cup])}

    # TODO update this once the interface is ready.
    goal_atoms = {ToolTransferDone([cup])}
    problem = PDDLProblem(domain.name, "AssistedFeeding", objects, init_atoms, goal_atoms)

    # Plan a sequence of high-level actions to execute.
    plan_strs = run_pyperplan_planning(str(domain), str(problem), heuristic="lmcut", search="astar")
    plan_ops = parse_pddl_plan(plan_strs, domain, problem)
    plan_hlas = pddl_plan_to_hla_plan(plan_ops, hlas)

    for hla, objects in plan_hlas:
        # Get commands to execute on the robot.
        robot_commands = hla.get_robot_commands(objects, sim)
        # Execute the commands.
        for robot_command in robot_commands:
            arm.execute_command(robot_command)
        # Update the simulator.
        hla.update_simulator(objects, sim)


if __name__ == "__main__":
    _main()
