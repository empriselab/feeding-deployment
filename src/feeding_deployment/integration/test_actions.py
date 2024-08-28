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
from pybullet_helpers.geometry import Pose
from pybullet_helpers.link import get_link_pose, get_relative_link_pose

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


def _main(
    run_on_robot: bool, make_videos: bool, max_motion_planning_time: float = 10
) -> None:
    """Testing components of the system."""

    # Initialize the interface to the robot.
    if run_on_robot:
        manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        manager.connect()
        robot_interface = manager.ArmInterface()  # type: ignore  # pylint: disable=no-member
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
        print("Running in simulation mode.")
    scene_description = SceneDescription(**kwargs)
    sim = FeedingDeploymentPyBulletSimulator(scene_description)

    # Create skills for high-level planning.
    hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}

    high_level_action = TransferToolHLA(sim, robot_interface, perception_interface, hla_hyperparams, run_on_robot)
    cup = Object("cup", tool_type)

    sim.held_object_name = "cup"
    sim.held_object_id = sim.cup_id
    sim.robot.set_finger_state(scene_description.tool_grasp_fingers_value)
    finger_frame_id = sim.robot.link_from_name("finger_tip")
    end_effector_link_id = sim.robot.link_from_name(sim.robot.tool_link_name)
    cup_from_end_effector = get_relative_link_pose(
        sim.robot.robot_id, finger_frame_id, end_effector_link_id, sim.physics_client_id
    )
    sim.held_object_tf = cup_from_end_effector
    print(f"cup_from_end_effector: {cup_from_end_effector}")

    sim_traj = high_level_action.execute_action(objects=[cup], params={})

    if make_videos:
        outfile = Path(__file__).parent / "single_action.mp4"
        make_simulation_video(sim, sim_traj, outfile)
        print(f"Saved video to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    args = parser.parse_args()

    _main(args.run_on_robot, args.make_videos, args.max_motion_planning_time)
