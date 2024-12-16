"""The main entry point for running the integrated system."""

from pathlib import Path
from typing import Any

import json

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

from feeding_deployment.simulation.scene_description import (
    SceneDescription,
    create_scene_description_from_config,
)
from feeding_deployment.simulation.simulator import (
    FeedingDeploymentPyBulletSimulator,
    FeedingDeploymentSimulatorState,
)
from feeding_deployment.simulation.video import make_simulation_video

from feeding_deployment.simulation.control import _get_trajectory_to_pose

def _main(
    run_on_robot: bool, simulate_head_perception: bool, make_videos: bool, max_motion_planning_time: float = 10, tool: str = "fork"
) -> None:
    
    # Initialize the simulator.
    kwargs: dict[str, Any] = {}
    kwargs["initial_joints"] = [-2.291869562487007, -1.3006196994707935, -1.720891553199544, -2.208449769765801, -0.477821699620927, -0.1613864967943659, -2.9981518678009262, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    print("Running in simulation mode.")
    scene_description = SceneDescription(**kwargs)

    print("Scene Description loaded")
    sim = FeedingDeploymentPyBulletSimulator(scene_description, use_gui=True, ignore_user=True)

    print("Feeding Deployment Simulator loaded")
    
    target_pose = Pose(position=[-0.282, 0.540, 0.619], orientation=[0, 0.7071068, 0.7071068, 0 ])
    _get_trajectory_to_pose(target_pose=target_pose, sim=sim, max_control_time=10.0)

    # input("Press Enter to set utensil motors to 0.5, 0.5")
    # sim.set_utensil_motors([0.5, 0.5])

    # input("Press Enter to set utensil motors to 0.0, 0.0")
    # sim.set_utensil_motors([0.0, 0.0])

    # input("Press Enter to set utensil motors to -0.5, -0.5")
    # sim.set_utensil_motors([-0.5, -0.5])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--simulate_head_perception", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    parser.add_argument("--tool", type=str, default="fork")
    args = parser.parse_args()

    _main(args.run_on_robot, args.simulate_head_perception, args.make_videos, args.max_motion_planning_time, args.tool)
