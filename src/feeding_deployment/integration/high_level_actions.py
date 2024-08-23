"""High-level actions that we can simulate and execute."""

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Rajat ToDo: Remove this hacky addition
FLAIR_PATH = "/home/isacc/deployment_ws/src/FLAIR/bite_acquisition/scripts"
import sys

sys.path.append(FLAIR_PATH)
try:
    from skill_library import SkillLibrary

    FLAIR_IMPORTED = True
except ModuleNotFoundError:
    FLAIR_IMPORTED = False
    pass


from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)

from feeding_deployment.integration.perception_interface import PerceptionInterface
from feeding_deployment.integration.utils import simulated_trajectory_to_kinova_commands
from feeding_deployment.robot_controller.arm_client import Arm, KinovaCommand
from feeding_deployment.simulation.planning import (
    get_bite_transfer_plan,
    get_plan_to_grasp_cup,
    get_plan_to_grasp_utensil,
    get_plan_to_grasp_wiper,
    get_plan_to_stow_cup,
    get_plan_to_stow_utensil,
    get_plan_to_stow_wiper,
    get_plan_to_transfer_cup,
    get_plan_to_transfer_wiper,
    remap_trajectory_to_constant_distance,
)
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState

# Define some predicates that can be used for sequencing the high-level actions.
tool_type = Type("tool")  # utensil, cup, or wiping tool
GripperFree = Predicate("GripperFree", [])  # not holding any tool
Holding = Predicate("Holding", [tool_type])  # holding tool
ToolTransferDone = Predicate("ToolTransferDone", [tool_type])  # wiped, drank, or ate
ToolPrepared = Predicate("ToolPrepared", [tool_type])  # e.g., bite acquired


# Define high-level actions.
class HighLevelAction(abc.ABC):
    """Base class for high-level action."""

    def __init__(
        self,
        sim: FeedingDeploymentPyBulletSimulator,
        robot_interface: Arm,
        perception_interface: PerceptionInterface,
        hla_hyperparams: dict[str, Any],
        run_on_robot: bool,
    ) -> None:
        self._sim = sim
        self._robot_interface = robot_interface
        self._perception_interface = perception_interface
        self._hla_hyperparams = hla_hyperparams
        self._run_on_robot = run_on_robot

    @abc.abstractmethod
    def get_name(self) -> str:
        """Get a human-readable name for this HLA."""

    @abc.abstractmethod
    def get_operator(self) -> LiftedOperator:
        """Create a planning operator for this HLA."""

    @abc.abstractmethod
    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        """Execute the action on the robot and return simulated trajectory."""


@dataclass(frozen=True)
class GroundHighLevelAction:
    """A high-level action with objects and parameters specified.

    For example, the parameters for bite acquisition might include the
    preferred food item. These parameters can be populated automatically
    or set by the user.
    """

    hla: HighLevelAction  # want this to be executed
    objects: tuple[Object, ...]  # for grounding the high-level action
    params: dict = field(default_factory=lambda: {})  # see docstring

    def __str__(self) -> str:
        obj_str = ", ".join([o.name for o in self.objects])
        return f"{self.hla.get_name()}({obj_str})"

    def get_operator(self) -> GroundOperator:
        """Get the operator for this ground HLA."""
        return self.hla.get_operator().ground(self.objects)

    def get_preconditions(self) -> set[GroundAtom]:
        """Get the preconditions for executing this command."""
        return self.get_operator().preconditions

    def execute_action(self) -> list[FeedingDeploymentSimulatorState]:
        """Execute the command."""
        return self.hla.execute_action(self.objects, self.params)


class PlanExecuteHighLevelAction(HighLevelAction):
    """Base class for high-level actions that follow planning and then
    execution pipeline."""

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        """Default implementation uses get_simulated_trajectory,
        get_robot_commands, and execute_robot_commands in sequence, but
        subclasses can override to modify their execution."""
        sim_traj = self.get_simulated_trajectory(objects, params)
        robot_commands = self.get_robot_commands(objects, params, sim_traj)
        if self._run_on_robot:
            self.execute_robot_commands(robot_commands)
        return sim_traj

    @abc.abstractmethod
    def get_simulated_trajectory(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        """Update the given simulator assuming that the action was executed."""

    def get_robot_commands(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
        sim_traj: list[FeedingDeploymentSimulatorState],
    ) -> list[KinovaCommand]:
        """Default implementation follows sim_traj exactly, but subclasses can
        override to modify their execution."""
        del objects, params  # not used
        return simulated_trajectory_to_kinova_commands(sim_traj)

    def execute_robot_commands(self, robot_commands: list[KinovaCommand]) -> None:
        """Execute the given commands on the robot."""
        for robot_command in robot_commands:
            self._robot_interface.execute_command(robot_command)


class PickToolHLA(PlanExecuteHighLevelAction):
    """Pick up a tool (utensil, drink, or wipe)."""

    def get_name(self) -> str:
        return "PickTool"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={LiftedAtom(GripperFree, [])},
            add_effects={Holding([tool])},
            delete_effects={LiftedAtom(GripperFree, [])},
        )

    def get_simulated_trajectory(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        assert len(objects) == 1
        tool = objects[0]
        if tool.name == "cup":
            nominal_plan = get_plan_to_grasp_cup(
                self._sim,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            )
        elif tool.name == "wiper":
            nominal_plan = get_plan_to_grasp_wiper(
                self._sim,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            )
        elif tool.name == "utensil":
            grasp_utensil_plan= 

            collision_ids = sim.get_collision_ids()
            if exclude_collision_ids is not None:
                collision_ids -= exclude_collision_ids

            plan = run_smooth_motion_planning_to_pose(
                target_pose,
                robot,
                collision_ids,
                finger_from_end_effector,
                seed,
                max_time=max_motion_plan_time,
                held_object=sim.held_object_id,
                base_link_to_held_obj=sim.held_object_tf,
            )

            nominal_plan = get_plan_to_grasp_utensil(
                self._sim,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            )
        else:
            print(f"PickTool not yet implemented for {tool}")
            return []
        remapped_plan = remap_trajectory_to_constant_distance(nominal_plan, self._sim)
        return remapped_plan


class StowToolHLA(PlanExecuteHighLevelAction):
    """Stow a tool (utensil, drink, or wipe)."""

    def get_name(self) -> str:
        return "StowTool"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={Holding([tool])},
            add_effects={LiftedAtom(GripperFree, [])},
            delete_effects={Holding([tool])},
        )

    def get_simulated_trajectory(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        # TODO
        assert len(objects) == 1
        tool = objects[0]
        if tool.name == "cup":
            nominal_plan = get_plan_to_stow_cup(
                self._sim,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            )
        elif tool.name == "wiper":
            nominal_plan = get_plan_to_stow_wiper(
                self._sim,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            )
        elif tool.name == "utensil":
            nominal_plan = get_plan_to_stow_utensil(
                self._sim,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            )
        else:
            print(f"StowTool not yet implemented for {tool}")
            return []
        remapped_plan = remap_trajectory_to_constant_distance(nominal_plan, self._sim)
        return remapped_plan


class TransferToolHLA(PlanExecuteHighLevelAction):
    """Wipe, or transfer drink, or transfer bite."""

    def get_name(self) -> str:
        return "TransferTool"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={Holding([tool]), ToolPrepared([tool])},
            add_effects={LiftedAtom(ToolTransferDone, [tool])},
            delete_effects=set(),
        )

    def get_simulated_trajectory(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        # TODO
        assert len(objects) == 1
        tool = objects[0]
        if tool.name == "cup":
            nominal_plan = get_plan_to_transfer_cup(
                self._sim,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            )
        elif tool.name == "wiper":
            nominal_plan = get_plan_to_transfer_wiper(
                self._sim,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            )
        # if tool.name == "utensil":
        #     forque_target_pose = (
        #         self._perception_interface.get_head_perception_forque_target_pose()
        #     )
        #     nominal_plan = get_bite_transfer_plan(
        #         forque_target_pose,
        #         self._sim,
        #         max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
        #     )
        #     remapped_plan = remap_trajectory_to_constant_distance(
        #         nominal_plan, self._sim
        #     )
        #     return remapped_plan
        else:
            print(f"TransferTool not yet implemented for {tool}")
            return []
        remapped_plan = remap_trajectory_to_constant_distance(nominal_plan, self._sim)
        return remapped_plan


class PrepareToolHLA(HighLevelAction):
    """Bite acquisition; other tools are always prepared."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Rajat todo: how do I initialize FLAIR just for the utensil tool?
        print("Initializing Acquisition Skill Library")
        if FLAIR_IMPORTED:
            self.acquisition_skill_library = SkillLibrary(self._robot_interface)

    def get_name(self) -> str:
        return "PrepareTool"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={Holding([tool])},
            add_effects={ToolPrepared([tool])},
            delete_effects=set(),
        )

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        assert len(objects) == 1
        tool = objects[0]
        if tool.name == "utensil" and FLAIR_IMPORTED:
            # Do Bite Acquisition
            print("Doing Bite Acquisition")
            self.acquisition_skill_library.reset()
            camera_color_data, camera_info_data, camera_depth_data, _ = (
                self._perception_interface.get_camera_data()
            )
            self.acquisition_skill_library.skewering_skill(
                camera_color_data, camera_depth_data, camera_info_data
            )

            # skill_library.scooping_skill(camera_color_data, camera_depth_data, camera_info_data)

            # skill_library.dipping_skill(camera_color_data, camera_depth_data, camera_info_data)

            # skill_library.pushing_skill(camera_color_data, camera_depth_data, camera_info_data)

            # skill_library.twirling_skill(camera_color_data, camera_depth_data, camera_info_data)

            # skill_library.cutting_skill(camera_color_data, camera_depth_data, camera_info_data)

        else:
            # Other tools are always prepared
            pass
        return []


def pddl_plan_to_hla_plan(
    pddl_plan: list[GroundOperator], hlas: set[HighLevelAction]
) -> list[GroundHighLevelAction]:
    """Convert a PDDL plan into a sequence of HLA and objects."""
    hla_plan = []
    op_to_hla = {hla.get_operator(): hla for hla in hlas}
    for ground_operator in pddl_plan:
        hla = op_to_hla[ground_operator.parent]
        objects = tuple(ground_operator.parameters)
        ground_hla = GroundHighLevelAction(hla, objects)
        hla_plan.append(ground_hla)
    return hla_plan
