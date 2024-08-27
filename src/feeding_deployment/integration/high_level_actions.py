"""High-level actions that we can simulate and execute."""

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import time

# Rajat ToDo: Remove this hacky addition
FLAIR_PATH = "/home/isacc/deployment_ws/src/FLAIR/bite_acquisition/scripts"
import sys

sys.path.append(FLAIR_PATH)
try:
    raise ModuleNotFoundError  # Just to skip this block
    from skill_library import SkillLibrary

    FLAIR_IMPORTED = True
except ModuleNotFoundError:
    FLAIR_IMPORTED = False
    pass


from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.motion_planning import (
    get_joint_positions_distance,
    run_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
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
from pybullet_helpers.gui import visualize_pose

from feeding_deployment.integration.low_level_actions import (
    move_to_joint_positions,
    teleport_to_ee_pose,
    move_to_ee_pose,
)
from feeding_deployment.integration.perception_interface import PerceptionInterface
from feeding_deployment.integration.utils import simulated_trajectory_to_kinova_commands
from feeding_deployment.robot_controller.arm_client import Arm
from feeding_deployment.robot_controller.command_interface import (
    CartesianCommand,
    CloseGripperCommand,
    JointCommand,
    KinovaCommand,
    OpenGripperCommand,
)
from feeding_deployment.simulation.planning import (
    _get_motion_plan_for_robot_finger_tip,
    _get_plan_to_execute_grasp,
    _get_plan_to_execute_ungrasp,
    _plan_to_sim_state_trajectory,
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

# Feeding:
# - Bite acquisition: move_joint(above_plate) -> move_ee(pickup food) -> move_joint(above plate)
# - Bite transfer: move_joint(transfer_pos) -> plan_to_pose(infront_mouth) -> plan_to_pose(inside_mouth) -> plan_to_pose(infront_mouth) -> plan_to_pose(transfer_pos)
# - Stow utensil: move_joint(utensil neutral) -> move_joint(outside mount) -> move_ee(inside mount) -> move_ee(above mount)


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

    # Rajat ToDo: Ask Tom if this is bad practice
    def execute_robot_commands(self, robot_commands: list[KinovaCommand]) -> None:
        """Execute the given commands on the robot."""
        for robot_command in robot_commands:
            input("Execute next command?")
            self._robot_interface.execute_command(robot_command)


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


class PickToolHLA(HighLevelAction):
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

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        assert len(objects) == 1
        tool = objects[0]

        if tool.name == "cup":

            assert self._sim.held_object_name is None
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.cup_outside_mount_pos,
                sim_states,
                robot_commands,
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.cup_inside_mount,
                self._sim.scene_description.cup_inside_mount_pos,
                sim_states,
                robot_commands,
            )

            # open gripperscup_inside_mount_pos
            robot_commands.append(OpenGripperCommand())
            # only for sim: set held object
            sim_states.extend(_get_plan_to_execute_grasp(self._sim, "cup"))

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.cup_above_mount,
                self._sim.scene_description.cup_above_mount_pos,
                sim_states,
                robot_commands,
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)

            return sim_states

        if tool.name == "utensil":

            assert self._sim.held_object_name is None
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_infront_mount_pos,
                sim_states,
                robot_commands,
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_above_mount_pos,
                sim_states,
                robot_commands,
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_inside_mount,
                self._sim.scene_description.utensil_inside_mount_pos,
                sim_states,
                robot_commands,
            )

            # open grippers
            robot_commands.append(OpenGripperCommand())
            # only for sim: set held object
            sim_states.extend(_get_plan_to_execute_grasp(self._sim, "utensil"))
            # input("Press Enter to continue...")

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_outside_mount,
                self._sim.scene_description.utensil_outside_mount_pos,
                sim_states,
                robot_commands,
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_neutral_pos,
                sim_states,
                robot_commands,
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)

            return sim_states

        else:
            print(f"PickTool not yet implemented for {tool}")
            return []


class StowToolHLA(HighLevelAction):
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

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        assert len(objects) == 1
        tool = objects[0]

        if tool.name == "cup":

            assert self._sim.held_object_name == "cup"
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.cup_above_mount_pos,
                sim_states,
                robot_commands,
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.cup_inside_mount,
                self._sim.scene_description.cup_inside_mount_pos,
                sim_states,
                robot_commands,
            )

            # close grippers
            robot_commands.append(CloseGripperCommand())
            # only for sim: unset held object
            sim_states.extend(_get_plan_to_execute_ungrasp(self._sim))

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.cup_outside_mount,
                self._sim.scene_description.cup_outside_mount_pos,
                sim_states,
                robot_commands,
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)

            return sim_states

        if tool.name == "utensil":

            assert self._sim.held_object_name == "utensil"
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_neutral_pos,
                sim_states,
                robot_commands,
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_outside_mount_pos,
                sim_states,
                robot_commands,
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_inside_mount,
                self._sim.scene_description.utensil_inside_mount_pos,
                sim_states,
                robot_commands,
            )

            # close grippers
            robot_commands.append(CloseGripperCommand())
            # only for sim: unset held object
            sim_states.extend(_get_plan_to_execute_ungrasp(self._sim))
            # input("Press Enter to continue...")

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_above_mount,
                self._sim.scene_description.utensil_above_mount_pos,
                sim_states,
                robot_commands,
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_infront_mount,
                self._sim.scene_description.utensil_infront_mount_pos,
                sim_states,
                robot_commands,
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)

            return sim_states

        else:
            print(f"StowTool not yet implemented for {tool}")
            return []


class TransferToolHLA(HighLevelAction):
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

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        assert len(objects) == 1
        tool = objects[0]

        if tool.name == "utensil":
            assert self._sim.held_object_name == "utensil"
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.before_transfer_pos,
                sim_states,
                robot_commands,
            )

            target_pose = self._perception_interface.get_head_perception_forque_target_pose()
            intermediate_pose = multiply_poses(
                target_pose, Pose(position=[0.0, 0.0, -0.1], orientation=[0.0, 0.0, 0.0, 1.0])
            ) # 10 cms away from the mouth

            visualize_pose(target_pose, self._sim.physics_client_id)
            input("Visualizing target pose. Press Enter to continue...")
            visualize_pose(intermediate_pose, self._sim.physics_client_id)
            input("Visualizing intermediate pose. Press Enter to continue...")

            move_to_ee_pose(sim=self._sim,
                target_pose=intermediate_pose,
                exclude_collision_ids=None,
                tip_from_end_effector=self._sim.scene_description.utensil_tip_from_end_effector,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
                sim_states=sim_states,
                robot_commands=robot_commands)

            move_to_ee_pose(sim=self._sim,
                target_pose=target_pose,
                exclude_collision_ids=None,
                tip_from_end_effector=self._sim.scene_description.utensil_tip_from_end_effector,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
                sim_states=sim_states,
                robot_commands=robot_commands)

            if self._run_on_robot:
                # Replay the trajectory before running on real robot
                input("Replaying the trajectory before running on real robot. Press Enter to continue...")

                for state in sim_states:
                    self._sim.sync(state)

                y = input("Does the trajectory look good? Press 'y' to execute on robot")
                if y == "y":
                    self.execute_robot_commands(robot_commands)
                else:
                    print("Trajectory not executed on robot")

            # Rajat ToDo: Implement the rest of bite transfer

            return sim_states
        elif tool.name == "cup":
            assert self._sim.held_object_name == "cup"
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.before_transfer_pos,
                sim_states,
                robot_commands,
            )

            target_pose = self._perception_interface.get_head_perception_forque_target_pose()
            print("target_pose", target_pose)

            intermediate_pose = multiply_poses(
                target_pose, Pose(position=[0.0, 0.0, -0.05], orientation=[0.0, 0.0, 0.0, 1.0])
            ) # 10 cms away from the mouth

            visualize_pose(target_pose, self._sim.physics_client_id)
            input("Visualizing target pose. Press Enter to continue...")
            visualize_pose(intermediate_pose, self._sim.physics_client_id)
            input("Visualizing intermediate pose. Press Enter to continue...")

            move_to_ee_pose(sim=self._sim,
                target_pose=intermediate_pose,
                exclude_collision_ids=None,
                tip_from_end_effector=self._sim.scene_description.drink_tip_from_end_effector,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
                sim_states=sim_states,
                robot_commands=robot_commands)

            move_to_ee_pose(sim=self._sim,
                target_pose=target_pose,
                exclude_collision_ids=None,
                tip_from_end_effector=self._sim.scene_description.drink_tip_from_end_effector,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
                sim_states=sim_states,
                robot_commands=robot_commands)

            if self._run_on_robot:
                # Replay the trajectory before running on real robot
                input("Replaying the trajectory before running on real robot. Press Enter to continue...")
                
                for state in sim_states:
                    self._sim.sync(state)
                    time.sleep(0.1)

                y = input("Does the trajectory look good? Press 'y' to execute on robot")
                if y == "y":
                    self.execute_robot_commands(robot_commands)
                else:
                    print("Trajectory not executed on robot")

            # Rajat ToDo: Implement the rest of cup transfer

            return sim_states

        print("Not implemented yet")


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

        if tool.name == "utensil":

            assert self._sim.held_object_name == "utensil"
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.above_plate_pos,
                sim_states,
                robot_commands,
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)

            if FLAIR_IMPORTED:
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

            return sim_states

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
