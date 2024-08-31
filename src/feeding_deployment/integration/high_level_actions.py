"""High-level actions that we can simulate and execute."""

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
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
from pybullet_helpers.joint import JointPositions
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
from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.robot_controller.command_interface import (
    CloseGripperCommand,
    KinovaCommand,
    JointTrajectoryCommand,
    OpenGripperCommand,
)
from feeding_deployment.simulation.planning import (
    _get_plan_to_execute_grasp,
    _get_plan_to_execute_ungrasp,
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
        robot_interface: ArmInterfaceClient,
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
                perception_interface=self._perception_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.cup_outside_mount_pos,
                sim_states,
                robot_commands,
                perception_interface=self._perception_interface
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
            self._perception_interface.rviz_tool_update(True, "cup", Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the cup

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
                perception_interface=self._perception_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_infront_mount_pos,
                sim_states,
                robot_commands,
                perception_interface=self._perception_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_above_mount_pos,
                sim_states,
                robot_commands,
                perception_interface=self._perception_interface
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
            self._perception_interface.rviz_tool_update(True, "utensil", Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the utensil

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
                perception_interface=self._perception_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                perception_interface=self._perception_interface
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
                perception_interface=self._perception_interface
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
            # update rviz
            self._perception_interface.rviz_tool_update(False, "cup", self._sim.scene_description.cup_pose) # stow the cup

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
                perception_interface=self._perception_interface
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
                perception_interface=self._perception_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_neutral_pos,
                sim_states,
                robot_commands,
                perception_interface=self._perception_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_outside_mount_pos,
                sim_states,
                robot_commands,
                perception_interface=self._perception_interface
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
            self._perception_interface.rviz_tool_update(False, "utensil", self._sim.scene_description.utensil_pose) # stow the utensil

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
                perception_interface=self._perception_interface
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
                perception_interface=self._perception_interface
            )

            # # target_pose = self._perception_interface.get_head_perception_forque_target_pose()
            # target_pose = Pose(position=(-0.17272330207928777, 0.6273752674813526, 0.5572539925006535), 
            #     orientation=(-0.42030807,  0.56739361,  0.47188225, -0.52795148))
            # intermediate_pose = multiply_poses(
            #     target_pose, Pose(position=[0.0, 0.0, -0.1], orientation=[0.0, 0.0, 0.0, 1.0])
            # ) # 10 cms away from the mouth

            # visualize_pose(target_pose, self._sim.physics_client_id)
            # input("Visualizing target pose. Press Enter to continue...")
            # visualize_pose(intermediate_pose, self._sim.physics_client_id)
            # input("Visualizing intermediate pose. Press Enter to continue...")

            # # NOTE: disabling collision checking here between held object and
            # # conservative bounding box.
            # move_to_ee_pose(sim=self._sim,
            #     target_pose=intermediate_pose,
            #     exclude_collision_ids=None,
            #     tip_from_end_effector=self._sim.scene_description.utensil_tip_from_end_effector,
            #     max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            #     sim_states=sim_states,
            #     robot_commands=robot_commands,
            #     check_held_object_collisions=False)

            # # NOTE: disabling collision checking here between held object and
            # # conservative bounding box.
            # move_to_ee_pose(sim=self._sim,
            #     target_pose=target_pose,
            #     exclude_collision_ids=None,
            #     tip_from_end_effector=self._sim.scene_description.utensil_tip_from_end_effector,
            #     max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            #     sim_states=sim_states,
            #     robot_commands=robot_commands,
            #     check_held_object_collisions=False)

            if self._run_on_robot:
                # Replay the trajectory before running on real robot
                input("Replaying the trajectory before running on real robot. Press Enter to continue...")

                for state in sim_states:
                    self._sim.sync(state)

                y = input("Does the trajectory look good? Press 'y' to execute on robot")
                if y == "y":
                    # input("Press enter to switch to joint compliant mode")
                    # self._robot_interface.switch_to_joint_compliant_mode()
                    self.execute_robot_commands(robot_commands)
                    # input("Press enter to switch out of joint compliant mode")
                    # self._robot_interface.switch_out_of_joint_compliant_mode()
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
                perception_interface=self._perception_interface
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            # robot_commands = []

            # sim_length = len(sim_states)

            # input("Press enter to perceive the target pose")
            # # target_pose = self._perception_interface.get_head_perception_forque_target_pose()
            # target_pose = Pose(position=(-0.17272330207928777, 0.6273752674813526, 0.5572539925006535), 
            #     orientation=(-0.42030807,  0.56739361,  0.47188225, -0.52795148))
            # print("target_pose", target_pose)

            # intermediate_pose = multiply_poses(
            #     target_pose, Pose(position=[0.0, 0.0, -0.10], orientation=[0.0, 0.0, 0.0, 1.0])
            # ) # 10 cms away from the mouth

            # visualize_pose(target_pose, self._sim.physics_client_id)
            # input("Visualizing target pose. Press Enter to continue...")
            # visualize_pose(intermediate_pose, self._sim.physics_client_id)
            # input("Visualizing intermediate pose. Press Enter to continue...")

            # # NOTE: disabling collision checking here between held object and
            # # conservative bounding box.
            # move_to_ee_pose(sim=self._sim,
            #     target_pose=intermediate_pose,
            #     exclude_collision_ids=None,
            #     tip_from_end_effector=self._sim.scene_description.drink_tip_from_end_effector,
            #     max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            #     sim_states=sim_states,
            #     robot_commands=robot_commands,
            #     check_held_object_collisions=False)
            
            # # input("Replaying the trajectory to check. Press Enter to continue...")
            # # for i in range(sim_length, len(sim_states)):
            # #     self._sim.sync(sim_states[i])
            # #     time.sleep(0.1)
            #     # input("Press Enter to continue...")

            # # sim_length = len(sim_states)
            # # robot_command_length = len(robot_commands)

            # # NOTE: disabling collision checking here between held object and
            # # conservative bounding box.
            # # move_to_ee_pose(sim=self._sim,
            # #     target_pose=target_pose,
            # #     exclude_collision_ids=None,
            # #     tip_from_end_effector=self._sim.scene_description.drink_tip_from_end_effector,
            # #     max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            # #     sim_states=sim_states,
            # #     robot_commands=robot_commands,
            # #     check_held_object_collisions=False)
            
            # # input("Replaying the trajectory to check. Press Enter to continue...")
            # # for i in range(sim_length, len(sim_states)):
            # #     self._sim.sync(sim_states[i])
            # #     time.sleep(0.1)
            #     # input("Press Enter to continue...")

            # # Rajat ToDo: Replace this wait with a ROS listener for button.
            # input("Press enter when drinking is finished")
            
            # # Reverse the transfer plan.
            # transfer_sim_states = sim_states[sim_length:]
            # sim_states.extend(transfer_sim_states[::-1])
            
            # transfer_robot_commands = robot_commands.copy()
            # reversed_robot_commands = []
            # for command in transfer_robot_commands[::-1]:
            #     assert isinstance(command, JointTrajectoryCommand), "Command not a joint trajectory command"
            #     reversed_robot_commands.append(JointTrajectoryCommand(command.traj[::-1]))
            
            # robot_commands.extend(reversed_robot_commands)

            # for i in range(len(robot_commands)):
            #     assert isinstance(robot_commands[i], JointTrajectoryCommand), "Command not a joint trajectory command"
            #     assert np.allclose(robot_commands[i].traj, robot_commands[-(i+1)].traj[::-1]), "Robot commands not a palindrome"

            # # Replay the trajectory before running on real robot
            # print("Replaying the trajectory before running on real robot..")

            # for state in sim_states:
            #     self._sim.sync(state)
            #     time.sleep(0.1)

            # if self._run_on_robot:
            #     y = input("Does the trajectory look good? Press 'y' to execute on robot")
            #     if y == "y":
            #         # input("Press enter to switch to joint compliant mode")
            #         # self._robot_interface.switch_to_joint_compliant_mode()
            #         self.execute_robot_commands(robot_commands)
            #         # input("Press enter to switch out of joint compliant mode")
            #         # self._robot_interface.switch_out_of_joint_compliant_mode()
            #     else:
            #         print("Trajectory not executed on robot")

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
                perception_interface=self._perception_interface
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
