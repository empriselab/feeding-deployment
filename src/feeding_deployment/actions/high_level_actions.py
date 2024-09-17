"""High-level actions that we can simulate and execute."""

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import json
import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
import time


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

from feeding_deployment.actions.low_level_actions import (
    move_to_joint_positions,
    teleport_to_ee_pose,
    move_to_ee_pose,
)
from feeding_deployment.actions.inside_mouth_transfer import InsideMouthTransfer

from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.web_interface import WebInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
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
tool_type = Type("tool")  # utensil, drink, or wiping tool
GripperFree = Predicate("GripperFree", [])  # not holding any tool
Holding = Predicate("Holding", [tool_type])  # holding tool
ToolTransferDone = Predicate("ToolTransferDone", [tool_type])  # wiped, drank, or ate
ToolPrepared = Predicate("ToolPrepared", [tool_type])  # e.g., bite acquired
PlateInView = Predicate("PlateInView", [])  # of the hand camera
IsUtensil = Predicate("IsUtensil", [tool_type])

# Define high-level actions.
class HighLevelAction(abc.ABC):
    """Base class for high-level action."""

    def __init__(
        self,
        sim: FeedingDeploymentPyBulletSimulator,
        robot_interface: ArmInterfaceClient,
        perception_interface: PerceptionInterface,
        rviz_interface: RVizInterface,
        web_interface: WebInterface,
        hla_hyperparams: dict[str, Any],
        run_on_robot: bool,
        wrist_controller,
        flair,
    ) -> None:
        self._sim = sim
        self._robot_interface = robot_interface
        self._perception_interface = perception_interface
        self._rviz_interface = rviz_interface
        self._web_interface = web_interface
        self._hla_hyperparams = hla_hyperparams
        self._run_on_robot = run_on_robot
        self.wrist_controller = wrist_controller
        self.flair = flair

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

        if tool.name == "drink":

            assert self._sim.held_object_name is None
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.drink_outside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.drink_inside_mount,
                self._sim.scene_description.drink_inside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            # open grippers
            robot_commands.append(OpenGripperCommand())
            # only for sim: set held object
            sim_states.extend(_get_plan_to_execute_grasp(self._sim, "drink"))
            self._rviz_interface.tool_update(True, "drink", Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the drink

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.drink_above_mount,
                self._sim.scene_description.drink_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)

            # Send message to web interface.
            self._web_interface.send_web_interface_message({"state": "drink_pickup", "status": "completed"})

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
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_inside_mount,
                self._sim.scene_description.utensil_inside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            # open grippers
            robot_commands.append(OpenGripperCommand())
            # only for sim: set held object
            sim_states.extend(_get_plan_to_execute_grasp(self._sim, "utensil"))
            # input("Press Enter to continue...")
            self._rviz_interface.tool_update(True, "utensil", Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the utensil

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            robot_commands = []

            if self.wrist_controller is not None:
                time.sleep(1.0) # wait for the utensil to be connected
                print("Resetting wrist controller ...")
                self.wrist_controller.set_velocity_mode()
                self.wrist_controller.reset()

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_outside_mount,
                self._sim.scene_description.utensil_outside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_outside_above_mount,
                self._sim.scene_description.utensil_outside_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)

            return sim_states
        
        if tool.name == "wipe":

            assert self._sim.held_object_name is None
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.wipe_infront_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.wipe_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.wipe_inside_mount,
                self._sim.scene_description.wipe_inside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            # open grippers
            robot_commands.append(OpenGripperCommand())
            # only for sim: set held object
            sim_states.extend(_get_plan_to_execute_grasp(self._sim, "wipe"))
            # input("Press Enter to continue...")
            self._rviz_interface.tool_update(True, "wipe", Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the wipe

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.wipe_outside_mount,
                self._sim.scene_description.wipe_outside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.wipe_neutral_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)

            # Send message to web interface.
            self._web_interface.send_web_interface_message({"state": "prepare_mouth_wiping", "status": "completed"})

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

        if tool.name == "drink":

            assert self._sim.held_object_name == "drink"
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.drink_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.drink_inside_mount,
                self._sim.scene_description.drink_inside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            # close grippers
            robot_commands.append(CloseGripperCommand())
            # only for sim: unset held object
            sim_states.extend(_get_plan_to_execute_ungrasp(self._sim))
            # update rviz
            self._rviz_interface.tool_update(False, "drink", self._sim.scene_description.drink_pose) # stow the drink

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.drink_outside_mount,
                self._sim.scene_description.drink_outside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
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
                self._sim.scene_description.utensil_outside_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_outside_mount,
                self._sim.scene_description.utensil_outside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_inside_mount,
                self._sim.scene_description.utensil_inside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            # close grippers
            robot_commands.append(CloseGripperCommand())
            # only for sim: unset held object
            sim_states.extend(_get_plan_to_execute_ungrasp(self._sim))
            # input("Press Enter to continue...")
            self._rviz_interface.tool_update(False, "utensil", self._sim.scene_description.utensil_pose) # stow the utensil

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_above_mount,
                self._sim.scene_description.utensil_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)

            return sim_states
        
        if tool.name == "wipe":

            assert self._sim.held_object_name == "wipe"
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.wipe_neutral_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.wipe_outside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.wipe_inside_mount,
                self._sim.scene_description.wipe_inside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            # close grippers
            robot_commands.append(CloseGripperCommand())
            # only for sim: unset held object
            sim_states.extend(_get_plan_to_execute_ungrasp(self._sim))
            # input("Press Enter to continue...")
            self._rviz_interface.tool_update(False, "wipe", self._sim.scene_description.wipe_pose) # stow the wipe

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.wipe_above_mount,
                self._sim.scene_description.wipe_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.wipe_infront_mount,
                self._sim.scene_description.wipe_infront_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)

            return sim_states

        else:
            print(f"StowTool not yet implemented for {tool}")
            return []


class TransferToolHLA(HighLevelAction):
    """Wipe, or transfer drink, or transfer bite."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inside_mouth_transfer = InsideMouthTransfer(perception_interface=self._perception_interface, robot_interface=self._robot_interface, rviz_interface=self._rviz_interface)

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
                rviz_interface=self._rviz_interface
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            robot_commands = []

            if self.wrist_controller is not None:
                # stop the keep horizontal thread
                self.wrist_controller.stop_horizontal_spoon_thread()

            if self._run_on_robot:
                input("Press enter to switch to task compliant mode")
                self._robot_interface.switch_to_task_compliant_mode()
                
                # Do inside-mouth transfer here
                self.inside_mouth_transfer.execute_transfer_loop()

                input("Press enter to switch out of compliant mode")
                self._robot_interface.switch_out_of_compliant_mode()

            return sim_states
        
        elif tool.name == "drink":

            assert self._sim.held_object_name == "drink"
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.before_transfer_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            robot_commands = []

            return sim_states

            sim_length = len(sim_states)

            # input("Press enter to perceive the target pose")
            # target_pose = self._perception_interface.get_head_perception_tool_tip_target_pose()
            target_pose = Pose(position=(-0.17272330207928777, 0.6273752674813526, 0.5572539925006535), 
                orientation=(-0.42030807,  0.56739361,  0.47188225, -0.52795148))
            print("target_pose", target_pose)

            intermediate_pose = multiply_poses(
                target_pose, Pose(position=[0.0, 0.0, -0.10], orientation=[0.0, 0.0, 0.0, 1.0])
            ) # 10 cms away from the mouth

            visualize_pose(target_pose, self._sim.physics_client_id)
            # input("Visualizing target pose. Press Enter to continue...")
            visualize_pose(intermediate_pose, self._sim.physics_client_id)
            # input("Visualizing intermediate pose. Press Enter to continue...")

            # NOTE: disabling collision checking here between held object and
            # conservative bounding box.
            move_to_ee_pose(sim=self._sim,
                target_pose=intermediate_pose,
                exclude_collision_ids=None,
                tip_from_end_effector=self._sim.scene_description.drink_tip_from_end_effector,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
                sim_states=sim_states,
                robot_commands=robot_commands,
                rviz_interface=self._rviz_interface,
                check_held_object_collisions=False)
            
            # input("Replaying the trajectory to check. Press Enter to continue...")
            # for i in range(sim_length, len(sim_states)):
            #     self._sim.sync(sim_states[i])
            #     time.sleep(0.1)
                # input("Press Enter to continue...")

            # sim_length = len(sim_states)
            # robot_command_length = len(robot_commands)

            # NOTE: disabling collision checking here between held object and
            # conservative bounding box.
            # move_to_ee_pose(sim=self._sim,
            #     target_pose=target_pose,
            #     exclude_collision_ids=None,
            #     tip_from_end_effector=self._sim.scene_description.drink_tip_from_end_effector,
            #     max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            #     sim_states=sim_states,
            #     robot_commands=robot_commands,
            #     rviz_interface=self._rviz_interface,
            #     check_held_object_collisions=False)
            
            # input("Replaying the trajectory to check. Press Enter to continue...")
            # for i in range(sim_length, len(sim_states)):
            #     self._sim.sync(sim_states[i])
            #     time.sleep(0.1)
                # input("Press Enter to continue...")

            if self._rviz_interface is not None:
                for sim_state in sim_states:
                    self._rviz_interface.joint_state_update(sim_state.robot_joints)
                    time.sleep(0.1)

            if self._run_on_robot:
                y = input("Does the trajectory look good? Press 'y' to execute on robot")
                if y == "y":
                    input("Press enter to switch to joint compliant mode")
                    self._robot_interface.switch_to_task_compliant_mode()
                    self.execute_robot_commands(robot_commands)
                    input("Press enter to switch out of joint compliant mode")
                    self._robot_interface.switch_out_of_compliant_mode()
                else:
                    print("Trajectory not executed on robot")
            
            # Wait for button press to indicate that transfer is finished.
            self._perception_interface.wait_for_user_continue_button()

            self._web_interface.send_web_interface_message({"state": "drink_transfer", "status": "completed"})

            # Reverse the transfer plan.
            transfer_sim_states = sim_states[sim_length:]
            sim_states.extend(transfer_sim_states[::-1])

            if self._rviz_interface is not None:
                for sim_state in transfer_sim_states[::-1]:
                    self._rviz_interface.joint_state_update(sim_state.robot_joints)
                    time.sleep(0.1)
            
            transfer_robot_commands = robot_commands.copy()
            reversed_robot_commands = []
            for command in transfer_robot_commands[::-1]:
                assert isinstance(command, JointTrajectoryCommand), "Command not a joint trajectory command"
                reversed_robot_commands.append(JointTrajectoryCommand(command.traj[::-1]))
            
            robot_commands.extend(reversed_robot_commands)

            for i in range(len(robot_commands)):
                assert isinstance(robot_commands[i], JointTrajectoryCommand), "Command not a joint trajectory command"
                assert np.allclose(robot_commands[i].traj, robot_commands[-(i+1)].traj[::-1]), "Robot commands not a palindrome"

            if self._run_on_robot:
                # y = input("Does the trajectory look good? Press 'y' to execute on robot")
                y = "n"
                if y == "y":
                    input("Press enter to switch to joint compliant mode")
                    self._robot_interface.switch_to_task_compliant_mode()
                    self.execute_robot_commands(reversed_robot_commands)
                    input("Press enter to switch out of joint compliant mode")
                    self._robot_interface.switch_out_of_compliant_mode()
                else:
                    print("Trajectory not executed on robot")

            return sim_states
        
        elif tool.name == "wipe":

            assert self._sim.held_object_name == "wipe"
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.before_transfer_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            robot_commands = []

            
            return sim_states

            sim_length = len(sim_states)

            # input("Press enter to perceive the target pose")
            # target_pose = self._perception_interface.get_head_perception_tool_tip_target_pose()
            target_pose = Pose(position=(-0.17272330207928777, 0.6273752674813526, 0.5572539925006535), 
                orientation=(-0.42030807,  0.56739361,  0.47188225, -0.52795148))
            print("target_pose", target_pose)

            intermediate_pose = multiply_poses(
                target_pose, Pose(position=[0.0, 0.0, -0.10], orientation=[0.0, 0.0, 0.0, 1.0])
            ) # 10 cms away from the mouth

            visualize_pose(target_pose, self._sim.physics_client_id)
            # input("Visualizing target pose. Press Enter to continue...")
            visualize_pose(intermediate_pose, self._sim.physics_client_id)
            # input("Visualizing intermediate pose. Press Enter to continue...")

            # NOTE: disabling collision checking here between held object and
            # conservative bounding box.
            move_to_ee_pose(sim=self._sim,
                target_pose=intermediate_pose,
                exclude_collision_ids=None,
                tip_from_end_effector=self._sim.scene_description.wipe_tip_from_end_effector,
                max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
                sim_states=sim_states,
                robot_commands=robot_commands,
                rviz_interface=self._rviz_interface,
                check_held_object_collisions=False)
            
            # input("Replaying the trajectory to check. Press Enter to continue...")
            # for i in range(sim_length, len(sim_states)):
            #     self._sim.sync(sim_states[i])
            #     time.sleep(0.1)
                # input("Press Enter to continue...")

            # sim_length = len(sim_states)
            # robot_command_length = len(robot_commands)

            # NOTE: disabling collision checking here between held object and
            # conservative bounding box.
            # move_to_ee_pose(sim=self._sim,
            #     target_pose=target_pose,
            #     exclude_collision_ids=None,
            #     tip_from_end_effector=self._sim.scene_description.wipe_tip_from_end_effector,
            #     max_motion_plan_time=self._hla_hyperparams["max_motion_planning_time"],
            #     sim_states=sim_states,
            #     robot_commands=robot_commands,
            #     rviz_interface=self._rviz_interface,
            #     check_held_object_collisions=False)
            
            # input("Replaying the trajectory to check. Press Enter to continue...")
            # for i in range(sim_length, len(sim_states)):
            #     self._sim.sync(sim_states[i])
            #     time.sleep(0.1)
                # input("Press Enter to continue...")


            if self._rviz_interface is not None:
                for sim_state in sim_states:
                    self._rviz_interface.joint_state_update(sim_state.robot_joints)
                    time.sleep(0.1)

            if self._run_on_robot:
                y = input("Does the trajectory look good? Press 'y' to execute on robot")
                if y == "y":
                    input("Press enter to switch to joint compliant mode")
                    self._robot_interface.switch_to_task_compliant_mode()
                    self.execute_robot_commands(robot_commands)
                    input("Press enter to switch out of joint compliant mode")
                    self._robot_interface.switch_out_of_compliant_mode()
                else:
                    print("Trajectory not executed on robot")
            
            # Wait for button press to indicate that transfer is finished.
            self._perception_interface.wait_for_user_continue_button()

            self._web_interface.send_web_interface_message({"state": "moved_to_wiping_position", "status": "completed"})

            # Reverse the transfer plan.
            transfer_sim_states = sim_states[sim_length:]
            sim_states.extend(transfer_sim_states[::-1])

            if self._rviz_interface is not None:
                for sim_state in transfer_sim_states[::-1]:
                    self._rviz_interface.joint_state_update(sim_state.robot_joints)
                    time.sleep(0.1)
            
            transfer_robot_commands = robot_commands.copy()
            reversed_robot_commands = []
            for command in transfer_robot_commands[::-1]:
                assert isinstance(command, JointTrajectoryCommand), "Command not a joint trajectory command"
                reversed_robot_commands.append(JointTrajectoryCommand(command.traj[::-1]))
            
            robot_commands.extend(reversed_robot_commands)

            for i in range(len(robot_commands)):
                assert isinstance(robot_commands[i], JointTrajectoryCommand), "Command not a joint trajectory command"
                assert np.allclose(robot_commands[i].traj, robot_commands[-(i+1)].traj[::-1]), "Robot commands not a palindrome"

            if self._run_on_robot:
                # y = input("Does the trajectory look good? Press 'y' to execute on robot")
                y = "n"
                if y == "y":
                    input("Press enter to switch to joint compliant mode")
                    self._robot_interface.switch_to_task_compliant_mode()
                    self.execute_robot_commands(reversed_robot_commands)
                    input("Press enter to switch out of joint compliant mode")
                    self._robot_interface.switch_out_of_compliant_mode()
                else:
                    print("Trajectory not executed on robot")

            return sim_states

        print("Not implemented yet")


class LookAtPlateHLA(HighLevelAction):
    """Look at plate in preparation of bite acquisition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preferences_set = False

    def get_name(self) -> str:
        return "LookAtPlate"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={Holding([tool]), IsUtensil([tool])},
            add_effects={LiftedAtom(PlateInView, [])},
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
            
            print("In LookAtPlateHLA")
            assert self._sim.held_object_name == "utensil"
            sim_states: list[FeedingDeploymentSimulatorState] = []
            robot_commands = []

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.above_plate_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface
            )
            

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            
            if self.wrist_controller is not None:
                self.wrist_controller.set_velocity_mode()
                self.wrist_controller.reset()

            if self.flair is not None:

                # Run FLAIR perception.
                camera_color_data, camera_info_data, camera_depth_data, _ = (
                    self._perception_interface.get_camera_data()
                )

                items = self.flair.identify_plate(camera_color_data)
                # flair.set_food_items(items)
                # self.flair.set_food_items(['banana', 'baby carrot'])
                self.flair.set_food_items(['cantaloupe'])
                items_detection = self.flair.detect_items(camera_color_data, camera_depth_data, camera_info_data, log_path=None)
                
                if not self._preferences_set:

                    # Handle one-time preference setting.
    
                    food_type_to_data = items_detection['food_type_to_bounding_boxes_plate']
                    n_food_types = len(food_type_to_data)
                    data = [{k: v} for k, v in food_type_to_data.items()]

                    food_types = food_type_to_data.keys()

                    # TODO: generalize this...
                    ordering_options = [f"Eat all the {food_type}s first" for food_type in food_types]
                    ordering_options += ["No preference"]

                    self._web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
                    time.sleep(1.0) # simulate delay, also needed for web interface
                    self._web_interface.update_web_interface_image(items_detection['plate_image'])
                    time.sleep(1.0)  # simulate delay, also needed for web interface
                    self._web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data})
                    self._web_interface.send_web_interface_message({"n_ordering": len(ordering_options), "data": ordering_options})

                    # Wait for web interface to report order selection.
                    print("WAITING TO GET PREFERENCE")
                    while self._web_interface.user_preference is None:
                        time.sleep(1e-1)
                    print("FINISHED GETTING PREFERENCES")

                    self.flair.set_preferences(self._web_interface.user_preference)
                    self._preferences_set = True

                # Prepare for bite acquisition.
                print("Doing Bite Acquisition")
                self.wrist_controller.set_velocity_mode()
                self.wrist_controller.reset()

                next_action_prediction = self.flair.predict_next_action(camera_color_data, items_detection=None, log_path=None)

                next_food_item = next_action_prediction['labels_list'][next_action_prediction['food_id']]
                bite_mask_idx = next_action_prediction['bite_mask_idx']
                print(" --- Next Food Item Prediction:", next_action_prediction['labels_list'][next_action_prediction['food_id']])
                print(" --- Next Action Prediction:", next_action_prediction['action_type'])

                # remove next_food_item from data
                food_type_to_data = items_detection['food_type_to_bounding_boxes_plate']

                n_food_types = len(food_type_to_data)
                data = [{k: v} for k, v in food_type_to_data.items() if k != next_food_item]
                current_bite = {next_food_item: food_type_to_data[next_food_item]}

                self._web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data, "current_bite": current_bite})            
  
            else:
                # Test image.
                rng = np.random.default_rng(123)
                camera_color_data = rng.integers(0, 255, size=(512, 512, 3))

            return sim_states

        else:
            # Other tools are always prepared
            pass
        return []


class AcquireBiteHLA(HighLevelAction):
    """Bite acquisition; other tools are always prepared."""

    def get_name(self) -> str:
        return "AcquireBite"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={Holding([tool]), IsUtensil([tool]), LiftedAtom(PlateInView, [])},
            add_effects={ToolPrepared([tool])},
            delete_effects={LiftedAtom(PlateInView, [])},
        )

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        assert len(objects) == 1
        tool = objects[0]

        if tool.name == "utensil":

            print("params", params)
            input("Press Enter to continue...")

            if self.flair is not None:

                print("Doing Bite Acquisition")
                camera_color_data, camera_info_data, camera_depth_data, _ = (
                    self._perception_interface.get_camera_data()
                )

                if params["status"] == 0:

                    detections = self.flair.get_items_detection()
                    plate_bounds = detections["plate_bounds"]

                    skewer_center = (int(params["positions"][0]["x"] + plate_bounds[0]), int(params["positions"][0]["y"] + plate_bounds[1]))
                    skewer_angle = 0

                    self.flair.skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = skewer_center, major_axis = skewer_angle)

                    # Manual - execute action from web interface
                    # action_type = "Skewer"
                    # self.flair.execute_manual_action(action_type, camera_color_data, camera_depth_data, camera_info_data):

                elif params["status"] == "aquire_food":
                    detections = self.flair.get_items_detection()
                    food_type_to_masks = detections["food_type_to_masks"]
                    food_type_to_skill = detections["food_type_to_skill"]
                    
                    food_type = params["data"][0]
                    item_id = params["data"][1] - 1

                    mask = food_type_to_masks[food_type][item_id]
                    skill = food_type_to_skill[food_type]

                    if skill == "Skewer":
                        skewer_point, skewer_angle = self.flair.inference_server.get_skewer_action(mask)
                        self.flair.skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = skewer_point, major_axis = skewer_angle)
                    elif skill == "Scoop":
                        raise NotImplementedError("Scoop skill not yet implemented")

                sim_states: list[FeedingDeploymentSimulatorState] = []
                robot_commands = []

                move_to_joint_positions(
                    self._sim,
                    self._sim.scene_description.above_plate_pos,
                    sim_states,
                    robot_commands,
                    rviz_interface=self._rviz_interface
                )

                if self._run_on_robot:
                    self.execute_robot_commands(robot_commands)

                # set the wrist controller to always keep utensil horizontal
                if self.wrist_controller is not None:
                    self.wrist_controller.start_horizontal_spoon_thread()
    
            else:
                time.sleep(2.0)  # simulate delay, also needed for web interface

            # Send message to web interface indicating that robot is done with acquisition.
            self._web_interface.send_web_interface_message({"state": "bite_pickup", "status": "completed"})

            return []

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
