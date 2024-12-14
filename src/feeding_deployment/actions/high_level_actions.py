"""High-level actions that we can simulate and execute."""

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation

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
from feeding_deployment.actions.outside_mouth_transfer import OutsideMouthTransfer

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
ResetPos = Predicate("ResetPos", [])  # robot in reset position
IsUtensil = Predicate("IsUtensil", [tool_type])

# Rajat ToDo: Move this to a config file
INSIDE_MOUTH_TRANSFER = False 
DISTANCE_INFRONT_MOUTH = 0.20

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
        no_waits=False,
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
        self.no_waits = no_waits

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
            if not self.no_waits:
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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )
            # close grippers
            robot_commands.append(CloseGripperCommand())



            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.drink_gaze_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            robot_commands = []

            drink_poses = self._perception_interface.perceive_drink_pickup_poses()

            # move_to_joint_positions(
            #     self._sim,
            #     self._sim.scene_description.drink_pre_staging_pos,
            #     sim_states,
            #     robot_commands,
            #     rviz_interface=self._rviz_interface if not self.no_waits else None,
            # )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.drink_staging_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            # NOTE: this just does nothing when executed and I don't know why

            # move_to_ee_pose(
            #     self._sim,
            #     drink_poses['pre_grasp_pose'],
            #     {self._sim.drink_id},
            #     Pose.identity(),
            #     max_motion_plan_time=10,
            #     sim_states=sim_states,
            #     robot_commands=robot_commands,
            #     rviz_interface=self._rviz_interface if not self.no_waits else None,
            # )

            teleport_to_ee_pose(
                self._sim,
                drink_poses['pre_grasp_pose'],
                None,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            teleport_to_ee_pose(
                self._sim,
                drink_poses['inside_bottom_pose'],
                None,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            teleport_to_ee_pose(
                self._sim,
                drink_poses['inside_top_pose'],
                None,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            # open grippers
            robot_commands.append(OpenGripperCommand())
            # only for sim: set held object
            sim_states.extend(_get_plan_to_execute_grasp(self._sim, "drink"))
            self._rviz_interface.tool_update(True, "drink", Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the drink

            teleport_to_ee_pose(
                self._sim,
                drink_poses['post_grasp_pose'],
                None,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            robot_commands = []

            self._perception_interface.record_drink_pickup_joint_pos()

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.before_transfer_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )
            # close grippers
            robot_commands.append(CloseGripperCommand())

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_inside_mount,
                self._sim.scene_description.utensil_inside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.before_transfer_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )
            # close grippers
            robot_commands.append(CloseGripperCommand())

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.wipe_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.wipe_inside_mount,
                self._sim.scene_description.wipe_inside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.wipe_outside_above_mount,
                self._sim.scene_description.wipe_outside_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.before_transfer_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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
                self._sim.scene_description.before_transfer_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            last_drink_poses, last_drink_pickup_joint_pos = self._perception_interface.get_last_drink_pickup_configs(study_poses=True)

            move_to_joint_positions(
                self._sim,
                last_drink_pickup_joint_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            teleport_to_ee_pose(
                self._sim,
                last_drink_poses['inside_top_pose'],
                None,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            # close grippers
            robot_commands.append(CloseGripperCommand())
            # only for sim: unset held object
            sim_states.extend(_get_plan_to_execute_ungrasp(self._sim))
            # update rviz
            self._rviz_interface.tool_update(False, "drink", self._sim.scene_description.drink_pose) # stow the drink

            teleport_to_ee_pose(
                self._sim,
                last_drink_poses['place_inside_bottom_pose'],
                None,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            teleport_to_ee_pose(
                self._sim,
                last_drink_poses['place_pre_grasp_pose'],
                None,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.drink_staging_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.utensil_outside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.utensil_inside_mount,
                self._sim.scene_description.utensil_inside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.wipe_outside_above_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.wipe_outside_mount,
                self._sim.scene_description.wipe_outside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            teleport_to_ee_pose(
                self._sim,
                self._sim.scene_description.wipe_inside_mount,
                self._sim.scene_description.wipe_inside_mount_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.retract_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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

        self.tool = None

        self.ready_for_transfer_interaction = "silent" # "silent", "voice" or "led"
        self.initiate_transfer_interaction = "open_mouth" # "button", "open_mouth" or "auto_timeout"
        self.transfer_complete_interaction = "sense" # "button", "sense" or "auto_timeout"
    
    def set_tool(self, tool):
        self.tool = tool

    def detect_initiate_transfer(self):
        if self.initiate_transfer_interaction == "button":
            self._perception_interface.detect_button_press()
        elif self.initiate_transfer_interaction == "open_mouth":
            self._perception_interface.detect_mouth_open()
        elif self.initiate_transfer_interaction == "auto_timeout":
            self._perception_interface.auto_timeout()
        print("Initiating transfer")

    def detect_transfer_complete(self):
        if self.transfer_complete_interaction == "button":
            self._perception_interface.detect_button_press()
        elif self.transfer_complete_interaction == "sense":
            if self.tool == "fork":
                self._perception_interface.detect_force_trigger()
            elif self.tool == "drink":
                self._perception_interface.detect_head_shake()
            elif self.tool == "wipe":
                self._perception_interface.detect_head_still()
        elif self.transfer_complete_interaction == "auto_timeout":
            self._perception_interface.auto_timeout()
        print("Detected transfer completion")

    def relay_ready_for_transfer(self):
        if self.ready_for_transfer_interaction == "silent":
            pass
        elif self.ready_for_transfer_interaction == "voice":
            self._perception_interface.speak("Please ooopen your mouth when ready")

    def publishTaskCommand(self, tip_pose):

        tip_pose[:3, :3] = Rotation.from_quat([0.478, -0.505, -0.515, 0.502]).as_matrix()

        # Rajat ToDo: Get this transform from pybullet (remove ROS dependency)
        if self.tool == "fork":
            tip_to_wrist = self.tf_utils.getTransformationFromTF('fork_tip', 'tool_frame')
        elif self.tool == "drink":
            tip_to_wrist = self.tf_utils.getTransformationFromTF('drink_tip', 'tool_frame')
        elif self.tool == "wipe":
            tip_to_wrist = self.tf_utils.getTransformationFromTF('wipe_tip', 'tool_frame')
        else:
            raise ValueError("Tool not recognized")
        tool_frame_target = tip_pose @ tip_to_wrist

        # self.visualizer.visualize_fork(tip_pose)
        self.tf_utils.publishTransformationToTF('base_link', 'tool_frame_target_viz', tool_frame_target)
      
        tool_frame_pos = tool_frame_target[:3,3].reshape(1,3).tolist()[0] # one dimensional list
        tool_frame_quat = Rotation.from_matrix(tool_frame_target[:3,:3]).as_quat()
        self.robot_interface.execute_command(CartesianCommand(tool_frame_pos, tool_frame_quat))

    def execute_transfer_loop(self, maintain_position_at_goal = False):
        
        assert self._perception_interface.head_perception_thread_is_running(), "Head perception thread is not running"
        assert self.tool is not None, "Tool is not set"

        self.relay_ready_for_transfer()
        self.detect_initiate_transfer()

        print("Setting state to 1")

        # move to infront of mouth
        forque_target_base = self._perception_interface.get_head_perception_tool_tip_target_pose()
        servo_point_forque_target = np.identity(4)
        servo_point_forque_target[:3,3] = np.array([0, 0, -DISTANCE_INFRONT_MOUTH]).reshape(1,3)
        infront_mouth_target = forque_target_base @ servo_point_forque_target
        self.publishTaskCommand(infront_mouth_target)

        self.detect_transfer_complete()
        # shutdown the head perception thread
        self._perception_interface.stop_head_perception_thread()

        # move to before transfer position
        final_target = self._perception_interface.get_tool_tip_pose_at_staging()
        self.publishTaskCommand(final_target)

        # incase for some reason the head perception thread is still running
        self._perception_interface.stop_head_perception_thread()            
        print("Exiting transfer loop")

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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            robot_commands = []

            if self.wrist_controller is not None:
                # stop the keep horizontal thread
                self.wrist_controller.stop_horizontal_spoon_thread()

            self._perception_interface.set_head_perception_tool("fork")
            self._perception_interface.start_head_perception_thread()
            time.sleep(5.0) # let head perception thread warmstart / robot to stabilize
            self._robot_interface.set_tool("fork")
            self.set_tool("fork")

            # Rajat Hack: Just to test interface
            if self._run_on_robot:

                if INSIDE_MOUTH_TRANSFER:
                    if not self.no_waits:
                        input("Press enter to switch to task compliant mode")
                    self._robot_interface.switch_to_task_compliant_mode()
                
                # Do inside-mouth transfer here
                self.execute_transfer_loop()

                if INSIDE_MOUTH_TRANSFER:                
                    if not self.no_waits:
                        input("Press enter to switch out of compliant mode")
                    self._robot_interface.switch_out_of_compliant_mode()

            # Send message to web interface indicating transfer is done.
            self._web_interface.send_web_interface_message({"state": "bite_transfer", "status": "completed"})

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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            robot_commands = []

            self._perception_interface.set_head_perception_tool("drink")
            self._perception_interface.start_head_perception_thread()
            time.sleep(5.0) # let head perception thread warmstart / robot to stabilize
            self._robot_interface.set_tool("drink")
            self.set_tool("drink")

            if self._run_on_robot:
                if INSIDE_MOUTH_TRANSFER:
                    if not self.no_waits:
                        input("Press enter to switch to task compliant mode")
                    self._robot_interface.switch_to_task_compliant_mode()
                
                # Do inside-mouth transfer here
                self.execute_transfer_loop(maintain_position_at_goal=True)

                if INSIDE_MOUTH_TRANSFER:
                    if not self.no_waits:
                        input("Press enter to switch out of compliant mode")
                    self._robot_interface.switch_out_of_compliant_mode()

            # Send message to web interface indicating transfer is done.
            self._web_interface.send_web_interface_message({"state": "drink_transfer", "status": "completed"})
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
                rviz_interface=self._rviz_interface if not self.no_waits else None
            )

            if self._run_on_robot:
                self.execute_robot_commands(robot_commands)
            robot_commands = []

            self._perception_interface.set_head_perception_tool("wipe")
            self._perception_interface.start_head_perception_thread()
            time.sleep(5.0) # let head perception thread warmstart / robot to stabilize
            self._robot_interface.set_tool("wipe")
            self.set_tool("wipe")

            if self._run_on_robot:
                if INSIDE_MOUTH_TRANSFER:
                    if not self.no_waits:
                        input("Press enter to switch to task compliant mode")
                    self._robot_interface.switch_to_task_compliant_mode()
                
                # Do inside-mouth transfer here
                self.execute_transfer_loop(maintain_position_at_goal=True)

                if INSIDE_MOUTH_TRANSFER:
                    if not self.no_waits:
                        input("Press enter to switch out of compliant mode")
                    self._robot_interface.switch_out_of_compliant_mode()

            # Send message to web interface indicating transfer is done.
            self._web_interface.send_web_interface_message({"state": "moved_to_wiping_position", "status": "completed"})
            return sim_states

        else:
            print(f"TransferTool not yet implemented for {tool}")
            return []


class LookAtPlateHLA(HighLevelAction):
    """Look at plate in preparation of bite acquisition."""

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

            # stop the keep horizontal thread (incase we're trying to re-acquire a bite)
            if self.wrist_controller is not None:
                self.wrist_controller.stop_horizontal_spoon_thread()

            move_to_joint_positions(
                self._sim,
                self._sim.scene_description.above_plate_pos,
                sim_states,
                robot_commands,
                rviz_interface=self._rviz_interface if not self.no_waits else None
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
                self.flair.set_food_items(['cantaloupe', 'banana'])
                # self.flair.set_food_items(['chicken nugget'])
                # self.flair.set_food_items(['banana'])
                items_detection = self.flair.detect_items(camera_color_data, camera_depth_data, camera_info_data, log_path=None)
                
                if not self.flair.is_preference_set():

                    # Handle one-time preference setting.
    
                    food_type_to_data = items_detection['food_type_to_bounding_boxes_plate']
                    n_food_types = len(food_type_to_data)
                    data = [{k: v} for k, v in food_type_to_data.items()]

                    food_types = food_type_to_data.keys()

                    # TODO: generalize this...
                    ordering_options = [f"Eat all the {food_type}s first" for food_type in food_types]
                    ordering_options += ["No preference"]

                    # # save plate image, plate bounds, and original image pickle
                    # import pickle
                    # with open("plate_log.pkl", "wb") as f:
                    #     plate_log = {
                    #         "plate_image": items_detection['plate_image'],
                    #         "plate_bounds": items_detection['plate_bounds'],
                    #         "original_image": camera_color_data,
                    #     }
                    #     pickle.dump(plate_log, f)

                    self._web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
                    time.sleep(0.2) # simulate delay for web interface
                    self._web_interface.send_web_interface_image(items_detection['plate_image'])
                    self._web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data})
                    self._web_interface.send_web_interface_message({"n_ordering": len(ordering_options), "data": ordering_options})

                    # save all of this in a pickle file:
                    # import pickle
                    # with open("meal_start_log.pkl", "wb") as f:
                    #     meal_start_log = {
                    #         "plate_image": items_detection['plate_image'],
                    #         "plate_bounds": items_detection['plate_bounds'],
                    #         "food_type_to_data": food_type_to_data,
                    #         "ordering_options": ordering_options,
                    #     }
                    #     pickle.dump(meal_start_log, f)

                    # self._web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
                    # time.sleep(1.0) # simulate delay, also needed for web interface
                    # self._web_interface.update_web_interface_image(items_detection['plate_image'])
                    # time.sleep(1.0)  # simulate delay, also needed for web interface
                    # self._web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data})
                    # self._web_interface.send_web_interface_message({"n_ordering": len(ordering_options), "data": ordering_options})

                    # Wait for web interface to report order selection.
                    print("WAITING TO GET PREFERENCE")
                    while self._web_interface.user_preference is None:
                        time.sleep(1e-1)
                    print("FINISHED GETTING PREFERENCES")
                    print("User Preference:", self._web_interface.user_preference)
                    self.flair.set_preference(self._web_interface.user_preference)
                # else:
                #     self._web_interface.send_web_interface_image(items_detection['plate_image'])
                #     time.sleep(1.0)  # simulate delay, also needed for web interface

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

                self._web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
                time.sleep(0.2) # simulate delay, needed for web interface
                self._web_interface.send_web_interface_image(items_detection['plate_image'])
                time.sleep(0.1)
                self._web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data, "current_bite": current_bite})            

                # import pickle
                # with open("next_bite_log.pkl", "wb") as f:
                #     next_bite_log = {
                #         "plate_image": items_detection['plate_image'],
                #         "plate_bounds": items_detection['plate_bounds'],
                #         "food_type_to_data": food_type_to_data,
                #         "next_food_item": next_food_item,
                #     }
                #     pickle.dump(next_bite_log, f)
  
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

            if self.flair is not None:

                print("Doing Bite Acquisition")
                camera_color_data, camera_info_data, camera_depth_data, _ = (
                    self._perception_interface.get_camera_data()
                )

                if params["status"] == 0:

                    detections = self.flair.get_items_detection()
                    plate_bounds = detections["plate_bounds"]
                    pos = params["positions"][0]

                    point_x = int(pos["x"]*plate_bounds[2]) + plate_bounds[0]
                    point_y = int(pos["y"]*plate_bounds[3]) + plate_bounds[1]

                    print("Plate Bounds:", plate_bounds)
                    print("Positions:", params["positions"])
                    print("Point:", point_x, point_y)

                    if not self.no_waits:
                        # visualize point on camera color image
                        viz = camera_color_data.copy()
                        for pos in params["positions"]:
                            cv2.circle(viz, (point_x, point_y), 5, (0, 255, 0), -1)
                        cv2.imshow("viz", viz)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    skewer_center = (point_x, point_y)
                    skewer_angle = -np.pi/2

                    self.flair.skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = skewer_center, major_axis = skewer_angle)

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
                    rviz_interface=self._rviz_interface if not self.no_waits else None
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

class ResetHLA(HighLevelAction):
    """Move the robot to retract position without any tool."""

    def get_name(self) -> str:
        return "Reset"

    def get_operator(self) -> LiftedOperator:
        return LiftedOperator(
            self.get_name(),
            parameters=[],
            preconditions={LiftedAtom(GripperFree, [])},
            add_effects={LiftedAtom(ResetPos, [])},
            delete_effects=set(),
        )

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> list[FeedingDeploymentSimulatorState]:
        assert len(objects) == 0
        assert self._sim.held_object_name is None
        sim_states: list[FeedingDeploymentSimulatorState] = []
        robot_commands = []

        move_to_joint_positions(
            self._sim,
            self._sim.scene_description.retract_pos,
            sim_states,
            robot_commands,
            rviz_interface=self._rviz_interface if not self.no_waits else None
        )

        if self._run_on_robot:
            self.execute_robot_commands(robot_commands)

        # set FLAIR preferences to None
        if self.flair is not None:
            self.flair.clear_preference()

        return sim_states
        

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
