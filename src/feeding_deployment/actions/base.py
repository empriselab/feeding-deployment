"""High-level actions that we can simulate and execute."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from gymnasium.spaces import Space, Box, Text
from tomsutils.spaces import EnumSpace
import functools
from operator import attrgetter
import itertools
import string

import yaml

import numpy as np
import time

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.spaces import PoseSpace
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

from tomsutils.llm import LargeLanguageModel

from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.web_interface import WebInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.control.wrist_controller.wrist_controller import WristInterface
from feeding_deployment.control.robot_controller.command_interface import (
    CartesianCommand,
    CloseGripperCommand,
    JointCommand,
    KinovaCommand,
    OpenGripperCommand,
)

from feeding_deployment.simulation.planning import (
    _get_plan_to_execute_grasp,
    _get_plan_to_execute_ungrasp,
)
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentWorldState

# Define some predicates that can be used for sequencing the high-level actions.
tool_type = Type("tool")  # utensil, drink, or wiping tool
GripperFree = Predicate("GripperFree", [])  # not holding any tool
Holding = Predicate("Holding", [tool_type])  # holding tool
ToolTransferDone = Predicate("ToolTransferDone", [tool_type])  # wiped, drank, or ate
EmulateTransferDone = Predicate("EmulateTransferDone", [])  # emulated transfer
ToolPrepared = Predicate("ToolPrepared", [tool_type])  # e.g., bite acquired
PlateInView = Predicate("PlateInView", [])  # of the hand camera
ResetPos = Predicate("ResetPos", [])  # robot in reset position
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
        wrist_interface: WristInterface,
        flair,
        behavior_tree_dir: Path,
        no_waits=False,
        log_path=None
    ) -> None:
        self.sim = sim
        self.robot_interface = robot_interface
        self.perception_interface = perception_interface
        self.rviz_interface = rviz_interface
        self.web_interface = web_interface
        self.hla_hyperparams = hla_hyperparams
        self.wrist_interface = wrist_interface
        self.flair = flair
        self.behavior_tree_dir = behavior_tree_dir
        self.gesture_detector_filepath = self.behavior_tree_dir / "synthesized_gesture_detectors.py"
        self.no_waits = no_waits
        self.log_path = log_path
        # NOTE: assuming 7-dof and that first 7 entries are arm joints (not gripper).
        self.arm_joint_lower_limits = self.sim.robot.joint_lower_limits[:7]
        self.arm_joint_upper_limits = self.sim.robot.joint_upper_limits[:7]
        # Used for generating unique names for user-added nodes.
        self.new_node_counter = itertools.count()

    @abc.abstractmethod
    def get_name(self) -> str:
        """Get a human-readable name for this HLA."""

    @abc.abstractmethod
    def get_operator(self) -> LiftedOperator:
        """Create a planning operator for this HLA."""

    @abc.abstractmethod
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        """Get the YAML filename (not path, just name) for the behavior tree."""

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        """Execute the action on the robot."""
        bt_filename = self.get_behavior_tree_filename(objects, params)
        bt_filepath = self.behavior_tree_dir / bt_filename
        assert bt_filepath.exists()
        bt = load_behavior_tree(bt_filepath, self)
        bt.tick()

    def process_behavior_tree_node_addition(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
        new_node_type: str,
        new_node_parameters: dict[str, Any],
        anchor_node_name: str,
        before_or_after: str
    ) -> None:
        """Validate and add a new node into the behavior tree for this HLA."""
        # Create the dictionary for the new node.
        new_node_name = f"UserGeneratedNode{next(self.new_node_counter)}"
        new_node_dict = self.create_user_addition_node_dict(new_node_name, new_node_type, new_node_parameters)
        if new_node_dict is None:  # node addition was misspecified / not safe
            print(f"BT UPDATE FAILED in node addition (see error above).")
            return
        if before_or_after not in ("before", "after"):
            print(f"BT UPDATE FAILED. Invalid before or after: {before_or_after}")
            return
        # Create the node itself.
        new_node = _parse_node(new_node_dict)
        # Load the current behavior tree.
        bt_filename = self.get_behavior_tree_filename(objects, params)
        bt_filepath = self.behavior_tree_dir / bt_filename
        assert bt_filepath.exists()
        bt = load_behavior_tree(bt_filepath, self)
        # Get the anchor node.
        anchor_node = bt.get_node(anchor_node_name)
        if anchor_node is None or not isinstance(anchor_node, ParameterizedActionBehaviorTreeNode):
            print(f"BT UPDATE FAILED. Invalid node name: {anchor_node_name}")
            return
        # Special case: the anchor node is not part of a sequence.
        if not isinstance(anchor_node.parent, SequenceBehaviorTreeNode):
            # For now we'll assume that this case only happens when the node is
            # the root itself.
            assert anchor_node.parent is None
            assert anchor_node is bt
            # Wrap it in a sequence node.
            bt = SequenceBehaviorTreeNode("Autogen", "Autogenerated sequence node", [anchor_node])
        # Now the anchor node is part of a sequence.
        assert isinstance(anchor_node.parent, SequenceBehaviorTreeNode)
        # Add the node to the tree.
        anchor_node.parent.add_child(new_node, anchor_node, before_or_after)
        print(f"BT UPDATE SUCCEEDED! New {new_node_type} node added")
        # Write the change to disk.
        save_behavior_tree(bt, bt_filepath, self)

    def process_behavior_tree_parameter_update(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
        node_name: str,
        parameter_name: str,
        new_parameter_value: Any
    ) -> None:
        """Validate and update the behavior tree for this HLA."""
        # Load the current behavior tree.
        bt_filename = self.get_behavior_tree_filename(objects, params)
        bt_filepath = self.behavior_tree_dir / bt_filename
        assert bt_filepath.exists()
        bt = load_behavior_tree(bt_filepath, self)
        # Get the node.
        node = bt.get_node(node_name)
        if node is None or not isinstance(node, ParameterizedActionBehaviorTreeNode):
            print(f"BT UPDATE FAILED. Invalid node name: {node_name}")
            return
        # Get the parameter.
        parameter = node.get_parameter(parameter_name)
        if parameter is None:
            print(f"BT UPDATE FAILED. Invalid parameter name: {parameter_name}")
            return
        # Ensure the parameter is allowed to be edited.
        if not parameter.is_user_editable:
            print(f"BT UPDATE FAILED. Parameter not user editable: {parameter_name}")
            return
        # Ensure the new value is in bounds.
        if not parameter.space.contains(new_parameter_value):
            print(f"BT UPDATE FAILED. Parameter value is out of bounds: {new_parameter_value} for {parameter_name}")
            return
        # Update is valid! So update the tree.
        print(f"BT UPDATE SUCCEEDED! New: {new_parameter_value} for {parameter_name}")
        node.set_parameter(parameter, new_parameter_value)
        # Write the change to disk.
        save_behavior_tree(bt, bt_filepath, self)

    def create_user_addition_node_dict(
        self,
        new_node_name: str,
        new_node_type: str,
        new_node_parameters: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Validate and create a new node from a user request."""
        if new_node_type == "Pause":
            if "duration" not in new_node_parameters:
                print("BT UPDATE FAILED: missing parameter duration")
                return
            duration = new_node_parameters["duration"]
            min_allowed_pause = 0.0
            max_allowed_pause = 10.0
            if not min_allowed_pause <= duration <= max_allowed_pause:
                print(f"BT UPDATE FAILED: duration {duration} outside bounds")
                return
            return {
                "type": "Behavior",
                "name": new_node_name,
                "description": "User-added pause.",
                "parameters": [
                {
                    "name": "Duration",
                    "space": {
                        "type": "Box",
                        "lower": min_allowed_pause,
                        "upper": max_allowed_pause,
                    },
                    "description": "Pause duration.",
                    "is_user_editable": True,
                    "value": duration,
                },
                ],
                "fn": self.pause,
            }
        
        if new_node_type == "WaitForGesture":
            if "gesture_fn_name" not in new_node_parameters:
                print("BT UPDATE FAILED: missing parameter gesture_fn_name")
                return
            gesture_fn_name = new_node_parameters["gesture_fn_name"]
            return {
                "type": "Behavior",
                "name": new_node_name,
                "description": "User-added gesture detector.",
                "parameters": [
                {
                    "name": "GestureDetector",
                    "space": {
                        "type": "Text",
                    },
                    "description": "Gesture detection function name.",
                    "is_user_editable": True,
                    "value": gesture_fn_name,
                },
                ],
                "fn": self.wait_for_gesture,
            }


        print(f"BT UPDATE FAILED: invalid new node type {new_node_type}")
        return None

    def move_to_joint_positions(self, joint_positions: list[float]) -> None:
        
        plan = None 
        if not self.no_waits:
            plan = self.sim.plan_to_joint_positions(joint_positions)
        if self.robot_interface is None:
            self.sim.visualize_plan(plan)
        else:
            self.execute_robot_command(JointCommand(pos=joint_positions), plan)
            
    def move_to_ee_pose(self, pose: Pose) -> None:

        plan = None
        if not self.no_waits:
            plan = self.sim.plan_to_ee_pose(pose)
        if self.robot_interface is None:
            self.sim.visualize_plan(plan)
        else:
            self.execute_robot_command(CartesianCommand(pos=pose.position, quat=pose.orientation), plan)
    
    def grasp_tool(self, tool: str) -> None:
        self.sim.grasp_object(tool)
        if self.robot_interface is not None:
            self.execute_robot_command(OpenGripperCommand(), tool_update=tool)

    def ungrasp_tool(self, tool: str) -> None:
        self.sim.ungrasp_object()
        if self.robot_interface is not None:
            self.execute_robot_command(CloseGripperCommand(), tool_update=tool)

    def open_gripper(self) -> None:
        if self.robot_interface is None:
            self.sim.robot.open_fingers()
        else:
            self.execute_robot_command(OpenGripperCommand())
    
    def close_gripper(self) -> None:
        if self.robot_interface is None:
            self.sim.robot.close_fingers()
        else:
            self.execute_robot_command(CloseGripperCommand())

    def reset_wrist(self) -> None:
        if self.wrist_interface is not None:
            time.sleep(1.0) # wait for the utensil to be connected
            print("Resetting wrist controller ...")
            self.wrist_interface.set_velocity_mode()
            self.wrist_interface.reset()

    def pause(self, duration: float) -> None:
        time.sleep(duration)

    def wait_for_gesture(self, gesture_fn_name: str) -> None:
        # Create a local namespace with gesture detector code.
        local_namespace = globals().copy()

        # Add the saved gesture detection code to the local namespace.
        with open(self.gesture_detector_filepath, "r", encoding="utf-8") as f:
            gesture_detector_code_text = f.read()
        exec(gesture_detector_code_text, local_namespace)
        assert gesture_fn_name in local_namespace

        # Add constants that we will pass to the gesture detector.
        local_namespace["PERCEPTION_INTERFACE"] = self.perception_interface
        local_namespace["TIMEOUT"] = 5.0

        # Create a snippet that will actually run gesture detection in a loop.
        wait_for_gesture_code_text = f"""{gesture_fn_name}(PERCEPTION_INTERFACE, TIMEOUT)"""
        # Run the code.
        exec(wait_for_gesture_code_text, local_namespace)

    def execute_robot_command(self, robot_command: KinovaCommand, plan_viz: list[FeedingDeploymentWorldState] = None, tool_update: str = None) -> None:
        """Execute the given commands on the robot."""
        if self.robot_interface is None:
            raise ValueError("Robot interface is not available to execute commands.")
        
        if not self.no_waits:
            if tool_update is not None:
                self.rviz_interface.tool_update(True, tool_update, Pose((0, 0, 0), (0, 0, 0, 1))) # pickup the drink
            if plan_viz is not None:
                self.rviz_interface.visualize_plan(plan_viz)
            input("Execute next command?")
        self.robot_interface.execute_command(robot_command)


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

    def execute_action(self) -> None:
        """Execute the command."""
        self.hla.execute_action(self.objects, self.params)

    def process_behavior_tree_node_addition(self, new_node_type: str, new_node_parameters: dict[str, Any],
                                            anchor_node_name: str, before_or_after: str) -> None:
        """Validate and add a new node into the behavior tree."""
        self.hla.process_behavior_tree_node_addition(self.objects, self.params, new_node_type, new_node_parameters, anchor_node_name, before_or_after)

    def process_behavior_tree_parameter_update(self, node_name: str, parameter_name: str, new_parameter_value: Any) -> None:
        """Validate and update the behavior tree for this ground HLA."""
        self.hla.process_behavior_tree_parameter_update(self.objects, self.params, node_name, parameter_name, new_parameter_value)


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
    
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        # Behavior trees not used for this HLA
        raise NotImplementedError

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        assert len(objects) == 0
        assert self.sim.held_object_name is None

        self.move_to_joint_positions(self.sim.scene_description.retract_pos)

        # set FLAIR preferences to None
        if self.flair is not None:
            self.flair.clear_preference()


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


@dataclass(frozen=True)
class BehaviorTreeParameter:
    """A single parameter for a policy in a behavior tree.
    
    One policy may have multiple parameters.
    """
    
    name: str
    description: str
    space: Space
    is_user_editable: bool

    def __hash__(self):
        return hash(self.name)  # assume unique names
    
    def __eq__(self, other: Any):
        return isinstance(other, BehaviorTreeParameter) and other.name == self.name
    
    def get_yaml_dict(self) -> dict[str, Any]:
        """Get a YAML dict for this parameter."""
        space_dict = get_space_yaml_dict(self.space)
        return {
            "name": self.name,
            "description": self.description,
            "is_user_editable": self.is_user_editable,
            "space": space_dict
        }


class BehaviorTreeParameterizedPolicy(abc.ABC):
    """A parameterized policy for an action node in a behavior tree."""

    def __init__(self, name: str) -> None:
        self._name = name

    @abc.abstractmethod
    def get_parameters(self) -> list[BehaviorTreeParameter]:
        """Expose ordered parameters for this policy."""

    @abc.abstractmethod
    def get_function_name(self) -> str:
        """Get the name of the function for dumping to YAML."""

    def run(self, bindings: dict[BehaviorTreeParameter, Any]) -> None:
        """Run the policy given values for the parameters."""
        parameters = self.get_parameters()
        assert set(bindings) == set(parameters)
        ordered_parameter_values = []
        for parameter in parameters:
            value = bindings[parameter]
            assert parameter.space.contains(value), (
                f"Value {value} invalid for parameter {parameter}")
            ordered_parameter_values.append(value)
        
        print(f"Executing parameterized policy {self._name} with bindings:")
        for parameter, value in zip(parameters, ordered_parameter_values, strict=True):
            print(f"  {parameter.name} = {value}")

        self._execute(*ordered_parameter_values)

    @abc.abstractmethod
    def _execute(self, *args: Any) -> None:
        """Execute the policy given ordered values for the parameters."""


class FunctionalBehaviorTreeParameterizedPolicy(BehaviorTreeParameterizedPolicy):
    """A parameterized policy defined by a given function."""

    def __init__(self, name: str, parameters: list[BehaviorTreeParameter],
                 fn: Callable[[Any], None]) -> None:
        super().__init__(name)
        self._parameters = parameters
        self._fn = fn

    def get_parameters(self) -> list[BehaviorTreeParameter]:
        return list(self._parameters)

    def get_function_name(self) -> str:
        return self._fn.__name__
        
    def _execute(self, *args: Any) -> None:
        return self._fn(*args)
    

class BehaviorTreeNode(abc.ABC):
    """A node in a behavior tree."""

    # using for creating an execution log of the behavior tree
    _execution_log_path = Path(__file__).parent.parent / "integration" / "log" / "execution_log.txt"

    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description
        self.parent = None  # may be set by parent during tree construction

    @abc.abstractmethod
    def tick(self) -> bool:
        """Execute the node for one step and return a status."""

    @abc.abstractmethod
    def get_node(self, name: str) -> BehaviorTreeNode | None:
        """Get a node in this subtree with the given name."""

    @abc.abstractmethod
    def get_yaml_dict(self, hla: HighLevelAction) -> dict[str, Any]:
        """Get a dictionary to pass to yaml.dump()."""

    def log_start(self) -> None:
        """Log the start of execution."""
        with open(self._execution_log_path, 'a') as f:
            f.write(f"Starting node: {self._name}\n")

    def log_end(self) -> None:
        """Log the end of execution."""
        with open(self._execution_log_path, 'a') as f:
            f.write(f"Finished executing node: {self._name}\n")


class ParameterizedActionBehaviorTreeNode(BehaviorTreeNode):
    """A node in a behavior tree that executes a parameterized action open-loop.
    
    For now, the status after execution is not checked.

    Parameters may or may not be user-editable.
    """
    def __init__(self, name: str, description: str, policy: BehaviorTreeParameterizedPolicy,
                 bindings: dict[BehaviorTreeParameter, Any]) -> None:
        super().__init__(name, description)
        self._policy = policy
        self._bindings = bindings

    def tick(self) -> bool:
        self.log_start()
        self._policy.run(self._bindings)
        self.log_end()
        return True  # assume this worked

    def get_node(self, name: str) -> BehaviorTreeNode | None:
        if name == self._name:
            return self
        return None
    
    def get_yaml_dict(self, hla: HighLevelAction) -> dict[str, Any]:
        fn_name = self._policy.get_function_name()
        assert hasattr(hla, fn_name)
        fn_str = f"!hla {fn_name}"
        parameter_dicts = []
        for parameter in self._policy.get_parameters():
            parameter_dict = parameter.get_yaml_dict().copy()
            parameter_dict["value"] = self._bindings[parameter]
            parameter_dicts.append(parameter_dict)
        return {
            "name": self._name,
            "description": self._description,
            "type": "Behavior",
            "parameters": parameter_dicts,
            "fn": fn_str,
        }

    def get_parameter(self, name: str) -> BehaviorTreeParameter | None:
        """Access a policy parameter by name."""
        for parameter in self._policy.get_parameters():
            if parameter.name == name:
                return parameter
        return None
    
    def set_parameter(self, parameter: BehaviorTreeParameter, value: Any) -> None:
        """Update the parameter binding."""
        assert parameter.space.contains(value)
        self._bindings[parameter] = value


class SequenceBehaviorTreeNode(BehaviorTreeNode):
    """A sequence node in a behavior tree.
    
    For now, the status after execution is not checked.
    """
    def __init__(self, name: str, description: str, children: list[BehaviorTreeNode]) -> None:
        super().__init__(name, description)
        self._children = children
        for child in children:
            assert child.parent is None
            child.parent = self

    def tick(self) -> bool:
        self.log_start()
        for child in self._children:
            child.tick()
        self.log_end()
        return True  # assume everything worked

    def get_node(self, name: str) -> BehaviorTreeNode | None:
        for child in self._children:
            if child.get_node(name) is not None:
                return child
        return None

    def get_yaml_dict(self, hla: HighLevelAction) -> dict[str, Any]:
        child_dicts = [child.get_yaml_dict(hla) for child in self._children]
        return {
            "name": self._name,
            "description": self._description,
            "type": "Sequence",
            "children": child_dicts,
        }
    
    def add_child(self, node: BehaviorTreeNode, anchor_node: BehaviorTreeNode, before_or_after: str) -> None:
        """Add a new child node."""
        assert anchor_node in self._children
        anchor_index = self._children.index(anchor_node)
        if before_or_after == "before":
            new_child_index = anchor_index - 1
        elif before_or_after == "after":
            new_child_index = anchor_index + 1
        else:
            raise ValueError(f"Invalid {before_or_after=}")
        self._children.insert(new_child_index, node)
        node.parent = self


def _eval_expression(obj, loader, node):
    value = loader.construct_scalar(node)
    try:
        # Need to use attrgetter instead of getattr for nested attributes.
        return attrgetter(value)(obj)
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{value}': {e}")


def load_behavior_tree(filepath: Path, hla: HighLevelAction) -> BehaviorTreeNode:

    with open(filepath, "r", encoding="utf-8") as f:
        yaml_text = f.read()

    class CustomLoader(yaml.SafeLoader):
        pass

    CustomLoader.add_constructor('!hla', functools.partial(_eval_expression, hla))
    CustomLoader.add_constructor('!scene_description', functools.partial(_eval_expression,
                                                                         hla.sim.scene_description))
    root_dict = yaml.load(yaml_text, Loader=CustomLoader)
    return _parse_node(root_dict)


def save_behavior_tree(behavior_tree: BehaviorTreeNode, filepath: Path, hla: HighLevelAction) -> None:

    yaml_dict = behavior_tree.get_yaml_dict(hla)
    yaml_str = yaml.dump(yaml_dict, sort_keys=False)

    # Cannot find any better way to do this... the issue is that the YAML dumper
    # puts single quotes around the !hla expression, but then loading doesn't
    # work. I tried many things on both loading and dumping and then gave up.
    lines = []
    for line in yaml_str.splitlines():
        if "'!hla" in line:
            assert line.endswith("'")
            line = line[:-1]
            line = line.replace("'!hla", "!hla")
        lines.append(line)
    yaml_str = "\n".join(lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(yaml_str)


def _parse_node(node_dict: dict) -> BehaviorTreeNode:
    """Recursively parse a node from the loaded YAML structure."""
    node_name = node_dict["name"]
    node_description = node_dict["description"]
    node_type = node_dict["type"]

    if node_type == "Sequence":
        children_dicts = node_dict.get("children", [])
        children_nodes = [_parse_node(child) for child in children_dicts]
        return SequenceBehaviorTreeNode(node_name, node_description, children_nodes)

    elif node_type == "Behavior":
        params_list = node_dict.get("parameters", [])
        parameters = []
        bindings = {}
        for p in params_list:
            p_name = p["name"]
            p_description = p["description"]
            p_is_user_editable = p["is_user_editable"]
            space_spec = p["space"]
            space_type = space_spec["type"]
            if space_type == "Box":
                space = Box(np.array(space_spec["lower"]),
                            np.array(space_spec["upper"]),
                            dtype=np.float64)
            elif space_type == "Enum":
                space = EnumSpace(space_spec["elements"])
            elif space_type == "PoseSpace":
                space = PoseSpace()
            elif space_type == "Text":
                space = Text(max_length=100000000, charset=frozenset(string.printable))
            else:
                raise ValueError(f"Unrecognized space type: {space_type}")
            param_obj = BehaviorTreeParameter(
                name=p_name,
                description=p_description,
                space=space,
                is_user_editable=p_is_user_editable
            )
            parameters.append(param_obj)
            param_value = p["value"]
            bindings[param_obj] = param_value
        fn = node_dict["fn"]  
        policy = FunctionalBehaviorTreeParameterizedPolicy(
            name=node_name,
            parameters=parameters,
            fn=fn
        )
        return ParameterizedActionBehaviorTreeNode(node_name, node_description, policy, bindings)

    else:
        raise ValueError(f"Unknown node type: {node_type}")


def get_space_yaml_dict(space: Space) -> dict[str, Any]:
    """Get a YAML dict for a space."""
    if isinstance(space, Box):
        return {
            "type": "Box",
            "lower": space.low.tolist(),
            "upper": space.high.tolist(),
        }
            
    if isinstance(space, EnumSpace):
        return {
            "type": "Enum",
            "elements": space.elements,
        }
    
    if isinstance(space, PoseSpace):
        return {
            "type": "PoseSpace",
        }
    
    if isinstance(space, Text):
        return {
            "type": "Text",
        }

    raise ValueError(f"Unrecognized space type: {space}")


@dataclass(frozen=True)
class UserUpdateRequest:
    """A structured request for updating a behavior tree."""

    hla_name: str
    hla_object_names: tuple[str, ...]
    hla_parameters: None   # not currently used


@dataclass(frozen=True)
class NodeModificationUserUpdateRequest(UserUpdateRequest):
    """A request to change the parameters of an existing node."""

    node_name: str
    parameter_name: str
    new_value: Any


def interpret_user_update_request(
    request_txt: str,
    llm: LargeLanguageModel,
    available_hla_object_names: list[str],
    behavior_log_path,
) -> list[UserUpdateRequest]:
    """Use an LLM to convert natural language into a user update request."""

    # TODO refactor to avoid this copied code from query_llm.py. I'm not yet
    # sure where this code should live.
    bite = ["pick_utensil", "look_at_plate", "acquire_bite", "transfer_utensil", "stow_utensil"]
    drink = ["pick_drink", "transfer_drink", "stow_drink"]
    wipe = ["pick_wipe", "transfer_wipe", "stow_wipe"]

    all_nodes_description = ""
    
    # Load the behavior trees.
    all_nodes_description += "Bite:\n"
    for bite_node in bite:
        with open(behavior_log_path / f"{bite_node}.yaml", 'r') as f:
            node_description = f.read()
        all_nodes_description += node_description + "\n---\n"

    all_nodes_description += "Drink:\n"
    for drink_node in drink:
        with open(behavior_log_path / f"{drink_node}.yaml", 'r') as f:
            node_description = f.read()
        all_nodes_description += node_description + "\n---\n"

    all_nodes_description += "Wipe:\n"
    for wipe_node in wipe:
        with open(behavior_log_path / f"{wipe_node}.yaml", 'r') as f:
            node_description = f.read()
        all_nodes_description += node_description + "\n---\n"

    hla_object_name_str = "\n".join(available_hla_object_names)

    prompt = """Your job is to convert the following command into one or more structured outputs in a format that I will describe next.

The command is: %s

The available structured output types are:

@dataclass(frozen=True)
class UserUpdateRequest:
    hla_name: str
    hla_object_names: tuple[str, ...]
    hla_parameters: None   # not currently used

@dataclass(frozen=True)
class NodeModificationUserUpdateRequest(UserUpdateRequest):
    node_name: str
    parameter_name: str
    new_value: Any

The "hla" stands for high-level action. Each hla can be grounded with zero or more object names. The possible hla and object combinations are:
%s

IMPORTANT: make sure your hla_object_names and hla_name appear together in the list above!

The hla parameters can be ignored for now.

A NodeModificationUserUpdateRequest a request to modify one parameter for one node in a behavior tree associated with an hla.

Here are all behavior trees:
%s

Based on the information given, convert the original command into a list of one or more structured outputs.

Return your answer in a format where calling eval() in python will directly produce a list of UserUpdateRequest instances.
""" % (request_txt, hla_object_name_str, all_nodes_description)

    response = llm.sample_completions(prompt, imgs=None, temperature=0.0, seed=0)[0]
    if response.startswith("```python"):
        response = response[len("```python"):]
    if response.endswith("```"):
        response = response[:-len("```")]

    # This is really not safe but I'm not actually worried.
    pythonic_response = eval(response)

    return pythonic_response
