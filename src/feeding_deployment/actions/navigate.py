import os
from pathlib import Path
from typing import Any, Tuple

import yaml

try:
    import actionlib
    import rospy
    from actionlib_msgs.msg import GoalStatus
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

    ROS_NAV_IMPORTED = True
except ModuleNotFoundError:
    ROS_NAV_IMPORTED = False

from relational_structs import (
    LiftedAtom,
    LiftedOperator,
    Object,
    Variable,
)
from feeding_deployment.actions.base import (
    HighLevelAction,
    nav_target_type,
    InFrontOf,
    DoorClosed,
    SafeToNavigate,
    GripperFree,
)


class NavigateHLA(HighLevelAction):
    """Navigate from one target to another."""

    _VALID_TARGETS = ("fridge", "microwave", "sink", "table")

    def _default_location_yaml(self) -> Path:
        return Path(__file__).resolve().parents[3] / "config" / "nav_named_locations.yaml"

    def _location_yaml(self) -> Path:
        user_path = os.environ.get("FEEDING_NAV_LOCATIONS_FILE", "").strip()
        if user_path:
            return Path(user_path).expanduser().resolve()
        return self._default_location_yaml()

    def _load_target_pose(self, location_name: str) -> dict[str, Any]:
        loc_file = self._location_yaml()
        if not loc_file.exists():
            raise FileNotFoundError(
                f"Named-location file not found at {loc_file}. "
                "Run capture_named_locations.py first."
            )

        with open(loc_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        locations = data.get("locations", {})
        if location_name not in locations:
            raise KeyError(
                f"Missing location '{location_name}' in {loc_file}. "
                f"Available locations: {sorted(locations)}"
            )

        target = locations[location_name]
        frame_id = target.get("frame_id", data.get("frame_id", "map"))
        position = target.get("position", {})
        orientation = target.get("orientation", {})
        return {
            "frame_id": frame_id,
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position.get("z", 0.0)),
            "qx": float(orientation["x"]),
            "qy": float(orientation["y"]),
            "qz": float(orientation["z"]),
            "qw": float(orientation["w"]),
        }

    def _speed_to_timeout(self, speed: str) -> float:
        return {
            "low": 300.0,
            "medium": 300.0,
            "high": 180.0,
        }.get(speed, 300.0)

    def _get_move_base_client(self):
        if not ROS_NAV_IMPORTED:
            raise RuntimeError("ROS navigation dependencies are not available")

        if not rospy.core.is_initialized():
            rospy.init_node(
                "feeding_deployment_navigate_hla",
                anonymous=True,
                disable_signals=True,
            )

        if not hasattr(self, "_move_base_client"):
            # Connect to the shared_autonomy_manager's "navigate" action server
            # (a transparent passthrough to move_base) rather than move_base
            # directly, so a human takeover can still report SUCCEEDED. Set the
            # FEEDING_NAV_ACTION env var to "move_base" to bypass the manager.
            nav_action = os.environ.get("FEEDING_NAV_ACTION", "navigate").strip()
            self._move_base_client = actionlib.SimpleActionClient(
                nav_action, MoveBaseAction
            )
            if not self._move_base_client.wait_for_server(rospy.Duration(15.0)):
                raise RuntimeError(
                    f"Timed out waiting for navigation action server '{nav_action}'"
                )
        return self._move_base_client

    def _navigate_to_target(self, location_name: str, speed: str) -> None:
        # if self.robot_interface is None:
        #     print(f"[SIM] Would navigate to {location_name} (speed={speed}).")
        #     return

        if not ROS_NAV_IMPORTED:
            raise RuntimeError(
                "ROS navigation modules not found. "
                "Please run in a ROS environment with move_base installed."
            )

        pose = self._load_target_pose(location_name)
        client = self._get_move_base_client()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = str(pose["frame_id"])
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = pose["x"]
        goal.target_pose.pose.position.y = pose["y"]
        goal.target_pose.pose.position.z = pose["z"]
        goal.target_pose.pose.orientation.x = pose["qx"]
        goal.target_pose.pose.orientation.y = pose["qy"]
        goal.target_pose.pose.orientation.z = pose["qz"]
        goal.target_pose.pose.orientation.w = pose["qw"]

        timeout_s = self._speed_to_timeout(speed)
        print(
            f"Navigating to {location_name} with timeout={timeout_s:.1f}s "
            f"using {self._location_yaml()} ..."
        )
        client.send_goal(goal)
        finished = client.wait_for_result(rospy.Duration(timeout_s))
        if not finished:
            client.cancel_goal()
            raise TimeoutError(
                f"Navigation to {location_name} timed out after {timeout_s:.1f}s"
            )

        state = client.get_state()
        if state != GoalStatus.SUCCEEDED:
            raise RuntimeError(
                f"move_base failed for {location_name}. Action state code={state}"
            )

        print(f"Reached {location_name}.")

    def get_name(self) -> str:
        return "Navigate"

    def get_operator(self) -> LiftedOperator:
        src = Variable("?from", nav_target_type)
        dst = Variable("?to", nav_target_type)

        return LiftedOperator(
            self.get_name(),
            parameters=[src, dst],
            preconditions={
                LiftedAtom(InFrontOf, [src]),
                LiftedAtom(SafeToNavigate, []),
                LiftedAtom(GripperFree, []),
            },
            add_effects={
                LiftedAtom(InFrontOf, [dst]),
            },
            delete_effects={
                LiftedAtom(InFrontOf, [src]),
            },
        )

    def get_behavior_tree_filename(
        self,
        objects: Tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params
        assert len(objects) == 2
        _, dst = objects
        assert self.sim.scene_description.scene_label == "vention"
        assert dst.name in self._VALID_TARGETS
        return f"navigate_to_{dst.name}.yaml"

    def navigate_to_fridge(self, speed: str) -> None:
        self._navigate_to_target("fridge", speed)

    def navigate_to_microwave(self, speed: str) -> None:
        self._navigate_to_target("microwave", speed)

    def navigate_to_sink(self, speed: str) -> None:
        self._navigate_to_target("sink", speed)

    def navigate_to_table(self, speed: str) -> None:
        self._navigate_to_target("table", speed)