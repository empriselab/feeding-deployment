#!/usr/bin/env python3
"""
shared_autonomy_manager.py

The "manager" that sits in front of move_base for shared autonomy.

It exposes a `navigate` action server (move_base_msgs/MoveBaseAction) that looks
identical to move_base to its caller (NavigateHLA in actions/navigate.py, which
now connects to "navigate" instead of "move_base"). It forwards each goal to the
real move_base and relays the outcome, with one twist:

  * AUTONOMOUS (default): forward the goal to move_base; relay whatever move_base
    decides (SUCCEEDED -> succeeded, anything terminal-else -> aborted).
  * On /shared_autonomy/takeover: the human is taking over. Cancel the move_base
    goal (so it stops planning/driving and is no longer fighting the human) and
    switch to TELEOP. Localization (Cartographer / VIO) is NOT touched, so TF
    stays valid throughout.
  * On /shared_autonomy/done while in TELEOP: the human has parked the robot.
    Report SUCCEEDED to the caller ourselves ("blind success") -- move_base
    cannot, because it was cancelled.

This is why the manager exists: an action client only believes the terminal
status from the server it is connected to, so when a human (not move_base)
finishes the goal, *something* must be the server that can legitimately say
SUCCEEDED. That something is this node.
"""

import threading

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseResult
from std_msgs.msg import Empty

# State labels
AUTONOMOUS = "AUTONOMOUS"
TELEOP = "TELEOP"

_TERMINAL_FAILURE = (
    GoalStatus.PREEMPTED,
    GoalStatus.ABORTED,
    GoalStatus.REJECTED,
    GoalStatus.RECALLED,
    GoalStatus.LOST,
)


class SharedAutonomyManager:
    def __init__(self) -> None:
        self.navigate_action = rospy.get_param("~navigate_action", "navigate")
        self.move_base_action = rospy.get_param("~move_base_action", "move_base")
        self.takeover_topic = rospy.get_param(
            "~takeover_topic", "/shared_autonomy/takeover"
        )
        self.done_topic = rospy.get_param("~done_topic", "/shared_autonomy/done")
        self.loop_hz = float(rospy.get_param("~loop_hz", 20.0))
        self.move_base_wait_s = float(rospy.get_param("~move_base_wait_s", 30.0))

        # Edge flags set by topic callbacks, consumed in the execute loop.
        self._lock = threading.Lock()
        self._takeover_req = False
        self._done_req = False

        # Client to the real move_base.
        self.mb_client = actionlib.SimpleActionClient(
            self.move_base_action, MoveBaseAction
        )
        rospy.loginfo(
            "shared_autonomy_manager: waiting for '%s' action server...",
            self.move_base_action,
        )
        if not self.mb_client.wait_for_server(rospy.Duration(self.move_base_wait_s)):
            raise RuntimeError(
                f"Timed out waiting for move_base action server "
                f"'{self.move_base_action}'"
            )

        rospy.Subscriber(self.takeover_topic, Empty, self._on_takeover, queue_size=1)
        rospy.Subscriber(self.done_topic, Empty, self._on_done, queue_size=1)

        # Our own action server. auto_start=False so we can start() after setup.
        self.server = actionlib.SimpleActionServer(
            self.navigate_action,
            MoveBaseAction,
            execute_cb=self._execute,
            auto_start=False,
        )
        self.server.start()
        rospy.loginfo(
            "shared_autonomy_manager ready. Serving '%s', forwarding to '%s'.",
            self.navigate_action,
            self.move_base_action,
        )

    # ------------------------------------------------------------------ #
    # Topic callbacks
    # ------------------------------------------------------------------ #
    def _on_takeover(self, _msg: Empty) -> None:
        with self._lock:
            self._takeover_req = True

    def _on_done(self, _msg: Empty) -> None:
        with self._lock:
            self._done_req = True

    def _consume_flags(self):
        with self._lock:
            takeover, done = self._takeover_req, self._done_req
            self._takeover_req = False
            self._done_req = False
        return takeover, done

    # ------------------------------------------------------------------ #
    # Action execution
    # ------------------------------------------------------------------ #
    def _execute(self, goal) -> None:
        # Clear any stale intents from before this goal started.
        self._consume_flags()
        state = AUTONOMOUS

        rospy.loginfo("shared_autonomy_manager: new goal -> forwarding to move_base.")
        self.mb_client.send_goal(goal)

        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            # Upstream caller cancelled (e.g. NavigateHLA timed out and cancelled).
            if self.server.is_preempt_requested():
                self.mb_client.cancel_goal()
                rospy.logwarn("shared_autonomy_manager: caller preempted the goal.")
                self.server.set_preempted()
                return

            takeover, done = self._consume_flags()

            if takeover and state == AUTONOMOUS:
                state = TELEOP
                self.mb_client.cancel_goal()
                rospy.loginfo(
                    "shared_autonomy_manager: TAKEOVER -> move_base cancelled, "
                    "human in control. Waiting for 'done'."
                )

            if done and state == TELEOP:
                rospy.loginfo(
                    "shared_autonomy_manager: DONE -> reporting SUCCEEDED "
                    "(human-completed)."
                )
                self.server.set_succeeded(
                    MoveBaseResult(), "Goal completed by human teleoperation."
                )
                return

            if state == AUTONOMOUS:
                mb_state = self.mb_client.get_state()
                if mb_state == GoalStatus.SUCCEEDED:
                    rospy.loginfo(
                        "shared_autonomy_manager: move_base SUCCEEDED -> relaying."
                    )
                    result = self.mb_client.get_result() or MoveBaseResult()
                    self.server.set_succeeded(result, "move_base reached the goal.")
                    return
                if mb_state in _TERMINAL_FAILURE:
                    rospy.logwarn(
                        "shared_autonomy_manager: move_base ended in state %d "
                        "without a takeover -> aborting.",
                        mb_state,
                    )
                    result = self.mb_client.get_result() or MoveBaseResult()
                    self.server.set_aborted(
                        result, f"move_base terminated in state {mb_state}."
                    )
                    return

            rate.sleep()

        # rospy shutting down mid-goal.
        self.mb_client.cancel_goal()


def main() -> None:
    rospy.init_node("shared_autonomy_manager")
    SharedAutonomyManager()
    rospy.spin()


if __name__ == "__main__":
    main()
