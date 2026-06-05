#!/usr/bin/env python3
"""
shared_autonomy_teleop.py

Xbox-controller teleop for the Vention base, for use in SHARED AUTONOMY.

Unlike vention_teleop_controller.py (which opens the Arduino serial port directly
and therefore cannot coexist with autonomous navigation), this node does NOT
touch the motors. It only:

  * reads the Xbox controller (via pygame, same as the standalone script), and
  * publishes geometry_msgs/Twist to /cmd_vel ONLY while the deadman is held, and
  * emits two high-level intents to the shared_autonomy_manager:
        - /shared_autonomy/takeover (std_msgs/Empty) on the rising edge of the
          deadman button  -> "human is taking over, cancel the autopilot"
        - /shared_autonomy/done     (std_msgs/Empty) on the rising edge of the
          done button     -> "human has parked the robot, report success"

Because it publishes Twist (instead of grabbing serial), it can run side-by-side
with autonomous navigation. When the deadman is not held it publishes nothing,
so it has zero effect on the robot.

Safety:
  * Deadman: motion is published only while the deadman button is held. On
    release we publish a single zero Twist, then go silent; the bridge watchdog
    keeps the base stopped.
  * Velocity is clamped to the autonomy's limits (max_vel_x / max_vel_theta) so a
    human cannot out-drive the safe envelope.
  * A missing/disconnected controller never blocks autonomy: the node stays
    dormant and keeps trying to (re)connect.
"""

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

import pygame


def apply_deadband(value: float, deadband: float) -> float:
    if abs(value) < deadband:
        return 0.0
    return value


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SharedAutonomyTeleop:
    def __init__(self) -> None:
        # ---- Topics ----
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.takeover_topic = rospy.get_param(
            "~takeover_topic", "/shared_autonomy/takeover"
        )
        self.done_topic = rospy.get_param("~done_topic", "/shared_autonomy/done")

        # ---- Controller mapping (pygame indices; Xbox defaults) ----
        # Left stick: axis 1 = forward/back (up is negative), axis 0 = left/right.
        self.axis_linear = int(rospy.get_param("~axis_linear", 1))
        self.axis_angular = int(rospy.get_param("~axis_angular", 0))
        # RB (right bumper) is commonly button 5; Start/Menu is commonly button 7.
        self.deadman_button = int(rospy.get_param("~deadman_button", 5))
        self.done_button = int(rospy.get_param("~done_button", 7))

        # ---- Velocity limits (match config/nav/teb_local_planner.yaml) ----
        self.max_vel_x = float(rospy.get_param("~max_vel_x", 0.2))
        self.max_vel_theta = float(rospy.get_param("~max_vel_theta", 0.5))
        self.deadband = float(rospy.get_param("~deadband", 0.12))

        # Flip these if a stick drives the robot the wrong way on real hardware.
        self.invert_linear = bool(rospy.get_param("~invert_linear", True))
        self.invert_angular = bool(rospy.get_param("~invert_angular", True))

        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.joystick_id = int(rospy.get_param("~joystick_id", 0))

        # ---- Publishers ----
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.takeover_pub = rospy.Publisher(self.takeover_topic, Empty, queue_size=1)
        self.done_pub = rospy.Publisher(self.done_topic, Empty, queue_size=1)

        # ---- State ----
        self.joystick = None
        self.prev_deadman = False
        self.prev_done = False

        pygame.init()
        pygame.joystick.init()
        self._try_open_joystick()

    # ------------------------------------------------------------------ #
    # Joystick connection handling
    # ------------------------------------------------------------------ #
    def _try_open_joystick(self) -> bool:
        """(Re)open the controller. Returns True if a controller is available."""
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() <= self.joystick_id:
                self.joystick = None
                return False
            self.joystick = pygame.joystick.Joystick(self.joystick_id)
            self.joystick.init()
            rospy.loginfo(
                "shared_autonomy_teleop: controller connected: %s",
                self.joystick.get_name(),
            )
            return True
        except pygame.error as exc:
            rospy.logwarn_throttle(
                5.0, "shared_autonomy_teleop: joystick init failed: %s", exc
            )
            self.joystick = None
            return False

    # ------------------------------------------------------------------ #
    # Stick mixing
    # ------------------------------------------------------------------ #
    def _compute_twist(self) -> Twist:
        x = apply_deadband(self.joystick.get_axis(self.axis_angular), self.deadband)
        y = apply_deadband(self.joystick.get_axis(self.axis_linear), self.deadband)

        # Stick up (negative axis) -> forward.
        lin = -y if self.invert_linear else y
        # Stick left (negative axis) -> turn left (positive angular.z, REP-103).
        ang = -x if self.invert_angular else x

        twist = Twist()
        twist.linear.x = clamp(lin * self.max_vel_x, -self.max_vel_x, self.max_vel_x)
        twist.angular.z = clamp(
            ang * self.max_vel_theta, -self.max_vel_theta, self.max_vel_theta
        )
        return twist

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def spin(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo(
            "shared_autonomy_teleop running. Hold button %d (deadman) to take over "
            "and drive; press button %d (done) to report goal reached.",
            self.deadman_button,
            self.done_button,
        )
        while not rospy.is_shutdown():
            # No controller: stay dormant, keep trying to (re)connect.
            if self.joystick is None:
                self._try_open_joystick()
                rate.sleep()
                continue

            try:
                pygame.event.pump()
                deadman = bool(self.joystick.get_button(self.deadman_button))
                done = bool(self.joystick.get_button(self.done_button))
            except pygame.error as exc:
                rospy.logwarn("shared_autonomy_teleop: controller read failed: %s", exc)
                self.joystick = None
                # Make sure we don't leave the base creeping if it dropped mid-drive.
                if self.prev_deadman:
                    self.cmd_pub.publish(Twist())
                self.prev_deadman = False
                self.prev_done = False
                rate.sleep()
                continue

            # Rising edge of deadman = the takeover request.
            if deadman and not self.prev_deadman:
                self.takeover_pub.publish(Empty())
                rospy.loginfo("shared_autonomy_teleop: takeover requested.")

            # Rising edge of done = goal-reached request.
            if done and not self.prev_done:
                self.done_pub.publish(Empty())
                rospy.loginfo("shared_autonomy_teleop: done (goal reached) requested.")

            if deadman:
                # Drive while held.
                self.cmd_pub.publish(self._compute_twist())
            elif self.prev_deadman:
                # Falling edge: publish a single stop, then stay silent.
                self.cmd_pub.publish(Twist())

            self.prev_deadman = deadman
            self.prev_done = done
            rate.sleep()


def main() -> None:
    rospy.init_node("shared_autonomy_teleop")
    SharedAutonomyTeleop().spin()


if __name__ == "__main__":
    main()
