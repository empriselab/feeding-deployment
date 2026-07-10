#!/usr/bin/env python3
"""
zupt_publisher.py -- Zero-velocity UPdaTe (ZUPT) pseudo-measurement for the
fused-odometry EKF.

When the wheel encoders AND the ZED gyro INDEPENDENTLY agree the robot is truly
stationary, publish a zero-twist Odometry (tiny covariance) on /odometry/zupt.
Wired as an EKF input (odom2), this actively pins the fused velocity to zero
while parked, cancelling the ZED VIO "confident creep" that otherwise drags the
estimate (worst in window sunlight). It is the soft, principled form of "only
trust ZED when the wheels move": instead of gating ZED, we assert zero velocity
when we can PROVE the robot is still.

Detector -- both must hold, continuously for `still_window_s`, with both streams
fresh (a stale stream => state unknown => publish nothing):
  |wheel vx|       < vx_still  (m/s)    -- the trustworthy wheel channel
  |gyro angular.z| < wz_still  (rad/s)  -- independent of the VIO visual pipeline
Requiring BOTH prevents firing during an in-place rotation (wheels differential-
cancel to ~0 net vx, but the gyro still sees the turn).

While stationary it publishes at ~pub_hz; otherwise it publishes NOTHING, so the
EKF's sensor_timeout drops the channel and no stale zero fights real motion.
Only the twist (vx, vy, vyaw) is consumed by the EKF; the pose is unused.
"""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

HUGE = 1e6  # non-fused / unused covariance axes


class ZuptPublisher:
    def __init__(self):
        self.out_topic = rospy.get_param("~output_topic", "/odometry/zupt")
        wheel_topic = rospy.get_param("~wheel_odom_topic", "/wheel_odom")
        imu_topic = rospy.get_param(
            "~imu_topic", "/zed_mini/zed_node/imu/data")

        # Stationarity thresholds (well below the capped teleop speeds, so real
        # motion is never mistaken for still; wheel encoders resolve sub-mm so
        # a genuine creep clears vx_still easily).
        self.vx_still = float(rospy.get_param("~vx_still", 0.01))       # m/s
        self.wz_still = float(rospy.get_param("~wz_still", 0.01))       # rad/s
        # Must be still continuously this long before ZUPT engages (rejects a
        # momentary zero-crossing during real motion).
        self.still_window_s = float(rospy.get_param("~still_window_s", 0.4))
        # A stream older than this => robot state unknown => do NOT ZUPT.
        self.fresh_timeout_s = float(rospy.get_param("~fresh_timeout_s", 0.3))
        self.pub_hz = float(rospy.get_param("~pub_hz", 20.0))
        # Small but non-zero: strong enough to pull the estimate to zero and
        # clamp the ZED leak, not so tiny it fights the wheel (which reads ~0
        # when still anyway). Comparable to the wheel vx covariance (1e-4).
        self.zupt_var = float(rospy.get_param("~zupt_var", 1e-3))
        self.base_frame = rospy.get_param("~base_frame", "vention_base_link")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")

        self.last_vx = None
        self.last_wz = None
        self.last_wheel_t = None    # rospy.Time of last wheel msg
        self.last_imu_t = None      # rospy.Time of last imu msg
        self.still_since = None     # rospy.Time stillness began, else None

        self.pub = rospy.Publisher(self.out_topic, Odometry, queue_size=20)
        rospy.Subscriber(wheel_topic, Odometry, self._cb_wheel, queue_size=20)
        rospy.Subscriber(imu_topic, Imu, self._cb_imu, queue_size=50)
        rospy.Timer(rospy.Duration(1.0 / self.pub_hz), self._tick)
        rospy.loginfo(
            "zupt_publisher: %s (wheel vx) + %s (gyro z) -> %s "
            "[still: |vx|<%.3f & |wz|<%.3f for %.2fs]",
            wheel_topic, imu_topic, self.out_topic,
            self.vx_still, self.wz_still, self.still_window_s)

    def _cb_wheel(self, msg):
        self.last_vx = msg.twist.twist.linear.x
        self.last_wheel_t = rospy.Time.now()

    def _cb_imu(self, msg):
        self.last_wz = msg.angular_velocity.z
        self.last_imu_t = rospy.Time.now()

    def _fresh(self, t, now):
        return t is not None and (now - t).to_sec() < self.fresh_timeout_s

    def _tick(self, _evt):
        now = rospy.Time.now()
        if not (self._fresh(self.last_wheel_t, now)
                and self._fresh(self.last_imu_t, now)):
            self.still_since = None          # state unknown -> no ZUPT
            return
        still = (abs(self.last_vx) < self.vx_still
                 and abs(self.last_wz) < self.wz_still)
        if not still:
            self.still_since = None
            return
        if self.still_since is None:
            self.still_since = now
        if (now - self.still_since).to_sec() < self.still_window_s:
            return                            # not settled long enough yet
        self._publish_zero(now)

    def _publish_zero(self, now):
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        # zero twist (linear.x/y and angular.z default to 0.0)
        v = self.zupt_var
        odom.twist.covariance = [
            v,    0,    0,    0,    0,    0,
            0,    v,    0,    0,    0,    0,
            0,    0,    HUGE, 0,    0,    0,
            0,    0,    0,    HUGE, 0,    0,
            0,    0,    0,    0,    HUGE, 0,
            0,    0,    0,    0,    0,    v,
        ]
        self.pub.publish(odom)


def main():
    rospy.init_node("zupt_publisher")
    ZuptPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
