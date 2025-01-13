# Author: Jimmy Wu, adapted by Rajat Kumar Jenamani for Emprise Lab

import math
import os
import queue
import subprocess
import threading
import time
import copy

import numpy as np
import pinocchio as pin

# Rajat ToDo: Move all ROS stuff to a separate interface
# import rospy
try:
    from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import (
        ActuatorConfigClient,
    )
    from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.client_stubs.ControlConfigClientRpc import (
        ControlConfigClient,
    )
    from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import (
        DeviceManagerClient,
    )
    from kortex_api.autogen.messages import (
        ActuatorConfig_pb2,
        ActuatorCyclic_pb2,
        Base_pb2,
        BaseCyclic_pb2,
        Common_pb2,
        ControlConfig_pb2,
        Session_pb2,
        DeviceConfig_pb2,
    )
    from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
    from kortex_api.SessionManager import SessionManager
    from kortex_api.TCPTransport import TCPTransport
    from kortex_api.UDPTransport import UDPTransport
except ModuleNotFoundError:
    pass
from scipy.spatial.transform import Rotation as R

# for joint space compliant control
from feeding_deployment.control.robot_controller.compliant_controller import CompliantController

# from std_msgs.msg import Bool


class DeviceConnection:
    IP_ADDRESS = "192.168.1.10"
    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection():
        return DeviceConnection(port=DeviceConnection.TCP_PORT)

    @staticmethod
    def createUdpConnection():
        return DeviceConnection(port=DeviceConnection.UDP_PORT)

    def __init__(
        self, ip_address=IP_ADDRESS, port=TCP_PORT, credentials=("admin", "admin")
    ):
        self.ip_address = ip_address
        self.port = port
        self.credentials = credentials
        self.session_manager = None
        self.transport = (
            TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        )
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ip_address, self.port)
        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000  # (milliseconds)
            session_info.connection_inactivity_timeout = 2000  # (milliseconds)
            self.session_manager = SessionManager(self.router)
            print("Logging as", self.credentials[0], "on device", self.ip_address)
            self.session_manager.CreateSession(session_info)
        return self.router

    def __exit__(self, *_):
        if self.session_manager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.session_manager.CloseSession(router_options)
        self.transport.disconnect()


class KinovaArm:
    ACTION_TIMEOUT_DURATION = 60

    def __init__(self):

        # Using Hack: fix joint 6
        self.fix_joint_hack = True

        # rospy.init_node("kinova_controller", anonymous=True)

        # self.estop_sub = rospy.Subscriber(
        #     "/estop", Bool, self.estop_callback, queue_size=10
        # )

        # Check whether arm is connected
        try:
            subprocess.run(
                ["ping", "-c", "1", "192.168.1.10"],
                check=True,
                timeout=1,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as e:
            raise Exception("Could not communicate with arm") from e

        # Lock file to enforce single instance
        self.lock_file = "/tmp/kinova.lock"
        if os.path.exists(self.lock_file):
            with open(self.lock_file, "r") as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)
            except OSError:
                print(f"Removing stale lock file (PID {pid})")
                os.remove(self.lock_file)
            else:
                raise Exception(
                    f"Another instance of the arm is already running (PID {pid})"
                )
        with open(self.lock_file, "w") as f:
            f.write(str(os.getpid()))

        # General Kortex setup
        self.tcp_connection = DeviceConnection.createTcpConnection()
        self.udp_connection = DeviceConnection.createUdpConnection()
        self.base = BaseClient(self.tcp_connection.__enter__())
        self.base_cyclic = BaseCyclicClient(self.udp_connection.__enter__())

        self.device_config = DeviceConfigClient(self.base.router)
        self.actuator_config = ActuatorConfigClient(self.base.router)
        self.actuator_count = self.base.GetActuatorCount().count
        self.control_config = ControlConfigClient(self.base.router)
        device_manager = DeviceManagerClient(self.base.router)
        device_handles = device_manager.ReadAllDevices()
        self.actuator_device_ids = [
            handle.device_identifier
            for handle in device_handles.device_handle
            if handle.device_type
            in [Common_pb2.BIG_ACTUATOR, Common_pb2.SMALL_ACTUATOR]
        ]
        self.send_options = RouterClientSendOptions()
        self.send_options.timeout_ms = 3

        # clear faults
        self.clear_faults()

        # Command and feedback setup
        self.base_command = BaseCyclic_pb2.Command()
        for _ in range(self.actuator_count):
            self.base_command.actuators.add()
        self.motor_cmd = self.base_command.interconnect.gripper_command.motor_cmd.add()
        self.base_feedback = BaseCyclic_pb2.Feedback()

        # Make sure actuators are in position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value(
            "POSITION"
        )
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Make sure arm is in high-level servoing mode
        self.set_arm_servoing_mode("high")

        print("Setting joint following error threshold to 10 degrees")
        # increase joint following error threshold to 10 degrees (upper hard limit)
        joint_following_safety_threshold = DeviceConfig_pb2.SafetyThreshold()
        joint_following_safety_threshold.handle.identifier = ActuatorConfig_pb2.SafetyIdentifierBankA.Value("FOLLOWING_ERROR")
        joint_following_safety_threshold.value = 10
        for device_id in self.actuator_device_ids:
            self.device_config.SetSafetyErrorThreshold(joint_following_safety_threshold, device_id)

        # Torque control setup
        # Note: Torque commands are converted to current commands since
        # Kinova's torque controller is unable to achieve commanded torques.
        # See relevant GitHub issue: https://github.com/Kinovarobotics/kortex/issues/38

        if self.fix_joint_hack:
            self.torque_constant = np.array([11.0, 11.0, 11.0, 11.0, 7.6, 7.6])
            self.current_limit_max = np.array([10.0, 10.0, 10.0, 10.0, 6.0, 6.0])
        else:
            self.torque_constant = np.array([11.0, 11.0, 11.0, 11.0, 7.6, 7.6, 7.6])
            self.current_limit_max = np.array([10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 6.0])
        self.current_limit_min = -self.current_limit_max

        # Cyclic thread setup
        self.cyclic_thread = None
        self.kill_the_thread = False
        self.cyclic_running = False

        if self.fix_joint_hack:
            self.n_compliant_dofs = self.actuator_count - 1 # Hack: fix joint 6
        else:
            self.n_compliant_dofs = self.actuator_count
            
        # Robot state setup (only used in low-level servoing mode)
        self.q = np.zeros(self.n_compliant_dofs)
        self.dq = np.zeros(self.n_compliant_dofs)
        self.tau = np.zeros(self.n_compliant_dofs)
        self.x = np.zeros(7)
        self.dx = np.zeros(7)
        self.gripper_pos = 0

        # Pinocchio setup (only used in low-level servoing mode)
        self.file_path = os.path.dirname(os.path.realpath(__file__)) 
        self.file_path = os.path.join(self.file_path, "urdfs")

        if self.fix_joint_hack:
            self.model = pin.buildModelFromUrdf(
                os.path.join(self.file_path, "hack_gen3_robotiq_2f_85.urdf")
            )
        else:
            self.model = pin.buildModelFromUrdf(
                os.path.join(self.file_path, "gen3_robotiq_2f_85.urdf")
            )
        self.data = self.model.createData()
        self.q_pin = np.zeros(self.model.nq)

        self.tool_frame_id = self.model.getFrameId("tool_frame")

        # Action topic notifications
        self.end_or_abort_event = threading.Event()
        self.end_or_abort_event.set()

        def check_for_end_or_abort(e):
            def check(notification, e=e):
                # print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
                if notification.action_event in (
                    Base_pb2.ACTION_END,
                    Base_pb2.ACTION_ABORT,
                ):
                    e.set()

            return check

        self.notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(self.end_or_abort_event),
            Base_pb2.NotificationOptions(),
        )

    def set_tool(self, tool):
        assert not self.cyclic_running, "Arm must be in high-level servoing mode to set tool"
        
        # updates the urdf being used by pinocchio to include tool mass
        if tool == "fork":
            # Pinocchio setup (only used in low-level servoing mode)
            self.file_path = os.path.dirname(os.path.realpath(__file__))
            self.file_path = os.path.join(self.file_path, "urdfs")

            if self.fix_joint_hack:
                self.model = pin.buildModelFromUrdf(
                    os.path.join(self.file_path, "hack_gen3_robotiq_2f_85_fork.urdf")
                )
            else:
                self.model = pin.buildModelFromUrdf(
                    os.path.join(self.file_path, "gen3_robotiq_2f_85_fork.urdf")
                )
            self.data = self.model.createData()
            self.q_pin = np.zeros(self.model.nq)
            self.tool_frame_id = self.model.getFrameId("fork_tip")
            
            print("Tool set to fork")
        elif tool == "wipe":
            # Pinocchio setup (only used in low-level servoing mode)
            self.file_path = os.path.dirname(os.path.realpath(__file__))
            self.file_path = os.path.join(self.file_path, "urdfs")

            if self.fix_joint_hack:
                self.model = pin.buildModelFromUrdf(
                    os.path.join(self.file_path, "hack_gen3_robotiq_2f_85_wipe.urdf")
                )
            else:
                self.model = pin.buildModelFromUrdf(
                    os.path.join(self.file_path, "gen3_robotiq_2f_85_wipe.urdf")
                )
            self.data = self.model.createData()
            self.q_pin = np.zeros(self.model.nq)
            self.tool_frame_id = self.model.getFrameId("wipe_tip")

            print("Tool set to wipe")
        elif tool == "drink":

            ee_force = self.get_ee_force()
            all_drink_urdfs = {
                6: "hack_gen3_robotiq_2f_85_drink_6.urdf",
                7.5: "hack_gen3_robotiq_2f_85_drink_7_5.urdf",
                9: "hack_gen3_robotiq_2f_85_drink_9.urdf",
                10.5: "hack_gen3_robotiq_2f_85_drink_10_5.urdf",
                12: "hack_gen3_robotiq_2f_85_drink_12.urdf",
            }
            drink_approximate_mass = -ee_force[2] 
            current_drink_urdf = all_drink_urdfs[min(all_drink_urdfs.keys(), key=lambda x: abs(x - drink_approximate_mass))]

            # Pinocchio setup (only used in low-level servoing mode)
            self.file_path = os.path.dirname(os.path.realpath(__file__))
            self.file_path = os.path.join(self.file_path, "urdfs")

            if self.fix_joint_hack:
                self.model = pin.buildModelFromUrdf(
                    os.path.join(self.file_path, current_drink_urdf)
                )
            else:
                self.model = pin.buildModelFromUrdf(
                    os.path.join(self.file_path, "gen3_robotiq_2f_85_drink.urdf")
                )
            self.data = self.model.createData()
            self.q_pin = np.zeros(self.model.nq)
            self.tool_frame_id = self.model.getFrameId("drink_tip")

            print(f"Tool set to drink with approximate mass {drink_approximate_mass} and urdf {current_drink_urdf}")
        else:
            raise ValueError("Invalid tool")

    # def estop_callback(self, msg):
    #     if msg.data:
    #         self.apply_emergency_stop()
    #         # kill ros node
    #         rospy.signal_shutdown("Emergency stop")

    def disconnect(self):
        if self.cyclic_running:
            self.stop_cyclic()

        self.base.Unsubscribe(
            self.notification_handle
        )  # Rajat ToDo: Check if this is necessary before switching to low-level servoing mode
        self.tcp_connection.__exit__()
        self.udp_connection.__exit__()
        os.remove(self.lock_file)

    def ready(self):
        return self.end_or_abort_event.is_set()

    def wait_ready(self):
        self.end_or_abort_event.wait(KinovaArm.ACTION_TIMEOUT_DURATION)

    def set_arm_servoing_mode(self, mode):
        if mode == "high":
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)
        elif mode == "low":
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)
        else:
            raise ValueError("Invalid servoing mode")

    def _execute_reference_action(self, action_name, blocking=True):
        assert not self.cyclic_running, "Arm must be in high-level servoing mode"

        # Retrieve reference action
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == action_name:
                action_handle = action.handle
        if action_handle is None:
            return

        # Execute action
        self.end_or_abort_event.clear()
        self.base.ExecuteActionFromReference(action_handle)
        if blocking:
            self.end_or_abort_event.wait(KinovaArm.ACTION_TIMEOUT_DURATION)

    def home(self):
        self._execute_reference_action("Home")

    def retract(self):
        self._execute_reference_action("Retract")

    def zero(self):
        self._execute_reference_action("Zero")

    def set_intermediate_zero_config(self):
        intermediate_zero_config = [
            0.0, 0.0, -3.12, 0.0, 0.0, 0.0, 0.0
        ]
        self.move_angular(intermediate_zero_config)

    def get_ee_force(self):
        base_feedback = self.base_cyclic.RefreshFeedback()
        ee_force = np.array(
            [
                base_feedback.base.tool_external_wrench_force_x,
                base_feedback.base.tool_external_wrench_force_y,
                base_feedback.base.tool_external_wrench_force_z,
            ]
        )
        return ee_force

    def get_state(self):

        # Rajat ToDo: this isn't exactly the same as get_state as it returns tool tip pose instead of tool_frame pose
        if self.cyclic_running:
            q, dq, tau, x, gripper_pos = self.get_update_state()
            if self.fix_joint_hack:
                q = np.insert(q, 5, -1.18039928)

                # Not returning below values as they are not correct
                # dq = np.insert(dq, 5, 0)
                # tau = np.insert(tau, 5, 0)

            return {
                "position": q,
                "velocity": np.zeros(self.actuator_count),
                "effort": np.zeros(self.actuator_count),
                "ee_pos": x,
                "gripper_pos": gripper_pos,
            }
            
        assert (
            not self.cyclic_running
        ), "Arm must be in high-level servoing mode"  # Rajat ToDo: Combine this with self.update_state()

        base_feedback = self.base_cyclic.RefreshFeedback()

        q, dq, tau = (
            np.zeros(self.actuator_count),
            np.zeros(self.actuator_count),
            np.zeros(self.actuator_count),
        )

        ee_pos, ee_vel, ee_force = (
            np.zeros(7),
            np.zeros(7),
            np.zeros(6),
        )

        # Robot joint state
        for i in range(self.actuator_count):
            q[i] = math.radians(base_feedback.actuators[i].position)
            if q[i] > np.pi:
                q[i] -= 2 * np.pi
            dq[i] = math.radians(base_feedback.actuators[i].velocity)
            tau[i] = -base_feedback.actuators[i].torque

        # Robot cartesian state
        ee_pos[:3] = (base_feedback.base.tool_pose_x, base_feedback.base.tool_pose_y, base_feedback.base.tool_pose_z)
        tool_rot = np.array([base_feedback.base.tool_pose_theta_x, base_feedback.base.tool_pose_theta_y, base_feedback.base.tool_pose_theta_z])
        ee_pos[3:] = R.from_euler("xyz", np.deg2rad(tool_rot)).as_quat()

        ee_vel[:3] = (base_feedback.base.tool_twist_linear_x, base_feedback.base.tool_twist_linear_y, base_feedback.base.tool_twist_linear_z)
        tool_rot_vel = np.array([base_feedback.base.tool_twist_angular_x, base_feedback.base.tool_twist_angular_y, base_feedback.base.tool_twist_angular_z])
        ee_vel[3:] = R.from_euler("xyz", np.deg2rad(tool_rot_vel)).as_quat()

        ee_force[:3] = (base_feedback.base.tool_external_wrench_force_x, base_feedback.base.tool_external_wrench_force_y, base_feedback.base.tool_external_wrench_force_z)

        # print("End Effector Force: ", ee_force)

        gripper_pos = (
            base_feedback.interconnect.gripper_feedback.motor[0].position / 100.0
        )

        return {
            "position": q,
            "velocity": dq,
            "effort": tau,
            "ee_pos": ee_pos,
            "gripper_pos": gripper_pos,
        }

    def move_angular_trajectory(self, trajectory_joint_angles, blocking=True):

        assert not self.cyclic_running, "Arm must be in high-level servoing mode"
        assert len(trajectory_joint_angles) > 0, "Invalid trajectory"
        assert (
            len(trajectory_joint_angles[0]) == self.actuator_count
        ), "Invalid number of joint angles"

        jointPoses = [
            [math.degrees(angle) for angle in jointPose]
            for jointPose in trajectory_joint_angles
        ]

        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0
        waypoints.use_optimal_blending = False

        index = 0
        for jointPose in jointPoses:
            waypoint = waypoints.waypoints.add()
            waypoint.name = "waypoint_" + str(index)
            waypoint.angular_waypoint.angles.extend(jointPose)
            waypoint.angular_waypoint.duration = 0.5
            index = index + 1

        result = self.base.ValidateWaypointList(waypoints)
        if len(result.trajectory_error_report.trajectory_error_elements) == 0:
            print("Reaching angular pose trajectory...")

            self.end_or_abort_event.clear()
            self.base.ExecuteWaypointTrajectory(waypoints)

            if blocking:
                print("Waiting for trajectory to finish ...")
                finished = self.end_or_abort_event.wait(
                    KinovaArm.ACTION_TIMEOUT_DURATION
                )
                if finished:
                    print("Angular movement completed")
                else:
                    print("Timeout on action notification wait")
        else:
            print("Error found in trajectory")
            print(result.trajectory_error_report)

    def move_angular(self, joint_angles, blocking=True):

        assert not self.cyclic_running, "Arm must be in high-level servoing mode"
        assert (
            len(joint_angles) == self.actuator_count
        ), "Invalid number of joint angles"

        # Create action
        action = Base_pb2.Action()
        for i in range(self.actuator_count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = i
            joint_angle.value = math.degrees(joint_angles[i])
        self.end_or_abort_event.clear()
        self.base.ExecuteAction(action)
        if blocking:
            print("Waiting for angular movement to finish ...")
            self.end_or_abort_event.wait(KinovaArm.ACTION_TIMEOUT_DURATION)
            # read states and check if the arm actually reached the desired position
            current_state = self.get_state()
            q = current_state["position"]
            # find error while wrapping angles
            error = np.degrees(q - joint_angles)
            while np.any(error > 180) or np.any(error < -180):
                error = np.where(error > 180, error - 360, error) 
                error = np.where(error < -180, error + 360, error)

            if np.any(np.abs(error) > 5): # 5 degrees
                print("Arm did not reach desired position")
                self.stop()
                print("Arm stopped")
                self.disconnect()
                print("Arm disconnected")
            else:
                print("Angular movement completed")


    def move_cartesian(self, xyz, xyz_quat, blocking=True):

        assert not self.cyclic_running, "Arm must be in high-level servoing mode"

        theta_xyz = R.from_quat(xyz_quat).as_euler("xyz")

        # Create action
        action = Base_pb2.Action()
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = xyz[0]
        cartesian_pose.y = xyz[1]
        cartesian_pose.z = xyz[2]
        cartesian_pose.theta_x = math.degrees(theta_xyz[0])
        cartesian_pose.theta_y = math.degrees(theta_xyz[1])
        cartesian_pose.theta_z = math.degrees(theta_xyz[2])
        self.end_or_abort_event.clear()
        self.base.ExecuteAction(action)
        if blocking:
            print("Waiting for cartesian movement to finish ...")
            self.end_or_abort_event.wait(KinovaArm.ACTION_TIMEOUT_DURATION)
            # read states and check if the arm actually reached the desired position
            current_state = self.get_state()
            x = current_state["ee_pos"]
            if not np.allclose(x[:3], xyz, atol=0.01): # 1 cm
                print("Arm did not reach desired position")
                self.stop()
                print("Arm stopped")
                self.disconnect()
                print("Arm disconnected")
            else:
                print("Cartesian movement completed")

    def _gripper_position_command(self, value, blocking=True, timeout=1.0):
        assert not self.cyclic_running, "Arm must be in high-level servoing mode"

        # Send gripper command
        gripper_command = Base_pb2.GripperCommand()
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger = gripper_command.gripper.finger.add()
        finger.value = value
        self.base.SendGripperCommand(gripper_command)

        if blocking:
            # Wait for reported position to match value
            gripper_request = Base_pb2.GripperRequest()
            gripper_request.mode = Base_pb2.GRIPPER_POSITION
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < timeout:
                gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
                if abs(value - gripper_measure.finger[0].value) < 0.01:
                    break
                time.sleep(0.01)

    def open_gripper(self, blocking=True):
        self._gripper_position_command(0, blocking)

    def close_gripper(self, blocking=True):
        self._gripper_position_command(1, blocking)

    def set_joint_limits(
        self,
        speed_limits=(60, 60, 60, 60, 60, 60, 60),
        acceleration_limits=(80, 80, 80, 80, 80, 80, 80),
        cartesian=False,
    ):
        if cartesian:
            joint_speed_soft_limits = ControlConfig_pb2.JointSpeedSoftLimits()
            joint_speed_soft_limits.control_mode = (
                ControlConfig_pb2.CARTESIAN_TRAJECTORY
            )
            joint_speed_soft_limits.joint_speed_soft_limits.extend(speed_limits)
            self.control_config.SetJointSpeedSoftLimits(joint_speed_soft_limits)
        else:
            joint_speed_soft_limits = ControlConfig_pb2.JointSpeedSoftLimits()
            joint_speed_soft_limits.control_mode = ControlConfig_pb2.ANGULAR_TRAJECTORY
            joint_speed_soft_limits.joint_speed_soft_limits.extend(speed_limits)
            self.control_config.SetJointSpeedSoftLimits(joint_speed_soft_limits)
            joint_acceleration_soft_limits = (
                ControlConfig_pb2.JointAccelerationSoftLimits()
            )
            joint_acceleration_soft_limits.control_mode = (
                ControlConfig_pb2.ANGULAR_TRAJECTORY
            )
            joint_acceleration_soft_limits.joint_acceleration_soft_limits.extend(
                acceleration_limits
            )
            self.control_config.SetJointAccelerationSoftLimits(
                joint_acceleration_soft_limits
            )

    def set_max_joint_limits(self):
        speed_limits = self.control_config.GetKinematicHardLimits().joint_speed_limits
        acceleration_limits = (
            self.control_config.GetKinematicHardLimits().joint_acceleration_limits
        )
        self.set_joint_limits(speed_limits, acceleration_limits)

    def reset_joint_limits(self):
        control_mode_information = ControlConfig_pb2.ControlModeInformation()
        for control_mode in [
            ControlConfig_pb2.ANGULAR_JOYSTICK,
            ControlConfig_pb2.CARTESIAN_JOYSTICK,
            ControlConfig_pb2.ANGULAR_TRAJECTORY,
            ControlConfig_pb2.CARTESIAN_TRAJECTORY,
            ControlConfig_pb2.CARTESIAN_WAYPOINT_TRAJECTORY,
        ]:
            control_mode_information.control_mode = control_mode
            self.control_config.ResetJointSpeedSoftLimits(control_mode_information)
        for control_mode in [
            ControlConfig_pb2.ANGULAR_JOYSTICK,
            ControlConfig_pb2.ANGULAR_TRAJECTORY,
        ]:
            control_mode_information.control_mode = control_mode
            self.control_config.ResetJointAccelerationSoftLimits(
                control_mode_information
            )

    def set_twist_linear_limit(self, limit):
        twist_linear_soft_limit = ControlConfig_pb2.TwistLinearSoftLimit()
        twist_linear_soft_limit.control_mode = ControlConfig_pb2.CARTESIAN_TRAJECTORY
        twist_linear_soft_limit.twist_linear_soft_limit = limit
        self.control_config.SetTwistLinearSoftLimit(twist_linear_soft_limit)

    def set_max_twist_linear_limit(self):
        limit = self.control_config.GetKinematicHardLimits().twist_linear  # 0.5
        self.set_twist_linear_limit(limit)

    def reset_twist_linear_limit(self):
        control_mode_information = ControlConfig_pb2.ControlModeInformation()
        control_mode_information.control_mode = ControlConfig_pb2.CARTESIAN_TRAJECTORY
        self.control_config.ResetTwistLinearSoftLimit(control_mode_information)

    # Rajat ToDo: Check how the following work:
    def pause_action(self):
        self.base.PauseAction()

    def resume_action(self):
        self.base.ResumeAction()

    def stop_action(self):
        self.base.StopAction()

    def stop(self):
        if self.cyclic_running:
            self.stop_cyclic()
        else:
            self.base.Stop()

    # Not using this as we haven't tested it
    # def apply_emergency_stop(self):
    #     self.base.ApplyEmergencyStop()

    def clear_faults(self):
        if self.base.GetArmState().active_state == Base_pb2.ARMSTATE_IN_FAULT:
            self.base.ClearFaults()
            while (
                self.base.GetArmState().active_state != Base_pb2.ARMSTATE_SERVOING_READY
            ):
                time.sleep(0.1)

    def zero_torque_offsets(self):
        assert not self.cyclic_running, "Arm must be in high-level servoing mode"

        input(
            "Arm will now be moved to retract configuration, press <Enter> to continue..."
        )
        self.retract()

        input(
            "Arm will now be moved to intermediate zero (candelstick with 3rd joint twisted) configuration. Please make sure arm is clear of obstacles and then press <Enter> to continue..."
        )
        self.set_intermediate_zero_config()

        # Move arm to zero configuration
        print("Arm will be moved to the candlestick configuration")
        input(
            "Arm will be moved to the candlestick configuration. Press <Enter> to continue..."
        )
        self.zero()

        # Wait for arm to become fully still
        input(
            "Please wait until the the arm is fully still and then press <Enter> to continue..."
        )

        # Set zero torque offsets
        for device_id in self.actuator_device_ids:
            torque_offset = ActuatorConfig_pb2.TorqueOffset()
            self.actuator_config.SetTorqueOffset(torque_offset, device_id)
        print("Torque offsets have been set to zero")

        # Move arm to home configuration
        input(
            "Arm will now be moved to intermediate zero (candelstick with 3rd joint twisted) configuration, press <Enter> to continue..."
        )
        self.set_intermediate_zero_config()

        input(
            "Arm will now be moved to retract configuration, press <Enter> to continue..."
        )
        self.retract()

    def init_cyclic(self, control_callback):
        assert not self.cyclic_running, "Cyclic thread is already running"

        # Set real-time scheduling policy
        try:
            os.sched_setscheduler(
                0,
                os.SCHED_FIFO,
                os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)),
            )
        except PermissionError:
            print(
                "Failed to set real-time scheduling policy, please edit /etc/security/limits.d/99-realtime.conf"
            )

        # Initialize command frame
        self.base_feedback = self.base_cyclic.RefreshFeedback()
        for i in range(self.actuator_count):
            self.base_command.actuators[i].flags = ActuatorCyclic_pb2.SERVO_ENABLE
            self.base_command.actuators[i].position = self.base_feedback.actuators[
                i
            ].position
            self.base_command.actuators[i].current_motor = self.base_feedback.actuators[
                i
            ].current_motor
        self.motor_cmd.position = (
            self.base_feedback.interconnect.gripper_feedback.motor[0].position
        )
        self.motor_cmd.velocity = 0
        self.motor_cmd.force = 100

        # Set arm to low-level servoing mode
        self.set_arm_servoing_mode("low")
        print("Arm is in low-level servoing mode")

        # Send first frame and update robot state
        self.base_feedback = self.base_cyclic.Refresh(
            self.base_command, 0, self.send_options
        )

        # Set actuators to current control mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value(
            "CURRENT"
        )
        for device_id in self.actuator_device_ids:
            print(f"Setting actuator {device_id} to current control mode")
            if self.fix_joint_hack and device_id == 6:
                print("Hack Activated: Skipping actuator 6")
                continue
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Start cyclic thread
        self.kill_the_thread = False
        self.cyclic_thread = threading.Thread(
            target=self.run_cyclic, args=(control_callback,), daemon=True
        )
        self.cyclic_thread.start()

    def run_cyclic(self, control_callback):
        self.cyclic_running = True
        failed_cyclic_count = 0
        # cyclic_count = 0
        # data = []

        # Update state before entering loop
        self.update_state()

        t_now = time.time()
        t_cyclic = t_now
        while not self.kill_the_thread:
            t_now = time.time()
            step_time = t_now - t_cyclic
            if step_time >= 0.001:  # 1 kHz
                t_cyclic = t_now
                if step_time > 0.004:  
                    print(
                        f"Warning: Step time {1000 * step_time:.3f} ms in {self.__class__.__name__} run_cyclic"
                    )

                # Get torque command
                torque_command, gripper_command = control_callback(self)

                # Convert to current command (Note: Current refers to electrical current)
                current_command = np.divide(torque_command, self.torque_constant)

                # Clamp using current limits
                np.clip(
                    current_command,
                    self.current_limit_min,
                    self.current_limit_max,
                    out=current_command,
                )

                # Increment frame ID to ensure actuators can reject out-of-order frames
                self.base_command.frame_id = (self.base_command.frame_id + 1) % 65536

                # Update arm command
                for i in range(self.actuator_count):

                    if self.fix_joint_hack: 

                        # Ignore actuator 6
                        if i == 5:
                            continue

                        # Update position command to avoid triggering following error
                        self.base_command.actuators[i].position = (
                            self.base_feedback.actuators[i].position
                        )
                        
                        # Update current command
                        if i < 5:
                            self.base_command.actuators[i].current_motor = current_command[i]
                        else:
                            self.base_command.actuators[i].current_motor = current_command[i - 1]

                    else:

                        # Update position command to avoid triggering following error
                        self.base_command.actuators[i].position = (
                            self.base_feedback.actuators[i].position
                        )

                        # Update current command
                        self.base_command.actuators[i].current_motor = current_command[i]

                    # Update command ID
                    self.base_command.actuators[i].command_id = (
                        self.base_command.frame_id
                    )

                # # Update gripper command
                # self.motor_cmd.position = 100 * gripper_command
                # self.motor_cmd.velocity = np.clip(
                #     abs(400 * (gripper_command - self.gripper_pos)), 0, 100
                # )

                # Send command frame
                try:
                    # Note: This call takes up most of the 1000 us cyclic step time
                    self.base_feedback = self.base_cyclic.Refresh(
                        self.base_command, 0, self.send_options
                    )
                except:
                    failed_cyclic_count += 1

                # Update robot state
                self.update_state()

                # data.append({
                #     'timestamp': t_now,
                #     'position': [actuator.position for actuator in self.base_feedback.actuators],
                #     'velocity': [actuator.velocity for actuator in self.base_feedback.actuators],
                #     'torque': [actuator.torque for actuator in self.base_feedback.actuators],
                #     'current_motor': [actuator.current_motor for actuator in self.base_feedback.actuators],
                #     'gripper_pos': self.base_feedback.interconnect.gripper_feedback.motor[0].position,
                #     'torque_command': torque_command.tolist(),
                #     'gripper_command': gripper_command,
                #     'current_command': current_command.tolist(),
                # })
                # cyclic_count += 1
                # if cyclic_count >= 5000:
                #     break

        # import pickle
        # output_path = 'actuator-states.pkl'
        # with open(output_path, 'wb') as f:
        #     pickle.dump(data, f)
        # print(f'Data saved to {output_path}')

        self.cyclic_running = False

    def stop_cyclic(self):
        # Kill cyclic thread
        if self.cyclic_running:
            self.kill_the_thread = True
            self.cyclic_thread.join()

        # Set actuators back to position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value(
            "POSITION"
        )
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Set arm back to high-level servoing mode
        self.set_arm_servoing_mode("high")
        print("Arm is back in high-level servoing mode")

    def update_state(self):

        # Robot state
        for i in range(self.n_compliant_dofs):
            if self.fix_joint_hack and i >= 5:
                self.q[i] = self.base_feedback.actuators[i + 1].position
                self.dq[i] = self.base_feedback.actuators[i + 1].velocity
                self.tau[i] = self.base_feedback.actuators[i + 1].torque
            else:
                self.q[i] = self.base_feedback.actuators[i].position
                self.dq[i] = self.base_feedback.actuators[i].velocity
                self.tau[i] = self.base_feedback.actuators[i].torque
        np.deg2rad(self.q, out=self.q)
        np.deg2rad(self.dq, out=self.dq)
        np.negative(
            self.tau, out=self.tau
        )  # Raw torque readings are negative relative to actuator direction
        self.gripper_pos = (
            self.base_feedback.interconnect.gripper_feedback.motor[0].position / 100.0
        )

        if self.fix_joint_hack:
            # Pinocchio joint configuration
            self.q_pin = np.array(
                [
                    math.cos(self.q[0]),
                    math.sin(self.q[0]),
                    self.q[1],
                    math.cos(self.q[2]),
                    math.sin(self.q[2]),
                    self.q[3],
                    math.cos(self.q[4]),
                    math.sin(self.q[4]),
                    math.cos(self.q[5]),
                    math.sin(self.q[5]),
                ]
            )
        else:
            # Pinocchio joint configuration
            self.q_pin = np.array(
                [
                    math.cos(self.q[0]),
                    math.sin(self.q[0]),
                    self.q[1],
                    math.cos(self.q[2]),
                    math.sin(self.q[2]),
                    self.q[3],
                    math.cos(self.q[4]),
                    math.sin(self.q[4]),
                    self.q[5],
                    math.cos(self.q[6]),
                    math.sin(self.q[6]),
                ]
            )

        pin.framesForwardKinematics(self.model, self.data, self.q_pin)
        tool_pose = self.data.oMf[self.tool_frame_id]
        self.x[:3] = tool_pose.translation.copy()
        self.x[3:] = R.from_matrix(tool_pose.rotation).as_quat()
    
    def get_update_state(self):

        assert self.cyclic_running, "Arm must be in low-level servoing mode"

        q = self.q.copy()
        dq = self.dq.copy()
        tau = self.tau.copy()
        x = self.x.copy()
        gripper_pos = copy.copy(self.gripper_pos)

        # normalize q
        for pos in range(len(q)):
            if q[pos] > np.pi:
                q[pos] -= 2 * np.pi
        
        return q, dq, tau, x, gripper_pos

    def gravity(self):
        assert self.cyclic_running, "Arm must be in low-level servoing mode"
        return pin.computeGeneralizedGravity(self.model, self.data, self.q_pin)

    def get_tool_pose(self):
        assert self.cyclic_running, "Arm must be in low-level servoing mode"
        pin.framesForwardKinematics(self.model, self.data, self.q_pin)
        tool_pose = self.data.oMf[self.tool_frame_id]
        pos = tool_pose.translation.copy()  # This copy() is important!
        # quat = pin.Quaternion(tool_pose.rotation).coeffs()  # This causes a segfault
        quat = R.from_matrix(tool_pose.rotation).as_quat()
        return pos, quat

    def get_fk(self, q):
        assert self.cyclic_running, "Arm must be in low-level servoing mode"
        if self.fix_joint_hack:
            # Pinocchio joint configuration
            q_pin = np.array(
                [
                    math.cos(q[0]),
                    math.sin(q[0]),
                    q[1],
                    math.cos(q[2]),
                    math.sin(q[2]),
                    q[3],
                    math.cos(q[4]),
                    math.sin(q[4]),
                    math.cos(q[5]),
                    math.sin(q[5]),
                ]
            )
        else:
            # Pinocchio joint configuration
            q_pin = np.array(
                [
                    math.cos(q[0]),
                    math.sin(q[0]),
                    q[1],
                    math.cos(q[2]),
                    math.sin(q[2]),
                    q[3],
                    math.cos(q[4]),
                    math.sin(q[4]),
                    q[5],
                    math.cos(q[6]),
                    math.sin(q[6]),
                ]
            )

        pin.computeJointJacobians(self.model, self.data, q_pin)
        pin.framesForwardKinematics(self.model, self.data, q_pin)
        tool_pose = self.data.oMf[self.tool_frame_id]

        # Copy the rotation matrix
        rotation_matrix = tool_pose.rotation.copy()

        # # Make first and second columns negative to account for axis convention
        # rotation_matrix[:, 0] = -rotation_matrix[:, 0]
        # rotation_matrix[:, 1] = -rotation_matrix[:, 1]

        # Construct pose
        pose = np.zeros(7)
        pose[:3] = tool_pose.translation.copy()
        pose[3:] = R.from_matrix(rotation_matrix).as_quat()
        
        J = pin.getFrameJacobian(
            self.model, self.data, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        
        return pose, J

    def switch_to_gravity_compensation_mode(self):
        def grav_comp_control_callback(arm):
            torque_command = arm.gravity()
            gripper_command = arm.gripper_pos
            return torque_command, gripper_command

        current_state = self.get_state()
        self.gripper_pos = current_state["gripper_pos"] # set gripper position to current position to avoid sudden jumps

        # if compliant control is already running, stop it (but do not switch back to high-level servoing mode)
        if self.cyclic_running:
            self.kill_the_thread = True
            self.cyclic_thread.join()
            
        self.init_cyclic(grav_comp_control_callback)
        while not self.cyclic_running:
            time.sleep(0.01)

        print("Arm is in gravity compensation mode")

    def switch_to_joint_compliant_mode(self, command_queue, gravity_compensation_external_event, gravity_compensation_internal_event):

        controller = CompliantController(command_queue, gravity_compensation_external_event, gravity_compensation_internal_event, control_type="joint", fix_joint_hack=self.fix_joint_hack)
        self.init_cyclic(controller.control_callback)
        while not self.cyclic_running:
            time.sleep(0.01)

        print("Arm is in joint compliant mode")

    def switch_to_task_compliant_mode(self, command_queue, gravity_compensation_external_event, gravity_compensation_internal_event):

        controller = CompliantController(command_queue, gravity_compensation_external_event, gravity_compensation_internal_event, control_type="task", fix_joint_hack=self.fix_joint_hack)
        self.init_cyclic(controller.control_callback)
        while not self.cyclic_running:
            time.sleep(0.01)

        print("Arm is in task compliant mode")

    def switch_out_of_compliant_mode(self):
        if self.cyclic_running:
            self.stop_cyclic()
        else:
            print("Not switching as arm is not in compliant mode")

def main():
    arm = KinovaArm()
    try:
        input("Press Enter to zero torque offsets")
        arm.zero_torque_offsets()
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
