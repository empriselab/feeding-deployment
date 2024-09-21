# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the real-time low-level
# control and causing latency spikes.

import queue
import time
import threading

import numpy as np
from multiprocess.managers import BaseManager as MPBaseManager

RPC_AUTHKEY = b"secret-key"
NUC_HOSTNAME = "192.168.1.3"
ARM_RPC_PORT = 5000

class ArmInterface:
    def __init__(self, arm_instance):
        self.arm = arm_instance
        # self.arm.set_joint_limits(speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,)))
        
        self.command_queue = queue.Queue(1)
        self.gravity_compensation_event = threading.Event()
        self.in_compliant_mode = False

        self.emergency_stop_active = False
        self.controller = None

        # Lock to handle a corner case where the gravity compensation event is set by self.emergency_stop(),
        # but cleared by self.switch_out_of_compliant_mode().
        self.gravity_compensation_event_lock = threading.Lock()  

    def get_state(self):
        arm_pos, ee_pose, gripper_pos = self.arm.get_state()
        return arm_pos, ee_pose, gripper_pos

    def reset(self):
        # Go to home position
        print("Moving to home position")
        self.arm.home()

    def set_tool(self, tool: str):
        self.arm.set_tool(tool)

    def switch_to_task_compliant_mode(self):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "Already in compliant mode"

        # clear command queue
        print("Clearing command queue")
        while not self.command_queue.empty():
            self.command_queue.get()

        # switch to joint compliant mode
        print("Switching to joint compliant mode")
        self.arm.switch_to_task_compliant_mode(self.command_queue, self.gravity_compensation_event)
        self.in_compliant_mode = True

    def switch_out_of_compliant_mode(self):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert self.in_compliant_mode, "Not in compliant mode"

        # first move to gravity compensation 
        print("Moving to gravity compensation")
        self.gravity_compensation_event.set()
        time.sleep(1.0) # Wait for the arm to settle

        with self.gravity_compensation_event_lock:
            # switch out of joint compliant mode
            if self.emergency_stop_active:
                print("Cannot switch out of compliant mode due to emergency stop")
                return

            print("Switching out of joint compliant mode")
            self.arm.switch_out_of_compliant_mode()
            self.in_compliant_mode = False

            self.gravity_compensation_event.clear()

    def compliant_set_ee_pose(self, xyz, xyz_quat):
            
        assert not self.emergency_stop_active, "Emergency stop is active"
        assert self.in_compliant_mode, "Not in compliant mode"

        command_pose = np.zeros(7)
        command_pose[:3] = xyz
        command_pose[3:] = xyz_quat

        print(f"Received compliant cartesian pose command: {xyz}, {xyz_quat}")
        gripper_pos = 0
        self.command_queue.put((command_pose, gripper_pos))

    def set_joint_position(self, command_pos):
        
        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print(f"Received joint pos command: {command_pos}")
        self.arm.move_angular(command_pos)

    def set_joint_trajectory(self, trajectory_command):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print(
            f"Received joint trajectory command with {len(trajectory_command)} waypoints"
        )
        self.arm.move_angular_trajectory(trajectory_command)

    def set_ee_pose(self, xyz, xyz_quat):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print(f"Received cartesian pose command: {xyz}, {xyz_quat}")
        self.arm.move_cartesian(xyz, xyz_quat)

    def set_gripper(self, gripper_pos):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print(f"Received gripper pos command: {gripper_pos}")
        self.arm._gripper_position_command(gripper_pos)

    def open_gripper(self):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print("Received open gripper command")
        self.arm.open_gripper()

    def close_gripper(self):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print("Received close gripper command")
        self.arm.close_gripper()

    def close(self):
        print("Close arm command received")
        if self.in_compliant_mode:
            print("Switching out of compliant mode through emergency stop")
            self.emergency_stop()
            time.sleep(1.0) # Wait for the arm to settle

        self.arm.stop() # Exit low level servoing mode incase it was in compliant mode, otherwise stop arm
        print("Arm stopped")
        self.arm.disconnect()
        print("Arm disconnected")

    def retract(self):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        self.arm.retract()

    def emergency_stop(self):
        with self.gravity_compensation_event_lock:
            self.emergency_stop_active = True
            if self.in_compliant_mode:
                self.gravity_compensation_event.set()
            else: # If not in compliant mode, stop arm (otherwise, arm is already stopped)
                self.arm.stop()

            print("Emergency stop activated, will not take any more commands")

class ArmManager(MPBaseManager):
    pass
