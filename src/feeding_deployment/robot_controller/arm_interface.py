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
        try:
            arm_pos, ee_pose, gripper_pos = self.arm.get_state()
        except Exception as e:
            print(f"Error in get_state: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in get_state: {str(e)}") from None # suppress original exception

        # also check if gravity compensation has been set
        with self.gravity_compensation_event_lock:
            if self.gravity_compensation_event.is_set():
                # print("Emergency stop (gravity compensation) activated by controller, will not take any more commands")
                self.emergency_stop_active = True
                if self.in_compliant_mode:
                    self.in_compliant_mode = False

        return arm_pos, ee_pose, gripper_pos

    def reset(self):
        # Go to home position
        print("Moving to home position")
        try:
            self.arm.home()
        except Exception as e:
            print(f"Error in reset: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in reset: {str(e)}") from None # suppress original exception

    def set_tool(self, tool: str):
        print(f"Setting tool to {tool}")
        try:
            self.arm.set_tool(tool)
        except Exception as e:
            print(f"Error in set_tool: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in set_tool: {str(e)}") from None # suppress original exception

    def switch_to_task_compliant_mode(self):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "Already in compliant mode"

        # clear command queue
        print("Clearing command queue")
        while not self.command_queue.empty():
            self.command_queue.get()

        # switch to joint compliant mode
        print("Switching to joint compliant mode")

        try:
            self.arm.switch_to_task_compliant_mode(self.command_queue, self.gravity_compensation_event)
        except Exception as e:
            print(f"Error in switch_to_task_compliant_mode: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in switch_to_task_compliant_mode: {str(e)}") from None # suppress original exception
        self.in_compliant_mode = True

    def switch_to_joint_compliant_mode(self):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "Already in compliant mode"

        # clear command queue
        print("Clearing command queue")
        while not self.command_queue.empty():
            self.command_queue.get()

        # switch to joint compliant mode
        print("Switching to joint compliant mode")

        try:
            self.arm.switch_to_joint_compliant_mode(self.command_queue, self.gravity_compensation_event)
        except Exception as e:
            print(f"Error in switch_to_joint_compliant_mode: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in switch_to_joint_compliant_mode: {str(e)}") from None
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
            try:
                self.arm.switch_out_of_compliant_mode()
            except Exception as e:
                print(f"Error in switch_out_of_compliant_mode: {e}")
                # Re-raise a simplified exception to avoid pickling issues
                raise Exception(f"Error in switch_out_of_compliant_mode: {str(e)}") from None # suppress original exception
            self.in_compliant_mode = False

            self.gravity_compensation_event.clear()

    def compliant_set_joint_position(self, command_pos):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert self.in_compliant_mode, "Not in compliant mode"

        # print(f"Received compliant joint pos command: {command_pos}")
        gripper_pos = 0
        self.command_queue.put((command_pos, gripper_pos))

    def compliant_set_ee_pose(self, xyz, xyz_quat):
            
        assert not self.emergency_stop_active, "Emergency stop is active"
        assert self.in_compliant_mode, "Not in compliant mode"

        command_pose = np.zeros(7)
        command_pose[:3] = xyz
        command_pose[3:] = xyz_quat

        # print(f"Received compliant cartesian pose command: {xyz}, {xyz_quat}")
        gripper_pos = 0
        self.command_queue.put((command_pose, gripper_pos))

    def set_joint_position(self, command_pos):
        
        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print(f"Received joint pos command: {command_pos}")

        try:
            self.arm.move_angular(command_pos)
        except Exception as e:
            print(f"Error in set_joint_position: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in set_joint_position: {str(e)}") from None # suppress original exception

    def set_joint_trajectory(self, trajectory_command):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print(
            f"Received joint trajectory command with {len(trajectory_command)} waypoints"
        )

        try:
            self.arm.move_angular_trajectory(trajectory_command)
        except Exception as e:
            print(f"Error in set_joint_trajectory: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in set_joint_trajectory: {str(e)}") from None # suppress original exception

    def set_ee_pose(self, xyz, xyz_quat):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print(f"Received cartesian pose command: {xyz}, {xyz_quat}")

        try:
            self.arm.move_cartesian(xyz, xyz_quat)
        except Exception as e:
            print(f"Error in set_ee_pose: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in set_ee_pose: {str(e)}") from None # suppress original exception

    def set_gripper(self, gripper_pos):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print(f"Received gripper pos command: {gripper_pos}")

        try:
            self.arm._gripper_position_command(gripper_pos)
        except Exception as e:
            print(f"Error in set_gripper: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in set_gripper: {str(e)}") from None # suppress original exception

    def open_gripper(self):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print("Received open gripper command")

        try:
            self.arm.open_gripper()
        except Exception as e:
            print(f"Error in open_gripper: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in open_gripper: {str(e)}") from None # suppress original exception

    def close_gripper(self):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print("Received close gripper command")

        try:
            self.arm.close_gripper()
        except Exception as e:
            print(f"Error in close_gripper: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in close_gripper: {str(e)}") from None # suppress original exception

    def close(self):
        print("Close arm command received")
        if self.in_compliant_mode:
            print("Switching out of compliant mode through emergency stop")
            self.emergency_stop()
            time.sleep(1.0) # Wait for the arm to settle

        try:
            self.arm.stop() # Exit low level servoing mode incase it was in compliant mode, otherwise stop arm
            print("Arm stopped")
            self.arm.disconnect()
            print("Arm disconnected")
        except Exception as e:
            print(f"Error in close: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in close: {str(e)}") from None # suppress original exception

    def retract(self):

        assert not self.emergency_stop_active, "Emergency stop is active"
        assert not self.in_compliant_mode, "In compliant mode"

        print("Received retract command")

        try:
            self.arm.retract()
        except Exception as e:
            print(f"Error in retract: {e}")
            # Re-raise a simplified exception to avoid pickling issues
            raise Exception(f"Error in retract: {str(e)}") from None # suppress original exception

    def emergency_stop(self):
        with self.gravity_compensation_event_lock:
            self.emergency_stop_active = True
            if self.in_compliant_mode:
                self.in_compliant_mode = False
                self.gravity_compensation_event.set()
            else: # If not in compliant mode, stop arm (otherwise, arm is already stopped)
                try:
                    self.arm.stop()
                except Exception as e:
                    print(f"Error in emergency_stop: {e}")
                    # Re-raise a simplified exception to avoid pickling issues
                    raise Exception(f"Error in emergency_stop: {str(e)}") from None # suppress original exception

            print("Emergency stop activated by user, will not take any more commands")

class ArmManager(MPBaseManager):
    pass
