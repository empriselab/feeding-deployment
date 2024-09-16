# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the real-time low-level
# control and causing latency spikes.

import queue
import time

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
        self.controller = None
        self.in_compliant_mode = False
        self.arm_stopped = False

    def get_state(self):
        arm_pos, ee_pose, gripper_pos = self.arm.get_state()
        return arm_pos, ee_pose, gripper_pos
    
    def get_update_state(self):
        arm_pos, arm_vel, _, gripper_pos = self.arm.get_update_state()
        return arm_pos, arm_vel, gripper_pos

    def reset(self):
        # Go to home position
        print("Moving to home position")
        self.arm.home()

    def switch_to_task_compliant_mode(self):

        assert not self.arm_stopped, "Arm is stopped"
        assert not self.in_compliant_mode, "Already in compliant mode"

        # clear command queue
        print("Clearing command queue")
        while not self.command_queue.empty():
            self.command_queue.get()

        # switch to joint compliant mode
        print("Switching to joint compliant mode")
        self.arm.switch_to_task_compliant_mode(self.command_queue)
        self.in_compliant_mode = True

    def switch_out_of_compliant_mode(self):

        assert not self.arm_stopped, "Arm is stopped"
        assert self.in_compliant_mode, "Not in compliant mode"

        # switch out of joint compliant mode
        print("Switching out of joint compliant mode")
        self.arm.switch_out_of_compliant_mode()
        self.in_compliant_mode = False

    def compliant_set_ee_pose(self, xyz, xyz_quat):
            
        assert not self.arm_stopped, "Arm is stopped"
        assert self.in_compliant_mode, "Not in compliant mode"

        command_pose = np.zeros(7)
        command_pose[:3] = xyz
        command_pose[3:] = xyz_quat

        print(f"Received compliant cartesian pose command: {xyz}, {xyz_quat}")
        gripper_pos = 0
        self.command_queue.put((command_pose, gripper_pos))

    # def compliant_set_joint_position(self, command_pos):

    #     assert not self.arm_stopped, "Arm is stopped"
    #     assert self.in_compliant_mode, "Not in compliant mode"

    #     print(f"Received compliant joint pos command: {command_pos}")
    #     gripper_pos = 0
    #     self.command_queue.put((command_pos, gripper_pos))

    # def compliant_set_joint_trajectory(self, trajectory_command):
    #     print(
    #         f"Received compliant joint trajectory command with {len(trajectory_command)} waypoints"
    #     )
    #     gripper_pos = 0
    #     for command_pos in trajectory_command:
    #         while True:
    #             time.sleep(0.01)
    #             q, _, _ = self.arm.get_update_state()
    #             error = np.linalg.norm(np.array(command_pos) - np.array(q))
    #             # threshold = 0.03*np.sqrt(7)
    #             threshold = 0.3
    #             print(f"Error: {error}, Threshold: {threshold}")
    #             # When near (distance < threshold) next waypoint, update to next waypoint
    #             if error < threshold:
    #                 self.command_queue.put((command_pos, gripper_pos))
    #                 break

    # def compliant_set_joint_trajectory(self, trajectory_command):

    #     assert not self.arm_stopped, "Arm is stopped"
    #     assert self.in_compliant_mode, "Not in compliant mode"

    #     print(
    #         f"Received compliant joint trajectory command with {len(trajectory_command)} waypoints"
    #     )

    #     for command_pos in trajectory_command:
    #         for i in range(len(command_pos)):
    #             if command_pos[i] > np.pi:
    #                 command_pos[i] -= 2 * np.pi
    #             if command_pos[i] < -np.pi:
    #                 command_pos[i] += 2 * np.pi

    #     assert self.command_queue.empty(), "Before trajectory execution - command queue not empty"

    #     gripper_pos = 0
    #     for command_pos in trajectory_command:
    #         self.command_queue.put((command_pos, gripper_pos))
    #         time.sleep(0.01)

    #     assert self.command_queue.empty(), "After trajectory execution - command queue not empty"

    def set_joint_position(self, command_pos):
        
        assert not self.arm_stopped, "Arm is stopped"
        assert not self.in_compliant_mode, "In compliant mode"

        print(f"Received joint pos command: {command_pos}")
        self.arm.move_angular(command_pos)

    def set_joint_trajectory(self, trajectory_command):

        assert not self.arm_stopped, "Arm is stopped"
        assert not self.in_compliant_mode, "In compliant mode"

        print(
            f"Received joint trajectory command with {len(trajectory_command)} waypoints"
        )
        self.arm.move_angular_trajectory(trajectory_command)

    def set_ee_pose(self, xyz, xyz_quat):

        assert not self.arm_stopped, "Arm is stopped"
        assert not self.in_compliant_mode, "In compliant mode"

        print(f"Received cartesian pose command: {xyz}, {xyz_quat}")
        self.arm.move_cartesian(xyz, xyz_quat)

    def set_gripper(self, gripper_pos):

        assert not self.arm_stopped, "Arm is stopped"
        assert not self.in_compliant_mode, "In compliant mode"

        print(f"Received gripper pos command: {gripper_pos}")
        self.arm._gripper_position_command(gripper_pos)

    def open_gripper(self):

        assert not self.arm_stopped, "Arm is stopped"
        assert not self.in_compliant_mode, "In compliant mode"

        print("Received open gripper command")
        self.arm.open_gripper()

    def close_gripper(self):

        assert not self.arm_stopped, "Arm is stopped"
        assert not self.in_compliant_mode, "In compliant mode"

        print("Received close gripper command")
        self.arm.close_gripper()

    def close(self):
        print("Closing arm connection")
        self.arm.stop()
        self.arm.disconnect()

    def retract(self):

        assert not self.arm_stopped, "Arm is stopped"
        assert not self.in_compliant_mode, "In compliant mode"

        self.arm.retract()

    def stop(self):
        self.arm_stopped = True
        print("No longer accepting commands")
        if self.in_compliant_mode:
            self.in_compliant_mode = False
            self.arm.switch_to_gravity_compensation_mode()
        else:
            self.arm.stop()
            print("Stopped arm")

class ArmManager(MPBaseManager):
    pass
