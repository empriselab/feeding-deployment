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

# from ik_solver import IKSolver
from feeding_deployment.robot_controller.command_interface import KinovaCommand

try:
    from feeding_deployment.robot_controller.kinova import KinovaArm
except ImportError:
    print(
        "KinovaArm import failed, continuing without executing arm commands on real robot"
    )

class ArmInterface:
    def __init__(self):
        self.arm = KinovaArm()
        # self.arm.set_joint_limits(speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,)))
        self.command_queue = queue.Queue(1)
        self.controller = None

    def get_state(self):
        arm_pos, ee_pose, gripper_pos = self.arm.get_state()
        return arm_pos, ee_pose, gripper_pos
    
    def get_update_state(self):
        arm_pos, arm_vel, gripper_pos = self.arm.get_update_state()
        return arm_pos, arm_vel, gripper_pos

    def reset(self):
        # Go to home position
        print("Moving to home position")
        self.arm.home()

    def switch_to_joint_compliant_mode(self):

        # clear command queue
        print("Clearing command queue")
        while not self.command_queue.empty():
            self.command_queue.get()

        # switch to joint compliant mode
        print("Switching to joint compliant mode")
        self.arm.switch_to_joint_compliant_mode(self.command_queue)

    def switch_out_of_joint_compliant_mode(self):
        # switch out of joint compliant mode
        print("Switching out of joint compliant mode")
        self.arm.switch_out_of_joint_compliant_mode()

    def compliant_set_joint_position(self, command_pos):
        print(f"Received compliant joint pos command: {command_pos}")
        gripper_pos = 0
        self.command_queue.put((command_pos, gripper_pos))

    def compliant_set_joint_trajectory(self, trajectory_command):
        print(
            f"Received compliant joint trajectory command with {len(trajectory_command)} waypoints"
        )
        gripper_pos = 0
        for command_pos in trajectory_command:
            while True:
                time.sleep(0.01)
                q, _, _ = self.arm.get_update_state()
                error = np.linalg.norm(np.array(command_pos) - np.array(q))
                # threshold = 0.03*np.sqrt(7)
                threshold = 0.3
                print(f"Error: {error}, Threshold: {threshold}")
                # When near (distance < threshold) next waypoint, update to next waypoint
                if error < threshold:
                    self.command_queue.put((command_pos, gripper_pos))
                    break

    # def compliant_set_joint_trajectory(self, trajectory_command):
    #     print(
    #         f"Received compliant joint trajectory command with {len(trajectory_command)} waypoints"
    #     )
    #     gripper_pos = 0
    #     for command_pos in trajectory_command:
    #         self.command_queue.put((command_pos, gripper_pos))
    #         time.sleep(0.1)

    def set_joint_position(self, command_pos):
        print(f"Received joint pos command: {command_pos}")
        self.arm.move_angular(command_pos)

    def set_joint_trajectory(self, trajectory_command):
        print(
            f"Received joint trajectory command with {len(trajectory_command)} waypoints"
        )
        self.arm.move_angular_trajectory(trajectory_command)

    def set_ee_pose(self, xyz, xyz_quat):
        print(f"Received cartesian pose command: {xyz}, {xyz_quat}")
        self.arm.move_cartesian(xyz, xyz_quat)

    def set_gripper(self, gripper_pos):
        print(f"Received gripper pos command: {gripper_pos}")
        self.arm._gripper_position_command(gripper_pos)

    def open_gripper(self):
        print("Received open gripper command")
        self.arm.open_gripper()

    def close_gripper(self):
        print("Received close gripper command")
        self.arm.close_gripper()

    def close(self):
        print("Closing arm connection")
        self.arm.disconnect()

    def retract(self):
        self.arm.retract()

    def stop(self):
        print("Stopping arm")
        self.arm.stop()

    def execute_command(self, cmd: KinovaCommand) -> None:

        if cmd.__class__.__name__ == "JointTrajectoryCommand":
            return self.set_joint_trajectory(cmd.traj)

        if cmd.__class__.__name__ == "JointCommand":
            return self.set_joint_position(cmd.pos)

        if cmd.__class__.__name__ == "CartesianCommand":
            return self.set_ee_pose(cmd.pos, cmd.quat)

        if cmd.__class__.__name__ == "OpenGripperCommand":
            return self.open_gripper()

        if cmd.__class__.__name__ == "CloseGripperCommand":
            return self.close_gripper()

        raise NotImplementedError(f"Unrecognized command: {cmd}")


class ArmManager(MPBaseManager):
    pass

ArmManager.register("ArmInterface", ArmInterface)
