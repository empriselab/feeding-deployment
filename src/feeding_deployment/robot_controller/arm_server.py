# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the real-time low-level
# control and causing latency spikes.

import queue
import time
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
# from arm_controller import JointCompliantController
# from constants import RPC_AUTHKEY, ARM_RPC_PORT
RPC_AUTHKEY = b'secret-key'
ARM_RPC_PORT = 5000
# from ik_solver import IKSolver
from kinova import KinovaArm

class Arm:
    def __init__(self):
        self.arm = KinovaArm()
        # self.arm.set_joint_limits(speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,)))
        self.command_queue = queue.Queue(1)
        self.controller = None

    def get_state(self):
        arm_pos, gripper_pos = self.arm.get_state()
        return arm_pos, gripper_pos
    
    def reset(self):
        # Go to home position
        self.arm.home()

    def switch_to_joint_compliant_mode(self):
        # switch to joint compliant mode
        self.arm.switch_to_joint_compliant_mode(self.command_queue)

    def switch_out_of_joint_compliant_mode(self):
        # switch out of joint compliant mode
        self.arm.switch_out_of_joint_compliant_mode()
    
    def compliant_set_joint_position(self, command_pos):
        print(f"Received compliant joint pos command: {command_pos}")
        gripper_pos = 0
        self.command_queue.put((command_pos, gripper_pos))

    def set_joint_position(self, command_pos):
        print(f"Received joint pos command: {command_pos}")
        self.arm.move_angular(command_pos)

    def set_ee_pose(self, xyz, theta_xyz):
        print(f"Received cartesian pose command: {xyz}, {theta_xyz}")
        self.arm.move_cartesian(xyz, theta_xyz)

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
        self.arm.disconnect()

class ArmManager(MPBaseManager):
    pass

ArmManager.register('Arm', Arm)

if __name__ == '__main__':
    hostname = '192.168.1.3'
    manager = ArmManager(address=(hostname, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'Arm manager server started at {hostname}:{ARM_RPC_PORT}')
    server.serve_forever()
    # import numpy as np
    # from constants import POLICY_CONTROL_PERIOD
    # manager = ArmManager(address=(hostname, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    # manager.connect()
    # arm = manager.Arm()
    # try:
    #     arm.reset()
    #     for i in range(50):
    #         arm.execute_action({
    #             'arm_pos': np.array([0.135, 0.002, 0.211]),
    #             'arm_quat': np.array([0.706, 0.707, 0.029, 0.029]),
    #             'gripper_pos': np.zeros(1),
    #         })
    #         print(arm.get_state())
    #         state = arm.get_state()
    #         state['arm_quat'][0] = 3
    #         time.sleep(POLICY_CONTROL_PERIOD)
    # finally:
    #     arm.close()
