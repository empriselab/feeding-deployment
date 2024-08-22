# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the real-time low-level
# control and causing latency spikes.

import queue
import time
from dataclasses import dataclass
from multiprocessing.managers import BaseManager as MPBaseManager

import numpy as np
from numpy.typing import NDArray

# from arm_controller import JointCompliantController
# from constants import RPC_AUTHKEY, ARM_RPC_PORT
RPC_AUTHKEY = b"secret-key"
ARM_RPC_PORT = 5000
# import rospy

# from ik_solver import IKSolver
from feeding_deployment.robot_controller.kinova import KinovaArm

# from sensor_msgs.msg import JointState

NUC_HOSTNAME = "192.168.1.3"


class KinovaCommand:
    """Establish an interface for commands that can be sent to the robot."""


@dataclass(frozen=True)
class JointTrajectoryCommand(KinovaCommand):
    """Command to follow an joint trajectory."""

    traj: list[NDArray]

    def __post_init__(self):
        num_dof = 7
        assert all(x.shape == (num_dof,) for x in self.traj)


class OpenGripperCommand(KinovaCommand):
    """Command to open the gripper."""


class CloseGripperCommand(KinovaCommand):
    """Command to close the gripper."""


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
        self.arm.disconnect()

    def retract(self):
        self.arm.retract()

    def stop(self):
        self.arm.stop()

    def execute_command(self, cmd: KinovaCommand) -> None:

        if cmd.__class__.__name__ == "JointTrajectoryCommand":
            return self.set_joint_trajectory(cmd.traj)

        if cmd.__class__.__name__ == "OpenGripperCommand":
            return self.open_gripper()

        if cmd.__class__.__name__ == "CloseGripperCommand":
            return self.close_gripper()

        raise NotImplementedError(f"Unrecognized command: {cmd}")


class ArmManager(MPBaseManager):
    pass


ArmManager.register("Arm", Arm)

if __name__ == "__main__":
    # manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    # server = manager.get_server()
    # print(f'Arm manager server started at {NUC_HOSTNAME}:{ARM_RPC_PORT}')
    # server.serve_forever()
    import numpy as np

    # from constants import POLICY_CONTROL_PERIOD
    POLICY_CONTROL_PERIOD = 0.1
    manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    manager.connect()
    arm = manager.Arm()
    try:
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        rospy.init_node("arm_client", anonymous=True)
        
        def emergency_stop_callback(msg):
            arm.stop()

        rospy.Subscriber("/estop", Bool, emergency_stop_callback)

        # # publish joint states
        # joint_states_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)

        # while not rospy.is_shutdown():
        #     arm_pos, gripper_pos = arm.get_state()
        #     joint_state_msg = JointState()
        #     joint_state_msg.header.stamp = rospy.Time.now()
        #     joint_state_msg.name = [
        #         "joint_1",
        #         "joint_2",
        #         "joint_3",
        #         "joint_4",
        #         "joint_5",
        #         "joint_6",
        #         "joint_7",
        #         "finger_joint",
        #     ]
        #     joint_state_msg.position = arm_pos.tolist() + [gripper_pos]
        #     joint_state_msg.velocity = [0.0] * 8
        #     joint_state_msg.effort = [0.0] * 8
        #     joint_states_pub.publish(joint_state_msg)
        #     time.sleep(0.01)

        input("Press Enter to retract the arm...")
        arm.retract()

        input("Press Enter to move to home position...")
        arm.reset()

        input("Press Enter to switch to joint compliant mode...")
        arm.switch_to_joint_compliant_mode()

        input("Press Enter to move to joint compliant position...")
        arm.compliant_set_joint_position([0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.9])

        input("Press Enter to switch out of joint compliant mode...")
        arm.switch_out_of_joint_compliant_mode()

        # print("Current Arm State:", arm.get_state())

        # home_pos = [
        #     2.2912759438800285,
        #     0.7308686750765581,
        #     2.082994642398784,
        #     4.109475142253324,
        #     0.2853091081120964,
        #     5.818345985240578,
        #     5.988186420599291,
        # ]

        # inside_mount_pose = (np.array([-0.147, -0.17, 0.07]), np.array([0.7071068, -0.7071068, 0, 0]))

        # outside_mount_pose = (np.array([-0.147, -0.29, 0.07]), np.array([0.7071068, -0.7071068, 0, 0]))

        # outside_mount_joint_states = [
        #     2.6266411620509817,
        #     0.6992626121546339,
        #     2.306749708761716,
        #     4.053362604401464,
        #     0.9559379448584164,
        #     5.655628973165609,
        #     5.80065247559031,
        # ]

        # above_mount_pose = (np.array([-0.147, -0.17, 0.15]), np.array([0.7071068, -0.7071068, 0, 0]))

        # above_mount_joint_states = [
        #     3.300153003835367,
        #     0.39120874346320217,
        #     1.8613410764520344,
        #     3.862447510072517,
        #     0.6143839397882825,
        #     5.583536137192727,
        #     6.276739392077158,
        # ]

        # infront_mount_pose = (np.array([0.0, -0.17, 0.15]), np.array([0.7071068, -0.7071068, 0, 0]))

        # infront_mount_joint_states = [
        #     2.835106221647441,
        #     0.18716812654374576,
        #     1.7554270267415284,
        #     -2.5582927305707517,
        #     0.3492644556371586,
        #     -0.5794207625752312,
        #     -0.3984099643402903,
        # ]

        # input("Press enter to move to home joint pos...")
        # next_pos = home_pos.copy()
        # arm.set_joint_position(next_pos)

        # input("Press enter to move to outside mount joint pos...")
        # next_pos = outside_mount_joint_states.copy()
        # arm.set_joint_position(next_pos)

        # input("Press enter to move to inside mount pose...")
        # arm.set_ee_pose(inside_mount_pose[0], inside_mount_pose[1])

        # input("Press enter to release the utensil...")
        # arm.set_gripper(1.0)

        # input("Press enter to move up...")
        # arm.set_ee_pose(above_mount_pose[0], above_mount_pose[1])

        # input("Press enter to move forward...")
        # arm.set_ee_pose(infront_mount_pose[0], infront_mount_pose[1])

        # input("Press enter to move to home joint pos...")
        # next_pos = home_pos.copy()
        # arm.set_joint_position(next_pos)

        # input("Press enter to move to infront mount joint pos...")
        # next_pos = infront_mount_joint_states.copy()
        # arm.set_joint_position(next_pos)

        # input("Press enter to move to above mount joint states...")
        # next_pos = above_mount_joint_states.copy()
        # arm.set_joint_position(next_pos)

        # input("Press enter to inside mount pose...")
        # arm.set_ee_pose(inside_mount_pose[0], inside_mount_pose[1])

        # input("Press enter to grab the utensil...")
        # arm.set_gripper(0.5)

        # input("Press enter to move to outside mount pose...")
        # arm.set_ee_pose(outside_mount_pose[0], outside_mount_pose[1])

        # input("Press enter to move to home joint pos...")
        # next_pos = home_pos.copy()
        # arm.set_joint_position(next_pos)

        # input("Press Enter to set arm pos...")
        # next_pos = [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.8]
        # arm.execute_action(next_pos)
        # input("Press Enter to exit...")
        # while True:
        # time.sleep(1)
        # for i in range(50):
        #     arm.execute_action({
        #         'arm_pos': np.array([0.135, 0.002, 0.211]),
        #         'arm_quat': np.array([0.706, 0.707, 0.029, 0.029]),
        #         'gripper_pos': np.zeros(1),
        #     })
        #     print(arm.get_state())
        #     state = arm.get_state()
        #     state['arm_quat'][0] = 3
        #     time.sleep(POLICY_CONTROL_PERIOD)
    except (EOFError, ConnectionRefusedError, BrokenPipeError) as e:
        print(f"Server connection lost: {e}")
    except KeyboardInterrupt:
        print("Client interrupted, shutting down...")
    finally:
        try:
            arm.close()  # Ensure the arm is disconnected properly
        except Exception as e:
            print(f"Error during client shutdown: {e}")
