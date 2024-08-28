# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the real-time low-level
# control and causing latency spikes.

import queue
import threading
import time

import numpy as np
from multiprocess.managers import BaseManager as MPBaseManager

# from arm_controller import JointCompliantController
# from constants import RPC_AUTHKEY, ARM_RPC_PORT
RPC_AUTHKEY = b"secret-key"
ARM_RPC_PORT = 5000
# import rospy

# from ik_solver import IKSolver
from feeding_deployment.robot_controller.command_interface import KinovaCommand

try:
    from feeding_deployment.robot_controller.kinova import KinovaArm
except ImportError:
    print(
        "KinovaArm import failed, continuing without executing arm commands on real robot"
    )

# from sensor_msgs.msg import JointState

NUC_HOSTNAME = "192.168.1.3"


class Arm:
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
    joint_state_thread = None
    try:
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        rospy.init_node("arm_client", anonymous=True)

        def emergency_stop_callback(msg):
            arm.stop()

        rospy.Subscriber("/estop", Bool, emergency_stop_callback)

        def publish_joint_states(arm):

            # publish joint states
            joint_states_pub = rospy.Publisher(
                "/robot_joint_states", JointState, queue_size=10
            )

            while not rospy.is_shutdown():
                arm_pos, ee_pose, gripper_pos = arm.get_state()
                joint_state_msg = JointState()
                joint_state_msg.header.stamp = rospy.Time.now()
                joint_state_msg.name = [
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "joint_7",
                    "finger_joint",
                ]
                joint_state_msg.position = arm_pos.tolist() + [gripper_pos]
                joint_state_msg.velocity = [0.0] * 8
                joint_state_msg.effort = [0.0] * 8
                joint_states_pub.publish(joint_state_msg)
                time.sleep(0.01)

        joint_state_thread = threading.Thread(target=publish_joint_states, args=(arm,))
        joint_state_thread.start()

        before_transfer_pos = [-2.86554642, -1.61951779, -2.60986085, -1.37302839,  1.11779249, -1.18028264,  2.05515862]
        
        input("Press enter to move to before transfer pose...")
        arm.set_joint_position(before_transfer_pos)

        input("Press Enter to switch to joint compliant mode...")
        arm.switch_to_joint_compliant_mode()

        compliant_trajectory = \
        [[ 3.41763889, -1.61951779, -2.60986085, -1.37302839,  1.11779249, -1.18028264, -4.22802669], \
        [-2.86177592, -1.61810199, -2.59101636, -1.41315263,  1.13104346, -1.15088751,  2.01918406], \
        [-2.85938399, -1.61746006, -2.57356532, -1.45686722,  1.14847337, -1.12112886,  1.97867146], \
        [-2.85699205, -1.61681813, -2.55611428, -1.5005818 ,  1.16590328, -1.09137022,  1.93815887], \
        [-2.85979598, -1.61647689, -2.54476893, -1.54186077,  1.1921008 , -1.06885054,  1.89284936], \
        [-2.86853489, -1.61647909, -2.54039782, -1.58035766,  1.22831315, -1.05459958,  1.84206055], \
        [-2.8772738 , -1.6164813 , -2.53602671, -1.61885454,  1.26452549, -1.04034862,  1.79127174], \
        [-2.88672904, -1.61644742, -2.53217158, -1.65680369,  1.30144078, -1.0273325 ,  1.74020177], \
        [-2.90328911, -1.61605558, -2.53343407, -1.68932024,  1.34532813, -1.02656399,  1.68634301], \
        [-2.91984918, -1.61566374, -2.53469656, -1.72183679,  1.38921548, -1.02579548,  1.63248426], \
        [-2.93640926, -1.6152719 , -2.53595905, -1.75435334,  1.43310283, -1.02502697,  1.57862551], \
        [-2.95091056, -1.61056262, -2.53082475, -1.78301695,  1.47261615, -1.02237995,  1.52686039], \
        [-2.96429599, -1.60351324, -2.5222233 , -1.80959223,  1.5097587 , -1.01871475,  1.47623006], \
        [-2.97768142, -1.59646386, -2.51362185, -1.8361675 ,  1.54690125, -1.01504955,  1.42559973], \
        [-2.99135705, -1.58946825, -2.5052373 , -1.86250795,  1.58409559, -1.01192551,  1.37512468], \
        [-3.01231347, -1.58382185, -2.50229433, -1.88295666,  1.62258906, -1.02237821,  1.32854554], \
        [-3.0332699 , -1.57817545, -2.49935137, -1.90340537,  1.66108253, -1.03283091,  1.2819664 ], \
        [-3.05422633, -1.57252904, -2.49640841, -1.92385408,  1.699576  , -1.04328362,  1.23538726], \
        [-3.07484656, -1.56581272, -2.49182133, -1.94333075,  1.73700034, -1.05321743,  1.18926345], \
        [-3.09418075, -1.55500373, -2.48094514, -1.95908914,  1.77033499, -1.06116636,  1.1448814 ], \
        [-3.11351495, -1.54419474, -2.47006895, -1.97484754,  1.80366965, -1.06911529,  1.10049935], \
        [-3.13284914, -1.53338575, -2.45919276, -1.99060593,  1.8370043 , -1.07706422,  1.05611729], \
        [ 3.12668153, -1.5248626 , -2.45171954, -2.00457495,  1.87018345, -1.09165458,  1.01553995], \
        [ 3.09451769, -1.52084146, -2.45094859, -2.01501973,  1.9030563 , -1.11932541,  0.98245608], \
        [ 3.06235385, -1.51682032, -2.45017764, -2.02546452,  1.93592916, -1.14699625,  0.94937221], \
        [ 3.03019002, -1.51279918, -2.44940669, -2.03590931,  1.96880201, -1.17466709,  0.91628833], \
        [ 2.99663442, -1.50842462, -2.44688751, -2.0443399 ,  2.0004814 , -1.20287614,  0.88544804], \
        [ 2.96116052, -1.50356293, -2.4419587 , -2.04999427,  2.0305158 , -1.23182704,  0.85770013], \
        [ 2.92568662, -1.49870124, -2.43702988, -2.05564863,  2.0605502 , -1.26077794,  0.82995223], \
        [ 2.89021272, -1.49383955, -2.43210107, -2.06130299,  2.0905846 , -1.28972884,  0.80220433], \
        [ 2.84880974, -1.49147307, -2.42777859, -2.06286484,  2.12114659, -1.32327376,  0.78068953], \
        [ 2.80508202, -1.49008493, -2.42369386, -2.06282204,  2.15191545, -1.35861997,  0.76161868], \
        [ 2.76135429, -1.4886968 , -2.41960912, -2.06277924,  2.18268431, -1.39396618,  0.74254784], \
        [ 2.71934301, -1.48619572, -2.41247589, -2.06068723,  2.21212124, -1.4254812 ,  0.72368812], \
        [ 2.68073937, -1.48148513, -2.39929054, -2.05452696,  2.23891391, -1.44939028,  0.70524755], \
        [ 2.64213573, -1.47677455, -2.3861052 , -2.04836669,  2.26570658, -1.47329935,  0.68680699], \
        [ 2.60353209, -1.47206397, -2.37291986, -2.04220642,  2.29249926, -1.49720842,  0.66836642]]

        # normalize compliant trajectory
        for pos in compliant_trajectory:
            for i in range(7):
                if pos[i] > np.pi:
                    pos[i] -= 2 * np.pi
                if pos[i] < -np.pi:
                    pos[i] += 2 * np.pi
    
        input("Press Enter to move to compliant trajectory...")
        arm.compliant_set_joint_trajectory(compliant_trajectory)

        # input("Press Enter to move to joint compliant position...")
        # arm.compliant_set_joint_position(
            # [-2.86554642, -1.61951779, -2.60986085, -1.37302839,  1.11779249, -1.18028264,  2.05515862]
        # )

        # def error_from_goal(arm):
        #     goal = [-2.86554642, -1.61951779, -2.60986085, -1.37302839,  1.11779249, -1.18028264,  2.05515862]
        #     # print upto 4 decimal places
        #     np.set_printoptions(precision=4, suppress=True)

        #     while True:
        #         current, _, _ = arm.get_update_state()
        #         current = np.array(current)
        #         goal = np.array(goal)
        #         # print("Current: ", current)
        #         # print("Goal: ", goal)
        #         error = goal - current
        #         norm = np.linalg.norm(error)
        #         # print("Error: ", error)
        #         print(f"Norm: {norm} Error: {error}")

        # # start thread to print error from goal
        # error_thread = threading.Thread(target=error_from_goal, args=(arm,))
        # error_thread.start()

        input("Press Enter to switch out of joint compliant mode...")
        arm.switch_out_of_joint_compliant_mode()

        # error_thread.join()

        input("Press Enter to move to before transfer pose...")
        arm.set_joint_position(before_transfer_pos)

        # print("Current Arm State:", arm.get_state())

        # transfer_pose = [-2.76117261, -1.18670831, -1.7014329 , -1.81186993,  0.26973719, -0.09092458,  2.4944019]
        # input("Press Enter to set the arm to transfer pose")
        # arm.set_joint_position(transfer_pose)

        # outside_cup_pose = (
        #     np.array([0.545, 0.45, 0.270]),
        #     np.array([-0.2126311, -0.6743797, -0.6743797, 0.2126311]),
        # )
        # outside_cup_pos = [
        #     -3.100185292329023,
        #     -1.0924888665911388,
        #     -0.5706994426374399,
        #     -1.424560020809773,
        #     -1.4250553687725285,
        #     -1.041275746196697,
        #     -2.8561579774322996,
        # ]

        # cup_inside_mount = (
        #     np.array([0.545, 0.518, 0.270]),
        #     np.array([-0.2126311, -0.6743797, -0.6743797, 0.2126311]),
        # )
        # cup_inside_mount_pos = [
        #     3.042634381172411,
        #     -1.168988168903665,
        #     -0.5663478374162505,
        #     -1.2153447487342381,
        #     -1.3638740364179194,
        #     -1.0536210957458687,
        #     -2.9810178882956833,
        # ]

        # above_cup_pose = (
        #     np.array([0.545, 0.518, 0.370]),
        #     np.array([-0.2126311, -0.6743797, -0.6743797, 0.2126311]),
        # )
        # above_cup_pos = [
        #     3.0622933037071576,
        #     -0.9648787092700299,
        #     -0.5952463310369183,
        #     -1.2963117700914815,
        #     -1.4352504820575698,
        #     -0.9462605500892867,
        #     -3.085153612188289,
        # ]

        # before_transfer_pos = [
        #     -2.86554642,
        #     -1.61951779,
        #     -2.60986085,
        #     -1.37302839,
        #     1.11779249,
        #     -1.18028264,
        #     2.05515862,
        # ]

        # input("Press enter to move to outside cup pose...")
        # arm.set_joint_position(outside_cup_pos)

        # input("Press enter to move to outside cup mount pose...")
        # arm.set_ee_pose(outside_cup_pose[0], outside_cup_pose[1])

        # input("Press enter to move to inside cup mount pose...")
        # arm.set_ee_pose(cup_inside_mount[0], cup_inside_mount[1])

        # input("Press enter to grasp the cup...")
        # arm.set_gripper(0.5)

        # input("Press enter to pickup the cup...")
        # arm.set_ee_pose(above_cup_pose[0], above_cup_pose[1])

        # input("Press enter to move to before transfer pose...")
        # arm.set_joint_position(before_transfer_pos)

        # input("Press enter to move above the cup...")
        # arm.set_joint_position(above_cup_pos)

        # input("Press enter to move to inside cup mount pose...")
        # arm.set_ee_pose(cup_inside_mount[0], cup_inside_mount[1])

        # input("Press enter to release the cup...")
        # arm.set_gripper(1.0)

        # input("Press enter to move to outside cup pose...")
        # arm.set_ee_pose(outside_cup_pose[0], outside_cup_pose[1])

        # home_pos = [
        #     2.2912759438800285,
        #     0.7308686750765581,
        #     2.082994642398784,
        #     4.109475142253324,
        #     0.2853091081120964,
        #     5.818345985240578,
        #     5.988186420599291,
        # ]

        # inside_mount_pose = (
        #     np.array([-0.147, -0.17, 0.07]),
        #     np.array([0.7071068, -0.7071068, 0, 0]),
        # )

        # outside_mount_pose = (
        #     np.array([-0.147, -0.29, 0.07]),
        #     np.array([0.7071068, -0.7071068, 0, 0]),
        # )

        # outside_mount_joint_states = [
        #     2.6266411620509817,
        #     0.6992626121546339,
        #     2.306749708761716,
        #     4.053362604401464,
        #     0.9559379448584164,
        #     5.655628973165609,
        #     5.80065247559031,
        # ]

        # above_mount_pose = (
        #     np.array([-0.147, -0.17, 0.15]),
        #     np.array([0.7071068, -0.7071068, 0, 0]),
        # )

        # above_mount_joint_states = [
        #     3.300153003835367,
        #     0.39120874346320217,
        #     1.8613410764520344,
        #     3.862447510072517,
        #     0.6143839397882825,
        #     5.583536137192727,
        #     6.276739392077158,
        # ]

        # infront_mount_pose = (
        #     np.array([0.0, -0.17, 0.15]),
        #     np.array([0.7071068, -0.7071068, 0, 0]),
        # )

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
            if joint_state_thread is not None:
                joint_state_thread.join()
            arm.close()  # Ensure the arm is disconnected properly
        except Exception as e:
            print(f"Error during client shutdown: {e}")
