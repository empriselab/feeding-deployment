'''
Entrypoint for controlling the robot arm on compute machine. Additionally runs two important threads:
1. A thread that checks no safety anomalies have occurred using the watchdog
2. A thread that publishes joint states to ROS
'''

import threading
import time
import numpy as np

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose
# from netft_rdt_driver.srv import String_cmd

from feeding_deployment.robot_controller.arm_interface import ArmInterface, ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY
from feeding_deployment.robot_controller.command_interface import KinovaCommand, JointTrajectoryCommand, JointCommand, CartesianCommand, OpenGripperCommand, CloseGripperCommand
# from feeding_deployment.safety.watchdog import WATCHDOG_MONITOR_FREQUENCY, PeekableQueue

class ArmInterfaceClient:
    def __init__(self):

        # make sure watchdog is running
        print("Waiting for Watchdog status...")
        rospy.wait_for_message("/watchdog_status", Bool)
        print("Watchdog is running, continuing...")

        # Register ArmInterface (no lambda needed on the client-side)
        ArmManager.register("ArmInterface")

        # Client setup
        self.manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        self.manager.connect()

        # This will now use the single, shared instance of ArmInterface
        self._arm_interface = self.manager.ArmInterface()
        self.in_compliant_mode = False

    def switch_to_task_compliant_mode(self):
        assert not self.in_compliant_mode, "Already in compliant mode"
        self._arm_interface.switch_to_task_compliant_mode()
        self.in_compliant_mode = True

    def switch_to_joint_compliant_mode(self):
        assert not self.in_compliant_mode, "Already in compliant mode"
        self._arm_interface.switch_to_joint_compliant_mode()
        self.in_compliant_mode = True

    def switch_out_of_compliant_mode(self):
        assert self.in_compliant_mode, "Not in compliant mode"
        # time.sleep(2.0) # Wait for the arm to settle
        self._arm_interface.switch_out_of_compliant_mode()
        self.in_compliant_mode = False

    def get_state(self):
        return self._arm_interface.get_state()

    def set_tool(self, tool: str):
        assert not self.in_compliant_mode, "Cannot set tool in compliant mode"
        self._arm_interface.set_tool(tool)

    def execute_command(self, cmd: KinovaCommand) -> None:

        # if not self.in_compliant_mode:
            # input("Press enter to execute command...")

        if cmd.__class__.__name__ == "JointTrajectoryCommand":
            return self._arm_interface.set_joint_trajectory(cmd.traj)

        if cmd.__class__.__name__ == "JointCommand":
            if self.in_compliant_mode:
                return self._arm_interface.compliant_set_joint_position(cmd.pos)
            else:
                return self._arm_interface.set_joint_position(cmd.pos)

        if cmd.__class__.__name__ == "CartesianCommand":
            if self.in_compliant_mode:
                return self._arm_interface.compliant_set_ee_pose(cmd.pos, cmd.quat)
            else:
                return self._arm_interface.set_ee_pose(cmd.pos, cmd.quat)

        if cmd.__class__.__name__ == "OpenGripperCommand":
            return self._arm_interface.open_gripper()

        if cmd.__class__.__name__ == "CloseGripperCommand":
            return self._arm_interface.close_gripper()

        raise NotImplementedError(f"Unrecognized command: {cmd}")

if __name__ == "__main__":

    rospy.init_node("arm_interface_client", anonymous=True)
    arm_client_interface = ArmInterfaceClient()

    run_commands = input("Press 'y' to run commands")

    if run_commands != "y":
        exit()

    before_transfer_pos = [
        -2.8655331,  
        -1.61973777, 
        -2.6097253, 
        -1.37301134, 
        1.11781087,
        -1.18039928,
        2.05515662
    ]

    arm_client_interface.execute_command(JointCommand(before_transfer_pos))

    input("Press enter to go to compliance mode...")
    arm_client_interface.switch_to_task_compliant_mode()

    arm_pos, ee_pose, gripper_pos = arm_client_interface.get_state()
    drop_test_pose = np.zeros(7)
    drop_test_pose[:3] = [0.45, 0.62, 0.6]
    drop_test_pose[3:] = [-0.03083443277876381, 0.7132803649800029, 0.7001853591905794, -0.00456305428030798]
    drop_test_task_command = CartesianCommand(pos=drop_test_pose[:3], quat=drop_test_pose[3:])
    
    input("Press Enter to move to drop test pos")
    arm_client_interface.execute_command(drop_test_task_command)

    input('Press Enter to switch out of compliant mode')
    arm_client_interface.switch_out_of_compliant_mode()

    # close gripper
    # arm_client_interface.execute_command(CloseGripperCommand())

    # home_pos = [2.0099529289564592e-05, 0.26191187306569164, -3.1415742777782714, -2.269018308753582, -1.1185276577840852e-05, 0.9598948696060562, 1.5707649014940337]

    # print("Moving to home position...")
    # arm_client_interface.execute_command(JointCommand(home_pos))

    # test_positions = [
    #     [1.5966822288586847e-05, -0.3490701114566095, -3.141557766179513, -2.548308539644045, 3.904089478140269e-06, -0.8726486470306245, 1.5709589127794459],
    #     [-0.007946872692796703, -0.5015446059791344, 3.0858849162236246, -2.580108280956245, -0.025630794963140247, -0.9928940986732275, 1.5269496431336684],
    #     [-0.00580569117637264, -0.5003387266375467, -3.070703566429628, -2.337005881906106, 0.0373198167840403, -1.3192836650222874, 1.6169241410878321],
    #     [-0.01612437513599385, -0.5525894145926831, -3.0655101359878643, -2.5410106793089007, 0.04257673448947779, -1.0858053975194766, 1.6028609194775052]
    # ]

    # test_poses = [
    #     [0.1338980495929718, -0.0005073380307294428, 0.21062198281288147, 0.712547437295058, 0.700687805660259, 0.025881994666155476, 0.02535490002657547],
    #     [0.09481269866228104, 0.009776144288480282, 0.21593131124973297, 0.712496060477749, 0.7007277901717651, 0.02601313944266246, 0.025558647480240525],
    #     [0.10039354115724564, -0.015133513137698174, 0.2866659164428711, -0.7132249227562583, -0.700909127758032, 0.0036400298249693623, 0.004832635686417539],
    #     [0.08909790962934494, -0.01270302850753069, 0.23117974400520325, 0.7126763019999464, 0.7006222160108023, 0.025051701023530416, 0.024360034392870284]
    # ]

    # # print("Moving to retract position...")
    # # arm_client_interface.execute_command(JointCommand(test_positions[0]))

    # for i in range(4):
    #     # send move to pose command
    #     # arm_client_interface.execute_command(CartesianCommand(pos=test_poses[0][:3],quat=test_poses[0][3:]))
    #     arm_client_interface.execute_command(JointCommand(test_positions[0]))

    #     arm_client_interface.execute_command(JointCommand(home_pos))

    #     # arm_client_interface.execute_command(JointCommand(home_pos))
    #     # # send gripper open command
    #     arm_client_interface.execute_command(OpenGripperCommand())

    #     # send move to pose command
    #     arm_client_interface.execute_command(JointCommand(test_positions[1]))
    #     # arm_client_interface.execute_command(CartesianCommand(pos=test_poses[1][:3],quat=test_poses[1][3:]))

    #     # arm_client_interface.execute_command(JointCommand(home_pos))

    #     # send gripper close command
    #     arm_client_interface.execute_command(CloseGripperCommand())

    # print("Moving through test positions...")
    # for i in range(20):
    #     arm_client_interface.execute_command(JointCommand(test_positions[i % 4]))

    # print("Moving through test poses...")
    # for i in range(20):
    #     arm_client_interface.execute_command(CartesianCommand(pos=test_poses[i % 4][:3],quat=test_poses[i % 4][3:]))

    # print("Moving through test positions and poses...")
    # for i in range(20):
    #     if i % 4 < 2:
    #         arm_client_interface.execute_command(JointCommand(test_positions[i % 4]))
    #     else:
    #         arm_client_interface.execute_command(CartesianCommand(pos=test_poses[i % 4][:3], quat=test_poses[i % 4][3:]))

    # utensil_inside_mount = (
    #     np.array([0.242, -0.077, 0.07]),
    #     np.array([-1, 0, 0, 0]),
    # )

    # utensil_outside_mount = (
    #     utensil_inside_mount[0].copy(),
    #     utensil_inside_mount[1].copy(),
    # )
    # utensil_outside_mount[0][0] += 0.13

    # utensil_outside_above_mount = (
    #     utensil_outside_mount[0].copy(),
    #     utensil_outside_mount[1].copy(),
    # )
    # utensil_outside_above_mount[0][2] += 0.1

    # utensil_above_mount = (
    #     utensil_inside_mount[0].copy(),
    #     utensil_inside_mount[1].copy(),
    # )
    # utensil_above_mount[0][2] += 0.1

    # utensil_inside_mount_pos = [0.03610898146611135, 0.46058565446690675, -2.7959276599029583, -2.4802831496040416, -0.5912420020218754, -0.27659484555119374, 0.9346522303717946]

    # # depend on the offsets set above
    # utensil_above_mount_pos = [-0.3081224117999879, 0.1449308244187662, -2.4515079603418446, -2.3539334664268674, -0.14376009880356744, -0.6872590793313744, 0.5028097739444904]
    # utensil_outside_mount_pos = [-0.13616795916796942, 0.6152003983994736, -2.673508269862523, -2.09060888627143, -0.5157179493005364, -0.5540355110645967, 0.7160963868259976]
    # utensil_outside_above_mount_pos = [-0.2692035082617874, 0.4127082432063301, -2.513398492494741, -1.9930522355357558, -0.31928105676741936, -0.8392446174777604, 0.5472652562309106]

    # above_plate_pos = [
    #     -2.86495014,
    #     -1.61460533,
    #     -2.6115943,
    #     -1.37673391,
    #     1.11842806,
    #     -1.17904586,
    #     -2.6957422,
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

    # # input("Press enter to move to above utensil mount pose...")
    # # arm_client_interface.execute_command(CartesianCommand(utensil_above_mount[0], utensil_above_mount[1]))

    # input("Press enter to move to above utensil pos...")
    # arm_client_interface.execute_command(JointCommand(utensil_above_mount_pos))

    # input("Press enter to move to inside utensil mount pose...")
    # arm_client_interface.execute_command(CartesianCommand(utensil_inside_mount[0], utensil_inside_mount[1]))

    # input("Press enter to grasp the utensil...")
    # arm_client_interface.execute_command(OpenGripperCommand())

    # input("Press enter to move outside the mount...")
    # arm_client_interface.execute_command(CartesianCommand(utensil_outside_mount[0], utensil_outside_mount[1]))

    # input("Press enter to move to above outside utensil pose...")
    # arm_client_interface.execute_command(CartesianCommand(utensil_outside_above_mount[0], utensil_outside_above_mount[1]))

    # # input("Press enter to move to before transfer pose...")
    # # arm_client_interface.execute_command(JointCommand(before_transfer_pos))

    # input("Press enter to move to above plate pose...")
    # arm_client_interface.execute_command(JointCommand(above_plate_pos))

    # input("Press enter to move to above outside utensil mount pos...")
    # arm_client_interface.execute_command(JointCommand(utensil_outside_above_mount_pos))

    # input("Press enter to move outside the mount...")
    # arm_client_interface.execute_command(JointCommand(utensil_outside_mount_pos))

    # input("Press enter to move to inside utensil mount pose...")
    # arm_client_interface.execute_command(CartesianCommand(utensil_inside_mount[0], utensil_inside_mount[1]))

    # input("Press enter to ungrasp the utensil...")
    # arm_client_interface.execute_command(CloseGripperCommand())

    # input("Press enter to move to above utensil pose...")
    # arm_client_interface.execute_command(CartesianCommand(utensil_above_mount[0], utensil_above_mount[1]))
