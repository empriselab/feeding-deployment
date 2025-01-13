"""An interface for perception (robot joints, human head poses, etc.)."""

import threading
import time
from pathlib import Path
import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
from scipy.spatial.transform import Rotation as R
import json
import pickle
import serial

LED_SERIAL_PORT = '/dev/ttyACM0'
LED_BAUD_RATE = 115200

try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String, Bool
    import tf2_ros
    from geometry_msgs.msg import WrenchStamped, Point, Pose as PoseMsg
    from netft_rdt_driver.srv import String_cmd

    from feeding_deployment.perception.head_perception.ros_wrapper import HeadPerceptionROSWrapper
    from feeding_deployment.perception.aruco_perception.aruco_perception import ArUcoPerception
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.utils.camera_utils import CustomCameraInfo

class PerceptionInterface:
    """An interface for perception (robot joints, human head poses, etc.)."""

    def __init__(self, robot_interface: ArmInterfaceClient | None, record_goal_pose: bool = False, simulate_head_perception: bool = False, log_dir: str | None = None) -> None:
        self.robot_interface = robot_interface
        self._simulate_head_perception = simulate_head_perception
        self.log_dir = log_dir

        # run head perception
        if self.robot_interface is None:
            self.simulation = True
            self._head_perception = None
            self._aruco_perception = None
        else:
            self.simulation = False
            self.tfBuffer = tf2_ros.Buffer()
            self.listener = tf2_ros.TransformListener(self.tfBuffer)

            self._head_perception = HeadPerceptionROSWrapper(record_goal_pose)
            
            # warm start head perception only if we're not recording the goal pose
            if not record_goal_pose:
                self._head_perception.set_tool("fork")
                for _ in range(10):
                    self._head_perception.run_head_perception()

            # Rajat ToDo: pass perception queues to all perception classes instead of having them use ros subscribers which spawn threads
            self._aruco_perception = ArUcoPerception()

            self.speak_pub = rospy.Publisher('/speak', String, queue_size=1)

            self.transfer_button = False
            self.transfer_button_sub = rospy.Subscriber('/transfer_button', Bool, self.transfer_button_callback)
            
            self.ft_threshold_exceeded = False
            self.ft_sensor_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, self.ft_callback)

        self.head_perception_data_lock = threading.Lock()
        # this term is updated in the run_head_perception method and read in the get_tool_tip_pose method
        self.head_perception_data = None

        # Head perception thread setup
        self.head_perception_thread = None
        self.kill_the_thread = False
        self.head_perception_running = False

        # set led brightness
        self.set_led_brightness()

    def zero_ft_sensor(self):
        print("Zeroing FT sensor")
        if self.simulation:
            return
        bias = rospy.ServiceProxy('/forque/bias_cmd', String_cmd)
        bias('bias')
        
    def speak(self, text):
        print("Speaking: ", text)
        if self.simulation:
            return
        self.speak_pub.publish(String(data=text))

    def set_led_brightness(self, brightness: float = 0.2):
        print("Setting LED Brightness")
        if self.simulation:
            return
        with serial.Serial(LED_SERIAL_PORT, LED_BAUD_RATE, timeout=1) as ser:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            # Convert brightness to string, encode to bytes, and concatenate
            command = f"BRIGHTNESS {brightness}\r\n".encode()
            ser.write(command)

    def turn_on_led(self):
        with serial.Serial(LED_SERIAL_PORT, LED_BAUD_RATE, timeout=1) as ser:
            ser.reset_input_buffer()  # Clear input buffer
            ser.reset_output_buffer()  # Clear output buffer
            ser.write(b"ON\r\n")  # Send the command

    def turn_off_led(self):
        with serial.Serial(LED_SERIAL_PORT, LED_BAUD_RATE, timeout=1) as ser:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            ser.write(b"OFF\r\n")

    def detect_button_press(self):
        print("Waiting for button press")
        if self.simulation:
            return True
        
        self.transfer_button = False
        # wait for button press
        while not rospy.is_shutdown() and not self.transfer_button:
            time.sleep(0.05)
        self.transfer_button = False
        return True
    
    def detect_force_trigger(self):
        print("Waiting for force torque threshold to be exceeded")
        if self.simulation:
            return True
        
        self.ft_threshold_exceeded = False
        # wait for force torque threshold to be exceeded
        while not rospy.is_shutdown() and not self.ft_threshold_exceeded:
            time.sleep(0.05)
        self.ft_threshold_exceeded = False
        return True

    def ft_callback(self, msg):

        ft_reading = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

        down_torque = ft_reading[3]
        if np.abs(down_torque) > 0.1:
            self.ft_threshold_exceeded = True

    def transfer_button_callback(self, msg):
        print("Transfer button pressed")
        self.transfer_button = True
        
    def get_robot_joints(self) -> "JointState":
        """Get the current robot joint state."""
        joint_state_msg = rospy.wait_for_message("/robot_joint_states", JointState)
        q = np.array(joint_state_msg.position[:7])
        gripper_position = joint_state_msg.position[7]
        
        joint_state = q.tolist() + [
            gripper_position,
            gripper_position,
            gripper_position,
            gripper_position,
            -gripper_position,
            -gripper_position,
        ]
        return joint_state

    def get_camera_data(self):  # Rajat ToDo: Add return type
        camera_color_data, camera_info_data, camera_depth_data, _ = self._head_perception.get_camera_data()
        camera_info = CustomCameraInfo(fx=camera_info_data.K[0], fy=camera_info_data.K[4], cx=camera_info_data.K[2], cy=camera_info_data.K[5])
        return camera_color_data, camera_info, camera_depth_data
    
    def set_head_perception_tool(self, tool: str) -> None:
        """Set the tool for head perception."""
        self.tool = tool
        if self._head_perception is not None:
            self._head_perception.set_tool(tool)

    def head_perception_thread_is_running(self) -> bool:
        return self.head_perception_running

    def start_head_perception_thread(self):
        assert not self.head_perception_running, "Head perception thread is already running" 

        # Start head perception thread
        self.kill_the_thread = False
        self.head_perception_thread = threading.Thread(
            target=self.run_head_perception_thread, args=(), daemon=True
        )
        self.head_perception_thread.start()
        print("Head perception thread started")

    def run_head_perception_thread(self):
        self.head_perception_running = True

        t_init = time.time()
        while not self.kill_the_thread:
            t_now = time.time()
            step_time = t_now - t_init
            if step_time >= 0.02:  # 50 Hz
                if self._head_perception is not None and not self._simulate_head_perception:
                    head_perception_data = self._head_perception.run_head_perception()
                else:
                    try:
                        # read from logged data
                        with open(self.log_dir / f'head_perception_data_{self.tool}.pkl', 'rb') as f:
                            head_perception_data = pickle.load(f)
                    except FileNotFoundError:
                        raise FileNotFoundError("No transfer logged data found for tool: ", self.tool)
                with self.head_perception_data_lock:
                    self.head_perception_data = head_perception_data
        self.head_perception_running = False

    def stop_head_perception_thread(self):
        if self.head_perception_running:
            self.kill_the_thread = True
            self.head_perception_thread.join()
            print("Head perception thread stopped")
        else:
            print("Head perception thread is not running")

    def get_head_perception_data(self) -> dict:
        """Get head perception data (head pose, face keypoints, tool tip target pose)."""

        with self.head_perception_data_lock:
            head_perception_data = self.head_perception_data

        # # Just for testing
        # benjamin_tool_tip_target_pose = np.eye(4)
        # benjamin_tool_tip_target_pose[:3, 3] = [-0.282, 0.540, 0.619]
        # benjamin_tool_tip_target_pose[:3, :3] = R.from_quat([-0.490, 0.510, 0.511, -0.489]).as_matrix()
        # head_perception_data["tool_tip_target_pose"] = benjamin_tool_tip_target_pose

        # save them in a pickle file
        if self.robot_interface is not None and self.log_dir is not None and self._simulate_head_perception == False:
            with open(self.log_dir / f'head_perception_data_{self.tool}.pkl', 'wb') as f:
                pickle.dump(head_perception_data, f)
            
        return head_perception_data
        
    def get_tool_tip_pose(self) -> np.ndarray:

        current_state = self.robot_interface.get_state()
        ee_pose = current_state["ee_pos"]

        tool_tip_pose = np.eye(4)
        tool_tip_pose[:3, 3] = ee_pose[:3]
        tool_tip_pose[:3, :3] = R.from_quat(ee_pose[3:]).as_matrix()

        return tool_tip_pose
    
    def get_tool_tip_pose_at_staging(self) -> np.ndarray:

        tool_tip_staging_pose = np.eye(4)

        # Rajat ToDo: Fix these hardcoded values
        if self.tool == "fork":
            tool_tip_staging_pose[:3, 3] = [0.250, 0.272, 0.518]
            tool_tip_staging_pose[:3, :3] = R.from_quat([0.523, -0.503, -0.469, 0.503]).as_matrix()
        elif self.tool == "drink":
            tool_tip_staging_pose[:3, 3] = [0.289, 0.315, 0.587]
            tool_tip_staging_pose[:3, :3] = R.from_quat([0.523, -0.503, -0.469, 0.503]).as_matrix()
        elif self.tool == "wipe":
            tool_tip_staging_pose[:3, 3] = [0.367, 0.277, 0.506]
            tool_tip_staging_pose[:3, :3] = R.from_quat([0.523, -0.503, -0.469, 0.503]).as_matrix()

        return tool_tip_staging_pose

    def getTransformationFromTF(self, source_frame, target_frame):

        while not rospy.is_shutdown():
            try:
                # print("Looking for transform")
                transform = self.tfBuffer.lookup_transform(source_frame, target_frame, rospy.Time())
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.control_rate.sleep()
                continue

        T = np.zeros((4,4))
        T[:3,:3] = R.from_quat([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]).as_matrix()
        T[:3,3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]).reshape(1,3)
        T[3,3] = 1

        return T

    def perceive_drink_pickup_poses(self):
        
        if self.simulation:
            # load them from a pickle file
            with open(Path(__file__).parent.parent / 'integration' / 'log' / 'drink_pickup_pos.pkl', 'rb') as f:
                drink_pickup_pos = pickle.load(f)
            drink_poses = drink_pickup_pos["last_drink_poses"]

        else:
            # Rajat Hack: Wait one second for the aruco mean to be correct, does this actually help though?
            time.sleep(1)

            aruco_pose_msg = rospy.wait_for_message("/aruco_pose", PoseMsg)
            position = (aruco_pose_msg.position.x, aruco_pose_msg.position.y, aruco_pose_msg.position.z)
            orientation = (aruco_pose_msg.orientation.x, aruco_pose_msg.orientation.y, aruco_pose_msg.orientation.z, aruco_pose_msg.orientation.w)
            self.aruco_pose = (position, orientation)

            drink_poses  = {}
            drink_poses['pre_grasp_pose'] = self.get_aruco_relative_pose(self.get_pre_grasp_transform())
            drink_poses['inside_bottom_pose'] = self.get_aruco_relative_pose(self.get_inside_bottom_transform())
            drink_poses['inside_top_pose'] = self.get_aruco_relative_pose(self.get_inside_top_transform())
            drink_poses['post_grasp_pose'] = self.get_aruco_relative_pose(self.get_post_grasp_pose())
            drink_poses['place_inside_bottom_pose'] = self.get_aruco_relative_pose(self.get_place_inside_bottom_transform())
            drink_poses['place_pre_grasp_pose'] = self.get_aruco_relative_pose(self.get_place_pre_grasp_transform())

        self.last_drink_poses = drink_poses

        return drink_poses
    
    def record_drink_pickup_joint_pos(self):
        if self.simulation:
            return
        
        self.drink_pickup_joint_pos = self.get_robot_joints()[:7]
        # save them in a pickle file
        drink_pickup_pos = {
            "last_drink_poses": self.last_drink_poses,
            "drink_pickup_joint_pos": self.drink_pickup_joint_pos
        }
        with open(Path(__file__).parent.parent / 'integration' / 'log' / 'drink_pickup_pos.pkl', 'wb') as f:
            pickle.dump(drink_pickup_pos, f)
        print("Drink pickup poses recorded")

    def get_last_drink_pickup_configs(self, study_poses = False):
        if study_poses:
            with open(Path(__file__).parent.parent / 'integration' / 'log' / 'study_drink_pickup_pos.pkl', 'rb') as f:
                drink_pickup_pos = pickle.load(f)
            last_drink_poses = drink_pickup_pos["last_drink_poses"]
            drink_pickup_joint_pos = drink_pickup_pos["drink_pickup_joint_pos"]
        else:
            try:
                last_drink_poses = self.last_drink_poses
                drink_pickup_joint_pos = self.drink_pickup_joint_pos
            except Exception as e:
                print("Error loading drink pickup poses from script, using values from file instead")
                with open(Path(__file__).parent.parent / 'integration' / 'log' / 'drink_pickup_pos.pkl', 'rb') as f:
                    drink_pickup_pos = pickle.load(f)
                last_drink_poses = drink_pickup_pos["last_drink_poses"]
                drink_pickup_joint_pos = drink_pickup_pos["drink_pickup_joint_pos"]
        
        return last_drink_poses, drink_pickup_joint_pos

    def get_aruco_relative_pose(self, transform, override_angles = True):
        aruco_pos_mat = self.pose_to_matrix(self.aruco_pose)
        goal_frame = np.dot(aruco_pos_mat, transform)
        goal_pose = self.matrix_to_pose(goal_frame)

        # If true, use 2 hardcoded angle values.
        if override_angles:
            rot = R.from_quat(goal_pose[1])
            roll = np.pi / 2
            pitch = 0
            _, _, yaw = rot.as_euler("xyz")
            new_rot = R.from_euler("xyz", [roll, pitch, yaw])
            goal_pose = Pose(goal_pose[0], new_rot.as_quat())

        return goal_pose

    def get_pre_grasp_transform(self):
        tf = np.zeros((4, 4))
        tf[:3, :3] = R.from_euler("xyz", [np.pi, 0, np.pi / 2]).as_matrix()
        tf[:3, 3] = np.array([0.09, -0.02, 0.1]) 
        tf[3, 3] = 1
        return tf

    def get_inside_bottom_transform(self):
        tf = self.get_pre_grasp_transform()
        tf[2, 3] = 0.01
        return tf

    def get_inside_top_transform(self):
        tf = self.get_inside_bottom_transform()
        tf[0, 3] = 0.14
        return tf
    
    def get_post_grasp_pose(self):
        tf = self.get_inside_top_transform()
        tf[0, 3] = 0.25
        return tf
    
    def get_place_inside_bottom_transform(self):
        tf = self.get_inside_bottom_transform()
        # tf[1, 3] = 0.0
        return tf

    def get_place_pre_grasp_transform(self):
        tf = self.get_pre_grasp_transform()
        # tf[1, 3] = 0.0
        return tf

    def pose_to_matrix(self, pose):
        position = pose[0]
        orientation = pose[1]
        pose_matrix = np.zeros((4, 4))
        pose_matrix[:3, 3] = position
        pose_matrix[:3, :3] = R.from_quat(orientation).as_matrix()
        pose_matrix[3, 3] = 1
        return pose_matrix
    
    def matrix_to_pose(self, mat):
        position = mat[:3, 3]
        orientation = R.from_matrix(mat[:3, :3]).as_quat()
        return Pose(position, orientation) 