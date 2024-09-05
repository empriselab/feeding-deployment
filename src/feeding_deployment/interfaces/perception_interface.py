"""An interface for perception (robot joints, human head poses, etc.)."""

import threading
import time

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
from scipy.spatial.transform import Rotation as R
import json


try:
    import rospy
    from sensor_msgs.msg import JointState, CompressedImage
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from cv_bridge import CvBridge


    from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper
except ModuleNotFoundError:
    pass

from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient

class PerceptionInterface:
    """An interface for perception (robot joints, human head poses, etc.)."""

    def __init__(self, robot_interface: ArmInterfaceClient | None, record_goal_pose: bool = False) -> None:
        self._robot_interface = robot_interface
        
        # Create a shared publisher for rviz simulation.
        self.sim_joint_publishers = rospy.Publisher("/sim/robot_joint_states", JointState, queue_size=10)
        self.static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Create a publisher for communication with the web interface.
        self.web_interface_publisher = rospy.Publisher("/ServerComm", String, queue_size=10)
        self.web_interface_image_publisher = rospy.Publisher("/camera/image/compressed", CompressedImage, queue_size=10)
        self.image_bridge = CvBridge()
        self.last_captured_ros_image = None
        self.update_web_interface_image(np.zeros((512, 512, 3)))
        thread = threading.Thread(target=self._publish_web_interface_image)
        thread.start()
        # The following is a hacky leaky abstraction to handle the one-time preference
        # setting step at the beginning of FLAIR.
        self.user_preference = None
        self.web_interface_sub = rospy.Subscriber(
            "WebAppComm", String, self._web_interface_callback
        )

        # This doesn't work in simulation.
        # rospy.Timer(rospy.Duration(1.0), self._web_interface_image_callback)

        # run head perception
        if robot_interface is None:
            self._head_perception = None
        else:
            # self._head_perception = None
            self._head_perception = HeadPerceptionROSWrapper(record_goal_pose)
            
            # warm start head perception
            # self._head_perception.set_tool("fork")
            # for _ in range(10):
            #     self._head_perception.run_head_perception()

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
        return self._head_perception.get_top_camera_data()

    def get_head_perception_forque_target_pose(self, simulation = False) -> Pose:
        """Get a target of the forque from head perception."""
        if self._head_perception is not None and not simulation:
            forque_target_transform = self._head_perception.run_head_perception()
            print("\n--\n---\n----Forque target transform: ", forque_target_transform)
        else:
            # Use a sensible default value for testing in simulation.
            forque_target_transform = np.array(
                [[ 2.39288367e-02,  8.46555150e-04, -9.99713306e-01, -9.36197722e-02],
                [-9.98958576e-01, -3.88389663e-02, -2.39436604e-02,  4.75341624e-01],
                [-3.88481010e-02,  9.99245124e-01, -8.36977532e-05,  6.02467578e-01],
                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
            )
        forque_target_pose = Pose(
            (
                forque_target_transform[0, 3],
                forque_target_transform[1, 3],
                forque_target_transform[2, 3],
            ),
            R.from_matrix(forque_target_transform[:3, :3]).as_quat(),
        )
        return forque_target_pose
    
    def rviz_joint_state_update(self, joints: JointPositions):
        self.sim_joint_publishers.publish(
                JointState(
                    name=[
                        "joint_1", "joint_2", "joint_3", 
                        "joint_4", "joint_5", "joint_6", 
                        "joint_7", "finger_joint"
                    ],
                    position=joints[:7] + [0.0]  # Assuming you want to add 0.0 for the finger_joint
                )
            )
        
    def rviz_tool_update(self, pick: bool, held_object: str, object_pose: Pose) -> None:

        if held_object == "utensil":
            tool_base = "forkbase"
        elif held_object == "drink":
            tool_base = "drinkbase"
        elif held_object == "wipe":
            tool_base = "wipebase"

        if pick:
            self.publish_static_transform("sim/finger_tip", "sim/" + tool_base, object_pose)
        else:
            self.publish_static_transform("sim/base_link", "sim/" + tool_base, object_pose)
    
    def publish_static_transform(self, parent_frame: str, child_frame: str, pose: Pose) -> None:

        static_transform_stamped = TransformStamped()

        static_transform_stamped.header.stamp = rospy.Time.now()
        static_transform_stamped.header.frame_id = parent_frame
        static_transform_stamped.child_frame_id = child_frame

        static_transform_stamped.transform.translation.x = pose.position[0]
        static_transform_stamped.transform.translation.y = pose.position[1]
        static_transform_stamped.transform.translation.z = pose.position[2]

        static_transform_stamped.transform.rotation.x = pose.orientation[0]
        static_transform_stamped.transform.rotation.y = pose.orientation[1]
        static_transform_stamped.transform.rotation.z = pose.orientation[2]
        static_transform_stamped.transform.rotation.w = pose.orientation[3]

        self.static_transform_broadcaster.sendTransform(static_transform_stamped)

    def update_web_interface_image(self, image):
        self.last_captured_ros_image = self.image_bridge.cv2_to_compressed_imgmsg(image)

    def _publish_web_interface_image(self):
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown():
            self.web_interface_image_publisher.publish(self.last_captured_ros_image)
            rate.sleep()

    def _web_interface_callback(self, msg: "String") -> None:
        """Callback for the web interface."""
        msg_dict = json.loads(msg.data)
        if msg_dict["state"] == "order_selection" and msg_dict["status"] != "ready_for_initial_data":
            self.user_preference =msg_dict["status"]

