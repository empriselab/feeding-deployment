"""An interface for perception (robot joints, human head poses, etc.)."""

import threading
import time
from typing import Any

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


class WebInterface:
    '''
    An interface to interact with the web interface.
    '''
    def __init__(self):
        
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
        self.web_interface_sub = rospy.Subscriber("WebAppCommPref", String, self._message_callback)

    def update_web_interface_image(self, image):
        self.last_captured_ros_image = self.image_bridge.cv2_to_compressed_imgmsg(image)

    def _publish_web_interface_image(self):
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown():
            self.web_interface_image_publisher.publish(self.last_captured_ros_image)
            rate.sleep()

    def _message_callback(self, msg: "String") -> None:
        """Callback for the web interface."""
        msg_dict = json.loads(msg.data)
        print(f"Received message: {msg_dict}")
        if msg_dict["state"] == "order_selection" and msg_dict["status"] != "ready_for_initial_data":
            self.user_preference = msg_dict["status"]
            print("SETTING USER PREFERENCE: ", self.user_preference)
        else:
            print("DID NOT SET USER PREFERENCE")

    def send_web_interface_message(self, msg_dict: dict[str, Any]) -> None:
        self.web_interface_publisher.publish(
            String(json.dumps(msg_dict))
        )

    def _send_web_interface_image(self, image) -> None:
        self._perception_interface.update_web_interface_image(image)