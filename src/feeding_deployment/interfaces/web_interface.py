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
        self.web_interface_sub = rospy.Subscriber("WebAppComm", String, self._message_callback, queue_size=100)
        time.sleep(1.0)  # Wait for the subscriber to connect

    def update_web_interface_image(self, image):
        self.last_captured_ros_image = self.image_bridge.cv2_to_compressed_imgmsg(image)

    def _publish_web_interface_image(self):
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown():
            self.web_interface_image_publisher.publish(self.last_captured_ros_image)
            rate.sleep()

    def _message_callback(self, msg: "String") -> None:
        """Callback for the web interface."""
        print("(web interface) Received message on WebAppComm: ", msg.data)
        # msg_dict = json.loads(msg.data)
        # print(f"Received message: {msg_dict}")
        # if msg_dict["state"] == "order_selection" and msg_dict["status"] != "ready_for_initial_data":
        #     self.user_preference = msg_dict["status"]
        #     print("SETTING USER PREFERENCE: ", self.user_preference)
        # else:
        #     print("DID NOT SET USER PREFERENCE")

    def send_web_interface_message(self, msg_dict: dict[str, Any]) -> None:
        self.web_interface_publisher.publish(
            String(json.dumps(msg_dict))
        )

    def _send_web_interface_image(self, image) -> None:
        self._perception_interface.update_web_interface_image(image)

if __name__ == "__main__":
    rospy.init_node("test_web_interface")
    web_interface = WebInterface()

    web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
    time.sleep(1.0) # simulate delay, also needed for web interface
    web_interface.update_web_interface_image(np.ones((512, 512, 3)))
    time.sleep(1.0)  # simulate delay, also needed for web interface


    # Wait for web interface to report order selection.
    print("WAITING TO GET PREFERENCE")
    while web_interface.user_preference is None:
        print("user preference is still None")
        time.sleep(1e-1)
    print("FINISHED GETTING PREFERENCES")

    print("User Preference:", web_interface.user_preference)
    input("Received user preference. Press Enter to continue...")
    rospy.spin()