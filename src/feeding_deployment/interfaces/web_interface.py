"""An interface for perception (robot joints, human head poses, etc.)."""

import threading
import time
from typing import Any
import pickle
import cv2
import argparse

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
from scipy.spatial.transform import Rotation as R
import json
import queue


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
    def __init__(self, hla_command_queue: queue.Queue) -> None:

        self.hla_command_queue = hla_command_queue
        
        # Create a publisher for communication with the web interface.
        self.web_interface_publisher = rospy.Publisher("/ServerComm", String, queue_size=10)
        self.web_interface_image_publisher = rospy.Publisher("/camera/image/compressed", CompressedImage, queue_size=10)
        self.image_bridge = CvBridge()
        self.last_captured_ros_image = None
        self.update_web_interface_image(np.zeros((512, 512, 3)))
        self.web_interface_image_thread = threading.Thread(target=self._publish_web_interface_image)
        self.web_interface_image_thread.start()
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
            # print("(web interface) publishing image...")
            self.web_interface_image_publisher.publish(self.last_captured_ros_image)
            rate.sleep()

    def _message_callback(self, msg: "String") -> None:
        """Callback for the web interface."""
        print("Received message on WebAppComm: ", msg.data)
        msg_dict = json.loads(msg.data)
        if msg_dict["state"] == "order_selection" and msg_dict["status"] != "ready_for_initial_data":
            self.user_preference = msg_dict["status"]
            print("SETTING USER PREFERENCE: ", self.user_preference)
        elif msg_dict["status"] in ["drink_pickup", "drink_transfer", "move_to_above_plate", "aquire_food", 0, "bite_skill_selection", "bite_transfer", "mouth_wiping", "return_to_main"]:
            print("Received high-level action message from web interface.")
            self.hla_command_queue.put(msg_dict)
        else:
            print("WARNING: Unrecognized message from web interface.")
            return

    def send_web_interface_message(self, msg_dict: dict[str, Any]) -> None:
        self.web_interface_publisher.publish(
            String(json.dumps(msg_dict))
        )

    def _send_web_interface_image(self, image) -> None:
        self._perception_interface.update_web_interface_image(image)

    # make sure thread is closed when object is deleted
    def __del__(self):
        self.web_interface_image_thread.join()

if __name__ == "__main__":
    rospy.init_node("test_web_interface")

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_manual_acquisition_pixels", action="store_true")
    parser.add_argument("--test_image_streaming", action="store_true")
    args = parser.parse_args()

    hla_command_queue = queue.Queue()
    web_interface = WebInterface(hla_command_queue)


    if args.test_manual_acquisition_pixels:
        plate_log = pickle.load(open("test_log/plate_log.pkl", "rb"))
        original_image = plate_log['original_image']
        plate_image = plate_log['plate_image']
        plate_bounds = plate_log['plate_bounds']
        print("Original image shape: ", original_image.shape)
        print("Plate image shape: ", plate_image.shape)
        print("Plate bounds: ", plate_bounds)

        cv2.imshow("plate_image", plate_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
        time.sleep(1.0) # simulate delay, also needed for web interface
        web_interface.update_web_interface_image(plate_image)
        time.sleep(1.0)  # simulate delay, also needed for web interface

        while not rospy.is_shutdown():
            try:
                msg_dict = hla_command_queue.get(timeout=1.0)
                if msg_dict["status"] == 0:
                    pos = msg_dict["positions"][0]

                    point_x = int(pos["x"]*plate_bounds[2]) + plate_bounds[0]
                    point_y = int(pos["y"]*plate_bounds[3]) + plate_bounds[1]

                    print("Plate Bounds:", plate_bounds)
                    print("Positions:", msg_dict["positions"])
                    print("Point:", point_x, point_y)

                    # visualize point on camera color image
                    viz = original_image.copy()
                    for pos in msg_dict["positions"]:
                        cv2.circle(viz, (point_x, point_y), 5, (0, 255, 0), -1)
                    
                    cv2.imshow("viz", viz)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except queue.Empty:
                continue

    elif args.test_image_streaming:
        for i in range(5):
            # read image
            image = cv2.imread(f"test_log/color_{i}.png")
            # Send the plate image to the web interface.
            input("Press Enter to send image to web interface.")
            web_interface.update_web_interface_image(image)

    
        

