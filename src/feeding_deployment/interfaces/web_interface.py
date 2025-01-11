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


    from feeding_deployment.perception.head_perception.ros_wrapper import HeadPerceptionROSWrapper
except ModuleNotFoundError:
    pass


class WebInterface:
    '''
    An interface to interact with the web interface.
    '''
    def __init__(self, hla_command_queue: queue.Queue = None) -> None:

        self.hla_command_queue = hla_command_queue
        
        # Create a publisher for communication with the web interface.
        self.web_interface_publisher = rospy.Publisher("/ServerComm", String, queue_size=10)
        self.web_interface_image_publisher = rospy.Publisher("/camera/image/compressed", CompressedImage, queue_size=10)
        self.image_bridge = CvBridge()
        self.user_preference = None
        self.web_interface_sub = rospy.Subscriber("WebAppComm", String, self._message_callback, queue_size=100)
        time.sleep(1.0)  # Wait for the subscriber to connect

        self.state = "task_selection" # task_selection, transparency, or adaptability

    def _message_callback(self, msg: "String") -> None:
        """Callback for the web interface."""
        print("Received message on WebAppComm: ", msg.data)

        # if msg.data is not JSON, it is a text message
        try:
            msg_dict = json.loads(msg.data)
        except json.JSONDecodeError:
            if self.state == "transparency" or self.state == "adaptability":
                print("Received personalization related text message from web interface.")
                request = {
                    "status": self.state,
                    "request": msg.data,
                    "state": None
                }
                self.hla_command_queue.put(request)
                return
            else:
                print("WARNING: Unrecognized message from web interface, cannot decode JSON.")
                return    
        
        # hack to not run into errors when message does not contain state
        if "state" not in msg_dict:
            msg_dict["state"] = None

        {"state":"order_selection","status":"opened_adaptability_page"}


        if (msg_dict["state"] == "order_selection" and msg_dict["status"] == "opened_transparency_page"):
            self.state = "transparency"
            return
        elif (msg_dict["state"] == "order_selection" and msg_dict["status"] == "opened_adaptability_page"):
            self.state = "adaptability"
            return
        else:
            self.state = "task_selection"

        if (msg_dict["state"] == "order_selection" and msg_dict["status"] != "ready_for_initial_data") \
            or (msg_dict["state"] == "voice"):
            self.user_preference = msg_dict["status"]
            print("SETTING USER PREFERENCE: ", self.user_preference)
        elif msg_dict["status"] in ["finish_feeding", "back", "move_to_wiping_position", "drink_pickup", "drink_transfer", "move_to_above_plate", "aquire_food", 0, "bite_skill_selection", "bite_transfer", "mouth_wiping", "return_to_main"]:
            print("Received high-level action message from web interface.")
            if self.hla_command_queue is not None:
                self.hla_command_queue.put(msg_dict)
        else:
            print("WARNING: Unrecognized message from web interface.")
            return

    def send_web_interface_message(self, msg_dict: dict[str, Any]) -> None:
        self.web_interface_publisher.publish(
            String(json.dumps(msg_dict))
        )

    def send_web_interface_text(self, text: str) -> None:
        self.web_interface_publisher.publish(
            String(text)
        )

    def send_web_interface_image(self, image) -> None:
        self.web_interface_image_publisher.publish(self.image_bridge.cv2_to_compressed_imgmsg(image))

if __name__ == "__main__":
    rospy.init_node("test_web_interface")

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_manual_acquisition_pixels", action="store_true")
    parser.add_argument("--test_image_streaming", action="store_true")
    parser.add_argument("--test_dry_run", action="store_true")
    args = parser.parse_args()

    hla_command_queue = queue.Queue()
    web_interface = WebInterface(hla_command_queue)

    if args.test_dry_run:
        meal_start_log = pickle.load(open("test_log/meal_start_log.pkl", "rb"))

        plate_image = meal_start_log['plate_image']
        plate_bounds = meal_start_log['plate_bounds']
        food_type_to_data = meal_start_log['food_type_to_data']
        ordering_options = meal_start_log['ordering_options']

        n_food_types = len(food_type_to_data)
        data = [{k: v} for k, v in food_type_to_data.items()]

        input("Press Enter to send look at plate finished message")
        web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
        time.sleep(0.2) # simulate delay, also needed for web interface
        web_interface.send_web_interface_image(plate_image)
        # time.sleep(1.0)  # simulate delay, also needed for web interface
        web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data})
        web_interface.send_web_interface_message({"n_ordering": len(ordering_options), "data": ordering_options})


        next_bite_log = pickle.load(open("test_log/next_bite_log.pkl", "rb"))
        plate_image = next_bite_log['plate_image']
        plate_bounds = next_bite_log['plate_bounds']
        food_type_to_data = next_bite_log['food_type_to_data']
        next_food_item = next_bite_log['next_food_item']
        n_food_types = len(food_type_to_data)
        data = [{k: v} for k, v in food_type_to_data.items() if k != next_food_item]
        current_bite = {next_food_item: food_type_to_data[next_food_item]}

        input("Press Enter to send bite acquisition ready message")
        web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
        time.sleep(0.2) # simulate delay, also needed for web interface
        web_interface.send_web_interface_image(plate_image)
        time.sleep(0.1)
        web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data, "current_bite": current_bite})            

        input("Press Enter to send bite acquisition completed message")
        web_interface.send_web_interface_message({"state": "bite_pickup", "status": "completed"})

        input("Press Enter to send bite transfer ready message")
        web_interface.send_web_interface_message({"state": "bite_transfer", "status": "completed"})

        input("Press Enter to send bite acquisition ready message")
        web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
        time.sleep(0.2) # simulate delay, also needed for web interface
        web_interface.send_web_interface_image(plate_image)
        time.sleep(0.1)
        web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data, "current_bite": current_bite})            

        input("Press Enter to send bite acquisition completed message")
        web_interface.send_web_interface_message({"state": "bite_pickup", "status": "completed"})

        input("Press Enter to send bite transfer ready message")
        web_interface.send_web_interface_message({"state": "bite_transfer", "status": "completed"})

        input("Press Enter to send drink pickup done message")
        web_interface.send_web_interface_message({"state": "drink_pickup", "status": "completed"})

        input("Press Enter to send drink transfer done message")
        web_interface.send_web_interface_message({"state": "drink_transfer", "status": "completed"})

        input("Press Enter to send mouth wiping picked up message")
        web_interface.send_web_interface_message({"state": "prepare_mouth_wiping", "status": "completed"})

        input("Press Enter to send mouth wiping done message")
        web_interface.send_web_interface_message({"state": "moved_to_wiping_position", "status": "completed"})

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
        web_interface.send_web_interface_image(plate_image)
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
            web_interface.send_web_interface_image(image)

    
        

