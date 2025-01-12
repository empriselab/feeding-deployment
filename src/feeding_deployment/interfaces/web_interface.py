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

from feeding_deployment.actions.base import GroundHighLevelAction
from 

class WebInterface:
    '''
    An interface to interact with the web interface.
    '''
    def __init__(self, task_selection_queue: queue.Queue = None, task_params_queue: queue.Queue = None) -> None:

        # Objects of task_selection_queue are dicts and can be of the following types:
        # {'task': 'meal_assistance', 'type': 'bite' / 'sip' / 'wipe'}
        # {'task': 'personalization', 'type': 'transparency' / 'adaptability' / 'gesture'}
        self.task_selection_queue = task_selection_queue

        # Used for setting params within a certain task
        self.task_params_queue = task_params_queue
        
        # Create a publisher for communication with the web interface.
        self.web_interface_publisher = rospy.Publisher("/ServerComm", String, queue_size=10)
        self.web_interface_image_publisher = rospy.Publisher("/camera/image/compressed", CompressedImage, queue_size=10)
        self.image_bridge = CvBridge()
        self.user_preference = None
        self.web_interface_sub = rospy.Subscriber("WebAppComm", String, self._message_callback, queue_size=100)
        time.sleep(1.0)  # Wait for the subscriber to connect

        self.state = "assistance_task" # task_selection, transparency, or adaptability
        self.current_page = "task_selection" # task_selection, meal_assistance, personalization, or explanation

    def ready_for_task_selection(self, last_task_type, autocontinue_timeout = 10) -> None:
        """Moves the web interface to the task selection page."""

        # after bite and after sip are special, because they have bite and sip preselected for autocontinue with a timeout
        if last_task_type == "bite":
            self.send_web_interface_message({"state": "task_selection_after_bite", "status": "jump", "autocontinue_timeout": autocontinue_timeout})
        elif last_task_type == "sip":
            self.send_web_interface_message({"state": "task_selection_after_sip", "status": "jump", "autocontinue_timeout": autocontinue_timeout})
        else:
            self.send_web_interface_message({"state": "task_selection", "status": "jump"})
        
        self.current_page = "task_selection"

    def _message_callback(self, msg: "String") -> None:
        """Callback for the web interface."""
        print("Received message on WebAppComm: ", msg.data)
        
        msg_dict = json.loads(msg.data)

        if msg_dict["state"] == "task_selection":
            if msg_dict["status"] == "bite":
                task_selected = {
                    "task": "meal_assistance",
                    "type": "bite",
                }
            elif msg_dict["status"] == "sip":
                task_selected = {
                    "task": "meal_assistance",
                    "type": "sip",
                }
            elif msg_dict["status"] == "wipe":
                task_selected = {
                    "task": "meal_assistance",
                    "type": "wipe",
                }
            elif msg_dict["status"] == "transparency":
                task_selected = {
                    "task": "personalization",
                    "type": "transparency",
                }
            elif msg_dict["status"] == "adaptability":
                task_selected = {
                    "task": "personalization",
                    "type": "adaptability",
                }
            elif msg_dict["status"] == "gesture":
                task_selected = {
                    "task": "personalization",
                    "type": "gesture",
                }
            else:
                print("Invalid task selection status received from interface: ", msg_dict["status"])
            
            self.task_selection_queue.put(task_selected)
        else:
            self.task_params_queue.put(msg_dict)

        self.ready_from_explanation = True

    def task_params_request(self, request_type: str) -> None:
        """
        Faciliates assistance task requesting params from the web interface.
        Web interface will open the appropriate page for the user to provide the required information, 
        and then return the provided information
        """
        while not self.task_params_queue.empty():
            self.task_params_queue.get()
        
        # enumerate all the possible request types for the HLAs / transparency / adaptability
        if request_type == "transparency":
            pass
        elif request_type == "adaptability":
            pass
        elif request_type == "flair_preference":
            pass
        elif request_type == "next_bite":
            pass
        elif request_type == "successful_pickup":
            pass
        elif request_type == "ready for transfer":
            pass

    def continuous_explanations(self) -> None:
        """
        If it has been two seconds since a task param
        """
        

    def jump_to_state(self, state: str) -> None:

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

    # at any point in time,
    # either the web interface is soliciting information from the user
    # or the web interface is being transparent
    # for soliciting, it would jump to a certain page
    # otherwise, it should show the white page (how?)
    

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

    
        

