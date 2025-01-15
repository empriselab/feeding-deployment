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
from pathlib import Path

try:
    import rospy
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    from cv_bridge import CvBridge

except ModuleNotFoundError:
    pass

from feeding_deployment.transparency.continuous_llm import TransparencyContinuous

class WebInterface:
    '''
    An interface to interact with the web interface.
    '''
    def __init__(self, task_selection_queue: queue.Queue = None) -> None:

        # Used for generating continuous explanations.
        self.transparency_continuous = TransparencyContinuous()

        # Objects of task_selection_queue are dicts and can be of the following types:
        # {'task': 'meal_assistance', 'type': 'bite' / 'sip' / 'wipe'}
        # {'task': 'personalization', 'type': 'transparency' / 'adaptability' / 'gesture'}
        self.task_selection_queue = task_selection_queue

        # Queue containing all messages from the web interface.
        self.received_web_interface_messages = queue.Queue()
        
        # Create a publisher for communication with the web interface.
        self.web_interface_publisher = rospy.Publisher("/ServerComm", String, queue_size=10)
        self.web_interface_image_publisher = rospy.Publisher("/camera/image/compressed", CompressedImage, queue_size=10)
        self.image_bridge = CvBridge()
        self.user_preference = None
        self.web_interface_sub = rospy.Subscriber("WebAppComm", String, self._message_callback, queue_size=100)
        time.sleep(1.0)  # Wait for the subscriber to connect

        self.current_page = "task_selection" # task_selection, transparency, adaptability
        self.explanation_lock = threading.Lock() # Lock for generating continuous explanations
        
        # Start the thread for generating continuous explanations.
        self.transparency_continuous_thread = threading.Thread(target=self.provide_continuous_explanations)
        self.transparency_continuous_thread.start()

    def _send_message(self, msg_dict: dict[str, Any]) -> None:
        self.web_interface_publisher.publish(String(json.dumps(msg_dict)))

    def _send_text(self, text: str) -> None:
        self.web_interface_publisher.publish(String(text))

    def _send_image(self, image) -> None:
        self.web_interface_image_publisher.publish(self.image_bridge.cv2_to_compressed_imgmsg(image))

    def _message_callback(self, msg: "String") -> None:
        """Callback for the web interface."""
        print("Received message on WebAppComm: ", msg.data)
        
        msg_dict = json.loads(msg.data)

        if msg_dict["state"] == "task_selection":
            if msg_dict["status"] == "take_bite":
                task_selected = {
                    "task": "meal_assistance",
                    "type": "bite",
                }
            elif msg_dict["status"] == "take_sip":
                task_selected = {
                    "task": "meal_assistance",
                    "type": "sip",
                }
            elif msg_dict["status"] == "mouth_wiping":
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
                return
            
            # remove explanation lock (if it exists)
            if self.explanation_lock.locked():
                self.explanation_lock.release()
            
            # set current page to task_selection (effectively reseting transparency and adaptability pages)
            self.current_page = "task_selection"
            
            self.task_selection_queue.put(task_selected)
        else:
            self.received_web_interface_messages.put(msg_dict)

    def ready_for_task_selection(self, last_task_type = None, autocontinue_timeout = 10) -> None:
        """Moves the web interface to the task selection page."""

        self.current_page = "task_selection"

        # after bite and after sip are special, because they have bite and sip preselected for autocontinue with a timeout
        if last_task_type == "bite":
            self._send_message({"state": "task_selection_after_bite", "status": "jump", "autocontinue_timeout": autocontinue_timeout})
        elif last_task_type == "sip":
            self._send_message({"state": "task_selection_after_sip", "status": "jump", "autocontinue_timeout": autocontinue_timeout})
        else:
            self._send_message({"state": "task_selection", "status": "jump"})

    def get_required_web_interface_message(self, condition) -> dict[str, Any]:
        """Parses through all messages received from the web interface and returns the oldest one satisfying the condition."""
        print_once = True
        while True:
            try:
                msg_dict = self.received_web_interface_messages.get_nowait()
                if condition(msg_dict):
                    print("Received required message from the web interface")
                    return msg_dict
            except queue.Empty:
                if print_once:
                    print("Waiting for required message from the web interface ...")
                    print_once = False
                time.sleep(0.1)
                continue

    def get_bite_ordering_preference(self, plate_image, n_food_types, data, ordering_options) -> None:

        with self.explanation_lock:
        
            # Jump to bite ordering page
            self._send_message({"state": "newmealpage", "status": "jump"})
            
            # Send required data for the bite ordering page
            self._send_image(plate_image)
            self._send_message({"n_food_types": n_food_types, "data": data})
            self._send_message({"n_ordering": len(ordering_options), "data": ordering_options})

            # Get the user's bite ordering preference
            msg_dict = self.get_required_web_interface_message(
                lambda msg_dict: (
                    (msg_dict["state"] == "order_selection" and msg_dict["status"] != "ready_for_initial_data")
                    or (msg_dict["state"] == "voice")
                )
            )

        bite_ordering_preference = msg_dict["status"]
        return bite_ordering_preference
    
    def get_next_bite_selection(self, plate_image, n_food_types, data, predicted_bite) -> None:

        with self.explanation_lock:

            # Jump to next bite selection page
            self._send_message({"state": "changefooditem2", "status": "jump"})

            # Send required data for the next bite selection page
            time.sleep(0.2) # simulate delay, needed for web interface
            self._send_image(plate_image)
            time.sleep(0.1)
            self._send_message({"n_food_types": n_food_types, "data": data, "current_bite": predicted_bite})  

            # Get the user's next bite selection
            msg_dict = self.get_required_web_interface_message(
                lambda msg_dict: (
                    (msg_dict["status"] == "aquire_food" or msg_dict["status"] == 0)
                )
            ) 

        if msg_dict["status"] == "aquire_food": # autonomous bite acquisition
            return "autonomous", msg_dict["data"]
        elif msg_dict["status"] == 0: # manual skewering
            return "manual_skewering", msg_dict["positions"]
        elif msg_dict["status"] == 1: # manual scooping
            return "manual_scooping", msg_dict["positions"]
        else:
            print("Unsupported message received from the web interface: ", msg_dict)

    def get_successful_food_acquisition_confirmation(self) -> None:

        with self.explanation_lock:
        
            # Jump to successful food acquisition page
            self._send_message({"state": "transfermeal", "status": "jump"})

            # Wait until the user confirms that the food has been acquired
            msg_dict = self.get_required_web_interface_message(
                lambda msg_dict: (
                    (msg_dict["state"] == "post_bite_pickup")
                )
            )

        if msg_dict["status"] == "bite_transfer":
            return True
        elif msg_dict["status"] == "return_to_main":
            return False
        else:
            print("Unsupported message received from the web interface: ", msg_dict)

    def get_drink_transfer_confirmation(self) -> None:

        with self.explanation_lock:
        
            # Jump to ready for drink transfer page
            self._send_message({"state": "transferdrinks", "status": "jump"})

            # Wait until the user confirms that the drink has been transferred
            self.get_required_web_interface_message(
                lambda msg_dict: (
                    (msg_dict["state"] == "post_drink_pickup" and msg_dict["status"] == "drink_transfer")
                )
            )

    def get_wipe_transfer_confirmation(self) -> None:

        with self.explanation_lock:

            # jump to ready for wipe transfer page
            self._send_message({"state": "wippingtrans", "status": "jump"})

            # Wait until the user confirms that the wipe has been transferred
            self.get_required_web_interface_message(
                lambda msg_dict: (
                    (msg_dict["state"] == "prepared_mouth_wiping" and msg_dict["status"] == "move_to_wiping_position")
                )
            )

    def get_transparency_query(self) -> None:
        
        if self.current_page != "transparency":
            
            # acquire explanation lock continuously until we move back to task selection page (which releases the lock)
            self.explanation_lock.acquire()

            self.current_page = "transparency"

            # Jump to transparency query page
            self._send_message({"state": "transparency", "status": "jump"})

        # Wait until the user provides a transparency query
        msg_dict = self.get_required_web_interface_message(
            lambda msg_dict: (
                (msg_dict["state"] == "transparency_query")
            )
        )

        return msg_dict["status"]
    
    def update_transparency_response(self, response: str) -> None:
        assert self.current_page == "transparency", "Cannot update transparency response when not on the transparency page."

        self._send_message({"state": "transparency_query", "status": response})
    
    def get_adaptability_query(self) -> None:
        
        if self.current_page != "adaptability":

            # acquire explanation lock continuously until we move back to task selection page (which releases the lock)
            self.explanation_lock.acquire()

            self.current_page = "adaptability"

            # Jump to adaptability query page
            self._send_message({"state": "adaptability", "status": "jump"})

        # Wait until the user provides an adaptability query
        msg_dict = self.get_required_web_interface_message(
            lambda msg_dict: (
                (msg_dict["state"] == "adaptability_query")
            )
        )

        self.current_page = "explanation"

        return msg_dict["status"]
    
    def update_adaptability_response(self, response: str) -> None:
        assert self.current_page == "adaptability", "Cannot update adaptability response when not on the adaptability page."

        self._send_message({"state": "adaptability_query", "status": response})

    def provide_continuous_explanations(self) -> None:
        while True:
            explanation_active = True
            start_time = time.time()

            while time.time() - start_time < 1.0:
                # Attempt to acquire the lock without blocking
                if not self.explanation_lock.acquire(blocking=False):
                    explanation_active = False
                    break

                # Ensure we release the lock immediately after acquiring it
                self.explanation_lock.release()

                time.sleep(0.1)  # Wait for 0.1 seconds before re-checking

            if explanation_active:
                current_explanation = self.transparency_continuous.get_explanation()
                with self.explanation_lock:
                    self._send_message({"state": "swithtodrink", "status": "jump"})
                    self.current_page = "explanation"