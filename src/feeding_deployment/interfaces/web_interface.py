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
        self.task_selection_jump = False

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

        # for escaping out of while loops
        self.active = True
        
        # Start the thread for generating continuous explanations.
        self.transparency_continuous_thread = threading.Thread(target=self.provide_continuous_explanations)
        self.transparency_continuous_thread.start()

    def stop_all_threads(self) -> None:
        self.active = False
        try:
            self.transparency_continuous_thread.join()
        except Exception as e:
            print("Error stopping transparency continuous thread: ", e)
        try:
            self.gesture_listener_thread.join()
        except Exception as e:
            print("Error stopping gesture listener thread: ", e)


    def _send_message(self, msg_dict: dict[str, Any]) -> None:
        self.web_interface_publisher.publish(String(json.dumps(msg_dict)))

    def _send_image(self, image) -> None:
        self.web_interface_image_publisher.publish(self.image_bridge.cv2_to_compressed_imgmsg(image))

    def _message_callback(self, msg: "String") -> None:
        """Callback for the web interface."""
        print("Received message on WebAppComm: ", msg.data)
        
        msg_dict = json.loads(msg.data)

        self.task_selection_jump = False
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
            elif msg_dict["status"] == "jump":
                self.task_selection_jump = True
                return
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

    def clear_received_messages(self) -> None:
        while not self.received_web_interface_messages.empty():
            self.received_web_interface_messages.get()

    def ready_for_task_selection(self, last_task_type = None, autocontinue_timeout = 7) -> None:
        """Moves the web interface to the task selection page."""

        self.current_page = "task_selection"

        print("Sending message to web interface to move to task selection page with last task type: ", last_task_type)

        # after bite and after sip are special, because they have bite and sip preselected for autocontinue with a timeout
        if last_task_type == "bite":
            self._send_message({"state": "afterbitetransfer", "status": "jump"})
            time.sleep(0.5)
            self._send_message({"state": "auto_time", "status": str(autocontinue_timeout)})
        elif last_task_type == "sip":
            self._send_message({"state": "afterdrinktransfer", "status": "jump"})
            time.sleep(0.5)
            self._send_message({"state": "auto_time", "status": str(autocontinue_timeout)})
        else:
            self._send_message({"state": "task_selection", "status": "jump"})

    def get_required_web_interface_message(self, condition) -> dict[str, Any]:
        """Parses through all messages received from the web interface and returns the oldest one satisfying the condition."""
        print_once = True
        while self.active:
            if self.task_selection_jump:
                return None
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

    #### Meal Assistance Pages ####

    def get_bite_ordering_preference(self, plate_image, n_food_types, data, ordering_options) -> None:

        self.current_page = "meal_assistance"

        with self.explanation_lock:
        
            # Jump to bite ordering page
            self._send_message({"state": "newmealpage", "status": "jump"})
            
            # Wait for the web interface to be ready for initial data
            time.sleep(0.2)
            
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
    
    def get_next_bite_selection(self, plate_image, n_food_types, data, predicted_bite, autocontinue_time=7) -> None:

        self.current_page = "meal_assistance"

        with self.explanation_lock:

            # Jump to next bite selection page
            self._send_message({"state": "acquirebite", "status": "jump"})

            # Send required data for the next bite selection page
            time.sleep(0.5) # simulate delay, needed for web interface

            self._send_image(plate_image)
            time.sleep(0.1)
            self._send_message({"n_food_types": n_food_types, "data": data, "current_bite": predicted_bite})  
            # set autocontinue time
            time.sleep(0.5)
            self._send_message({"state": "auto_time", "status": str(autocontinue_time)})

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

        self.current_page = "meal_assistance"

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

        self.current_page = "meal_assistance"

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

        self.current_page = "meal_assistance"

        with self.explanation_lock:

            # jump to ready for wipe transfer page
            print("Jumping to mouth wiping transfer page")
            self._send_message({"state": "wipingtrans", "status": "jump"})

            # Wait until the user confirms that the wipe has been transferred
            self.get_required_web_interface_message(
                lambda msg_dict: (
                    (msg_dict["state"] == "prepared_mouth_wiping" and msg_dict["status"] == "move_to_wiping_position")
                )
            )

    #### Transparency Pages ####

    def get_transparency_request(self) -> None:
        
        if self.current_page != "transparency":
            
            # acquire explanation lock continuously until we move back to task selection page (which releases the lock)
            self.explanation_lock.acquire()

            self.current_page = "transparency"

            # Jump to transparency query page
            self._send_message({"state": "transparency", "status": "jump"})

        # Wait until the user provides a transparency query
        msg_dict = self.get_required_web_interface_message(
            lambda msg_dict: (
                (msg_dict["state"] == "transparency_request")
            )
        )

        if msg_dict is None:
            return None
        return msg_dict["status"]
    
    def update_transparency_response(self, response: str) -> None:
        assert self.current_page == "transparency", "Cannot update transparency response when not on the transparency page."
        self._send_message({"state": "transparency_response", "status": response})

    #### Adaptability Pages ####
    
    def get_adaptability_request(self) -> None:
        
        if self.current_page != "adaptability":

            # acquire explanation lock continuously until we move back to task selection page (which releases the lock)
            self.explanation_lock.acquire()

            self.current_page = "adaptability"

            # Jump to adaptability query page
            self._send_message({"state": "adaptability", "status": "jump"})

        # Wait until the user provides an adaptability query
        msg_dict = self.get_required_web_interface_message(
            lambda msg_dict: (
                (msg_dict["state"] == "adaptability_request")
            )
        )
        print("Received adaptability request: ", msg_dict)

        if msg_dict is None:
            return None
        return msg_dict["status"]
    
    def update_adaptability_response(self, response: str) -> None:
        assert self.current_page == "adaptability", "Cannot update adaptability response when not on the adaptability page."
        self._send_message({"state": "adaptability_response", "status": response})

    #### Gesture Pages ####

    def get_gesture_type(self) -> None:
        """Get whether the user wants to add a gesture or test a gesture."""

        self.current_page = "gesture"

        msg_dict = self.get_required_web_interface_message(
            lambda msg_dict: (
                (msg_dict["state"] == "gesture_main")
            )
        )

        return msg_dict["status"]
    
    def get_new_gesture_details(self) -> None:
        """Get the gesture label and description from the user."""

        self.current_page = "record_gesture"

        # get first message from web interface (condition is always true)
        msg_dict = self.get_required_web_interface_message(lambda msg_dict: True)

        return msg_dict["state"], msg_dict["status"]
    
    def jump_to_test_gesture_page(self, available_gestures: list[str]) -> None:
        """Send available gestures to the web interface."""
        
        self.current_page = "test_gesture"

        # Jump to test gesture page
        self._send_message({"state": "gesturetest", "status": "jump"})

        # Send available gestures to the web interface
        print("Length of available gestures: ", len(available_gestures))
        print("Available gestures: ", available_gestures)
        time.sleep(0.1)
        self._send_message({"n_ordering": len(available_gestures), "data": available_gestures})
    
    def start_gesture_listener_thread(self, new_gesture_selected_event: threading.Event) -> None:
        """Start the gesture listener thread."""
        assert self.current_page == "test_gesture", "Cannot start gesture listener thread when not on the test gesture page."
        self.new_gesture_selected_event = new_gesture_selected_event
        self.selected_gesture = None
        self.gesture_listener_thread = threading.Thread(target=self.gesture_listener_thread)
        self.gesture_listener_thread.start()

    def gesture_listener_thread(self) -> None:
        
        while self.active:
            msg_dict = self.get_required_web_interface_message(
                lambda msg_dict: (
                    (msg_dict["state"] == "test_selection")
                )
            )

            if msg_dict is None or msg_dict["status"] == "back":
                self.selected_gesture = None
                self.new_gesture_selected_event.set()
                break

            self.selected_gesture = msg_dict["status"]
            self.new_gesture_selected_event.set()

    def get_selected_gesture(self) -> None:
        """Get the gesture selected by the user."""
        if self.selected_gesture:
            return self.selected_gesture

    def stop_gesture_listener_thread(self) -> None:
        """Stop the gesture listener thread."""
        self.gesture_listener_thread.join()

    def register_positive_gesture_detection(self) -> None:
        """Register a positive gesture detection."""
        self._send_message({"state": "gesture_response", "status": "Detected the selected gesture."})

    def register_negative_gesture_detection(self) -> None:
        """Register a negative gesture detection."""
        self._send_message({"state": "gesture_response", "status": "Did not detect the selected gesture."})

    def get_gesture_examples(self) -> None:
        """Get gesture examples from the user."""
        
        # Jump to gesturerecording page
        self._send_message({"state": "gesturerecording", "status": "jump"})

        positive_timestamps = self.record_gesture_examples()
        negative_timestamps = self.record_gesture_examples(positive=False)

        return positive_timestamps, negative_timestamps

    def record_gesture_examples(self, positive=True) -> None:
        """Record gesture examples."""

        if positive:
            trigger_message = "gesture_add"
        else:
            trigger_message = "gesture_add_negative"

        timestamps = []
        start_timestamp, end_timestamp = None, None
        while self.active:
            msg_dict = self.get_required_web_interface_message(
                lambda msg_dict: (
                    (msg_dict["state"] == trigger_message)
                )
            )
            if msg_dict is None:
                break

            if msg_dict["status"] == "start":
                if start_timestamp is not None:
                    self._send_message({"state": "gesture_response", "status": "Invalid example: received start before stop. Please click on stop and then click on start."})
                    print("Invalid example: received start before stop. Please click on stop and then click on start.")
                else:
                    start_timestamp = time.time()
                    # rostopic pub -1 /ServerComm std_msgs/String "{data: '{\"state\": \"gesture_response\", \"status\": \"This is a robot message\"}'}"
                    self._send_message({"state": "gesture_response", "status": f"Started recording gesture example: {len(timestamps) + 1}"})
                    print("Started recording gesture example: ", len(timestamps) + 1)
            elif msg_dict["status"] == "stop":
                if start_timestamp is not None:
                    end_timestamp = time.time()
                    self._send_message({"state": "gesture_response", "status": f"Recorded gesture example: {len(timestamps) + 1}"})
                    print("Recorded gesture example: ", len(timestamps) + 1)
                    timestamps.append((start_timestamp, end_timestamp))
                else:
                    self._send_message({"state": "gesture_response", "status": "Invalid example: received stop before start. Please click on start, then demonstrate the gesture, and then click on stop."})
                    print("Invalid example: received stop before start. Please click on start, then demonstrate the gesture, and then click on stop.")
                start_timestamp, end_timestamp = None, None
            elif msg_dict["status"] == "delete":
                if start_timestamp is None:
                    if len(timestamps) > 0:
                        self._send_message({"state": "gesture_response", "status": f"Deleted gesture example: {len(timestamps)}. {len(timestamps)-1} examples remain in recording history"})
                        print("Deleted gesture example: ", len(timestamps))
                        timestamps.pop()
                        print(f"{len(timestamps)} examples remain in recording history")
                    else:
                        self._send_message({"state": "gesture_response", "status": "No examples to delete"})
                        print("No examples to delete")
                else:
                    self._send_message({"state": "gesture_response", "status": "Cannot delete while recording. Please click on stop and then click on delete."})
                    print("Cannot delete while recording. Please click on stop and then click on delete.")
            elif msg_dict["status"] == "next" or msg_dict["status"] == "back":
                break
            else:
                print("Unsupported message received from the web interface: ", msg_dict)
        return timestamps

    #### Continuous Explanations ####

    def provide_continuous_explanations(self) -> None:
        while self.active:
            # Rajat ToDo: Do not provide explanations during transfer (when the user is expected to open their mouth)
            time.sleep(5.0)
            explanation_active = True
            start_time = time.time()
            current_explanation = self.transparency_continuous.get_explanation()
            self._send_message({"state": "explanation", "status": current_explanation})