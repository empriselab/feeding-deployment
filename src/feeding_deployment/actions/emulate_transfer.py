from typing import Any, Callable

import numpy as np
import time
import pickle
from scipy.spatial.transform import Rotation
import inspect
from pathlib import Path
import threading
import types
import json
import imageio

from pybullet_helpers.geometry import Pose

from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from feeding_deployment.actions.base import (
    HighLevelAction,
    tool_type,
    GripperFree,
    Holding,
    ToolPrepared,
    EmulateTransferDone,
)
from feeding_deployment.actions.feel_the_bite.outside_mouth_transfer import OutsideMouthTransfer
from feeding_deployment.perception.gestures_perception.synthesizer import PersonalizedGestureDetectorSynthesizer
import feeding_deployment.perception.gestures_perception.static_gesture_detectors as static_gesture_detectors

class EmulateTransferHLA(HighLevelAction):
    """Emulate transfer by bringing the empty gripper in front of the user's mouth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transfer = OutsideMouthTransfer(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.no_waits)

        self.gesture_examples_path = self.gesture_detectors_dir / "gesture_examples"
        self.synthesized_gestures_dict_path = self.gesture_detectors_dir / "synthesized_gestures_dict.json"
        if not self.gesture_examples_path.exists():
            self.gesture_examples_path.mkdir(parents=True)
        if not self.synthesized_gestures_dict_path.exists():
            with open(self.synthesized_gestures_dict_path, "w") as f:
                f.write("{}")
        self.detector_synthesizer = PersonalizedGestureDetectorSynthesizer(self.log_dir)

        self.test_mode = False

    def emulate_transfer(self, speed: str):

        if self.robot_interface is not None:
            self.robot_interface.set_speed(speed)

        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        if self.web_interface is not None:
            self.web_interface.fix_explanation("Moving to infront of mouth")

        self.perception_interface.set_head_perception_tool("fork")
        self.perception_interface.start_head_perception_thread()
        if self.robot_interface is not None:
            time.sleep(5.0) # let head perception thread warmstart / robot to stabilize
            self.robot_interface.set_tool("fork")
            self.perception_interface.zero_ft_sensor()
        else:
            time.sleep(1.0) # let sim head perception thread warmstart

        if self.robot_interface is not None:
            self.perception_interface.speak("Please press transfer button while looking towards the robot when ready")
            self.perception_interface.detect_button_press()

        self.transfer.set_tool("fork")
        self.transfer.move_to_transfer_state(outside_mouth_distance=0.15)

        if self.robot_interface is not None:
            self.perception_interface.speak("Ready for gestures")

        if self.web_interface is not None:
            self.web_interface.update_fixed_explanation("Ready for gestures")
            if self.test_mode:
                # Start with the static, given gestures.
                available_gestures = inspect.getmembers(static_gesture_detectors, inspect.isfunction)
                gesture_function_name_to_label = static_gesture_detectors.function_name_to_label.copy()

                # Load the synthesized gestures.
                synthesized_gestures = self.load_synthesized_gestures()
                available_gestures += synthesized_gestures
                
                # load from synthesized_gestures_dict_path
                with open(self.synthesized_gestures_dict_path, "r") as f:
                    synthesized_gestures_dict = json.load(f)
                gesture_function_name_to_label.update(synthesized_gestures_dict)

                gestures_function_name_to_function = {gesture[0]: gesture[1] for gesture in available_gestures}
                gesture_labels = [gesture_function_name_to_label[gesture[0]] for gesture in available_gestures]
                self.web_interface.jump_to_test_gesture_page(gesture_labels)

                # Create a termination event to signal when to switch detectors
                new_gesture_selected_event = threading.Event()
                self.web_interface.start_gesture_listener_thread(new_gesture_selected_event)
                print("Main thread is looking for gestures")

                while True:
                    # Get the selected gesture from the web interface
                    if not new_gesture_selected_event.is_set():
                        time.sleep(0.1)
                        continue
                    selected_gesture = self.web_interface.get_selected_gesture()
                    print(f"Selected gesture: {selected_gesture}")
                    self.web_interface.register_negative_gesture_detection()
                    new_gesture_selected_event.clear()
                    if selected_gesture:
                        # Call the gesture detection function
                        selected_gesture_function_name = None
                        for gesture_function_name, gesture_label in gesture_function_name_to_label.items():
                            if gesture_label == selected_gesture:
                                selected_gesture_function_name = gesture_function_name
                                break
                        if selected_gesture_function_name is None:
                            print(f"Gesture function not found for label: {selected_gesture}")
                            continue
                        print(f"Calling gesture detection function: {selected_gesture_function_name}")
                        gesture_detected = gestures_function_name_to_function[selected_gesture_function_name](self.perception_interface, timeout=600, termination_event=new_gesture_selected_event)
                        if gesture_detected:
                            self.web_interface.register_positive_gesture_detection()
                        else:
                            self.web_interface.register_negative_gesture_detection()
                    else: # None means the user has switched to another page
                        break
                self.web_interface.stop_gesture_listener_thread()

                # shutdown the head perception thread and move to before transfer state
                self.perception_interface.stop_head_perception_thread()
                self.web_interface.update_fixed_explanation("Moving back to before transfer position")
                self.transfer.move_to_before_transfer_state()  

            else:
                # start logging perception data while user selects when to record and delete on the web interface,
                # then extract relevant examples using timestamps

                # load from synthesized_gestures_dict_path
                with open(self.synthesized_gestures_dict_path, "r") as f:
                    synthesized_gestures_dict = json.load(f)

                # generate function name from gesture label by adding _
                synthesized_gesture_function_name = self.gesture_label.replace(" ", "_").lower()

                # check if the gesture is already synthesized / in static detectors
                if synthesized_gesture_function_name in synthesized_gestures_dict or synthesized_gesture_function_name in static_gesture_detectors.function_name_to_label:
                    print(f"Gesture {self.gesture_label} already synthesized or in static detectors")
                    # slightly hacky way to uniquely name the synthesized gesture (assumption: only one user generated gesture per label)
                    id = 0
                    new_function_name = synthesized_gesture_function_name
                    while new_function_name in synthesized_gestures_dict or new_function_name in static_gesture_detectors.function_name_to_label:
                        id += 1
                        new_function_name = synthesized_gesture_function_name + f"_{id}"
                    synthesized_gesture_function_name = new_function_name
                    self.gesture_label = self.gesture_label + f"_{id}"
                    print(f"Renaming to {self.gesture_label}")

                synthesized_gestures_dict[synthesized_gesture_function_name] = self.gesture_label
                with open(self.synthesized_gestures_dict_path, "w") as f:
                    json.dump(synthesized_gestures_dict, f)

                logging_start_time = self.perception_interface.start_logging_head_perception()
                positive_examples_timestamps, negative_examples_timestamps = self.web_interface.get_gesture_examples()
                self.perception_interface.stop_logging_head_perception()

                # shutdown the head perception thread and move to before transfer state
                self.perception_interface.stop_head_perception_thread()
                self.web_interface.update_fixed_explanation("Moving back to before transfer position")
                self.transfer.move_to_before_transfer_state()  
                self.web_interface.update_fixed_explanation("Synthesizing gesture detector, please wait ...")

                video_segments_path = self.gesture_detectors_dir / "video_segments" / synthesized_gesture_function_name
                if not video_segments_path.exists():
                    video_segments_path.mkdir(parents=True)

                def save_video(video_frames, video_path):
                    # Save the video using imageio
                    print(f"Saving {len(video_frames)} frames")
                    with imageio.get_writer(video_path, fps=10, codec='libx264') as writer:
                        for frame in video_frames:
                            # Convert to RGB format for imageio if needed
                            frame_rgb = frame[:, :, ::-1]  # Convert from BGR to RGB
                            writer.append_data(frame_rgb)
                
                if len(positive_examples_timestamps) > 0 and len(negative_examples_timestamps) > 0:
                    positive_examples = []
                    for timestamp in positive_examples_timestamps:
                        print("Extracting positive example")
                        positive_example, positive_video = self.perception_interface.extract_from_logged_head_perception_data(timestamp)
                        positive_examples.append(positive_example)
                        save_video(positive_video, video_segments_path / f"positive_{len(positive_examples)}.mp4")

                    negative_examples = []
                    for timestamp in negative_examples_timestamps:
                        print("Extracting negative example")
                        negative_example, negative_video = self.perception_interface.extract_from_logged_head_perception_data(timestamp)
                        negative_examples.append(negative_example)
                        save_video(negative_video, video_segments_path / f"negative_{len(negative_examples)}.mp4")

                    # delete the logged data
                    self.perception_interface.delete_logged_head_perception_data()

                    # save the examples
                    gesture_datapath = self.gesture_examples_path / f"{synthesized_gesture_function_name}.pkl"
                    with open(gesture_datapath, "wb") as f:
                        pickle.dump({
                            "gesture_label": synthesized_gesture_function_name,
                            "gesture_description": self.gesture_description,
                            "positive_examples": positive_examples, 
                            "negative_examples": negative_examples
                        }, f)

                    print("Synthesizing gesture detector ...")
                    generated_function_txt, accuracy = self.detector_synthesizer.generate_function(gesture_datapath)
                
                    # Hack to test the synthesizer
                    # hack_datapath = Path(__file__).parent.parent / "perception" / "gestures_perception" / "gestures_examples" / "shake_my_head_from_left_to_right.pkl"
                    # generated_function_txt = self.detector_synthesizer.generate_function(hack_datapath)
                    
                    if generated_function_txt is not None:
                        self.web_interface.update_fixed_explanation(f"Synthesized gesture detector for {self.gesture_label} with accuracy {accuracy*100:.2f}%")
                        self.register_gesture_detector(synthesized_gesture_function_name, generated_function_txt)
                    else:
                        self.web_interface.update_fixed_explanation("Did not generate valid detector function")
                        print("Did not generate valid detector function")
                    time.sleep(2.0) # let the user see the message
                else:
                    print("Gesture examples recording is not valid")
            self.web_interface.clear_explanation()
        else:
            print("Can record or test gestures only with real robot and web interface")

            # shutdown the head perception thread and move to before transfer state
            self.perception_interface.stop_head_perception_thread()
            self.transfer.move_to_before_transfer_state()        

    def get_name(self) -> str:
        return "EmulateTransfer"
    
    def get_operator(self) -> LiftedOperator:
        return LiftedOperator(
            self.get_name(),
            parameters=[],
            preconditions={LiftedAtom(GripperFree, [])},
            add_effects={LiftedAtom(EmulateTransferDone, [])},
            delete_effects=set(),
        )
    
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        return f"emulate_transfer.yaml"
    
    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        print("Params: ", params)
        if params["test_mode"]:
            self.test_mode = True
        else:
            self.test_mode = False
            self.gesture_label = params["gesture_label"]
            self.gesture_description = params["gesture_description"]
        return super().execute_action(objects, params)
