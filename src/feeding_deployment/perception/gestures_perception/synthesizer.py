import os
from openai import OpenAI
import numpy as np
import math
import time
from pathlib import Path
import textwrap
import pickle
from tomsutils.llm import OpenAILLM, synthesize_python_function_with_llm, GridSearchSynthesizedProgramArgumentOptimizer
from gymnasium.spaces import Box

from feeding_deployment.perception.gestures_perception.in_context_examples import detect_mouth_open, detect_head_nod


class MockPerceptionInterface:
    """
    Simulate the perception interface for the robot with get_head_perception_data method
    """
    def __init__(self, head_perception_data):

        self.face_keypoints = head_perception_data['face_keypoints']
        self.head_pose = head_perception_data['head_pose']
        self.current_frame = 0
        self.max_frame = len(self.face_keypoints)

    def get_head_perception_data(self):
        if self.current_frame < self.max_frame:
            self.current_frame += 1
            head_perception_data = {
                'head_pose': self.head_pose[self.current_frame-1],
                'face_keypoints': self.face_keypoints[self.current_frame-1]
            }
            return head_perception_data
        # Reset the current frame back to 0 to prepare for next call.
        self.current_frame = 0
        return None


class PersonalizedGestureDetectorSynthesizer:
    def __init__(self, log_dir):

        self.llm = OpenAILLM(
            model_name="gpt-4o",
            cache_dir=log_dir / "llm_cache",
            max_tokens=2500,
        )

        with open(Path(__file__).parent / "prompt.txt", 'r') as f:
            self.prompt_skeleton = f.read()

        self.adjustable_parameters_name = "ADJUSTABLE_PARAMETERS"

    def _load_from_data_path(self, gesture_datapath: Path):
        self.label = gesture_datapath.stem
        with open(gesture_datapath, 'rb') as f:
            gesture_data = pickle.load(f)
        self.language_description = gesture_data['gesture_description']
        self.function_label = gesture_data['gesture_label']
        self.positive_examples = gesture_data['positive_examples']
        self.negative_examples = gesture_data['negative_examples']
        timeout = 20.0
        self.input_output_examples = [
            ((MockPerceptionInterface(head_perception_data=example), None, timeout), True)
            for example in self.positive_examples 
        ] + [
            ((MockPerceptionInterface(head_perception_data=example), None, timeout), False)
            for example in self.negative_examples 
        ]
    
    def generate_function(self, gesture_datapath: Path):
        # label, language_description, examples_data_path
        # label is the name of the datapath (last part of the datapath.pkl
        self._load_from_data_path(gesture_datapath)
        prompt = self.prompt_skeleton % (self.language_description, self.function_label)

        code_prefix = """import time
import numpy as np
from gymnasium.spaces import Box

"""
        arg_optimizer = GridSearchSynthesizedProgramArgumentOptimizer()
        synthesized_program, synthesized_info = synthesize_python_function_with_llm(
            self.llm, self.function_label, self.input_output_examples, prompt,
            code_prefix=code_prefix,
            argument_optimizer=arg_optimizer,
            arg_index_to_space_var_name=self.adjustable_parameters_name,
        )

        # Replace parameters!
        function_code = synthesized_program.create_code_str_from_arg_values(synthesized_info.optimized_args)

        print("Best Optimized Arguments: ", synthesized_info.optimized_args)

        print("Generated Function Code: ", function_code)
        try:
            workspace = locals().copy()
            exec(function_code, workspace)
            assert self.function_label in workspace
            gesture_detector = workspace[self.function_label]

            # Make sure code snippet to replace is present in the generated function code (ensures breaking out of the loop)
            old_snippet = """
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            break
"""
            assert old_snippet.strip() in function_code, "head perception snippet not found in the generated function code"

            positive_accuracy, negative_accuracy = self.run_detector(gesture_detector, None, 20.0)
            print("Best Positive Accuracy: ", positive_accuracy)
            print("Best Negative Accuracy: ", negative_accuracy)
            accuracy = (positive_accuracy + negative_accuracy) / 2.0
            with open(Path(__file__).parent / "results" / f"{self.label}.txt", "a") as f:
                f.write(f"\nnBest Accuracy: {accuracy}")
        
            # Replacement snippet
            new_snippet = """
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            continue
        else:
            time.sleep(0.1) # Maintain 10 Hz rate
"""
            updated_function_code = function_code.replace(old_snippet.strip(), new_snippet.strip())

            return updated_function_code, accuracy
        except Exception as e:
            print("Error: ", e)
            with open(Path(__file__).parent / "results" / f"{self.label}.txt", "a") as f:
                f.write(f"\nError: {e}")
            return None, None
    
    def test_in_context_examples(self):
        self._load_from_data_path(Path(__file__).parent / "gestures_examples" / "open_mouth.pkl")
        positive_accuracy, negative_accuracy = self.run_detector(detect_mouth_open, None, 20)
        print("In-Context Example 1")
        print("Best Positive Accuracy: ", positive_accuracy)
        print("Best Negative Accuracy: ", negative_accuracy)

        # /home/rkjenamani/sim_experiments/feeding-deployment/src/feeding_deployment/integration/log/gesture_examples/open_mouth.pkl
        # self._load_from_data_path(Path(__file__).parent.parent.parent / "integration" / "log" / "gesture_examples" / "open_mouth.pkl")
        self._load_from_data_path(Path(__file__).parent.parent.parent / "integration" / "log" / "gesture_examples" / "head_nod_user_generated.pkl")
        positive_accuracy, negative_accuracy = self.run_detector(detect_head_nod, None, 20)
        print("In-Context Example 2")
        print("Best Positive Accuracy: ", positive_accuracy)
        print("Best Negative Accuracy: ", negative_accuracy)
    
    def run_detector(self, gesture_detector, *args):
        """
        Run the gesture detector on examples in self.examples_data_path
        """
        positive_correct = 0
        for positive_example in self.positive_examples:
            perception_interface = MockPerceptionInterface(head_perception_data=positive_example)
            if gesture_detector(perception_interface, *args):
                positive_correct += 1
        
        negative_correct = 0
        for negative_example in self.negative_examples:
            perception_interface = MockPerceptionInterface(head_perception_data=negative_example)
            if not gesture_detector(perception_interface, *args):
                negative_correct += 1
        
        return positive_correct/len(self.positive_examples), negative_correct/len(self.negative_examples)


def main():

    # gestures = {
    #     "blinking": ("detect_blinking", "blinking eyes open and closed"),
    #     "eyebrows_raised": ("detect_raised_eyebrows", "eyebrows are raised"),
    #     "head_nod": ("detect_head_nod", "nodding head up and down"),
    #     "head_still_atleast_three_secs": ("detect_head_still_atleast_three_secs", "head remains still for at least 3 seconds"),
    #     "look_at_robot_atleast_three_secs": ("detect_look_at_robot_atleast_three_secs", "head looks directly forward at the robot for at least 3 seconds"),
    #     "talking" : ("detect_talking", "mouth is talking"),
    #     "open_mouth": ("detect_mouth_open", "mouth is wide open"),
    #     "shake_my_head_from_left_to_right": ("detect_head_shake", "head is shaking from left to right")
    # }

    # # Re-configure gesture examples to include labels.
    # for gesture in gestures:
    #     gesture_datapath = Path(__file__).parent / "gestures_examples" / f"{gesture}.pkl"
    #     with open(gesture_datapath, "rb") as f:
    #         data = pickle.load(f)
    #     label, description = gestures[gesture]
    #     with open(gesture_datapath, "wb") as f:
    #         pickle.dump({
    #             "gesture_label": label,
    #             "gesture_description": description,
    #             "positive_examples": data["positive_examples"], 
    #             "negative_examples": data["negative_examples"], 
    #         }, f)
        

    synthesizer = PersonalizedGestureDetectorSynthesizer(log_dir=Path(__file__).parent / "log")
    gesture_data_path = Path(__file__).parent.parent.parent / "integration" / "log" / "gesture_examples" / "head_nod_user_generated.pkl"
    generated_function_txt, accuracy = synthesizer.generate_function(gesture_datapath=gesture_data_path)
    synthesizer.test_in_context_examples()

    # gesture_to_test = "shake_my_head_from_left_to_right"
    # gesture_datapath = Path(__file__).parent / "gestures_examples" / f"{gesture_to_test}.pkl"
    # generated_function, accuracy = synthesizer.generate_function(gesture_datapath)
    # if generated_function is not None:
    #     with open("synthesized_gesture_detectors.py", "a") as f:
    #         f.write(generated_function)



if __name__ == '__main__':

    main()    
    # gestures = ["blinking", "eyebrows_raised", "head_nod", "head_still_atleast_three_secs", "look_at_robot_atleast_three_secs", "talking", "open_mouth", "shake_my_head_from_left_to_right"]

    # for gesture in gestures:

    #     gesture_datapath = Path(__file__).parent / "gestures_examples" / f"{gesture}.pkl"
    #     with open(gesture_datapath, 'rb') as f:
    #         gesture_data = pickle.load(f)

    #     # update the gesture data
    #     for positive_example in gesture_data['positive_examples']:
    #         positive_example['head_pose'] = positive_example['head_poses']
    #         del positive_example['head_poses']

    #     for negative_example in gesture_data['negative_examples']:
    #         negative_example['head_pose'] = negative_example['head_poses']
    #         del negative_example['head_poses']
        
    #     # overwrite original file
    #     with open(gesture_datapath, 'wb') as f:
    #         pickle.dump(gesture_data, f)