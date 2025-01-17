import os
from openai import OpenAI
import numpy as np
import math
import time
from pathlib import Path
import textwrap
import pickle
from tomsutils.llm import OpenAILLM, synthesize_python_function_with_llm

from feeding_deployment.perception.gestures_perception.in_context_examples import in_context_example1, in_context_example2

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
        return None

class PersonalizedGestureDetectorSynthesizer:
    def __init__(self):
        log_dir = Path(__file__).parent.parent.parent / "integration" / "log"

        self.llm = OpenAILLM(
            model_name="gpt-4o",
            cache_dir=log_dir / "llm_cache",
        )

        with open(Path(__file__).parent / "prompt.txt", 'r') as f:
            self.prompt_skeleton = f.read()

        self.function_name = "gesture_detector"

    def _load_from_data_path(self, gesture_datapath: Path):
        self.label = gesture_datapath.stem
        with open(gesture_datapath, 'rb') as f:
            gesture_data = pickle.load(f)
        self.language_description = gesture_data['description']
        self.positive_examples = gesture_data['positive_examples']
        self.negative_examples = gesture_data['negative_examples']
        timeout = 20.0
        threshold = 0.5  # this will be optimized after the initial synthesis
        self.input_output_examples = [
            ((MockPerceptionInterface(head_perception_data=example), None, timeout, threshold), True)
            for example in self.positive_examples 
        ] + [
            ((MockPerceptionInterface(head_perception_data=example), None, timeout, threshold), False)
            for example in self.negative_examples 
        ]
    
    def generate_function(self, gesture_datapath: Path):
        # label, language_description, examples_data_path
        # label is the name of the datapath (last part of the datapath.pkl
        self._load_from_data_path(gesture_datapath)
        prompt = self.prompt_skeleton%(self.language_description)

        synthesized_program, _ = synthesize_python_function_with_llm(self.llm, self.function_name, self.input_output_examples, prompt)
        function_code = synthesized_program.code_str

        print("Generated Function Code: ", function_code)
        try:
            exec(function_code, globals())  # Executes code in the global namespace

            threshold, accuracy = self.search_threshold(gesture_detector)
            print("Best Threshold: ", threshold)
            print("Best Accuracy: ", accuracy)
            with open(Path(__file__).parent / "results" / f"{self.label}.txt", "a") as f:
                f.write(f"\nBest Threshold: {threshold}\nBest Accuracy: {accuracy}")
            
            # Code snippet to replace
            old_snippet = """
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            break
"""
            assert old_snippet.strip() in function_code, "head perception snippet not found in the generated function code"
            # Replacement snippet
            new_snippet = """
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            continue
        else:
            time.sleep(0.1) # Maintain 10 Hz rate
"""
            updated_function_code = function_code.replace(old_snippet.strip(), new_snippet.strip())  
            function_code_with_threshold = f"""
def {self.label}(perception_interface, termination_event, timeout):
    \"\"\"{self.language_description}\"\"\"
    threshold = {threshold}
{textwrap.indent(updated_function_code, "    ")}
    return gesture_detector(perception_interface, termination_event, timeout, threshold)
"""   
            return function_code_with_threshold
        except Exception as e:
            print("Error: ", e)
            with open(Path(__file__).parent / "results" / f"{self.label}.txt", "a") as f:
                f.write(f"\nError: {e}")
            return None
    
    def test_in_context_examples(self):
        self._load_from_data_path(Path(__file__).parent / "gestures_examples" / "shake_my_head_from_left_to_right.pkl")
        threshold1, accuracy1 = self.search_threshold(in_context_example1)
        print("In-Context Example 1")
        print("Best Threshold: ", threshold1)
        print("Best Accuracy: ", accuracy1)

        # /home/rkjenamani/sim_experiments/feeding-deployment/src/feeding_deployment/integration/log/gesture_examples/open_mouth.pkl
        # self._load_from_data_path(Path(__file__).parent.parent.parent / "integration" / "log" / "gesture_examples" / "open_mouth.pkl")
        self._load_from_data_path(Path(__file__).parent / "gestures_examples" / "open_mouth.pkl")
        threshold2, accuracy2 = self.search_threshold(in_context_example2)
        print("In-Context Example 2")
        print("Best Threshold: ", threshold2)
        print("Best Accuracy: ", accuracy2)
    
    def run_detector(self, gesture_detector, **kwargs):
        """
        Run the gesture detector on examples in self.examples_data_path
        """
        positive_correct = 0
        for positive_example in self.positive_examples:
            perception_interface = MockPerceptionInterface(head_perception_data=positive_example)
            if gesture_detector(perception_interface, **kwargs):
                positive_correct += 1
        
        negative_correct = 0
        for negative_example in self.negative_examples:
            perception_interface = MockPerceptionInterface(head_perception_data=negative_example)
            if not gesture_detector(perception_interface, **kwargs):
                negative_correct += 1
        
        return positive_correct/len(self.positive_examples), negative_correct/len(self.negative_examples)
    
    def search_threshold(self, gesture_detector, timeout=20.0, threshold_range=(0.0, 1.0), step=0.1):
        """
        Search for the best threshold for the given gesture detector
        """
        best_threshold = None
        best_accuracy = 0.0

        for threshold in np.arange(threshold_range[0], threshold_range[1], step):
            positive_accuracy, negative_accuracy = self.run_detector(gesture_detector, termination_event=None, timeout=timeout, threshold=threshold)
            # print("Threshold: ", threshold, "Positive Accuracy: ", positive_accuracy, "Negative Accuracy: ", negative_accuracy)
            accuracy = (positive_accuracy + negative_accuracy) / 2.0
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold, best_accuracy

def main():

    synthesizer = PersonalizedGestureDetectorSynthesizer()
    # synthesizer.test_in_context_examples()

    gestures = [
        # "blinking",
        # "eyebrows_raised",
        # "head_nod",
        # "head_still_atleast_three_secs",
        # "look_at_robot_atleast_three_secs",
        # "talking",
        "open_mouth",
    ]

    for gesture in gestures:
        
        gesture_datapath = Path(__file__).parent / "gestures_examples" / f"{gesture}.pkl"
        generated_function = synthesizer.generate_function(gesture_datapath)
        if generated_function is not None:
            with open("synthesized_gesture_detectors.py", "a") as f:
                f.write(generated_function)

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