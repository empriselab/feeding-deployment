import os
from openai import OpenAI
import numpy as np
import math
import time
from pathlib import Path
import textwrap
import pickle
from tomsutils.llm import OpenAILLM

from feeding_deployment.perception.gesture_synthesis.in_context_examples import in_context_example1, in_context_example2

class MockPerceptionInterface:
    """
    Simulate the perception interface for the robot with get_head_perception_data method
    """
    def __init__(self, data_path):

        # load data from pickle file data_path
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.face_keypoints = data['face_keypoints']
        self.head_pose = data['head_poses']
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

        with open('prompt.txt', 'r') as f:
            self.prompt_skeleton = f.read()
    
    def generate_function(self, label, language_description, examples_data_path):
        prompt = self.prompt_skeleton%(language_description)
        response = self.llm.sample_completions(prompt, imgs=None, temperature=0.0, seed=0)[0]
        function_code = response.strip("```python").strip("```")
        try:
            exec(function_code, globals())  # Executes code in the global namespace

            # Assumes that the data is stored in gesture_data/{label}/
            self.examples_data_path = examples_data_path
            threshold, accuracy = self.search_threshold(gesture_detector)
            print("Best Threshold: ", threshold)
            print("Best Accuracy: ", accuracy)
            with open(f"gesture_data/{label}/results/response.txt", "a") as f:
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
def {label}(perception_interface, timeout):
    \"\"\"{language_description}\"\"\"
    threshold = {threshold}
{textwrap.indent(updated_function_code, "    ")}
    return gesture_detector(perception_interface, timeout, threshold)
"""   
            return function_code_with_threshold
        except Exception as e:
            print("Error: ", e)
            with open(f"gesture_data/{label}/results/response.txt", "a") as f:
                f.write(f"\nError: {e}")
            return None
    
    def test_in_context_examples(self):
        self.examples_data_path = "gesture_data/shake_my_head_from_left_to_right"
        threshold1, accuracy1 = self.search_threshold(in_context_example1)
        print("In-Context Example 1")
        print("Best Threshold: ", threshold1)
        print("Best Accuracy: ", accuracy1)

        self.examples_data_path = "gesture_data/open_mouth"
        threshold2, accuracy2 = self.search_threshold(in_context_example2)
        print("In-Context Example 2")
        print("Best Threshold: ", threshold2)
        print("Best Accuracy: ", accuracy2)
    
    def run_detector(self, gesture_detector, **kwargs):
        """
        Run the gesture detector on examples in self.examples_data_path
        """
        positive_correct = 0
        for i in range(5):
            perception_interface = MockPerceptionInterface(self.examples_data_path + f'/positive_examples/{i}_parsed.pkl')
            if gesture_detector(perception_interface, **kwargs):
                positive_correct += 1
        
        negative_correct = 0
        for i in range(5):
            perception_interface = MockPerceptionInterface(self.examples_data_path + f'/negative_examples/{i}_parsed.pkl')
            if not gesture_detector(perception_interface, **kwargs):
                negative_correct += 1
        
        return positive_correct/5.0, negative_correct/5.0
    
    def search_threshold(self, gesture_detector, timeout=20.0, threshold_range=(0.0, 1.0), step=0.1):
        """
        Search for the best threshold for the given gesture detector
        """
        best_threshold = None
        best_accuracy = 0.0

        for threshold in np.arange(threshold_range[0], threshold_range[1], step):
            positive_accuracy, negative_accuracy = self.run_detector(gesture_detector, timeout=timeout, threshold=threshold)
            # print("Threshold: ", threshold, "Positive Accuracy: ", positive_accuracy, "Negative Accuracy: ", negative_accuracy)
            accuracy = (positive_accuracy + negative_accuracy) / 2.0
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold, best_accuracy


def main():

    synthesizer = PersonalizedGestureDetectorSynthesizer()
    synthesizer.test_in_context_examples()

    gestures = {
        "blinking": "eyes blinking",
        "eyebrows_raised": "eyebrows raised",
        "head_nod": "up-down head nod",
        "head_still_atleast_three_secs": "head is still for atleast three seconds",
        "look_at_robot_atleast_three_secs": "looking at robot with head still for atleast three seconds",
        "talking": "talking",
    }

    for gesture, language_description in gestures.items():

        if not os.path.exists(f"gesture_data/{gesture}/results"):
            os.makedirs(f"gesture_data/{gesture}/results")
        examples_data_path = f"gesture_data/{gesture}"
        generated_function = synthesizer.generate_function(gesture, language_description, examples_data_path)
        if generated_function is not None:
            with open("synthesized_gesture_detectors.py", "a") as f:
                f.write(generated_function)

if __name__ == '__main__':
    main()