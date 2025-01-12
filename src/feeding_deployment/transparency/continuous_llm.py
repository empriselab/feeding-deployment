import os
from openai import OpenAI
import pickle
import numpy as np
import math
import time
from pathlib import Path
from tomsutils.llm import OpenAILLM

from feeding_deployment.transparency.base import TransparencyBase

class TransparencyContinuous(TransparencyBase):

    def __init__(self):
        super().__init__()
        with open(Path(__file__).parent / "continuous_prompt.txt", 'r') as f:
            self.prompt_skeleton = f.read()
        self.explanation_history = ""

        self.last_behavior = self.load_behavior()
        self.last_execution = self.load_execution()
        self.last_sensor = self.load_sensor()

    def update_history(self):
        self.last_behavior = self.load_behavior()
        self.last_execution = self.load_execution()
        self.last_sensor = self.load_sensor()

    def get_explanation(self):
        """
        Explains what the robot is doing right now.
        """
        behavior = self.load_behavior()
        execution = self.load_execution()
        sensor = self.load_sensor()

        prompt = self.prompt_skeleton%(self.last_behavior, behavior, self.last_execution, execution, self.last_sensor, sensor, self.explanation_history)
        response = self.llm.sample_completions(prompt, imgs=None, temperature=0.0, seed=0)[0]
        self.explanation_history += response + "\n"

        self.last_behavior = behavior
        self.last_execution = execution
        self.last_sensor = sensor

        return response

    def run(self):
        while True:
            time.sleep(5)
            response = self.get_explanation()
            print(response)

def main():
    transparency_continuous = TransparencyContinuous()
    print("Generating continuous explanations...")
    transparency_continuous.run()

if __name__ == '__main__':
    main()
