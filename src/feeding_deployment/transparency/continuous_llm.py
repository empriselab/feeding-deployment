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

    def run(self):

        old_behavior = self.load_behavior()
        old_execution = self.load_execution()
        old_sensor = self.load_sensor()

        while True:
            time.sleep(1)
            behavior = self.load_behavior()
            execution = self.load_execution()
            sensor = self.load_sensor()

            if behavior != old_behavior or execution != old_execution or sensor != old_sensor:
                prompt = self.prompt_skeleton%(old_behavior, behavior, old_execution, execution, old_sensor, sensor, self.explanation_history)
                response = self.llm.sample_completions(prompt, imgs=None, temperature=0.0, seed=0)[0]
                print(response)
                self.explanation_history += response + "\n"
                old_behavior, old_execution, old_sensor = behavior, execution, sensor
            # else:
                # print("No change in behavior, execution, or sensor.")

def main():
    transparency_continuous = TransparencyContinuous()
    print("Generating continuous explanations...")
    transparency_continuous.run()

if __name__ == '__main__':
    main()
