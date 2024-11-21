import os
from openai import OpenAI

import numpy as np
import math
import time

from feeding_deployment.gesture_synthesis.robot import Robot, run_detector, search_threshold

class GPTInterface:
    def __init__(self):
        self.api_key =  os.environ.get('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        
    def chat_with_openai(self, prompt):
        """
        Sends the prompt to OpenAI API using the chat interface and gets the model's response.
        """
        message = {
                    'role': 'user',
                    'content': prompt
                  }
        response = self.client.chat.completions.create(
                   model='o1-mini',
                   messages=[message]
                  )
        chatbot_response = response.choices[0].message.content
        return chatbot_response.strip()

def main():
    gpt = GPTInterface()
    with open('prompt.txt', 'r') as f:
        source_prompt = f.read()

    gestures = {
        "blinking": "eyes blinking",
        "eyebrows_raised": "eyebrows raised",
        "head_nod": "up-down head nod",
        "head_still_atleast_three_secs": "head is still for atleast three seconds",
        "look_at_robot_atleast_three_secs": "looking at robot with head still for atleast three seconds",
        "talking": "talking",
    }

    for gesture, language_description in gestures.items():
        
        print("language_description: ", language_description)
        prompt = source_prompt%(language_description)
        # evaluate each gesture 10 times

        if not os.path.exists(f"gesture_data/{gesture}/results"):
            os.makedirs(f"gesture_data/{gesture}/results")
            
        for i in range(10):
            
            # print("=== Prompt ===")
            # print(prompt)
            response = gpt.chat_with_openai(prompt)
            print("=== Response ===")
            print(response)

            # save the response to a file
            with open(f"gesture_data/{gesture}/results/{i}.txt", "w") as f:
                f.write(response)

            # # read from response.txt
            # with open('response.txt', 'r') as f:
                # response = f.read()

            function_code = response.strip("```python").strip("```")
            # print("=== Function Code ===")
            # print(function_code)
            # print("==========")

            try:
                exec(function_code, globals())  # Executes code in the global namespace

                threshold, accuracy = search_threshold(f"gesture_data/{gesture}", gesture_detector)
                print("Test Example: ")
                print("Best Threshold: ", threshold)
                print("Best Accuracy: ", accuracy)
                with open(f"gesture_data/{gesture}/results/{i}.txt", "a") as f:
                    f.write(f"\nBest Threshold: {threshold}\nBest Accuracy: {accuracy}")
            except Exception as e:
                print("Error: ", e)
                with open(f"gesture_data/{gesture}/results/{i}.txt", "a") as f:
                    f.write(f"\nError: {e}")





if __name__ == '__main__':
    main()