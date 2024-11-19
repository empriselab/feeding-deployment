

# Step 1: Prompt LLM to generate code
# Step 2: Do parameter search for personalized detection

# Prompt:
# 1. Candidate funtions: head detection
# 2. In-context examples: Left to right head shake
# 3. Language command

import cv2
import time
import numpy as np
import math
import os

import base64
import requests
import json

from openai import OpenAI
import ast

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
                   model='gpt-4o',
                   messages=[message]
                  )
        chatbot_response = response.choices[0].message.content
        return chatbot_response.strip()

def main():
    gpt = GPTInterface()
    with open('prompt.txt', 'r') as f:
        prompt = f.read()

    language_description = "mouth open"
    prompt = prompt%(language_description)

    print("=== Prompt ===")
    print(prompt)
    response = gpt.chat_with_openai(prompt)
    print("=== Response ===")
    print(response)

# def gesture_detector(threshold, timeout=20.0):
#     start_time = time.time()
#     pitch_data = []
#     direction_changes = 0  # Counts the number of up-down or down-up changes

#     while time.time() - start_time < timeout:
#         (head_x, head_y, head_z, head_roll, head_yaw, head_pitch) = get_head_pose()
#         pitch_data.append(head_pitch)

#         if len(pitch_data) > 3:
#             if (pitch_data[-2] - pitch_data[-3] > threshold and pitch_data[-2] - pitch_data[-1] > threshold) or \
#                (pitch_data[-3] - pitch_data[-2] > threshold and pitch_data[-1] - pitch_data[-2] > threshold):
#                 direction_changes += 1

#         if direction_changes == 2:
#             return True

#     return False

if __name__ == '__main__':
    main()