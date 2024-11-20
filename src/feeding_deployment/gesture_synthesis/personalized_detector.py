import cv2
import time
import numpy as np
import math
import os
from scipy.spatial.transform import Rotation as R

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
                   model='o1-preview',
                   messages=[message]
                  )
        chatbot_response = response.choices[0].message.content
        return chatbot_response.strip()

def main():
    gpt = GPTInterface()
    with open('prompt.txt', 'r') as f:
        prompt = f.read()

    language_description = "talking"
    prompt = prompt%(language_description)

    # print("=== Prompt ===")
    # print(prompt)
    response = gpt.chat_with_openai(prompt)
    print("=== Response ===")
    print(response)

if __name__ == '__main__':
    main()