import os
from openai import OpenAI

import numpy as np
import math
import time

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
    

class Transparency:
    def __init__(self):
        self.gpt = GPTInterface()
        with open('prompt.txt', 'r') as f:
            self.prompt_skeleton = f.read()
    
    def load_execution(self):
        """
        Loads the current logs from the logs directory.
        """

    def load_behavior(self):
        """
        Loads the current behavior from the behavior directory.
        """

    def answer_query(self, query):
        """
        Answers the query using the GPT model.
        """

        behavior = self.load_behavior()
        execution = self.load_execution()
        
        prompt = self.prompt_skeleton%(behavior, execution, query)
        response = self.gpt.chat_with_openai(prompt)
        return response
    
def main():
    transparency = Transparency()
    print("Ready to answer queries. Type 'exit' to quit.")
    while True:
        query = input("Enter a query: ")
        if query == 'exit':
            break
        response = transparency.answer_query(query)
        print("Response: ", response)

if __name__ == '__main__':
    main()