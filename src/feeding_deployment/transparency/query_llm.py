import os
from openai import OpenAI
import pickle
import numpy as np
import math
import time
from pathlib import Path
from tomsutils.llm import OpenAILLM

from feeding_deployment.transparency.base import TransparencyBase

class TransparencyQuery(TransparencyBase):

    def __init__(self, log_dir):
        super().__init__(log_dir)
        with open(Path(__file__).parent / "query_prompt.txt", 'r') as f:
            self.prompt_skeleton = f.read()
        self.query_history = ""

    def answer_query(self, query):
        """
        Answers the query using the GPT model.
        """

        behavior = self.load_behavior()
        execution = self.load_execution()
        sensor = self.load_sensor()
        prompt = self.prompt_skeleton%(behavior, execution, sensor, self.query_history, query)
        response = self.llm.sample_completions(prompt, imgs=None, temperature=0.0, seed=0)[0]

        self.query_history += f"Query: {query}\nResponse: {response}\n\n"
        return response

def main():
    raise NotImplementedError("Set log directory correctly")
    transparency_query = TransparencyQuery(log_dir = Path(__file__).parent.parent / "integration" / "log")
    print("Ready to answer queries. Type 'exit' to quit.")
    while True:
        query = input("Enter a query: ")
        if query == 'exit':
            break
        response = transparency_query.answer_query(query)
        print("Response: ", response)

if __name__ == '__main__':
    main()
