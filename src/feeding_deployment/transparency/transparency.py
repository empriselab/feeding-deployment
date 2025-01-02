import os
from openai import OpenAI
import pickle
import numpy as np
import math
import time
from pathlib import Path

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

        self.behavior_log_path = Path(__file__).parent.parent / "integration" / "log" / "behavior_trees"
        self.execution_log_path = Path(__file__).parent.parent / "integration" / "log" / "execution_log.txt"
        self.sensor_log_path = Path(__file__).parent.parent / "integration" / "log"

        self.query_history = ""

    def load_behavior(self):
        """
        Loads the current behavior description from the log directory.
        """

        # For now I am sequencing them in a fixed order, but ideally this should reflect integration with the web interface.
        bite = ["pick_utensil", "look_at_plate", "acquire_bite", "transfer_utensil", "stow_utensil"]
        drink = ["pick_drink", "transfer_drink", "stow_drink"]
        wipe = ["pick_wipe", "transfer_wipe", "stow_wipe"]

        all_nodes_description = ""
        
        # Load the behavior trees
        all_nodes_description += "Bite:\n"
        for bite_node in bite:
            with open(self.behavior_log_path / f"{bite_node}.yaml", 'r') as f:
                node_description = f.read()
            all_nodes_description += node_description + "\n---\n"

        all_nodes_description += "Drink:\n"
        for drink_node in drink:
            with open(self.behavior_log_path / f"{drink_node}.yaml", 'r') as f:
                node_description = f.read()
            all_nodes_description += node_description + "\n---\n"

        all_nodes_description += "Wipe:\n"
        for wipe_node in wipe:
            with open(self.behavior_log_path / f"{wipe_node}.yaml", 'r') as f:
                node_description = f.read()
            all_nodes_description += node_description + "\n---\n"

        return all_nodes_description
    
    def load_execution(self):
        """
        Loads the current execution log from the log directory.
        """
        
        execution_description = ""
        with open(self.execution_log_path, 'r') as f:
            execution_description = f.read()

        return execution_description
    
    def load_sensor(self):
        """
        Loads the current sensor log from the log directory.
        """

        # For now I am just using food detection data, but I need to identify all the data that needs to be logged / used for transparency.

        # read from "food_detection_data.pkl"
        with open(self.sensor_log_path / "food_detection_data.pkl", 'rb') as f:
            food_detection_data = pickle.load(f)

        useful_info = {
            "labels_list": food_detection_data["items_detection"]["labels_list"],
            "per_food_portions": food_detection_data["items_detection"]["per_food_portions"],
            "food_type_to_skill": food_detection_data["items_detection"]["food_type_to_skill"]
        }

        sensor_data_description = "Food detection data:\n"
        for key, value in useful_info.items():
            sensor_data_description += f"{key}: {value}\n"

        return sensor_data_description

    def answer_query(self, query):
        """
        Answers the query using the GPT model.
        """

        behavior = self.load_behavior()
        execution = self.load_execution()
        sensor = self.load_sensor()
        prompt = self.prompt_skeleton%(behavior, execution, sensor, self.query_history, query)
        response = self.gpt.chat_with_openai(prompt)

        self.query_history += f"Query: {query}\nResponse: {response}\n\n"
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