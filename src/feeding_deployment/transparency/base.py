import os
import abc
from openai import OpenAI
import pickle
import numpy as np
import math
import time
from pathlib import Path
from tomsutils.llm import OpenAILLM

class TransparencyBase(abc.ABC):
    def __init__(self):

        log_dir = Path(__file__).parent.parent / "integration" / "log"

        self.llm = OpenAILLM(
            model_name="gpt-4o",
            cache_dir=log_dir / "llm_cache",
        )

        self.behavior_log_path = log_dir / "behavior_trees"
        self.execution_log_path = log_dir / "execution_log.txt"
        self.nuc_execution_log_path = log_dir / "nuc_execution_log.txt"
        self.sensor_log_path = log_dir

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
        if self.execution_log_path.exists():
            with open(self.execution_log_path, 'r') as f:
                execution_description = f.read()

            with open(self.nuc_execution_log_path, 'r') as f:
                nuc_execution_description = f.read()
            execution_description += nuc_execution_description

        return execution_description
    
    def load_sensor(self):
        """
        Loads the current sensor log from the log directory.
        """

        # For now I am just using food detection data, but I need to identify all the data that needs to be logged / used for TransparencyBase.

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