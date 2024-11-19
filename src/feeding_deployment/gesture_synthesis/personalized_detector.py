import cv2
import time
import numpy as np
import math
import os
import pickle

import base64
import requests
import json


from feeding_deployment.head_perception.deca_perception import (
    HeadPerception,
)

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
    
class Robot:
    def __init__(self):
        self.head_perception = HeadPerception(record_goal_pose=False)
    
    def get_head_pose(self):
        _, camera_color_data, camera_info_data, camera_depth_data = self.get_camera_data()

        if camera_color_data is not None:
            base_to_camera = np.ones((4, 4))

            (
                landmarks2d,
                landmarks3d,
                viz_image,
                mouth_state,
                average_head_point,
                tool_tip_target_pose,
                visualization_points_world_frame,
                reference_neck_frame,
                neck_frame,
                noisy_reading,
                neck_rotation,
            ) = self.head_perception.run_deca(
                camera_color_data,
                camera_info_data,
                camera_depth_data,
                base_to_camera,
                debug_print=False,
                visualize=False,
                filter_noisy_readings=False,
            )

            neck_position = neck_frame[:3, 3]
            neck_orientation = R.from_matrix(neck_frame[:3, :3]).as_euler("xyz", degrees=True)

            head_x = neck_position[0]
            head_y = neck_position[1]
            head_z = neck_position[2]
            head_roll = neck_orientation[0]
            head_yaw = neck_orientation[1]
            head_pitch = neck_orientation[2]

            return head_x, head_y, head_z, head_roll, head_yaw, head_pitch
        else:
            return None, None, None, None, None, None
    
    def get_face_keypoints(self):

        _, camera_color_data, camera_info_data, camera_depth_data = self.get_camera_data()

        if camera_color_data is not None:
            base_to_camera = np.ones((4, 4))

            (
                landmarks2d,
                landmarks3d,
                viz_image,
                mouth_state,
                average_head_point,
                tool_tip_target_pose,
                visualization_points_world_frame,
                reference_neck_frame,
                neck_frame,
                noisy_reading,
                neck_rotation,
            ) = self.head_perception.run_deca(
                camera_color_data,
                camera_info_data,
                camera_depth_data,
                base_to_camera,
                debug_print=False,
                visualize=False,
                filter_noisy_readings=False,
            )

            return landmarks2d
        else:
            return None
    
    def set_sample(self, data_path):

        # load data from pickle file data_path
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def set_start_time(self):
        self.start_time = time.time()
    
    def get_camera_data(self):
        current_time = time.time() 
        timestamp = current_time - self.start_time
        index = int(timestamp*10) # data is at 10Hz

        if index < len(self.data):
            camera_header = self.data['camera_header'][index]
            camera_color_data = self.data['camera_color_data'][index]
            camera_info_data = self.data['camera_info_data'][index]
            camera_depth_data = self.data['camera_depth_data'][index]
            return camera_header, camera_color_data, camera_info_data, camera_depth_data
        
        return None, None, None, None
    
def in_context_example(robot, timeout=20.0, threshold=0.1):
    """
    Verifies the in-context example code provided in the prompt
    """
    start_time = time.time()
    pitch_data = []
    direction_changes = 0  # Counts the number of up-down or down-up changes

    while time.time() - start_time < timeout:
        (head_x, head_y, head_z, head_roll, head_yaw, head_pitch) = robot.get_head_pose()
        pitch_data.append(head_pitch)

        if len(pitch_data) > 3:
            if (pitch_data[-2] - pitch_data[-3] > threshold and pitch_data[-2] - pitch_data[-1] > threshold) or \
               (pitch_data[-3] - pitch_data[-2] > threshold and pitch_data[-1] - pitch_data[-2] > threshold):
                direction_changes += 1

        if direction_changes == 2:
            return True

    return False

def main():

    robot = Robot()
    robot.set_sample('src/feeding_deployment/gesture_synthesis/gesture_data/shake_my_head_from_left_to_right/positive_examples/0.pkl')
    robot.set_start_time()
    in_context_example(robot)

    # gpt = GPTInterface()
    # with open('prompt.txt', 'r') as f:
    #     prompt = f.read()

    # language_description = "mouth open"
    # prompt = prompt%(language_description)

    # print("=== Prompt ===")
    # print(prompt)
    # response = gpt.chat_with_openai(prompt)
    # print("=== Response ===")
    # print(response)

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