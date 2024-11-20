import cv2
import time
import numpy as np
import math
import os
import pickle
from scipy.spatial.transform import Rotation as R

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
        self.head_perception.set_tool("fork")

    def warmup(self):
        """
        Run DECA 10 times to warm up the model
        """
        
        with open('warmup.pkl', 'rb') as f:
            warmup_data = pickle.load(f)

        camera_color_data = warmup_data['color']
        camera_info_data = warmup_data['info']
        camera_depth_data = warmup_data['depth']

        for i in range(10):
            base_to_camera = np.eye(4)
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
            print("Warmup: ", i)
    
    def get_head_pose(self):
        _, camera_color_data, camera_info_data, camera_depth_data = self.get_camera_data()

        if camera_color_data is not None:
            base_to_camera = np.eye(4)

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

            head_roll = neck_orientation[0]  # Rotation around x-axis
            head_pitch = neck_orientation[1]  # Rotation around y-axis
            head_yaw = neck_orientation[2]  # Rotation around z-axis

            # because our axis if not the conventional one, we will switch to conventional axis
            conventional_head_roll = head_yaw
            conventional_head_pitch = head_roll
            conventional_head_yaw = head_pitch

            return head_x, head_y, head_z, conventional_head_roll, conventional_head_pitch, conventional_head_yaw
        else:
            return None, None, None, None, None, None
    
    def get_face_keypoints(self):

        _, camera_color_data, camera_info_data, camera_depth_data = self.get_camera_data()

        if camera_color_data is not None:
            base_to_camera = np.eye(4)

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

        print("Number of frames: ", len(self.data['color']))

    def set_start_time(self):
        self.start_time = time.time()
    
    def get_camera_data(self):
        current_time = time.time() 
        timestamp = current_time - self.start_time 
        index = int(timestamp*10) # data is at 10Hz
        # print("Index: ", index, "Length: ", len(self.data['color']))

        if index < len(self.data['color']):
            camera_header = self.data['header'][index]
            camera_color_data = self.data['color'][index]
            camera_info_data = self.data['info'][index]
            camera_depth_data = self.data['depth'][index]

            return camera_header, camera_color_data, camera_info_data, camera_depth_data
        
        return None, None, None, None
    
def in_context_example1(robot, timeout=20.0, threshold=0.5):
    """
    Verifies the in-context example 1 code provided in the prompt
    """
    start_time = time.time()
    yaw_data = []
    direction_changes = 0  # Counts the number of left-right or right-left changes

    id = 0
    while time.time() - start_time < timeout:
        head_x, head_y, head_z, head_roll, head_pitch, head_yaw = robot.get_head_pose()
        # print("Head Rotation: ", head_roll, head_pitch, head_yaw)
        # id+= 1
        # time.sleep(0.5)
        if head_x is None:
            break # Handle case where head pose data is unavailable
        yaw_data.append(head_yaw)

        # Check if there is enough data to detect direction change
        if len(yaw_data) >= 3:

            if (yaw_data[-2] - yaw_data[-3] > threshold and yaw_data[-2] - yaw_data[-1] > threshold) or \
               (yaw_data[-3] - yaw_data[-2] > threshold and yaw_data[-1] - yaw_data[-2] > threshold):
                direction_changes += 1

            if direction_changes >= 2:
                return True

        # To avoid excessive memory usage, keep the yaw_data size small
        if len(yaw_data) > 100:
            yaw_data.pop(0)
    
    # If timeout expires without detecting the gesture, return False
    return False

def in_context_example2(robot, timeout=20.0, threshold=0.6):
    """
    Verifies the in-context example 2 code provided in the prompt
    """

    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    start_time = time.time()

    max_mar = 0.0

    # check for mouth open
    while time.time() - start_time < timeout:
        face_keypoints = robot.get_face_keypoints()

        if face_keypoints is None:
            break
        
        # Indices for mouth landmarks
        mouth_points = face_keypoints[48:68]
        
        # Calculate vertical distances
        A = euclidean_distance(mouth_points[2], mouth_points[10])  # 51, 59
        B = euclidean_distance(mouth_points[4], mouth_points[8])   # 53, 57
    
        # Calculate horizontal distance
        C = euclidean_distance(mouth_points[0], mouth_points[6])   # 49, 55

        mar = (A + B) / (2.0 * C)
        # print("MAR: ", mar, "Threshold: ", threshold)
        max_mar = max(max_mar, mar)
        if mar > threshold:
            # print("Max MAR: ", max_mar, "Detection: ", True)
            return True
    
    # print("Max MAR: ", max_mar, "Detection: ", False)
    return False

def validate_in_context_examples():

    robot = Robot()
    robot.warmup()

    data_path = 'gesture_data/shake_my_head_from_left_to_right'

    for i in range(5):
        robot.set_sample(data_path + f'/positive_examples/{i}.pkl')
        robot.set_start_time()
        print("Positive example output: ",in_context_example1(robot))

    for i in range(5):
        robot.set_sample(data_path + f'/negative_examples/{i}.pkl')
        robot.set_start_time()
        print("Negative example output: ",in_context_example1(robot))

    data_path = 'gesture_data/open_mouth'
    
    for i in range(5):
        robot.set_sample(data_path + f'/positive_examples/{i}.pkl')
        robot.set_start_time()
        print("Positive example output: ",in_context_example2(robot))

    for i in range(5):
        robot.set_sample(data_path + f'/negative_examples/{i}.pkl')
        robot.set_start_time()
        print("Negative example output: ",in_context_example2(robot))

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
    validate_in_context_examples()
    # main()