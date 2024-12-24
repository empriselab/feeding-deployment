import cv2
import time
import numpy as np
import math
import os
import pickle
from scipy.spatial.transform import Rotation as R

from feeding_deployment.perception.head_perception.deca_perception import (
    HeadPerception,
)

class HeadPerceptionLogWrapper:
    def __init__(self):
        self.head_perception = HeadPerception(record_goal_pose=False)
        self.head_perception.set_tool("fork")
        self.warmup()

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
    
    def perceive_head(self):
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

            head_pose = (head_x, head_y, head_z, conventional_head_roll, conventional_head_pitch, conventional_head_yaw)
            face_keypoints = landmarks2d

            return head_pose, face_keypoints
        else:
            return None, None
    
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
    
    def parse(self, data_path):
        # load data from pickle file data_path
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        print("Number of frames: ", len(self.data['color']))
        self.start_time = time.time()

        head_poses = []
        face_keypoints = []

        while True:
            head_pose, face_keypoint = self.perceive_head()
            if head_pose is None or face_keypoint is None:
                break
            head_poses.append(head_pose)
            face_keypoints.append(face_keypoint)

        # save in pickle file at same data_path but with _parsed
        parsed_data = {
            'head_poses': head_poses,
            'face_keypoints': face_keypoints
        }
        with open(data_path.replace('.pkl', '_parsed.pkl'), 'wb') as f:
            pickle.dump(parsed_data, f)

    def parse_dataset(self, source_path):
        
        for i in range(5):
            data_path = source_path + f'/positive_examples/{i}.pkl'
            self.parse(data_path)
        
        for i in range(5):
            data_path = source_path + f'/negative_examples/{i}.pkl'
            self.parse(data_path)

if __name__ == "__main__":
    head_perception_log_wrapper = HeadPerceptionLogWrapper()
    
    # source_paths = ['shake_my_head_from_left_to_right', 'open_mouth']
    source_paths = ['blinking', 'eyebrows_raised', 'head_nod', 'head_still_atleast_three_secs', 'look_at_robot_atleast_three_secs', 'talking']
    for source_path in source_paths:
        head_perception_log_wrapper.parse_dataset( 'gesture_data/' + source_path)

            
        