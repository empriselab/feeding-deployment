import time
import numpy as np

def mouth_open_detector(perception_interface, timeout):

    start_time = time.time()
    while time.time() - start_time < timeout:
        head_perception_data = perception_interface.get_head_perception_data()
        keypoints = head_perception_data["face_keypoints"]

        if len(keypoints) != 68:
            print("Not enough keypoints : ", np.array(keypoints).shape[0])
            continue

        lipDist = np.sqrt((keypoints[66][0] - keypoints[62][0]) ** 2 + (keypoints[66][1] - keypoints[62][1]) ** 2)

        lipThickness = float(np.sqrt((keypoints[51][0] - keypoints[62][0]) ** 2 + (keypoints[51][1] - keypoints[62][1]) ** 2) / 2
        ) + float(np.sqrt((keypoints[57][0] - keypoints[66][0]) ** 2 + (keypoints[57][1] - keypoints[66][1]) ** 2) / 2)

        if lipDist >= 1.5 * lipThickness:
            return True
        
    return False

def head_shake_detector(perception_interface, timeout):
    start_time = time.time()
    last_head_pose = None
    while time.time() - start_time < timeout:
        head_perception_data = perception_interface.get_head_perception_data()
        head_pose = head_perception_data["head_pose"]
        if last_head_pose is not None:
            if np.linalg.norm(np.array(head_pose) - np.array(last_head_pose)) < 0.1:
                return True
        last_head_pose = head_pose
    return False

def head_still_detector(perception_interface, timeout):
    """ Detect head still for 5 seconds """

    start_time = time.time()
    last_head_pose = None
    head_still_start_time = time.time()
    while time.time() - start_time < timeout:
        head_perception_data = perception_interface.get_head_perception_data()
        head_pose = head_perception_data["head_pose"]
        if last_head_pose is not None:
            if np.linalg.norm(np.array(head_pose) - np.array(last_head_pose)) > 0.1:
                head_still_start_time = time.time()
        last_head_pose = head_pose
        if time.time() - head_still_start_time > 5.0:
            return True
    return False