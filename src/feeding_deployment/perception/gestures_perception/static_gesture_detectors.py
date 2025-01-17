import time
import numpy as np

def mouth_open_detector(perception_interface, termination_event, timeout):
    """ Detect mouth open """
    threshold = 0.45

    def gesture_detector(perception_interface, termination_event, timeout, threshold):

        def euclidean_distance(p1, p2):
            """Calculate Euclidean distance between two points."""
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        start_time = time.time()
        while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            face_keypoints = head_perception_data["face_keypoints"]
            
            # Indices for mouth landmarks
            mouth_points = face_keypoints[48:68]
            
            # Calculate vertical distances
            A = euclidean_distance(mouth_points[2], mouth_points[10])  # 51, 59
            B = euclidean_distance(mouth_points[4], mouth_points[8])   # 53, 57
        
            # Calculate horizontal distance
            C = euclidean_distance(mouth_points[0], mouth_points[6])   # 49, 55

            mar = (A + B) / (2.0 * C)
            if mar > threshold:
                return True
    
        return False

    return gesture_detector(perception_interface, termination_event, timeout, threshold)

def head_shake_detector(perception_interface, termination_event, timeout):
    """ Detect head shake """
    threshold = 2.0

    def gesture_detector(perception_interface, termination_event, timeout, threshold):

        start_time = time.time()
        yaw_data = []
        direction_changes = 0  # Counts the number of left-right or right-left changes
        

        while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            head_pose = head_perception_data["head_pose"]

            (head_x, head_y, head_z, head_roll, head_pitch, head_yaw) = head_pose
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
    
    return gesture_detector(perception_interface, termination_event, timeout, threshold)

def head_still_detector(perception_interface, termination_event, timeout):
    """ Detect head still for 5 seconds """

    distance_threshold = 0.02
    angle_threshold = 5.0

    # Rajat ToDo: Add support for multiple thresholds to synthesizer
    def gesture_detector(perception_interface, termination_event, timeout, distance_threshold, angle_threshold):
        start_time = time.time()
        last_head_pose = None
        head_still_start_time = time.time()

        while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            head_pose = head_perception_data["head_pose"]
            if last_head_pose is not None:
                # head_pose is a tuple (x, y, z, roll, pitch, yaw) with x, y, z in meters and angles in degrees
                if (np.any(np.abs(np.array(head_pose[:3]) - np.array(last_head_pose[:3])) > distance_threshold)
                    or np.any(np.abs(np.array(head_pose[3:]) - np.array(last_head_pose[3:])) > angle_threshold)):
                    head_still_start_time = time.time()
            last_head_pose = head_pose
            print("Head still time:", time.time() - head_still_start_time)
            if time.time() - head_still_start_time > 5.0:
                return True
        return False
    
    return gesture_detector(perception_interface, termination_event, timeout, distance_threshold, angle_threshold)