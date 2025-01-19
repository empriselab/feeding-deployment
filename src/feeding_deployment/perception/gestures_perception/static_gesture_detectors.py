import time
import numpy as np

function_name_to_label = {
    "mouth_open": "mouth open",
    "head_nod": "head nod"
}

def mouth_open(perception_interface, termination_event, timeout):
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

def head_nod(perception_interface, termination_event, timeout):
    """ Detect head nod """

    threshold = 5.0

    def gesture_detector(perception_interface, termination_event, timeout, threshold):

        start_time = time.time()
        pitch_data = []
        direction_changes = 0  # Counts the number of up-down or down-up changes

        while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            head_pose = head_perception_data["head_pose"]

            (head_x, head_y, head_z, head_roll, head_pitch, head_yaw) = head_pose
            pitch_data.append(head_pitch)

            # Check if there is enough data to detect direction change
            if len(pitch_data) >= 3:

                print("Pitch differences: ", pitch_data[-2] - pitch_data[-3], pitch_data[-2] - pitch_data[-1])
                if (pitch_data[-2] - pitch_data[-3] > threshold and pitch_data[-2] - pitch_data[-1] > threshold) or \
                (pitch_data[-3] - pitch_data[-2] > threshold and pitch_data[-1] - pitch_data[-2] > threshold):
                    direction_changes += 1

                if direction_changes >= 2:
                    return True

            # To avoid excessive memory usage, keep the pitch_data size small
            if len(pitch_data) > 100:
                pitch_data.pop(0)

        # If timeout expires without detecting the gesture, return False
        return False
    
    return gesture_detector(perception_interface, termination_event, timeout, threshold)
