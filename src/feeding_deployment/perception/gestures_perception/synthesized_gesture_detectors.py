import time
import numpy as np

def eyebrows_raised(perception_interface, timeout):
    """eyebrows raised"""
    threshold = 0.0

    def gesture_detector(perception_interface, threshold, timeout):

        def euclidean_distance(p1, p2):
            """Calculate Euclidean distance between two points."""
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        start_time = time.time()
        while time.time() - start_time < timeout:
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            face_keypoints = head_perception_data["face_keypoints"]
        
            # Indices for eyebrow and eye landmarks
            left_eyebrow_points = face_keypoints[17:22]
            right_eyebrow_points = face_keypoints[22:27]
            left_eye_points = face_keypoints[36:42]
            right_eye_points = face_keypoints[42:48]
        
            # Calculate vertical distances between eyebrows and eyes
            left_eyebrow_eye_distance = euclidean_distance(left_eyebrow_points[1], left_eye_points[1])
            right_eyebrow_eye_distance = euclidean_distance(right_eyebrow_points[1], right_eye_points[1])

            if left_eyebrow_eye_distance > threshold and right_eyebrow_eye_distance > threshold:
                return True
    
        return False

    return gesture_detector(perception_interface, timeout, threshold)

def head_nod(perception_interface, timeout):
    """up-down head nod"""
    threshold = 0.6000000000000001

    def gesture_detector(perception_interface, threshold, timeout):

        start_time = time.time()
        pitch_data = []
        direction_changes = 0  # Counts the number of up-down or down-up changes

        while time.time() - start_time < timeout:
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

    return gesture_detector(perception_interface, timeout, threshold)

def head_still_atleast_three_secs(perception_interface, timeout):
    """head is still for atleast three seconds"""
    threshold = 0.0

    def gesture_detector(perception_interface, threshold, timeout):

        start_time = time.time()
        still_start_time = None

        while time.time() - start_time < timeout:
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            head_pose = head_perception_data["head_pose"]

            (head_x, head_y, head_z, head_roll, head_pitch, head_yaw) = head_pose

            # Check if head movement is below the threshold
            if abs(head_x) < threshold and abs(head_y) < threshold and abs(head_z) < threshold and \
               abs(head_roll) < threshold and abs(head_pitch) < threshold and abs(head_yaw) < threshold:
                if still_start_time is None:
                    still_start_time = time.time()
                elif time.time() - still_start_time >= 3:
                    return True
            else:
                still_start_time = None

        return False

    return gesture_detector(perception_interface, timeout, threshold)

def look_at_robot_atleast_three_secs(perception_interface, timeout):
    """looking at robot with head still for atleast three seconds"""
    threshold = 0.0

    def gesture_detector(perception_interface, threshold, timeout):

        start_time = time.time()
        still_start_time = None

        while time.time() - start_time < timeout:
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            head_pose = head_perception_data["head_pose"]

            (head_x, head_y, head_z, head_roll, head_pitch, head_yaw) = head_pose

            # Check if head is still within the threshold
            if abs(head_roll) < threshold and abs(head_pitch) < threshold and abs(head_yaw) < threshold:
                if still_start_time is None:
                    still_start_time = time.time()
                elif time.time() - still_start_time >= 3:
                    return True
            else:
                still_start_time = None

        return False

    return gesture_detector(perception_interface, timeout, threshold)

def talking(perception_interface, timeout):
    """talking"""
    threshold = 0.30000000000000004

    def gesture_detector(perception_interface, threshold, timeout):

        def euclidean_distance(p1, p2):
            """Calculate Euclidean distance between two points."""
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        start_time = time.time()
        mouth_open_frames = 0

        while time.time() - start_time < timeout:
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
                mouth_open_frames += 1
            else:
                mouth_open_frames = 0

            # If mouth is open for a certain number of consecutive frames, consider it as talking
            if mouth_open_frames >= 5:
                return True
    
        return False

    return gesture_detector(perception_interface, timeout, threshold)
