import time
import numpy as np

def blinking(perception_interface, termination_event, timeout):
    """eyes blinking"""
    threshold = 0.2

    def gesture_detector(perception_interface, termination_event, timeout, threshold):

        def euclidean_distance(p1, p2):
            """Calculate Euclidean distance between two points."""
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        start_time = time.time()
        blink_detected = False

        while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            face_keypoints = head_perception_data["face_keypoints"]

            # Indices for eye landmarks
            left_eye_points = face_keypoints[36:42]
            right_eye_points = face_keypoints[42:48]

            # Calculate vertical distances for left eye
            left_eye_vertical_1 = euclidean_distance(left_eye_points[1], left_eye_points[5])
            left_eye_vertical_2 = euclidean_distance(left_eye_points[2], left_eye_points[4])

            # Calculate horizontal distance for left eye
            left_eye_horizontal = euclidean_distance(left_eye_points[0], left_eye_points[3])

            # Calculate Eye Aspect Ratio (EAR) for left eye
            left_ear = (left_eye_vertical_1 + left_eye_vertical_2) / (2.0 * left_eye_horizontal)

            # Calculate vertical distances for right eye
            right_eye_vertical_1 = euclidean_distance(right_eye_points[1], right_eye_points[5])
            right_eye_vertical_2 = euclidean_distance(right_eye_points[2], right_eye_points[4])

            # Calculate horizontal distance for right eye
            right_eye_horizontal = euclidean_distance(right_eye_points[0], right_eye_points[3])

            # Calculate Eye Aspect Ratio (EAR) for right eye
            right_ear = (right_eye_vertical_1 + right_eye_vertical_2) / (2.0 * right_eye_horizontal)

            # Average EAR for both eyes
            ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below the threshold indicating a blink
            if ear < threshold:
                blink_detected = True
            else:
                if blink_detected:
                    return True
                blink_detected = False

        return False

    return gesture_detector(perception_interface, termination_event, timeout, threshold)

def eyebrows_raised(perception_interface, termination_event, timeout):
    """eyebrows raised"""
    threshold = 0.0

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
        
            # Indices for eyebrow landmarks
            left_eyebrow_points = face_keypoints[17:22]
            right_eyebrow_points = face_keypoints[22:27]
            left_eye_points = face_keypoints[36:42]
            right_eye_points = face_keypoints[42:48]
        
            # Calculate vertical distances between eyebrows and eyes
            left_eyebrow_eye_distance = euclidean_distance(left_eyebrow_points[2], left_eye_points[1])
            right_eyebrow_eye_distance = euclidean_distance(right_eyebrow_points[2], right_eye_points[1])

            # Average distance
            average_distance = (left_eyebrow_eye_distance + right_eyebrow_eye_distance) / 2.0

            if average_distance > threshold:
                return True

        return False

    return gesture_detector(perception_interface, termination_event, timeout, threshold)

def head_nod(perception_interface, termination_event, timeout):
    """up-down head nod"""
    threshold = 0.6000000000000001

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

def head_still_atleast_three_secs(perception_interface, termination_event, timeout):
    """head is still for atleast three seconds"""
    threshold = 0.0

    def gesture_detector(perception_interface, termination_event, timeout, threshold):

        start_time = time.time()
        still_start_time = None

        while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            head_pose = head_perception_data["head_pose"]

            (head_x, head_y, head_z, head_roll, head_pitch, head_yaw) = head_pose

            # Check if head is still by comparing the absolute values of roll, pitch, and yaw with the threshold
            if abs(head_roll) < threshold and abs(head_pitch) < threshold and abs(head_yaw) < threshold:
                if still_start_time is None:
                    still_start_time = time.time()
                elif time.time() - still_start_time >= 3:
                    return True
            else:
                still_start_time = None

        return False

    return gesture_detector(perception_interface, termination_event, timeout, threshold)

def look_at_robot_atleast_three_secs(perception_interface, termination_event, timeout):
    """looking at robot with head still for atleast three seconds"""
    threshold = 0.0

    def gesture_detector(perception_interface, termination_event, timeout, threshold):

        start_time = time.time()
        still_start_time = None

        while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            head_pose = head_perception_data["head_pose"]

            (head_x, head_y, head_z, head_roll, head_pitch, head_yaw) = head_pose

            # Check if head is still by comparing roll, pitch, and yaw to the threshold
            if abs(head_roll) < threshold and abs(head_pitch) < threshold and abs(head_yaw) < threshold:
                if still_start_time is None:
                    still_start_time = time.time()
                elif time.time() - still_start_time >= 3:
                    return True
            else:
                still_start_time = None

        return False

    return gesture_detector(perception_interface, termination_event, timeout, threshold)

def talking(perception_interface, termination_event, timeout):
    """talking"""
    threshold = 0.30000000000000004

    def gesture_detector(perception_interface, termination_event, timeout, threshold):

        def euclidean_distance(p1, p2):
            """Calculate Euclidean distance between two points."""
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        start_time = time.time()
        mouth_open_frames = 0
        total_frames = 0

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
                mouth_open_frames += 1

            total_frames += 1

            # Check if the mouth is open for a significant portion of the frames
            if total_frames > 0 and (mouth_open_frames / total_frames) > 0.5:
                return True

        return False

    return gesture_detector(perception_interface, termination_event, timeout, threshold)

def three_eye_blinks(perception_interface, termination_event, timeout):
    """three_eye_blinks"""
    threshold = 0.2

    def gesture_detector(perception_interface, termination_event, timeout, threshold):

        def euclidean_distance(p1, p2):
            """Calculate Euclidean distance between two points."""
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        start_time = time.time()
        blink_count = 0
        eye_aspect_ratio_threshold = threshold

        while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            face_keypoints = head_perception_data["face_keypoints"]
        
            # Indices for eye landmarks
            left_eye_points = face_keypoints[36:42]
            right_eye_points = face_keypoints[42:48]
        
            # Calculate Eye Aspect Ratio (EAR) for both eyes
            def calculate_ear(eye_points):
                A = euclidean_distance(eye_points[1], eye_points[5])  # Vertical distance
                B = euclidean_distance(eye_points[2], eye_points[4])  # Vertical distance
                C = euclidean_distance(eye_points[0], eye_points[3])  # Horizontal distance
                ear = (A + B) / (2.0 * C)
                return ear
        
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
        
            # Check if both eyes are closed
            if left_ear < eye_aspect_ratio_threshold and right_ear < eye_aspect_ratio_threshold:
                blink_count += 1
                # Wait for eyes to open again
                while left_ear < eye_aspect_ratio_threshold and right_ear < eye_aspect_ratio_threshold:
                    head_perception_data = perception_interface.get_head_perception_data()
                    if head_perception_data is None:
                        break
                    face_keypoints = head_perception_data["face_keypoints"]
                    left_eye_points = face_keypoints[36:42]
                    right_eye_points = face_keypoints[42:48]
                    left_ear = calculate_ear(left_eye_points)
                    right_ear = calculate_ear(right_eye_points)
        
            if blink_count >= 3:
                return True

        return False

    return gesture_detector(perception_interface, termination_event, timeout, threshold)
