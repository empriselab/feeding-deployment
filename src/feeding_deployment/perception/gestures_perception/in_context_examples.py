import time

def in_context_example1(perception_interface, termination_event, timeout, threshold):
    """
    Verifies the in-context example 1 code (head shake left to right) provided in the prompt
    """

    start_time = time.time()
    yaw_data = []
    direction_changes = 0  # Counts the number of left-right or right-left changes
    

    while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            break
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

def in_context_example2(perception_interface, termination_event, timeout, threshold):
    """ Verifies the in-context example 2 code (mouth open) provided in the prompt """

    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    start_time = time.time()
    while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            break
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