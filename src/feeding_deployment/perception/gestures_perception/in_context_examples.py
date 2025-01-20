import time

def detect_mouth_open(perception_interface, termination_event, timeout):

    mouth_open_threshold = 0.45

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
        if mar > mouth_open_threshold:
            return True

    return False


def detect_head_nod(perception_interface, termination_event, timeout):
    head_nod_threshold = 15.0
    required_direction_changes = 3

    start_time = time.time()
    direction_changes = 0
    last_min_extreme = float("inf")
    last_max_extreme = -float("inf")

    while (time.time() - start_time < timeout and 
           (termination_event is None or not termination_event.is_set())):
        
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            # No more data
            break
        
        head_pose = head_perception_data["head_pose"]
        (_, _, _, _, head_pitch, _) = head_pose

        if head_pitch - last_min_extreme > head_nod_threshold:
            direction_changes += 1
            last_min_extreme = float("inf")

        if last_max_extreme - head_pitch > head_nod_threshold:
            direction_changes += 1
            last_max_extreme = -float("inf")

        last_min_extreme = min(head_pitch, last_min_extreme)
        last_max_extreme = max(head_pitch, last_max_extreme)

        if direction_changes >= required_direction_changes:
            return True

    return False
