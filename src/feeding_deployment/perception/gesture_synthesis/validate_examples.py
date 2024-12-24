import time
from feeding_deployment.gesture_synthesis.robot import Robot, run_detector, search_threshold


def in_context_example1(robot, timeout=20.0, threshold=0.5):
    """
    Verifies the in-context example 1 code provided in the prompt
    """
    start_time = time.time()
    yaw_data = []
    direction_changes = 0  # Counts the number of left-right or right-left changes

    id = 0
    while time.time() - start_time < timeout:
        head_pose = robot.get_head_pose()
        # print("Head Rotation: ", head_roll, head_pitch, head_yaw)
        # id+= 1
        # time.sleep(0.5)
        if head_pose is None:
            break # Handle case where head pose data is unavailable

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

def test_example(robot, timeout, threshold):

    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    start_time = time.time()
    mouth_movement_count = 0
    last_mar_above_threshold = None

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

        # Check if mar crosses threshold
        mar_above_threshold = mar > threshold

        if last_mar_above_threshold is not None:
            if mar_above_threshold != last_mar_above_threshold:
                # Mouth has opened or closed
                mouth_movement_count += 1

        last_mar_above_threshold = mar_above_threshold

        # If enough mouth movements detected, return True
        if mouth_movement_count >= 5:
            return True

    # If timeout expires without detecting talking, return False
    return False

if __name__ == '__main__':
    # threshold1, accuracy1 = search_threshold('gesture_data/shake_my_head_from_left_to_right', in_context_example1)
    # print("In-Context Example 1")
    # print("Best Threshold: ", threshold1)
    # print("Best Accuracy: ", accuracy1)

    # threshold2, accuracy2 = search_threshold('gesture_data/open_mouth', in_context_example2)
    # print("In-Context Example 2")
    # print("Best Threshold: ", threshold2)
    # print("Best Accuracy: ", accuracy2)

    threshold, accuracy = search_threshold('gesture_data/talking', test_example)
    print("Test Example")
    print("Best Threshold: ", threshold)
    print("Best Accuracy: ", accuracy)