import pickle
import numpy as np

class Robot:
    """
    Simulate the robot with get_head_pose and get_face_keypoints methods
    """
    def __init__(self, data_path):

        # load data from pickle file data_path
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.face_keypoints = data['face_keypoints']
        self.head_pose = data['head_poses']
        self.current_frame = 0
        self.max_frame = len(self.face_keypoints)
    
    def get_head_pose(self):
        if self.current_frame < self.max_frame:
            self.current_frame += 1
            return self.head_pose[self.current_frame-1]
        return None
    
    def get_face_keypoints(self):
        if self.current_frame < self.max_frame:
            self.current_frame += 1
            return self.face_keypoints[self.current_frame-1]
        return None
    
def run_detector(data_path, gesture_detector, **kwargs):
    """
    Run the gesture detector on the given data_path
    """
    positive_correct = 0
    for i in range(5):
        robot = Robot(data_path + f'/positive_examples/{i}_parsed.pkl')
        if gesture_detector(robot, **kwargs):
            positive_correct += 1
    
    negative_correct = 0
    for i in range(5):
        robot = Robot(data_path + f'/negative_examples/{i}_parsed.pkl')
        if not gesture_detector(robot, **kwargs):
            negative_correct += 1
    
    return positive_correct/5.0, negative_correct/5.0

def search_threshold(data_path, gesture_detector, timeout=20.0, threshold_range=(0.0, 1.0), step=0.1):
    """
    Search for the best threshold for the given gesture detector
    """
    best_threshold = None
    best_accuracy = 0.0

    for threshold in np.arange(threshold_range[0], threshold_range[1], step):
        positive_accuracy, negative_accuracy = run_detector(data_path, gesture_detector, timeout=timeout, threshold=threshold)
        # print("Threshold: ", threshold, "Positive Accuracy: ", positive_accuracy, "Negative Accuracy: ", negative_accuracy)
        accuracy = (positive_accuracy + negative_accuracy) / 2.0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold, best_accuracy