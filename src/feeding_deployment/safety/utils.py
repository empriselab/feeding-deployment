from enum import Enum
import queue

class AnomalyStatus(Enum):
    UNEXPECTED_ERROR = -1
    NO_ANOMALY = 0
    CAMERA_FREQUENCY = 1
    CAMERA_UNEXPECTED = 2
    FT_FREQUENCY = 3
    FT_UNEXPECTED = 4
    COLLISION_FREE_FREQUENCY = 5
    COLLISION_FREE_UNEXPECTED = 6
    USER_ESTOP_FREQUENCY = 7
    USER_ESTOP_PRESSED = 8
    EXPERIMENTOR_ESTOP_FREQUENCY = 9
    EXPERIMENTOR_ESTOP_PRESSED = 10
    OUTSIDE_JOINT_LIMITS_FREQUENCY = 11
    OUTSIDE_JOINT_LIMITS_ERROR = 12

class PeekableQueue(queue.Queue):
    def peek(self):
        with self.mutex:  # Lock the queue to ensure thread safety
            if len(self.queue) > 0:
                return self.queue[0]  # Safely access the first element
            else:
                return float('inf') # Handle the case when the queue is empty: Do not pop from an empty queue
