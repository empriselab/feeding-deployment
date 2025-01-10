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

    @classmethod
    def get_error_message(cls, status):
        """Get the error message corresponding to an AnomalyStatus."""
        messages = {
            cls.UNEXPECTED_ERROR: "An unexpected error occurred.",
            cls.NO_ANOMALY: "No anomaly detected.",
            cls.CAMERA_FREQUENCY: "Camera frequency is below the expected rate. Please check the camera.",
            cls.CAMERA_UNEXPECTED: "Unexpected camera anomaly detected.",
            cls.FT_FREQUENCY: "Force-torque sensor frequency is below the expected rate. Please check the force-torque sensor.",
            cls.FT_UNEXPECTED: "Unexpected force-torque sensor anomaly detected.",
            cls.COLLISION_FREE_FREQUENCY: "Collision monitoring frequency is below the expected rate. Please check the collision monitoring system.",
            cls.COLLISION_FREE_UNEXPECTED: "Collision detected. Remove obstacles from the robot's path.",
            cls.USER_ESTOP_FREQUENCY: "User emergency stop frequency is below the expected rate. Please check the user emergency stop button.",
            cls.USER_ESTOP_PRESSED: "User emergency stop activated.",
            cls.EXPERIMENTOR_ESTOP_FREQUENCY: "Experimenter emergency stop frequency is below the expected rate. Please check the experimenter emergency stop button.",
            cls.EXPERIMENTOR_ESTOP_PRESSED: "Experimenter emergency stop activated.",
            cls.OUTSIDE_JOINT_LIMITS_FREQUENCY: "Joint limits monitoring frequency is below the expected rate. Please check the joint limits monitoring system.",
            cls.OUTSIDE_JOINT_LIMITS_ERROR: "Joint limits exceeded. Ensure the robot is operating within its joint limits."
        }
        return messages.get(status, "Unknown anomaly status.")
    

class PeekableQueue(queue.Queue):
    def peek(self):
        with self.mutex:  # Lock the queue to ensure thread safety
            if len(self.queue) > 0:
                return self.queue[0]  # Safely access the first element
            else:
                return float('inf') # Handle the case when the queue is empty: Do not pop from an empty queue
