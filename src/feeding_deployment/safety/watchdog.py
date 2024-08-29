'''
This script is a client-side (run on compute machine) watchdog for the robot's intended functionality. 
It validates the following:
1. All sensors are streaming correctly.
2. All sensor outputs are within the expected range.
    a. The ft sensor is not exceeding the threshold.
    b. The camera perception outputs are within the expected range (ToDo)
    c. The robot's current state is not near collision.
3. The robot is not in a state of emergency stop (from the user / experimentor emergency stop button).
If any of the above is not true, the watchdog will return the corresponding AnomalyStatus.
'''

import rospy
import numpy as np
import time
from enum import Enum
import queue

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool

CAMERA_FREQUENCY_THRESHOLD = 20
FT_FREQUENCY_THRESHOLD = 800
FT_THRESHOLD = [10.0, 10.0, 10.0, 0.5, 0.5, 0.5]
COLLISION_FREE_FREQUENCY_THRESHOLD = 80 
USER_ESTOP_FREQUENCY_THRESHOLD = 80
EXPERIMENTOR_ESTOP_FREQUENCY_THRESHOLD = 80

WATCHDOG_MONITOR_FREQUENCY = 1000

class AnomalyStatus(Enum):
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

class PeekableQueue(queue.Queue):
    def peek(self):
        with self.mutex:  # Lock the queue to ensure thread safety
            if self.qsize() > 0:
                return self.queue[0]  # Safely access the first element
            else:
                raise queue.Empty  # Handle the case when the queue is empty

class WatchDog:
    def __init__(self):
        rospy.init_node('WatchDog', anonymous=True)

        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.cameraCallback)
        self.camera_timestamps = PeekableQueue()
        self.camera_unexpected = False # ToDo - Implement this
        
        self.ft_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, self.ftCallback)
        self.ft_timestamps = PeekableQueue()
        self.ft_unexpected = False

        self.collision_free_sub = rospy.Subscriber('/collision_free', Bool, self.collisionFreeCallback)
        self.collision_free_timestamps = PeekableQueue()
        self.collision_free_unexpected = False

        self.user_emergency_stop_sub = rospy.Subscriber('/estop', Bool, self.userEmergencyStopCallback)
        self.user_emergency_stop_timestamps = PeekableQueue()
        self.user_emergency_stop_pressed = False

        self.experimentor_emergency_stop_sub = rospy.Subscriber('/experimentor_estop', Bool, self.experimentorEmergencyStopCallback)
        self.exeperimentor_emergency_stop_timestamps = PeekableQueue()
        self.experimentor_emergency_stop_pressed = False

        time.sleep(2.0) # Wait for all queues to fill up
        print("Initialized.")

    def cameraCallback(self, msg):

        self.camera_timestamps.put(time.time())
        self.camera_unexpected = False

    def ftCallback(self, msg):

        self.ft_timestamps.put(time.time())
        ft = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
        if not self.ft_unexpected:
            for i in range(6):
                if abs(ft[i]) > FT_THRESHOLD[i]:
                    self.ft_unexpected = True
                    break

    def collisionFreeCallback(self, msg):

        self.collision_free_timestamps.put(time.time())
        if not msg.data:
            self.collision_free_unexpected = True

    def userEmergencyStopCallback(self, msg):

        self.user_emergency_stop_timestamps.put(time.time())
        if msg.data:
            self.user_emergency_stop_pressed = True

    def experimentorEmergencyStopCallback(self, msg):

        self.exeperimentor_emergency_stop_timestamps.put(time.time())
        if msg.data:
            self.experimentor_emergency_stop_pressed = True


    def run(self):
        anomaly = AnomalyStatus.NO_ANOMALY

        start_time = time.time()
        for _queue, _threshold, _anomaly in [(self.camera_timestamps, CAMERA_FREQUENCY_THRESHOLD, AnomalyStatus.CAMERA_FREQUENCY), 
                                            (self.ft_timestamps, FT_FREQUENCY_THRESHOLD, AnomalyStatus.FT_FREQUENCY), 
                                            (self.collision_free_timestamps, COLLISION_FREE_FREQUENCY_THRESHOLD, AnomalyStatus.COLLISION_FREE_FREQUENCY), 
                                            (self.user_emergency_stop_timestamps, USER_ESTOP_FREQUENCY_THRESHOLD, AnomalyStatus.USER_ESTOP_FREQUENCY), 
                                            (self.exeperimentor_emergency_stop_timestamps, EXPERIMENTOR_ESTOP_FREQUENCY_THRESHOLD, AnomalyStatus.EXPERIMENTOR_ESTOP_FREQUENCY)]:
            while _queue.peek() < start_time - 1.0:
                _queue.get()
            if _queue.qsize() < _threshold:
                anomaly = _anomaly   

        for _unexpected, _anomaly in [(self.camera_unexpected, AnomalyStatus.CAMERA_UNEXPECTED),
                                    (self.ft_unexpected, AnomalyStatus.FT_UNEXPECTED),
                                    (self.collision_free_unexpected, AnomalyStatus.COLLISION_FREE_UNEXPECTED),
                                    (self.user_emergency_stop_pressed, AnomalyStatus.USER_ESTOP_PRESSED),
                                    (self.experimentor_emergency_stop_pressed, AnomalyStatus.EXPERIMENTOR_ESTOP_PRESSED)]:
            if _unexpected:
                anomaly = _anomaly

        return anomaly

if __name__ == '__main__':

    rospy.init_node('WatchDog', anonymous=True)
    watchdog = WatchDog()
    while True:
        status = watchdog.run()
        if status != AnomalyStatus.NO_ANOMALY:
            print(f"AnomalyStatus detected: {status}")
            break