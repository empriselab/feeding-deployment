'''
Runs a client-side (run on compute machine) watchdog for the robot's intended functionality. 
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
import signal
import sys

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool

import threading
import time
import numpy as np

import rospy
from std_msgs.msg import Bool
from netft_rdt_driver.srv import String_cmd

from feeding_deployment.robot_controller.arm_interface import ArmInterface, ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY

CAMERA_FREQUENCY_THRESHOLD = 20 # expected is 30 Hz
FT_FREQUENCY_THRESHOLD = 800 # expected is 1000 Hz
FT_THRESHOLD = [10.0, 10.0, 10.0, 0.5, 0.5, 0.5]
COLLISION_FREE_FREQUENCY_THRESHOLD = 100 # expected is 350 Hz (empirical)
USER_ESTOP_FREQUENCY_THRESHOLD = 800 # expected is 1000 Hz
experimentor_ESTOP_FREQUENCY_THRESHOLD = 800 # expected is 1000 Hz

WATCHDOG_RUN_FREQUENCY = 1000

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
    experimentor_ESTOP_FREQUENCY = 9
    experimentor_ESTOP_PRESSED = 10

class PeekableQueue(queue.Queue):
    def peek(self):
        with self.mutex:  # Lock the queue to ensure thread safety
            if len(self.queue) > 0:
                return self.queue[0]  # Safely access the first element
            else:
                return float('inf') # Handle the case when the queue is empty: Do not pop from an empty queue

class WatchDog:
    def __init__(self):

        # Register ArmInterface (no lambda needed on the client-side)
        ArmManager.register("ArmInterface")

        # Client setup
        self.manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        self.manager.connect()

        # This will now use the single, shared instance of ArmInterface
        self._arm_interface = self.manager.ArmInterface()

        # bias FT sensor
        bias = rospy.ServiceProxy('/forque/bias_cmd', String_cmd)
        bias('bias')
        time.sleep(2.0) # wait for bias to complete

        queue_size = 1000
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.cameraCallback, queue_size = queue_size, buff_size = 65536*queue_size)
        self.camera_timestamps = PeekableQueue()
        self.camera_unexpected = False # ToDo - Implement this
        
        self.ft_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, self.ftCallback, queue_size = queue_size, buff_size = 65536*queue_size)
        self.ft_timestamps = PeekableQueue()
        self.ft_unexpected = False

        self.collision_free_sub = rospy.Subscriber('/collision_free', Bool, self.collisionFreeCallback, queue_size = queue_size, buff_size = 65536*queue_size)
        self.collision_free_timestamps = PeekableQueue()
        self.collision_free_unexpected = False

        self.user_emergency_stop_sub = rospy.Subscriber('/user_estop', Bool, self.userEmergencyStopCallback, queue_size = queue_size, buff_size = 65536*queue_size)
        self.user_emergency_stop_timestamps = PeekableQueue()
        self.user_emergency_stop_pressed = False

        self.experimentor_emergency_stop_sub = rospy.Subscriber('/experimentor_estop', Bool, self.experimentorEmergencyStopCallback, queue_size = queue_size, buff_size = 65536*queue_size)
        self.experimentor_emergency_stop_timestamps = PeekableQueue()
        self.experimentor_emergency_stop_pressed = False

        self.watchdog_status_pub = rospy.Publisher("/watchdog_status", Bool, queue_size=1)

        self.second_counter = 0
        time.sleep(3.0) # Wait for all queues to fill up / collision monitor to start
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
                    print("FT threshold exceeded with magnitude: ", ft)
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

        self.experimentor_emergency_stop_timestamps.put(time.time())
        if msg.data:
            self.experimentor_emergency_stop_pressed = True

    def check_status(self):
        self.second_counter += 1
        anomaly = AnomalyStatus.NO_ANOMALY
        # start_time = time.time()
        # frequencies = []
        # for _queue, _threshold, _anomaly in [(self.camera_timestamps, CAMERA_FREQUENCY_THRESHOLD, AnomalyStatus.CAMERA_FREQUENCY), 
        #                                     (self.ft_timestamps, FT_FREQUENCY_THRESHOLD, AnomalyStatus.FT_FREQUENCY),
        #                                     (self.collision_free_timestamps, COLLISION_FREE_FREQUENCY_THRESHOLD, AnomalyStatus.COLLISION_FREE_FREQUENCY), 
        #                                     (self.user_emergency_stop_timestamps, USER_ESTOP_FREQUENCY_THRESHOLD, AnomalyStatus.USER_ESTOP_FREQUENCY), 
        #                                     (self.experimentor_emergency_stop_timestamps, experimentor_ESTOP_FREQUENCY_THRESHOLD, AnomalyStatus.experimentor_ESTOP_FREQUENCY)]:
        #     while _queue.peek() < start_time - 1.0:
        #         _queue.get()
        #     queue_size = _queue.qsize()
        #     if queue_size < _threshold:
        #         print(f"Frequency: {queue_size} for {_anomaly}")
        #         rospy.loginfo(f"Frequency: {queue_size} for {_anomaly}")
        #         anomaly = _anomaly
        #         break   
        #     frequencies.append(queue_size)

        # if self.second_counter == WATCHDOG_RUN_FREQUENCY:
        #     print("Watchdog running at expected frequency.")
        #     print(f"Frequencies:  Camera: {frequencies[0]}, FT: {frequencies[1]}, Collision Free: {frequencies[2]}, User EStop: {frequencies[3]}, Experimentor EStop: {frequencies[4]}")
        #     self.second_counter = 0

        # for _unexpected, _anomaly in [(self.camera_unexpected, AnomalyStatus.CAMERA_UNEXPECTED),
        #                             # (self.ft_unexpected, AnomalyStatus.FT_UNEXPECTED),
        #                             (self.collision_free_unexpected, AnomalyStatus.COLLISION_FREE_UNEXPECTED),
        #                             (self.user_emergency_stop_pressed, AnomalyStatus.USER_ESTOP_PRESSED),
        #                             (self.experimentor_emergency_stop_pressed, AnomalyStatus.experimentor_ESTOP_PRESSED)]:
        #     if _unexpected:
        #         print(f"Unexpected: {_anomaly}")
        #         rospy.loginfo(f"Unexpected: {_anomaly}")
        #         anomaly = _anomaly
        #         break

        # if anomaly != AnomalyStatus.NO_ANOMALY:
        #     print(f"AnomalyStatus detected: {anomaly}")
        #     rospy.loginfo(f"AnomalyStatus detected: {anomaly}")

        #     self._arm_interface.stop()

        self.watchdog_status_pub.publish(Bool(data=anomaly == AnomalyStatus.NO_ANOMALY))
        return anomaly
    
    def run(self):
        while True:
            start_time = time.time()
            status = self.check_status()
            if status != AnomalyStatus.NO_ANOMALY:
                print(f"AnomalyStatus detected: {status}")
                break
            end_time = time.time()
            # print(f"Time taken: {end_time - start_time}")
            time.sleep(max(0, 1.0/WATCHDOG_RUN_FREQUENCY - (end_time - start_time)))

    def signal_handler(self, signal, frame):

        print("\nprogram exiting gracefully")
        sys.exit(0)

if __name__ == '__main__':

    rospy.init_node('WatchDog', anonymous=True)
    
    watchdog = WatchDog()
    signal.signal(signal.SIGINT, watchdog.signal_handler) # ctrl+c
    
    watchdog.run()
    