'''
Runs a client-side (run on compute machine) watchdog for the robot's intended functionality. 
It validates the following:
1. All sensors are streaming correctly.
2. All sensor outputs are within the expected range.
    a. The ft sensor is not exceeding the threshold.
    b. The camera perception outputs are within the expected range (ToDo)
    c. The robot's current state is not near collision.
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

CAMERA_FREQUENCY_THRESHOLD = 0 # expected is 30 Hz
FT_FREQUENCY_THRESHOLD = 800 # expected is 1000 Hz
FT_THRESHOLD = [30.0, 30.0, 30.0, 2.0, 2.0, 2.0]
WITHIN_JOINT_LIMITS_FREQUENCY = 100 # expected is 100 Hz
COLLISION_FREE_FREQUENCY_THRESHOLD = 100 # expected is 350 Hz (empirical)

WATCHDOG_RUN_FREQUENCY = 1000

from feeding_deployment.safety.utils import PeekableQueue, AnomalyStatus

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

        self.within_joint_limits_sub = rospy.Subscriber('/within_joint_limits', Bool, self.withinJointLimitsCallback, queue_size = queue_size, buff_size = 65536*queue_size)
        self.within_joint_limits_timestamps = PeekableQueue()
        self.within_joint_limits_unexpected = False

        self.watchdog_status_pub = rospy.Publisher("/watchdog_status", Bool, queue_size=1)

        self.soft_anomaly = False

        self.second_counter = 0
        time.sleep(3.0) # Wait for all queues to fill up / collision monitor to start
        print("Initialized.")

    def cameraCallback(self, msg):

        self.camera_timestamps.put(time.time())
        self.camera_unexpected = False

    def withinJointLimitsCallback(self, msg):
        self.within_joint_limits_timestamps.put(time.time())
        if not msg.data:
            self.within_joint_limits_unexpected = True

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

    def check_status(self):
        self.second_counter += 1
        anomaly = AnomalyStatus.NO_ANOMALY
        start_time = time.time()
        frequencies = []
        for _queue, _threshold, _anomaly in [(self.camera_timestamps, CAMERA_FREQUENCY_THRESHOLD, AnomalyStatus.CAMERA_FREQUENCY), 
                                            (self.ft_timestamps, FT_FREQUENCY_THRESHOLD, AnomalyStatus.FT_FREQUENCY),
                                            (self.collision_free_timestamps, COLLISION_FREE_FREQUENCY_THRESHOLD, AnomalyStatus.COLLISION_FREE_FREQUENCY), 
                                            (self.user_emergency_stop_timestamps, USER_ESTOP_FREQUENCY_THRESHOLD, AnomalyStatus.USER_ESTOP_FREQUENCY), 
                                            (self.experimentor_emergency_stop_timestamps, EXPERIMENTOR_ESTOP_FREQUENCY_THRESHOLD, AnomalyStatus.EXPERIMENTOR_ESTOP_FREQUENCY)]:
            while _queue.peek() < start_time - 1.0:
                _queue.get()
            queue_size = _queue.qsize()
            if queue_size < _threshold:
                print(f"Frequency: {queue_size} for {_anomaly}")
                rospy.loginfo(f"Frequency: {queue_size} for {_anomaly}")
                anomaly = _anomaly
                break   
            frequencies.append(queue_size)

        if self.second_counter == WATCHDOG_RUN_FREQUENCY:
            print("Watchdog running at expected frequency.")
            # print(f"Frequencies:  Camera: {frequencies[0]}, FT: {frequencies[1]}, Collision Free: {frequencies[2]}, User EStop: {frequencies[3]}, Experimentor EStop: {frequencies[4]}")
            # print(f"Frequencies:  Camera: {frequencies[0]}, FT: {frequencies[1]}, User EStop: {frequencies[2]}, Experimentor EStop: {frequencies[3]}")
            self.second_counter = 0

        for _unexpected, _anomaly in [(self.camera_unexpected, AnomalyStatus.CAMERA_UNEXPECTED),
                                    (self.ft_unexpected, AnomalyStatus.FT_UNEXPECTED),
                                    (self.collision_free_unexpected, AnomalyStatus.COLLISION_FREE_UNEXPECTED),
                                    (self.user_emergency_stop_pressed, AnomalyStatus.USER_ESTOP_PRESSED),
                                    (self.experimentor_emergency_stop_pressed, AnomalyStatus.EXPERIMENTOR_ESTOP_PRESSED)]:
            if _unexpected:
                print(f"Unexpected: {_anomaly}")
                rospy.loginfo(f"Unexpected: {_anomaly}")
                anomaly = _anomaly
                break

        if anomaly != AnomalyStatus.NO_ANOMALY:
            print(f"AnomalyStatus detected: {anomaly}")
            rospy.loginfo(f"AnomalyStatus detected: {anomaly}")

            if anomaly == AnomalyStatus.OUTSIDE_JOINT_LIMITS_ERROR:
                print("Sending close command to arm.")
                self._arm_interface.close(no_grav_comp=True) # serious issue, so close the arm
            elif not self.soft_anomaly: # if a soft anomaly has already been detected, do not emergency stop the arm
                self.soft_anomaly = True
                self._arm_interface.emergency_stop() # less serious issue, so just emergency stop the arm

        self.watchdog_status_pub.publish(Bool(data=anomaly == AnomalyStatus.NO_ANOMALY))
        return anomaly
    
    def run(self):
        while True:
            start_time = time.time()
            status = self.check_status()
            if status != AnomalyStatus.NO_ANOMALY:
                print(f"AnomalyStatus detected: {status}")
                rospy.loginfo(f"AnomalyStatus detected: {status}")
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
    