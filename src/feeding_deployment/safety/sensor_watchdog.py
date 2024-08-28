import os, sys
import cv2
import numpy as np
import time
import math

from scipy.spatial.transform import Rotation

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros

from geometry_msgs.msg import Point, TransformStamped, WrenchStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float64MultiArray, Bool
from types import SimpleNamespace

import pyttsx3

from threading import Lock
import threading
from copy import deepcopy

class SensorWatchDog:
    def __init__(self):
        rospy.init_node('FoodBoundingBoxPerception')

        self.top_camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.topCameraCallback)
        self.top_camera_count_lock = threading.Lock()
        self.top_camera_count = 0

        self.bottom_camera_info_sub = rospy.Subscriber("/bottom_camera/color/camera_info", CameraInfo, self.bottomCameraCallback)
        self.bottom_camera_count_lock = threading.Lock()
        self.bottom_camera_count = 0
        
        self.ft_sensor_sub = rospy.Subscriber('/forque/forqueSensor', WrenchStamped, self.ftCallback)
        self.ft_sensor_count_lock = threading.Lock()
        self.ft_sensor_count = 0

        self.collisions_sub = rospy.Subscriber('/collision_detected', Bool, self.collisionCallback)
        self.collision_count_lock = threading.Lock()
        self.collision_count = 0

        self.engine = pyttsx3.init()
        print("Initialized.")

    def topCameraCallback(self, msg):

        with self.top_camera_count_lock:
            self.top_camera_count += 1

    def bottomCameraCallback(self, msg):

        with self.bottom_camera_count_lock:
            self.bottom_camera_count += 1

    def ftCallback(self, msg):

        with self.ft_sensor_count_lock:
            self.ft_sensor_count += 1

    def collisionCallback(self, msg):

        # Wait if collision detected.
        if msg.data:
            self.engine.say("Collision detected!")
            self.engine.runAndWait()
            print("Press [ENTER] to continue:")
            inp = input()

        with self.collision_count_lock:
            self.collision_count += 1

    def watchdog(self):

        top_camera_count = None
        bottom_camera_count = None
        ft_sensor_count = None
        time.sleep(0.2)
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            with self.top_camera_count_lock:
                top_camera_count = self.top_camera_count
                self.top_camera_count = 0
            with self.bottom_camera_count_lock:
                bottom_camera_count = self.bottom_camera_count
                self.bottom_camera_count = 0
            with self.top_camera_count_lock:
                ft_sensor_count = self.ft_sensor_count
                self.ft_sensor_count = 0
            with self.collision_count_lock:
                collision_count = self.collision_count
                self.collision_count = 0
            
            print("Frequencies: ", top_camera_count, bottom_camera_count, ft_sensor_count)
            if top_camera_count < 20:
                self.engine.say("Top camera not streaming correctly.")
                self.engine.runAndWait()
                print("Press [ENTER] to continue:")
                inp = input()

            if bottom_camera_count < 20:
                self.engine.say("Bottom camera not streaming correctly.")
                self.engine.runAndWait()
                print("Press [ENTER] to continue:")
                inp = input()

            if ft_sensor_count < 800:
                self.engine.say("FT Sensor not streaming correctly.")
                self.engine.runAndWait()
                print("Press [ENTER] to continue:")
                inp = input()

            if collision_count < 20:
                self.engine.say("Collision detection not streaming correctly.")
                self.engine.runAndWait()
                print("Press [ENTER] to continue:")
                inp = input()

            rate.sleep()

if __name__ == '__main__':

    sensor_watchdog = SensorWatchDog()
    sensor_watchdog.watchdog()