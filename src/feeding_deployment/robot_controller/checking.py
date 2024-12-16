"""
__________.__                                      _______               
\______   \  |   ____  ______ __________   _____   \      \ _____ ___  __
 |    |  _/  |  /  _ \/  ___//  ___/  _ \ /     \  /   |   \\__  \\  \/ /
 |    |   \  |_(  <_> )___ \ \___ (  <_> )  Y Y  \/    |    \/ __ \\   / 
 |______  /____/\____/____  >____  >____/|__|_|  /\____|__  (____  /\_/  
        \/                \/     \/            \/         \/     \/      

Copyright (c) 2024 Interactions Lab
License: MIT
Authors: Anthony Song and Nathan Dennler, Cornell University & University of Southern California
Project Page: https://github.com/interaction-lab/BlossomNav.git

This script contains an GUI that allows recording video footage from the Pi Zero 2 and saving to local computer

"""

import cv2
import os

class VideoStreamApp:
    def __init__(self, rtsp_link):
        self.rtsp_link = rtsp_link
        self.saving = False
        self.cap = cv2.VideoCapture(self.rtsp_link)
        self.frame_counter = 0
        self.image_dir = "saved_images"
        os.makedirs(self.image_dir, exist_ok=True)
        self.current_frame = None

    def capture_frames(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                current_frame = frame
        return current_frame

if __name__ == "__main__":
    app = VideoStreamApp('http://192.168.1.5:8083/')
    image = app.capture_frames()

    # visualize the image
    cv2.imshow('some_image', image)
    cv2.waitKey(0)