from pathlib import Path
from typing import Any

import json

import rospy
from std_msgs.msg import String
from feeding_deployment.interfaces.web_interface import WebInterface
import pickle

if __name__ == "__main__":
    rospy.init_node("test_actions")

    plate_log = pickle.load(open("/home/isacc/deployment_ws/src/feeding-deployment/src/feeding_deployment/interfaces/test_log/plate_log.pkl", "rb"))
    original_image = plate_log['original_image']
    plate_image = plate_log['plate_image']
    plate_bounds = plate_log['plate_bounds']

    web_interface = WebInterface()
    
    input("Press Enter to send look at plate finished message")
    web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
    
    input("Press Enter to send plate image")
    web_interface.send_web_interface_image(plate_image)