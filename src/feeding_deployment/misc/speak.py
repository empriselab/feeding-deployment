# Reads from a ROS topic called /speak and says the text using pyttsx3
import rospy
from std_msgs.msg import String
import pyttsx3
import time

class Speak:
    
    def __init__(self):
        rospy.init_node('Speak', anonymous=True)
        rospy.Subscriber("/speak", String, self.callback, queue_size=10)
        self.engine = pyttsx3.init()
        print("Speak node initialized")

    # Speak the text
    def callback(self, msg):        
        text = msg.data
        print("Speaking: ", text)
        self.engine.say(text)
        self.engine.runAndWait()

if __name__ == '__main__':
    speak = Speak()
    rospy.spin()