
import time
import argparse
import pyaudio

import rospy
from std_msgs.msg import Bool

from feeding_deployment.safety.button import Button 
BUTTON_CHECK_FREQUENCY = 1000

class TransferButtonListener:
    def __init__(self, button_id: int):

        self.button = Button(button_id)
        self.button_pub = rospy.Publisher("/transfer_complete", Bool, queue_size=1)

    def run(self):
        while not rospy.is_shutdown():
            start_time = time.time()
            button_pressed = self.button.check()

            if button_pressed:
                print("Transfer button pressed")
                self.button_pub.publish(Bool(data=button_pressed))
                self.button.reset()
                
            time.sleep(max(0, 1.0/BUTTON_CHECK_FREQUENCY - (time.time() - start_time)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--button_id", type=int, default=0)

    args = parser.parse_args()

    if args.button_id == 0:
        audio = pyaudio.PyAudio()
        device_indices = []
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:  # Only consider input devices
                device_indices.append(i)
                print(f"Device {i}: {device_info['name']}")
        raise ValueError("Please provide the input device index")
    
    rospy.init_node("transfer_button_listener")
    estop_publisher = TransferButtonListener(button_id=args.button_id)
    estop_publisher.run()