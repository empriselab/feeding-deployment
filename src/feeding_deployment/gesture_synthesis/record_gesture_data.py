FLAIR_PATH = "/home/isacc/deployment_ws/src/FLAIR/bite_acquisition/scripts"
import sys
sys.path.append(FLAIR_PATH)

import os
import rospy
import imageio
import time
from pynput import keyboard
import pickle

from rs_ros import RealSenseROS

# Variable to stop recording
recording_stopped = False

def on_press(key):
    global recording_stopped
    if key == keyboard.Key.space:  # Use the 'Space' key to stop recording
        recording_stopped = True
        return False  # Stop the listener

def record_example(camera, command, example_type, index):
    global recording_stopped
    header_stream = []
    color_stream = []
    depth_stream = []
    info_stream = []

    # Start listener for key press detection
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print(f"Recording {example_type} example {index}, press 'Space' to stop.")
    while not recording_stopped:
        header, color_data, info_data, depth_data = camera.get_camera_data()
        
        header_stream.append(header)
        color_stream.append(color_data)
        depth_stream.append(depth_data)
        info_stream.append(info_data)

        # Control the frame rate
        time.sleep(0.1)

    # Reset the flag for the next recording
    recording_stopped = False

    # Stop the listener
    listener.stop()
    
    datapoint = {
        "header": header_stream,
        "color": color_stream,
        "depth": depth_stream,
        "info": info_stream
    }

    # Save the video using pickle
    print(f"Saving {len(color_stream)} frames")
    with open(f"gesture_data/{command}/{example_type}_examples/{index}.pkl", "wb") as f:
        pickle.dump(datapoint, f)

    print(f"Saved {example_type} example {index}")

if __name__ == "__main__":
    rospy.init_node('record_gesture_data', anonymous=True)
    
    camera = RealSenseROS()

    command = input("What gesture do you want to create a detector for? ").strip().replace(" ", "_")
    os.makedirs(f"gesture_data/{command}", exist_ok=True)
    os.makedirs(f"gesture_data/{command}/positive_examples", exist_ok=True)
    os.makedirs(f"gesture_data/{command}/negative_examples", exist_ok=True)

    header, color_data, info_data, depth_data = camera.get_camera_data()
    frame_height, frame_width, _ = color_data.shape

    # Record 5 positive examples
    print(" -- Record positive examples --")
    for i in range(5):
        input(f"Press enter to record positive example {i}")
        record_example(camera, command, "positive", i)

    # Record 5 negative examples
    print(" -- Record negative examples --")
    for i in range(5):
        input(f"Press enter to record negative example {i}")
        record_example(camera, command, "negative", i)
