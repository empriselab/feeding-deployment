FLAIR_PATH = "/home/isacc/deployment_ws/src/FLAIR/bite_acquisition/scripts"
import sys
sys.path.append(FLAIR_PATH)

import os
import rospy
import imageio
import time
from pynput import keyboard

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
    example_video = []

    # Start listener for key press detection
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print(f"Recording {example_type} example {index}, press 'Space' to stop.")
    while not recording_stopped:
        header, color_data, info_data, depth_data = camera.get_camera_data()
        example_video.append(color_data)

        # Control the frame rate
        time.sleep(0.1)

    # Reset the flag for the next recording
    recording_stopped = False

    # Stop the listener
    listener.stop()

    # Save the video using imageio
    print(f"Saving {len(example_video)} frames")
    with imageio.get_writer(f"gesture_data/{command}/{example_type}_examples/{index}.mp4", fps=10, codec='libx264') as writer:
        for frame in example_video:
            # Convert to RGB format for imageio if needed
            frame_rgb = frame[:, :, ::-1]  # Convert from BGR to RGB
            writer.append_data(frame_rgb)

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
