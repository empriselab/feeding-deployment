import sys
import os
import rospy
import imageio
import time
import pickle
import argparse
import cv2

if __name__ == "__main__":
    rospy.init_node('visualize_gesture_data', anonymous=True)

    parser = argparse.ArgumentParser(description='Visualize gesture data')
    parser.add_argument('--path', type=str, help='Path to the gesture datapoint')

    args = parser.parse_args()

    with open(args.path, "rb") as f:
        datapoint = pickle.load(f)

    print(f"Loaded {len(datapoint['color'])} frames")

    for i in range(len(datapoint['color'])):
        color_frame = datapoint['color'][i]
        depth_frame = datapoint['depth'][i]

        cv2.imshow("Color", color_frame)
        cv2.imshow("Depth", depth_frame)
        cv2.waitKey(100)  # 10 fps

    cv2.destroyAllWindows()
        