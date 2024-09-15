#!/usr/bin/env python

import rospy
import rosbag
from tf2_msgs.msg import TFMessage

def extract_topics(input_bag, output_bag, topics_to_extract, frames_to_extract):
    # Open the input bag file
    with rosbag.Bag(input_bag, 'r') as in_bag:
        # Open the output bag file
        with rosbag.Bag(output_bag, 'w') as out_bag:
            for topic, msg, t in in_bag.read_messages():
                # Check if the message is in the topics to extract
                if topic in topics_to_extract:
                    out_bag.write(topic, msg, t)

                # Handle tf messages specifically
                elif topic == '/tf':
                    filtered_transforms = []
                    for transform in msg.transforms:
                        # Extract only the transforms between the desired frames
                        if (transform.header.frame_id == 'base_link' and 
                            transform.child_frame_id in frames_to_extract):
                            filtered_transforms.append(transform)
                    
                    # If any matching transforms are found, write them to the output bag
                    if filtered_transforms:
                        new_tf_msg = TFMessage(transforms=filtered_transforms)
                        out_bag.write('/tf', new_tf_msg, t)
                        
    rospy.loginfo(f"Extracted topics and specific tf frames saved to {output_bag}")

if __name__ == '__main__':
    # Initialize the rospy node (optional, you can remove it if not needed)
    rospy.init_node('rosbag_extractor', anonymous=True)
    
    # Input and output bag file paths
    input_bag = '/home/isacc/deployment_ws/2024-09-07-18-26-32.bag'  
    output_bag = '/home/isacc/deployment_ws/benjamin_head_perception.bag'

    # Topics to extract
    topics_to_extract = [
        '/head_perception/mouth_state',
        '/head_perception/tool/marker_array',
        '/head_perception/voxels/marker_array'
    ]
    
    # Frames to extract from the /tf topic
    frames_to_extract = [
        'tool_tip_target',
        'head_pose',
        'reference_head_pose'
    ]

    # Call the extraction function
    extract_topics(input_bag, output_bag, topics_to_extract, frames_to_extract)
