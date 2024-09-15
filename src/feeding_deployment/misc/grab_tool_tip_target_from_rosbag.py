import rosbag
import tf
import numpy as np
from tf import transformations

def extract_transforms_from_bag(bag_path, base_frame, target_frame):
    # Initialize the list to store the transforms
    transforms_list = []

    # Open the rosbag file
    with rosbag.Bag(bag_path, 'r') as bag:
        # Iterate through the bag, filtering messages from the /tf topic
        for topic, msg, t in bag.read_messages(topics=['/tf']):
            # Go through each transform in the tf message
            for transform in msg.transforms:
                # Check if the transform is between base_frame and target_frame
                if transform.header.frame_id == base_frame and transform.child_frame_id == target_frame:
                    # Extract translation
                    translation = (transform.transform.translation.x,
                                   transform.transform.translation.y,
                                   transform.transform.translation.z)

                    # Extract rotation as a quaternion
                    rotation = (transform.transform.rotation.x,
                                transform.transform.rotation.y,
                                transform.transform.rotation.z,
                                transform.transform.rotation.w)

                    # Convert rotation to a rotation matrix
                    rotation_matrix = transformations.quaternion_matrix(rotation)

                    # Combine translation and rotation into a 4x4 transformation matrix
                    transform_matrix = np.eye(4)
                    transform_matrix[:3, 3] = translation  # Add translation
                    transform_matrix[:3, :3] = rotation_matrix[:3, :3]  # Add rotation

                    # Append the transformation matrix as a row to the list
                    transforms_list.append(transform_matrix)

    # Convert the list of transforms into a 3D NumPy array
    transforms_array = np.array(transforms_list)

    return transforms_array

# Example usage
if __name__ == "__main__":
    # Specify the path to the bag file
    bag_file = "/home/isacc/deployment_ws/2024-09-07-18-26-32.bag"

    # Specify the frames of interest
    base_frame = 'base_link'
    target_frame = 'tool_tip_target'

    # Extract transforms and print the result
    transforms = extract_transforms_from_bag(bag_file, base_frame, target_frame)
    print(f"Extracted {transforms.shape[0]} transforms, with each transform having shape ({transforms.shape[1:]})")
    # print(transforms)

    np.save('benjamin_target_transforms.npy', transforms)