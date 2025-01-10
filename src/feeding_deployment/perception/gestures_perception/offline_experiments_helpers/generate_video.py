"""
Parses pickled files containing camera data and saves a RGB video of the data
"""

import os
import imageio
import pickle

def save_video(data_path):

    # Load the pickled file
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    video_frames = data['color']

    # Save the video using imageio
    print(f"Saving {len(video_frames)} frames")
    video_path = data_path.replace(".pkl", ".mp4")
    with imageio.get_writer(video_path, fps=10, codec='libx264') as writer:
        for frame in video_frames:
            # Convert to RGB format for imageio if needed
            frame_rgb = frame[:, :, ::-1]  # Convert from BGR to RGB
            writer.append_data(frame_rgb)

def parse_dataset(source_path):
    
    for i in range(5):
        data_path = source_path + f'/positive_examples/{i}.pkl'
        save_video(data_path)
    
    for i in range(5):
        data_path = source_path + f'/negative_examples/{i}.pkl'
        save_video(data_path)

if __name__ == "__main__":
    source_paths = ['shake_my_head_from_left_to_right', 'open_mouth', 'blinking', 'eyebrows_raised', 'head_nod', 'head_still_atleast_three_secs', 'look_at_robot_atleast_three_secs', 'talking']
    for source_path in source_paths:
        parse_dataset( 'gesture_data/' + source_path)