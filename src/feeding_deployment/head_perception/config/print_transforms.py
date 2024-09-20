# print contents of npy file

import numpy as np

transform_files = [
    "fork/tool_tip_transform.npy",
    "fork/old_tool_tip_transform.npy",
    "drink/tool_tip_transform.npy",
    "wipe/tool_tip_transform.npy"
]

for transform_file in transform_files:
    data = np.load(transform_file)
    print("Contents of", transform_file)
    print(data)