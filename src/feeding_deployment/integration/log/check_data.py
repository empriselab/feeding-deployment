import pickle

with open('../drink_pickup_pos.pkl', 'rb') as f:
    head_perception_pose_drink = pickle.load(f)

print(head_perception_pose_drink)