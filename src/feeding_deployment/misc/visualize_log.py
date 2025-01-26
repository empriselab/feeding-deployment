import pickle
import cv2

# load pickle
log_path = "/home/isacc/deployment_ws/src/feeding-deployment/src/feeding_deployment/integration/log/benjamin/default/food_detection_log/food_detection_data_0.pkl"

with open(log_path, "rb") as f:
    food_detection_data = pickle.load(f)

camera_color_data = food_detection_data["camera_color_data"]

# visualize camera_color_data
cv2.imshow('vis', camera_color_data)
cv2.waitKey(0)
cv2.destroyAllWindows()