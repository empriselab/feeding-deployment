import pickle
import cv2

# load pickle
log_path = "/home/isacc/deployment_ws/src/feeding-deployment/src/feeding_deployment/integration/log/aimee/tv/food_detection_log/food_detection_data_2.pkl"

with open(log_path, "rb") as f:
    food_detection_data = pickle.load(f)

camera_color_data = food_detection_data["camera_color_data"]

# visualize camera_color_data
cv2.imshow('vis', camera_color_data)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save image
cv2.imwrite("camera_color_data.jpg", camera_color_data)