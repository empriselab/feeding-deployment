import cv2
import ast
import numpy as np
from scipy.spatial.transform import Rotation
import math
import os
from pathlib import Path

from feeding_deployment.actions.flair.inference_class import BiteAcquisitionInference

HOME_ORIENTATION = Rotation.from_quat([1/math.sqrt(2), 1/math.sqrt(2), 0, 0]).as_matrix()
DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]

class FLAIR:
    def __init__(self):

        self.inference_server = BiteAcquisitionInference(mode='ours')
        print("inf server init")

        self.history_path = Path(__file__).parent.parent.parent / "integration" / "log" / "flair_history.txt"
        if not os.path.exists(self.history_path):
            self.bite_history = []
            self.user_preference = None
        else:
            # read in json format
            try:
                with open(self.history_path, 'r') as f:
                    logged_history = ast.literal_eval(f.read())
                self.bite_history = logged_history["bite_history"]
                self.user_preference = logged_history["user_preference"]
            except Exception as e:
                print("Error reading history file", e)
                print("Creating new history file...")
                self.bite_history = []
                self.user_preference = None

            print("Logged Bite History", self.bite_history)
            print("Logged User Preference", self.user_preference)

        # Continue food
        self.continue_food_label = None
        self.continue_dip_label = None

        self.visualize = True

        # itermediate variables
        self.items_detection = None
        self.next_action_prediction = None

    def log_history(self):
        with open(self.history_path, 'w') as f:
            f.write(str({"bite_history": self.bite_history, "user_preference": self.user_preference}))

    def update_bite_history(self, acquired_food_label):
        self.bite_history.append(acquired_food_label)
        self.log_history()
        
    def identify_plate(self, camera_color_data):

        items = self.inference_server.recognize_items(camera_color_data)
        print("Food Items recognized:", items)

        # k = input("Did the robot recognize the food items correctly?")
        k = 'n'
        if k == 'n':
            # Rajat ToDo: Implement manual input of food items        
            items = ['yellow banana', 'baby carrot']
            
        self.inference_server.FOOD_CLASSES = items
        return items

    def set_food_items(self, items):
        self.inference_server.FOOD_CLASSES = items

    def set_preference(self, user_preference):
        self.user_preference = user_preference
        self.log_history()

    def clear_preference(self):
        self.user_preference = None

    def is_preference_set(self):
        return self.user_preference is not None

    def detect_items(self, camera_color_data, camera_depth_data, camera_info_data, log_path):

        annotated_image, detections, item_masks, item_portions, item_labels, plate_bounds = self.inference_server.detect_items(camera_color_data, log_path)

        item_bounding_boxes = []
        item_bounding_boxes_plate = []
        # add bounding boxes corresponding to the detected item masks
        for mask in item_masks:
            non_zero_points = cv2.findNonZero(mask)
            x, y, w, h = cv2.boundingRect(non_zero_points)
            item_bounding_boxes.append([x, y, w, h]) # original image coordinates
            item_bounding_boxes_plate.append([x-plate_bounds[0], y-plate_bounds[1], w, h]) # plate image coordinates

        # cv2.imshow('vis', annotated_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # input("Visualzing the detected items. Press Enter to continue.")

        # k = input('Are detected items correct?')
        # while k not in ['y', 'n']:
        #     k = input('Are detected items correct?')
        #     if k == 'e':
        #         return None
        # while k == 'n':
        #     return None
            # print("Please manually give the correct labels")
            # print("Detected items:", item_labels)
            # label_id = int(input("What label to correct?"))
            # item_labels[label_id] = input("Correct label:")

            # annotated_image = self.inference_server.get_annotated_image(camera_color_data, detections, item_labels)

            # cv2.imshow('vis', annotated_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # input("Visualzing the detected items. Press Enter to continue.")

            # k = input('Are detected items correct now?')
            # while k not in ['y', 'n']:
            #     k = input('Are detected items correct now?')
        
        clean_item_labels, _ = self.inference_server.clean_labels(item_labels)

        # remove detections of blue plate
        if 'blue plate' in clean_item_labels:
            idx = clean_item_labels.index('blue plate')
            clean_item_labels.pop(idx)
            item_labels.pop(idx)
            item_masks.pop(idx)
            item_bounding_boxes.pop(idx)
            item_bounding_boxes_plate.pop(idx)
            item_portions.pop(idx)

        print("----- Clean Item Labels:", clean_item_labels)

        # cv2.imwrite(log_path + "_annotated.png", annotated_image)
        # cv2.imwrite(log_path + "_color.png", camera_color_data)
        # cv2.imwrite(log_path + "_depth.png", camera_depth_data)

        categories = self.inference_server.categorize_items(item_labels) 

        print("--------------------")
        print("Labels:", item_labels)
        print("Categories:", categories)
        print("Portions:", item_portions)
        print("--------------------")

        category_list = []
        labels_list = []
        per_food_masks = [] # For multiple items per food, ordered by prediction confidence
        per_food_portions = []
        per_food_bounding_boxes = []
        per_food_bounding_boxes_plate = []

        # for i in range(len(categories)):
        #     if categories[i] not in category_list:
        #         category_list.append(categories[i])
        #         labels_list.append(clean_item_labels[i])
        #         per_food_masks.append([item_masks[i]])
        #         per_food_portions.append(item_portions[i])
        #     else:
        #         index = category_list.index(categories[i])
        #         per_food_masks[index].append(item_masks[i])
        #         per_food_portions[index] += item_portions[i] 

        for i in range(len(categories)):
            if labels_list.count(clean_item_labels[i]) == 0:
                category_list.append(categories[i])
                labels_list.append(clean_item_labels[i])
                per_food_masks.append([item_masks[i]])
                per_food_bounding_boxes.append([item_bounding_boxes[i]])
                per_food_bounding_boxes_plate.append([item_bounding_boxes_plate[i]])
                per_food_portions.append(item_portions[i])
            else:
                index = labels_list.index(clean_item_labels[i])
                per_food_masks[index].append(item_masks[i])
                per_food_bounding_boxes[index].append(item_bounding_boxes[i])
                per_food_bounding_boxes_plate[index].append(item_bounding_boxes_plate[i])
                per_food_portions[index] += item_portions[i]
        
        print("Bite History", self.bite_history)
        print("Category List:", category_list)
        print("Labels List:", labels_list)
        print("Per Food Masks Len:", [len(x) for x in per_food_masks])
        print("Per Food Portions:", per_food_portions)

        plate_image = camera_color_data.copy()[plate_bounds[1]:plate_bounds[1]+plate_bounds[3], plate_bounds[0]:plate_bounds[0]+plate_bounds[2]]

        food_type_to_bounding_boxes = {label: [] for label in labels_list}
        food_type_to_bounding_boxes_plate = {label: [] for label in labels_list}
        food_type_to_masks = {label: [] for label in labels_list}
        food_type_to_skill = {label: None for label in labels_list}

        for i in range(len(labels_list)):
            food_type_to_bounding_boxes[labels_list[i]] = per_food_bounding_boxes[i]
            food_type_to_bounding_boxes_plate[labels_list[i]] = per_food_bounding_boxes_plate[i]
            food_type_to_masks[labels_list[i]] = per_food_masks[i]
            if categories[i] == 'noodles':
                food_type_to_skill[labels_list[i]] = 'Twirl'
            elif categories[i] == 'semisolid':
                food_type_to_skill[labels_list[i]] = 'Scoop'
            else:
                food_type_to_skill[labels_list[i]] = 'Skewer'

        # Rajat ToDo: Remove repeated code
        items_detection = {
            'annotated_image': annotated_image,
            'plate_image': plate_image,
            'plate_bounds': plate_bounds,
            'per_food_masks': per_food_masks, 
            'category_list': category_list, 
            'labels_list': labels_list, 
            'per_food_portions': per_food_portions,
            'food_type_to_bounding_boxes': food_type_to_bounding_boxes,
            'food_type_to_bounding_boxes_plate': food_type_to_bounding_boxes_plate,
            'food_type_to_masks': food_type_to_masks,
            'food_type_to_skill': food_type_to_skill
        }

        self.items_detection = items_detection
        return items_detection
    
    def get_items_detection(self):
        return self.items_detection
    
    def set_items_detection(self, items_detection):
        self.items_detection = items_detection

    def predict_next_action(self, camera_color_data, items_detection, log_path):

        if items_detection is None:
            items_detection = self.items_detection

        annotated_image = items_detection['annotated_image']
        per_food_masks = items_detection['per_food_masks']
        category_list = items_detection['category_list']
        per_food_masks = items_detection['per_food_masks']
        labels_list = items_detection['labels_list']
        per_food_portions = items_detection['per_food_portions']
        
        food, dip, bite_mask_idx = self.inference_server.get_autonomous_action(annotated_image, camera_color_data, per_food_masks, category_list, labels_list, per_food_portions, self.user_preference, self.bite_history, self.continue_food_label, self.continue_dip_label, log_path)
        if food is None:
            return None
        
        food_id, action_type, metadata = food
        if dip is not None:
            dip_id, dip_action_type, dip_metadata = dip
        else:
            dip_id, dip_action_type, dip_metadata = None, None, None
        
        # next bite food item
        next_action_prediction = {
            'food_id': food_id,
            'action_type': action_type,
            'metadata': metadata,
            'dip_id': dip_id,
            'dip_action_type': dip_action_type,
            'dip_metadata': dip_metadata,
            'labels_list': labels_list,
            'bite_mask_idx': bite_mask_idx
        }

        # Rajat ToDo: Update the detections with bite_mask_idx as the first mask for the food item
        
        self.next_action_prediction = next_action_prediction
        return next_action_prediction