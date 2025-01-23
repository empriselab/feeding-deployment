import cv2
import ast
import numpy as np
from scipy.spatial.transform import Rotation
import math
import os
from pathlib import Path

from feeding_deployment.actions.flair.inference_class import BiteAcquisitionInference
from feeding_deployment.actions.flair.new_meal_parser import NewMealParser

HOME_ORIENTATION = Rotation.from_quat([1/math.sqrt(2), 1/math.sqrt(2), 0, 0]).as_matrix()
DEFAULT_FORCE_THRESHOLD = [30.0, 30.0, 30.0, 30.0, 30.0, 30.0]

class FLAIR:
    def __init__(self, log_dir):

        self.inference_server = BiteAcquisitionInference(mode='ours')
        print("inf server init")

        self.new_meal_parser = NewMealParser(log_dir)   
        self.history_path = log_dir / "flair_history.txt"
        
        if not os.path.exists(self.history_path):
            self.bite_history = []
            self.inference_server.FOOD_CLASSES = None
            self.inference_server.FOOD_CATEGORIES = None
            self.user_preference = None
        else:
            # read in json format
            try:
                with open(self.history_path, 'r') as f:
                    logged_history = ast.literal_eval(f.read())
                self.bite_history = logged_history["bite_history"]
                self.inference_server.FOOD_CLASSES = logged_history["food_labels"]
                self.inference_server.FOOD_CATEGORIES = logged_history["food_categories"]
                self.user_preference = logged_history["user_preference"]
            except Exception as e:
                print("Error reading history file", e)
                print("Creating new history file...")
                self.bite_history = []
                self.inference_server.FOOD_CLASSES = None
                self.inference_server.FOOD_CATEGORIES = None
                self.user_preference = None

            print("Logged Bite History", self.bite_history)
            print("Logged User Preference", self.user_preference)

        # Continue food
        self.continue_food_label = None
        self.continue_dip_label = None

    def log_history(self):
        with open(self.history_path, 'w') as f:
            f.write(str({
                "bite_history": self.bite_history,
                "food_labels": self.inference_server.FOOD_CLASSES, 
                "food_categories": self.inference_server.FOOD_CATEGORIES,
                "user_preference": self.user_preference
            }))

    def update_bite_history(self, acquired_food_label):
        self.bite_history.append(acquired_food_label)
        self.log_history()

    def parse_new_meal(self, food_items, bite_ordering_preference):
        solid_items, dip_items, bite_ordering_preference = self.new_meal_parser.parse_user_message(food_items, bite_ordering_preference)
        
        food_items = {
            "solid": solid_items,
            "dip": dip_items
        }

        return food_items, bite_ordering_preference

    # def identify_plate(self, camera_color_data):

    #     items = self.inference_server.recognize_items(camera_color_data)
    #     print("Food Items recognized:", items)
    #     return items

    def set_food_items(self, food_items):
        self.inference_server.FOOD_CLASSES = food_items["solid"] + food_items["dip"]
        self.inference_server.FOOD_CATEGORIES = []
        for solid_food_item in food_items["solid"]:
            self.inference_server.FOOD_CATEGORIES.append('solid')
        for dip in food_items["dip"]:
            self.inference_server.FOOD_CATEGORIES.append('dip')

        self.log_history()

    def get_food_items(self):

        solid_items = []
        dip_items = []

        for i, category in enumerate(self.inference_server.FOOD_CATEGORIES):
            if category == 'solid':
                solid_items.append(self.inference_server.FOOD_CLASSES[i])
            else:
                dip_items.append(self.inference_server.FOOD_CLASSES[i])

        return {
            "solid": solid_items,
            "dip": dip_items
        }

    def set_preference(self, user_preference):
        self.user_preference = user_preference
        self.log_history()

    def get_preference(self):
        return self.user_preference

    def clear_preference(self):
        self.user_preference = None

    def is_preference_set(self):
        return self.user_preference is not None
    
    def crop_plate(self, camera_color_data):
        return self.inference_server.crop_plate(camera_color_data)

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

        cv2.imshow('vis', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        input("Visualzing the detected items. Press Enter to continue.")

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

        # categories = self.inference_server.categorize_items(item_labels, sim=False) 
        # set all items to solids
        categories = []
        for label in clean_item_labels:
            if label in self.inference_server.FOOD_CLASSES:
                idx = self.inference_server.FOOD_CLASSES.index(label)
                categories.append(self.inference_server.FOOD_CATEGORIES[idx])
            else:
                print("Label not found in food classes:", label, self.inference_server.FOOD_CLASSES)
                categories.append('solid')


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

        # # visualize plate_image
        # cv2.imshow('vis', plate_image)
        # cv2.waitKey(0)
        # input("Visualzing the plate image. Press Enter to continue.")
        # cv2.destroyAllWindows()

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
            elif categories[i] == 'dip':
                food_type_to_skill[labels_list[i]] = 'Dip'
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

        return items_detection

    def predict_next_action(self, camera_color_data, items_detection, log_path):

        annotated_image = items_detection['annotated_image']
        per_food_masks = items_detection['per_food_masks']
        category_list = items_detection['category_list']
        per_food_masks = items_detection['per_food_masks']
        labels_list = items_detection['labels_list']
        per_food_portions = items_detection['per_food_portions']
        
        food, dip, bite_mask_idx = self.inference_server.get_autonomous_action(annotated_image, per_food_masks, category_list, labels_list, per_food_portions, self.user_preference, self.bite_history)
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
        return next_action_prediction