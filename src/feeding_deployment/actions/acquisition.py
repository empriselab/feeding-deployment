from typing import Any

import time
import pickle
import numpy as np
import cv2

from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from feeding_deployment.actions.base import (
    HighLevelAction,
    tool_type,
    GripperFree,
    Holding,
    IsUtensil,
    PlateInView,
    ToolPrepared
)

from feeding_deployment.actions.flair.food_manipulation_skill_library import FoodManipulationSkillLibrary

class AcquireBiteHLA(HighLevelAction):
    """Bite acquisition; other tools are always prepared."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.food_manipulation_skill_library = FoodManipulationSkillLibrary(self.sim, self.robot_interface, self.wrist_interface, self.perception_interface, self.rviz_interface, self.no_waits)
        self.params = None

        self.food_detection_log_dir = self.log_dir / "food_detection_log"
        self.food_detection_log_dir.mkdir(exist_ok=True)

    def get_name(self) -> str:
        return "AcquireBiteWithTool"
    
    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={Holding([tool]), IsUtensil([tool])},
            add_effects={ToolPrepared([tool])},
            delete_effects=set(),
        )
    
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        assert len(objects) == 1
        tool = objects[0]
        assert tool.name == "utensil"
        return "acquire_bite.yaml"
    
    def acquire_bite(self, speed: str, dipping_depth: float, skewering_depth: float, skewering_orientation: str, autocontinue_timeout: float, ask_confirmation: bool) -> None:

        assert self.sim.held_object_name == "utensil"

        if self.robot_interface is not None:
            self.robot_interface.set_speed(speed)

        # stop the keep horizontal thread (incase we're trying to re-acquire a bite)
        if self.wrist_interface is not None:
            self.wrist_interface.stop_horizontal_spoon_thread()

        # self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos) # leads to safer motion
        self.move_to_joint_positions(self.sim.scene_description.above_plate_pos)
        
        while True:
            if self.wrist_interface is not None:
                self.wrist_interface.set_velocity_mode()
                self.wrist_interface.reset()

            try: # bite ordering and detection
                if self.robot_interface is not None:   

                    camera_color_data, camera_info_data, camera_depth_data = (
                        self.perception_interface.get_camera_data()
                    )

                    if not self.flair.is_preference_set():
                        plate_image = self.flair.crop_plate(camera_color_data)
                        if self.web_interface is not None:
                            while self.web_interface.active:
                                user_input_food_items, user_input_bite_ordering_preference = self.web_interface.get_new_meal_input(plate_image)
                                food_items, bite_ordering_preference = self.flair.parse_new_meal(user_input_food_items, user_input_bite_ordering_preference)
                                if food_items is not None and bite_ordering_preference is not None:
                                    break
                                else:
                                    print("Failed to parse user input. Trying again ...")
                                    time.sleep(1.0)
                        else:
                            # Use command line input for preference setting.
                            user_input_food_items = input("Enter food items as a python list: ")
                            user_input_bite_ordering_preference = input("Enter bite ordering preference: ")
                            food_items, bite_ordering_preference = self.flair.parse_new_meal(user_input_food_items, user_input_bite_ordering_preference)

                        self.flair.set_food_items(food_items)
                        self.flair.set_preference(bite_ordering_preference)

                    items_detection = self.flair.detect_items(camera_color_data, camera_depth_data, camera_info_data, log_path=None)

                    assert self.log_dir is not None, "Log path must be set to save food detection data"
                    # save food detection data
                    food_detection_data = {
                        "camera_color_data": camera_color_data,
                        "camera_info_data": camera_info_data,
                        "camera_depth_data": camera_depth_data,
                        "food_items": self.flair.get_food_items(),
                        "bite_ordering_preference": self.flair.get_preference(),
                        "items_detection": items_detection,
                    }

                    with open(self.log_dir / "food_detection_data.pkl", "wb") as f:
                        pickle.dump(food_detection_data, f)

                    # food detection continuous log
                    file_name = "food_detection_data"
                    id = 0
                    while (self.food_detection_log_dir / f"{file_name}_{id}.pkl").exists():
                        id += 1
                    with open(self.food_detection_log_dir / f"{file_name}_{id}.pkl", "wb") as f:
                        pickle.dump(food_detection_data, f)
                        
                else:
                    # read last logged data
                    try:
                        with open(self.log_dir / "food_detection_data.pkl", "rb") as f:
                            food_detection_data = pickle.load(f)

                        camera_color_data = food_detection_data["camera_color_data"]
                        camera_info_data = food_detection_data["camera_info_data"]
                        camera_depth_data = food_detection_data["camera_depth_data"]
                        food_items = food_detection_data["food_items"]
                        bite_ordering_preference = food_detection_data["bite_ordering_preference"]
                        items_detection = food_detection_data["items_detection"]

                        self.flair.set_food_items(food_items)
                        self.flair.set_preference(bite_ordering_preference)

                    except FileNotFoundError:
                        raise FileNotFoundError("No logged data found for bite acquisition")
            except Exception as e:
                print("Failed to detect items:", e)
                continue

            try: # actual acquisition

                # Prepare for bite acquisition.
                if self.wrist_interface is not None:
                    self.wrist_interface.set_velocity_mode()
                    self.wrist_interface.reset()

                next_action_prediction = self.flair.predict_next_action(camera_color_data, items_detection, log_path=None)

                next_food_item = next_action_prediction['labels_list'][next_action_prediction['food_id']]
                bite_mask_idx = next_action_prediction['bite_mask_idx']
                print(" --- Next Food Item Prediction:", next_action_prediction['labels_list'][next_action_prediction['food_id']])
                print(" --- Next Action Prediction:", next_action_prediction['action_type'])

                # remove next_food_item from data
                solid_food_type_to_data = {}
                for id in range(0, len(items_detection['labels_list'])):
                    if items_detection['category_list'][id] == "solid":
                        label = items_detection['labels_list'][id]
                        solid_food_type_to_data[label] = items_detection['food_type_to_bounding_boxes_plate'][label]

                n_food_types = len(solid_food_type_to_data)
                data = [{k: v} for k, v in solid_food_type_to_data.items() if k != next_food_item]
                predicted_bite = {next_food_item: solid_food_type_to_data[next_food_item]}

                dip_food_type_to_data = {}
                for id in range(0, len(items_detection['labels_list'])):
                    if items_detection['category_list'][id] == "dip":
                        label = items_detection['labels_list'][id]
                        dip_food_type_to_data[label] = items_detection['food_type_to_bounding_boxes_plate'][label]

                if len(dip_food_type_to_data) == 0: # no dips detected
                    dip_data = ["No dip"]     
                else:
                    if next_action_prediction['dip_id'] is None: 
                        dip_data = ["No dip"]
                        dip_data.extend(list(dip_food_type_to_data.keys()))
                    else: # some dip was predicted
                        next_dip_item = next_action_prediction['labels_list'][next_action_prediction['dip_id']]
                        dip_data = [next_dip_item]
                        dip_data.append("No dip")
                        dip_data.extend([k for k in dip_food_type_to_data.keys() if k != next_dip_item])
                n_dip_food_types = len(dip_data)

                if self.web_interface is not None:
                    skill_type, skill_params, dip_type = self.web_interface.get_next_bite_selection(items_detection['plate_image'], n_food_types, data, predicted_bite, n_dip_food_types, dip_data, autocontinue_timeout=autocontinue_timeout)   
                else:
                    # params must be set to the autonomously selected values
                    skill_type = "autonomous"
                    skill_params = [next_food_item, bite_mask_idx]
                    dip_type = "No dip"

                skill_success = False
                if skill_type == "autonomous":
                    food_type_to_masks = items_detection["food_type_to_masks"]
                    food_type_to_skill = items_detection["food_type_to_skill"]
                    
                    food_type = skill_params[0]
                    item_id = skill_params[1] - 1

                    # Rajat Imp ToDo: Update bite history after successful skill execution
                    self.flair.update_bite_history(food_type)

                    mask = food_type_to_masks[food_type][item_id]
                    skill = food_type_to_skill[food_type]

                    if skill == "Skewer":
                        skewer_point, skewer_angle = self.flair.inference_server.get_skewer_action(mask)
                        if skewering_orientation == "vertical":
                            skewer_angle = skewer_angle + np.pi / 2
                        skill_success = self.food_manipulation_skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = skewer_point, major_axis = skewer_angle, skewering_depth=skewering_depth)
                    elif skill == "Scoop":
                        raise NotImplementedError("Scoop skill not yet implemented")
                    
                    if dip_type != "No dip" and skill_success:
                        self.flair.update_bite_history(dip_type)
                        dip_mask = food_type_to_masks[dip_type][0]
                        dip_point = self.flair.inference_server.get_dip_action(dip_mask)
                        self.food_manipulation_skill_library.robot_reset()
                        skill_success = self.food_manipulation_skill_library.dipping_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = dip_point, dipping_depth=dipping_depth)
                
                elif skill_type == "manual_skewering":

                    plate_bounds = items_detection["plate_bounds"]
                    pos = skill_params[0]

                    point_x = int(pos["x"]*plate_bounds[2]) + plate_bounds[0]
                    point_y = int(pos["y"]*plate_bounds[3]) + plate_bounds[1]

                    print("Plate Bounds:", plate_bounds)
                    print("Positions:", skill_params)
                    print("Point:", point_x, point_y)

                    if not self.no_waits:
                        # visualize point on camera color image
                        viz = camera_color_data.copy()
                        for pos in skill_params:
                            cv2.circle(viz, (point_x, point_y), 5, (0, 255, 0), -1)
                        cv2.imshow("viz", viz)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    skewer_center = (point_x, point_y)
                    skewer_angle = -np.pi/2

                    skill_success = self.food_manipulation_skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = skewer_center, major_axis = skewer_angle, skewering_depth=skewering_depth)            
                elif skill_type == "manual_scooping":
                    raise NotImplementedError("Scoop skill not yet implemented")
                elif skill_type == "manual_dipping":

                    plate_bounds = items_detection["plate_bounds"]
                    pos = skill_params[0]

                    point_x = int(pos["x"]*plate_bounds[2]) + plate_bounds[0]
                    point_y = int(pos["y"]*plate_bounds[3]) + plate_bounds[1]

                    print("Plate Bounds:", plate_bounds)
                    print("Positions:", skill_params)
                    print("Point:", point_x, point_y)

                    if not self.no_waits:
                        # visualize point on camera color image
                        viz = camera_color_data.copy()
                        for pos in skill_params:
                            cv2.circle(viz, (point_x, point_y), 5, (0, 255, 0), -1)
                        cv2.imshow("viz", viz)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    dip_point = (point_x, point_y)

                    skill_success = self.food_manipulation_skill_library.dipping_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = dip_point, dipping_depth=dipping_depth)

                self.move_to_joint_positions(self.sim.scene_description.above_plate_pos)
                if not skill_success:
                    print("Skill failed. Retrying ...")
                    continue
            except Exception as e:
                print("Failed to acquire bite:", e)
                continue
            
            if self.web_interface is not None and ask_confirmation:
                get_success_confirmation = self.web_interface.get_successful_food_acquisition_confirmation()
                if get_success_confirmation:
                    break
            else:
                break

        # set the wrist controller to always keep utensil horizontal
        if self.wrist_interface is not None:
            self.wrist_interface.start_horizontal_spoon_thread()

        return []
