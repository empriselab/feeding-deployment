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
    
    def acquire_bite(self, speed: float) -> None:

        # TODO actually use speed
        print("ACQUIRE BITE CALLED WITH SPEED: ", speed)

        print("In AcquireBiteHLA")
        assert self.sim.held_object_name == "utensil"

        # stop the keep horizontal thread (incase we're trying to re-acquire a bite)
        if self.wrist_interface is not None:
            self.wrist_interface.stop_horizontal_spoon_thread()

        self.move_to_joint_positions(self.sim.scene_description.above_plate_pos)
        
        while True:
            if self.wrist_interface is not None:
                self.wrist_interface.set_velocity_mode()
                self.wrist_interface.reset()

            if self.robot_interface is not None:   

                # Run FLAIR perception.
                camera_color_data, camera_info_data, camera_depth_data = (
                    self.perception_interface.get_camera_data()
                )

                items_detection = None
                if not self.flair.is_preference_set():

                    # Handle one-time preference setting for new meal.
                    # items = self.flair.identify_plate(camera_color_data)
                    # print("Items detected:", items)
                    # input("Press Enter to continue...")
                    # items = ['cantaloupe', 'banana']
                    items = ['apple', 'mini donut']
                    # items = ['banana']
                    self.flair.set_food_items(items)
                    items_detection = self.flair.detect_items(camera_color_data, camera_depth_data, camera_info_data, log_path=None)

                    food_type_to_data = items_detection['food_type_to_bounding_boxes_plate']
                    n_food_types = len(food_type_to_data)
                    data = [{k: v} for k, v in food_type_to_data.items()]

                    food_types = food_type_to_data.keys()

                    # TODO: generalize this...
                    ordering_options = [f"Eat all the {food_type}s first" for food_type in food_types]
                    ordering_options += ["No preference"]

                    if self.web_interface is not None:
                        bite_ordering_preference = self.web_interface.get_bite_ordering_preference(items_detection['plate_image'], n_food_types, data, ordering_options)
                        print("User Preference:", bite_ordering_preference)
                        self.flair.set_preference(bite_ordering_preference)
                    else:
                        # Use command line input for preference setting.
                        print("Some example bite ordering preferences:")
                        for ordering_option in ordering_options:
                            print(" - ", ordering_option)
                        preference = input("Enter preference: ")
                        self.flair.set_preference(preference)

                if items_detection is None:
                    items_detection = self.flair.detect_items(camera_color_data, camera_depth_data, camera_info_data, log_path=None)

                assert self.log_path is not None, "Log path must be set to save food detection data"
                # save food detection data
                food_detection_data = {
                    "camera_color_data": camera_color_data,
                    "camera_info_data": camera_info_data,
                    "camera_depth_data": camera_depth_data,
                    "items": self.flair.get_food_items(),
                    "items_detection": items_detection,
                }
                with open(self.log_path / "food_detection_data.pkl", "wb") as f:
                    pickle.dump(food_detection_data, f)

            else:
                # read last logged data
                try:
                    with open(self.log_path / "food_detection_data.pkl", "rb") as f:
                        food_detection_data = pickle.load(f)

                    camera_color_data = food_detection_data["camera_color_data"]
                    camera_info_data = food_detection_data["camera_info_data"]
                    camera_depth_data = food_detection_data["camera_depth_data"]
                    items = food_detection_data["items"]
                    items_detection = food_detection_data["items_detection"]

                    self.flair.set_food_items(items)

                except FileNotFoundError:
                    raise FileNotFoundError("No logged data found for bite acquisition")

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
            food_type_to_data = items_detection['food_type_to_bounding_boxes_plate']

            n_food_types = len(food_type_to_data)
            data = [{k: v} for k, v in food_type_to_data.items() if k != next_food_item]
            predicted_bite = {next_food_item: food_type_to_data[next_food_item]}

            if self.web_interface is not None:
                skill_type, skill_params = self.web_interface.get_next_bite_selection(items_detection['plate_image'], n_food_types, data, predicted_bite)         
            else:
                # params must be set to the autonomously selected values
                skill_type = "autonomous"
                skill_params = [next_food_item, bite_mask_idx]

            if skill_type == "autonomous":
                food_type_to_masks = items_detection["food_type_to_masks"]
                food_type_to_skill = items_detection["food_type_to_skill"]
                
                food_type = skill_params[0]
                item_id = skill_params[1] - 1

                self.flair.update_bite_history(food_type)

                mask = food_type_to_masks[food_type][item_id]
                skill = food_type_to_skill[food_type]

                if skill == "Skewer":
                    skewer_point, skewer_angle = self.flair.inference_server.get_skewer_action(mask)
                    self.food_manipulation_skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = skewer_point, major_axis = skewer_angle)
                elif skill == "Scoop":
                    raise NotImplementedError("Scoop skill not yet implemented")

            if skill_type == "manual_skewering":

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

                self.food_manipulation_skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = skewer_center, major_axis = skewer_angle)            

            self.move_to_joint_positions(self.sim.scene_description.above_plate_pos)

            if self.web_interface is not None:
                get_success_confirmation = self.web_interface.get_successful_food_acquisition_confirmation()
                if get_success_confirmation:
                    break
            else:
                break

        # set the wrist controller to always keep utensil horizontal
        if self.wrist_interface is not None:
            self.wrist_interface.start_horizontal_spoon_thread()

        return []
