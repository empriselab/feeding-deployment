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

from feeding_deployment.actions.flair.flair import FLAIR
from feeding_deployment.actions.flair.food_manipulation_skill_library import FoodManipulationSkillLibrary

class LookAtPlateHLA(HighLevelAction):
    """Look at plate in preparation of bite acquisition."""

    def get_name(self) -> str:
        return "LookAtPlate"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={Holding([tool]), IsUtensil([tool])},
            add_effects={LiftedAtom(PlateInView, [])},
            delete_effects=set(),
        )

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        assert len(objects) == 1
        tool = objects[0]

        if tool.name == "utensil":
            
            print("In LookAtPlateHLA")
            assert self.sim.held_object_name == "utensil"

            # stop the keep horizontal thread (incase we're trying to re-acquire a bite)
            if self.wrist_interface is not None:
                self.wrist_interface.stop_horizontal_spoon_thread()

            self.move_to_joint_positions(self.sim.scene_description.above_plate_pos)
            
            if self.wrist_interface is not None:
                self.wrist_interface.set_velocity_mode()
                self.wrist_interface.reset()

            if self.flair is not None:

                if self.robot_interface is not None:
                    # Run FLAIR perception.
                    camera_color_data, camera_info_data, camera_depth_data, _ = (
                        self.perception_interface.get_camera_data()
                    )
                    # log the data
                    if self.log_path is not None:

                        # items = self.flair.identify_plate(camera_color_data)
                        # items = ['cantaloupe', 'banana']
                        items = ['banana']
                        self.flair.set_food_items(items)
                        items_detection = self.flair.detect_items(camera_color_data, camera_depth_data, camera_info_data, log_path=None)

                        # save food detection data
                        food_detection_data = {
                            "camera_color_data": camera_color_data,
                            "camera_info_data": camera_info_data,
                            "camera_depth_data": camera_depth_data,
                            "items": items,
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
                        self.flair.set_items_detection(items_detection)

                    except FileNotFoundError:
                        raise FileNotFoundError("No logged data found for bite acquisition")
                
                if not self.flair.is_preference_set():

                    # Handle one-time preference setting.
    
                    food_type_to_data = items_detection['food_type_to_bounding_boxes_plate']
                    n_food_types = len(food_type_to_data)
                    data = [{k: v} for k, v in food_type_to_data.items()]

                    food_types = food_type_to_data.keys()

                    # TODO: generalize this...
                    ordering_options = [f"Eat all the {food_type}s first" for food_type in food_types]
                    ordering_options += ["No preference"]

                    # # save plate image, plate bounds, and original image pickle
                    # import pickle
                    # with open("plate_log.pkl", "wb") as f:
                    #     plate_log = {
                    #         "plate_image": items_detection['plate_image'],
                    #         "plate_bounds": items_detection['plate_bounds'],
                    #         "original_image": camera_color_data,
                    #     }
                    #     pickle.dump(plate_log, f)

                    if self.web_interface is not None:
                        self.web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
                        time.sleep(0.2) # simulate delay for web interface
                        self.web_interface.send_web_interface_image(items_detection['plate_image'])
                        self.web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data})
                        self.web_interface.send_web_interface_message({"n_ordering": len(ordering_options), "data": ordering_options})

                        # save all of this in a pickle file:
                        # import pickle
                        # with open("meal_start_log.pkl", "wb") as f:
                        #     meal_start_log = {
                        #         "plate_image": items_detection['plate_image'],
                        #         "plate_bounds": items_detection['plate_bounds'],
                        #         "food_type_to_data": food_type_to_data,
                        #         "ordering_options": ordering_options,
                        #     }
                        #     pickle.dump(meal_start_log, f)

                        # self.web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
                        # time.sleep(1.0) # simulate delay, also needed for web interface
                        # self.web_interface.update_web_interface_image(items_detection['plate_image'])
                        # time.sleep(1.0)  # simulate delay, also needed for web interface
                        # self.web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data})
                        # self.web_interface.send_web_interface_message({"n_ordering": len(ordering_options), "data": ordering_options})

                        # Wait for web interface to report order selection.
                        print("WAITING TO GET PREFERENCE")
                        while self.web_interface.user_preference is None:
                            time.sleep(1e-1)
                        print("FINISHED GETTING PREFERENCES")
                        print("User Preference:", self.web_interface.user_preference)
                        self.flair.set_preference(self.web_interface.user_preference)
                    else:
                        # Use command line input for preference setting.
                        print("Some example bite ordering preferences:")
                        for ordering_option in ordering_options:
                            print(" - ", ordering_option)
                        preference = input("Enter preference: ")
                        self.flair.set_preference(preference)
                # else:
                #     self.web_interface.send_web_interface_image(items_detection['plate_image'])
                #     time.sleep(1.0)  # simulate delay, also needed for web interface

                # Prepare for bite acquisition.
                if self.wrist_interface is not None:
                    self.wrist_interface.set_velocity_mode()
                    self.wrist_interface.reset()

                next_action_prediction = self.flair.predict_next_action(camera_color_data, items_detection=None, log_path=None)

                next_food_item = next_action_prediction['labels_list'][next_action_prediction['food_id']]
                bite_mask_idx = next_action_prediction['bite_mask_idx']
                print(" --- Next Food Item Prediction:", next_action_prediction['labels_list'][next_action_prediction['food_id']])
                print(" --- Next Action Prediction:", next_action_prediction['action_type'])

                # remove next_food_item from data
                food_type_to_data = items_detection['food_type_to_bounding_boxes_plate']

                n_food_types = len(food_type_to_data)
                data = [{k: v} for k, v in food_type_to_data.items() if k != next_food_item]
                current_bite = {next_food_item: food_type_to_data[next_food_item]}

                if self.web_interface is not None:
                    self.web_interface.send_web_interface_message({"state": "prepare_bite", "status": "completed"})
                    time.sleep(0.2) # simulate delay, needed for web interface
                    self.web_interface.send_web_interface_image(items_detection['plate_image'])
                    time.sleep(0.1)
                    self.web_interface.send_web_interface_message({"n_food_types": n_food_types, "data": data, "current_bite": current_bite})            

                # import pickle
                # with open("next_bite_log.pkl", "wb") as f:
                #     next_bite_log = {
                #         "plate_image": items_detection['plate_image'],
                #         "plate_bounds": items_detection['plate_bounds'],
                #         "food_type_to_data": food_type_to_data,
                #         "next_food_item": next_food_item,
                #     }
                #     pickle.dump(next_bite_log, f)
  
            else:
                # Test image.
                rng = np.random.default_rng(123)
                camera_color_data = rng.integers(0, 255, size=(512, 512, 3))

        else:
            # Other tools are always prepared
            pass
        return []

class AcquireBiteHLA(HighLevelAction):
    """Bite acquisition; other tools are always prepared."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.food_manipulation_skill_library = FoodManipulationSkillLibrary(self.sim, self.robot_interface, self.wrist_interface, self.perception_interface, self.rviz_interface, self.no_waits)

    def get_name(self) -> str:
        return "AcquireBite"

    def get_operator(self) -> LiftedOperator:
        tool = Variable("?tool", tool_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[tool],
            preconditions={Holding([tool]), IsUtensil([tool]), LiftedAtom(PlateInView, [])},
            add_effects={ToolPrepared([tool])},
            delete_effects={LiftedAtom(PlateInView, [])},
        )

    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        assert len(objects) == 1
        tool = objects[0]

        if tool.name == "utensil":

            if self.flair is not None:

                if self.robot_interface is not None:
                    camera_color_data, camera_info_data, camera_depth_data, _ = (
                        self.perception_interface.get_camera_data()
                    )
                else:
                    with open(self.log_path / "food_detection_data.pkl", "rb") as f:
                            food_detection_data = pickle.load(f)

                    camera_color_data = food_detection_data["camera_color_data"]
                    camera_info_data = food_detection_data["camera_info_data"]
                    camera_depth_data = food_detection_data["camera_depth_data"]

                if self.web_interface is None:
                    # params must be set manually to the autonomously selected values
                    next_action_prediction = self.flair.get_next_action()
                    next_food_item = next_action_prediction['labels_list'][next_action_prediction['food_id']]
                    bite_mask_idx = next_action_prediction['bite_mask_idx']

                    params = {
                        "status": "aquire_food",
                        "data": [next_food_item, bite_mask_idx],
                    }

                if params["status"] == 0:

                    detections = self.flair.get_items_detection()
                    plate_bounds = detections["plate_bounds"]
                    pos = params["positions"][0]

                    point_x = int(pos["x"]*plate_bounds[2]) + plate_bounds[0]
                    point_y = int(pos["y"]*plate_bounds[3]) + plate_bounds[1]

                    print("Plate Bounds:", plate_bounds)
                    print("Positions:", params["positions"])
                    print("Point:", point_x, point_y)

                    if not self.no_waits:
                        # visualize point on camera color image
                        viz = camera_color_data.copy()
                        for pos in params["positions"]:
                            cv2.circle(viz, (point_x, point_y), 5, (0, 255, 0), -1)
                        cv2.imshow("viz", viz)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    skewer_center = (point_x, point_y)
                    skewer_angle = -np.pi/2

                    self.food_manipulation_skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = skewer_center, major_axis = skewer_angle)

                elif params["status"] == "aquire_food":
                    detections = self.flair.get_items_detection()
                    food_type_to_masks = detections["food_type_to_masks"]
                    food_type_to_skill = detections["food_type_to_skill"]
                    
                    food_type = params["data"][0]
                    item_id = params["data"][1] - 1

                    mask = food_type_to_masks[food_type][item_id]
                    skill = food_type_to_skill[food_type]

                    if skill == "Skewer":
                        skewer_point, skewer_angle = self.flair.inference_server.get_skewer_action(mask)
                        self.food_manipulation_skill_library.skewering_skill(camera_color_data, camera_depth_data, camera_info_data, keypoint = skewer_point, major_axis = skewer_angle)
                    elif skill == "Scoop":
                        raise NotImplementedError("Scoop skill not yet implemented")

                self.move_to_joint_positions(self.sim.scene_description.above_plate_pos)

                # set the wrist controller to always keep utensil horizontal
                if self.wrist_interface is not None:
                    self.wrist_interface.start_horizontal_spoon_thread()
    
            else:
                time.sleep(2.0)  # simulate delay, also needed for web interface

            # Send message to web interface indicating that robot is done with acquisition.
            if self.web_interface is not None:
                self.web_interface.send_web_interface_message({"state": "bite_pickup", "status": "completed"})

            return []

        else:
            # Other tools are always prepared
            pass
        return []