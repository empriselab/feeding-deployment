import cv2
import time
import os
import numpy as np
import os
from openai import OpenAI
import ast
import sys
import cmath
import math
import random

try:
    import supervision as sv
    import torch
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor
    PATH_TO_GROUNDED_SAM = '/home/isacc/Grounded-Segment-Anything'
    PATH_TO_DEPTH_ANYTHING = '/home/isacc/Depth-Anything'
    PATH_TO_SPAGHETTI_CHECKPOINTS = '/home/isacc/deployment_ws/src/feeding-deployment/src/feeding_deployment/actions/flair/spaghetti_checkpoints'
    USE_EFFICIENT_SAM = False

    sys.path.append(PATH_TO_DEPTH_ANYTHING)

    from depth_anything.dpt import DepthAnything
    # from depth_anything.dpt import DPT_DINOv2
    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

    from feeding_deployment.actions.flair.src.food_pos_ori_net.model.minispanet import MiniSPANet
    from feeding_deployment.actions.flair.src.spaghetti_segmentation.model import SegModel

    FOOD_MODELS_IMPORTS = True
except ModuleNotFoundError as e:
    # print(f"Module not found: {e}")
    FOOD_MODELS_IMPORTS = False

from feeding_deployment.actions.flair.vision_utils import detect_densest, new_detect_densest, detect_sparsest, detect_centroid, detect_angular_bbox, detect_convex_hull, detect_filling_push_noodles, detect_filling_push_semisolid, efficient_sam_box_prompt_segment, outpaint_masks, detect_blue, proj_pix2mask, cleanup_mask, visualize_keypoints, visualize_skewer, visualize_push, detect_plate, mask_weight, nearest_neighbor, nearest_point_to_mask, detect_furthest_unobstructed_boundary_point, calculate_heatmap_density, calculate_heatmap_entropy, resize_to_square, fill_enclosing_polygon, detect_fillings_in_mask, expanded_detect_furthest_unobstructed_boundary_point
from feeding_deployment.actions.flair.preference_planner import PreferencePlanner
# from feeding_deployment.actions.flair.food_identification import GPT4VFoodIdentification

class BiteAcquisitionInference:
    def __init__(self, mode):

        if FOOD_MODELS_IMPORTS:
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # GroundingDINO config and checkpoint
            self.GROUNDING_DINO_CONFIG_PATH = PATH_TO_GROUNDED_SAM + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            self.GROUNDING_DINO_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/groundingdino_swint_ogc.pth"

            print("Initializing Grounding Dino")        
            # Building GroundingDINO inference model
            self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH)
            self.use_efficient_sam = USE_EFFICIENT_SAM

            if self.use_efficient_sam:
                # Building MobileSAM predictor
                self.EFFICIENT_SAM_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/efficientsam_s_gpu.jit"
                self.efficientsam = torch.jit.load(self.EFFICIENT_SAM_CHECKPOINT_PATH)
            else:
                # Segment-Anything checkpoint
                SAM_ENCODER_VERSION = "vit_h"
                SAM_CHECKPOINT_PATH = PATH_TO_GROUNDED_SAM + "/sam_vit_h_4b8939.pth"

                print("Initializing SAM")
                # Building SAM Model and SAM Predictor
                sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
                sam.to(device=self.DEVICE)
                self.sam_predictor = SamPredictor(sam)

            print("Initializing Depth Anything")
            self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(self.DEVICE).eval()

            self.depth_anything_transform = transforms.Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])

            self.FOOD_CLASSES = None
            self.FOOD_CATEGORIES = None
            self.BOX_THRESHOLD = 0.3
            self.TEXT_THRESHOLD = 0.2
            self.NMS_THRESHOLD = 0.4

            torch.set_flush_denormal(True)
            checkpoint_dir = PATH_TO_SPAGHETTI_CHECKPOINTS
            
            self.minispanet = MiniSPANet(out_features=1)
            self.minispanet_crop_size = 100
            checkpoint = torch.load('%s/spaghetti_ori_net.pth'%checkpoint_dir, map_location=self.DEVICE)
            self.minispanet.load_state_dict(checkpoint)
            self.minispanet.eval()
            self.minispanet_transform = transforms.Compose([transforms.ToTensor()])

            self.seg_net = SegModel("FPN", "resnet34", in_channels=3, out_classes=1)
            ckpt = torch.load('%s/spaghetti_seg_resnet.pth'%checkpoint_dir, map_location=self.DEVICE)
            self.seg_net.load_state_dict(ckpt)
            self.seg_net.eval()
            self.seg_net.to(self.DEVICE)
            self.seg_net_transform = transforms.Compose([transforms.ToTensor()])

        # print("Initializing GPT4V Food Identification")
        # self.api_key =  os.environ['OPENAI_API_KEY']
        # self.gpt4v_client = GPT4VFoodIdentification(self.api_key, os.path.dirname(os.path.realpath(__file__)) + '/prompts/identification')
        # self.client = OpenAI(api_key=self.api_key)
        
        print("Initializing Preference Planner")
        self.preference_planner = PreferencePlanner()

        self.mode = mode

    # def recognize_items(self, image):
    #     response = self.gpt4v_client.prompt(image).strip()
    #     items = ast.literal_eval(response)
    #     return items

    def chat_with_openai(self, prompt):
        """
        Sends the prompt to OpenAI API using the chat interface and gets the model's response.
        """
        message = {
                    'role': 'user',
                    'content': prompt
                  }
    
        response = self.client.chat.completions.create(
                   model='gpt-3.5-turbo-1106',
                   messages=[message]
                  )
        
        # Extract the chatbot's message from the response.
        # Assuming there's at least one response and taking the last one as the chatbot's reply.
        chatbot_response = response.choices[0].message.content
        return chatbot_response.strip()

    def run_minispanet_inference(self, u, v, cv_img, crop_dim=15):

        assert FOOD_MODELS_IMPORTS, "MiniSPANet model required to run this function"

        cv_crop = cv_img[v-crop_dim:v+crop_dim, u-crop_dim:u+crop_dim]
        cv_crop_resized = cv2.resize(cv_crop, (self.minispanet_crop_size, self.minispanet_crop_size))
        rescale_factor = cv_crop.shape[0]/self.minispanet_crop_size

        img_t = self.minispanet_transform(cv_crop_resized)
        img_t = img_t.unsqueeze(0)
        H,W = self.minispanet_crop_size, self.minispanet_crop_size

        heatmap, pred = self.minispanet(img_t)

        heatmap = heatmap.detach().cpu().numpy()
        pred_rot = pred.detach().cpu().numpy().squeeze()

        heatmap = heatmap[0][0]
        pred_x, pred_y = self.minispanet_crop_size//2, self.minispanet_crop_size//2
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(cv_crop_resized, 0.55, heatmap, 0.45, 0)
        cv2.circle(heatmap, (pred_x,pred_y), 2, (255,255,255), -1)
        cv2.circle(heatmap, (W//2,H//2), 2, (0,0,0), -1)
        pt = cmath.rect(20, np.pi/2-pred_rot)  
        x2 = int(pt.real)
        y2 = int(pt.imag)
        rot_vis = cv2.line(cv_crop_resized, (pred_x-x2,pred_y+y2), (pred_x+x2, pred_y-y2), (255,255,255), 2)
        cv2.putText(heatmap,"Skewer Point",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(rot_vis,"Skewer Angle",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.circle(rot_vis, (pred_x,pred_y), 4, (255,255,255), -1)
        result = rot_vis

        global_x = u + pred_x*rescale_factor
        global_y = v + pred_y*rescale_factor
        pred_rot = math.degrees(pred_rot)
        return pred_rot, int(global_x), int(global_y), result

    def determine_action(self, density, entropy, valid_actions):
        
        DENSITY_THRESH = 0.64 # 0.8 for mashed potatoes
        ENTROPY_THRESH = 9.0
        if density > DENSITY_THRESH:
            if 'Acquire' in valid_actions:
                return 'Acquire'
            elif 'Push Filling' in valid_actions:
                return 'Push Filling'
        elif entropy > ENTROPY_THRESH:
            if 'Group' in valid_actions:
                return 'Group'
            elif 'Push Filling' in valid_actions:
                return 'Push Filling'
        elif 'Push Filling' in valid_actions:
            return 'Push Filling'
        return 'Acquire'

    def get_scoop_action(self, image, masks, categories, log_path = None):
        
        semisolid_mask = None
        for (category, mask) in zip(categories, masks):
            if category == 'semisolid':
                semisolid_mask = mask[0]

        filling_centroids = []
        filling_masks = []
        filling_push_start = None
        filling_push_end = None

        for i, (category, mask) in enumerate(zip(categories, masks)):
            if category in ['meat/seafood', 'vegetable', 'fruit']:
                for item_mask in mask:
                    centroid = detect_centroid(item_mask)
                    filling_centroids.append(centroid)
                    filling_masks.append(item_mask)
        
        if len(filling_masks) > 0:
            filling_masks, filling_centroids = detect_fillings_in_mask(filling_masks, filling_centroids, semisolid_mask)
        
        obstructions_masks_combined = np.zeros_like(semisolid_mask)
        for mask in filling_masks:
            obstruction_mask = cv2.dilate(mask, np.ones((20,20), np.uint8), iterations=1)
            obstructions_masks_combined = cv2.bitwise_or(obstructions_masks_combined, obstruction_mask)
        
        densest, heatmap = new_detect_densest(semisolid_mask, obstructions_masks_combined)
        
        furthest_unobstructed_boundary_point, furthest_unobstructed_boundary_point_vis = expanded_detect_furthest_unobstructed_boundary_point(densest, semisolid_mask, obstructions_masks_combined)

        heatmap = cv2.bitwise_and(heatmap, semisolid_mask)
        density = calculate_heatmap_density(heatmap)
        entropy = calculate_heatmap_entropy(heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
        
        if not len(filling_masks):
            valid_actions, action_vis_mask = ['Acquire'], np.zeros_like(image)
        else:
            
            hull = detect_convex_hull(semisolid_mask)
            filling_push, filling_push_start, filling_push_end = detect_filling_push_semisolid(filling_centroids, hull)
            
            H,W,C = image.shape
            vis = np.zeros((H,W))
            action_vis_mask = np.zeros((H,W), dtype=np.uint8)
            action_vis_mask = cv2.cvtColor(action_vis_mask, cv2.COLOR_GRAY2RGB)
            action_vis_mask = visualize_keypoints(action_vis_mask, [densest], radius=5, color=(0,0,255))
 
            if furthest_unobstructed_boundary_point is not None:
                valid_actions = ['Acquire', 'Push Filling']
                action_vis_mask = visualize_push(action_vis_mask, furthest_unobstructed_boundary_point, densest, color=(0,255,0))
                action_vis_mask = visualize_keypoints(action_vis_mask, [furthest_unobstructed_boundary_point], radius=5, color=(255,0,0))            
                action_vis_mask = cv2.putText(action_vis_mask, 'Valid Actions: %s'%valid_actions, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            else:
                valid_actions = ['Push Filling']
                action_vis_mask = cv2.putText(action_vis_mask, 'Valid Actions: %s'%valid_actions, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        action = self.determine_action(density, entropy, valid_actions)
        print('here', action)

        heatmap = cv2.putText(heatmap, 'Density Score: %.2f'%(density), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        heatmap = cv2.putText(heatmap, 'Entropy Score: %.2f'%(entropy), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        heatmap = cv2.putText(heatmap, 'Action: %s'%(action), (20,460), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        print("semisolid_mask: ", np.max(semisolid_mask), semisolid_mask.shape)
        print("heatmap: ", np.max(heatmap), heatmap.shape)

        if log_path is not None:
            cv2.imwrite(log_path + "_scoop_mask.png", semisolid_mask)
            cv2.imwrite(log_path + "_scoop_heatmap.png", heatmap)

        color_image_vis = image.copy()
        if furthest_unobstructed_boundary_point is not None:
            start_px = np.array(furthest_unobstructed_boundary_point)
            end_px = np.array(densest)

            start_px = start_px + 20 * (start_px - end_px)/np.linalg.norm(end_px - start_px)
            # end_px = end_px + (end_px - start_px)/np.linalg.norm(end_px - start_px) * 10

            end_px = start_px + 150 * (end_px - start_px)/np.linalg.norm(end_px - start_px) # Was 90 before

            start_px = start_px.astype(int)
            end_px = end_px.astype(int)

            # visualize on intersection which is a binary image
            cv2.circle(color_image_vis, tuple(start_px), 5, (255, 0, 0), -1) 
            cv2.circle(color_image_vis, tuple(end_px), 5, (0, 255, 0), -1)

            cv2.arrowedLine(color_image_vis, tuple(start_px), tuple(end_px),(0,0,255), 1)
            cv2.putText(color_image_vis, 'Scoop',  (start_px[0], start_px[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2, 2)
        else:
            start_px = None
            end_px = None

        if log_path is not None:
            cv2.imwrite(log_path + "_scoop_action_vis.png", color_image_vis)

        # visualize scoop action 
        cv2.imshow("Scoop Action", color_image_vis)
        cv2.waitKey(0)

        if action == 'Acquire':
            heatmap = visualize_push(heatmap, furthest_unobstructed_boundary_point, densest, radius=2, color=(0,255,0))
            heatmap = visualize_keypoints(heatmap, [densest], radius=5, color=(0,0,255))
            heatmap = visualize_keypoints(heatmap, [furthest_unobstructed_boundary_point], radius=5, color=(255,0,0))
        elif action == 'Push Filling':
            heatmap = visualize_push(heatmap, filling_push_start, filling_push_end, radius=2, color=(0,255,0))
            heatmap = visualize_keypoints(heatmap, [filling_push_start], radius=5, color=(0,0,255))
            heatmap = visualize_keypoints(heatmap, [filling_push_end], radius=5, color=(255,0,0))

        return densest, furthest_unobstructed_boundary_point, filling_push_start, filling_push_end, valid_actions, action_vis_mask, heatmap, action, start_px, end_px, color_image_vis, semisolid_mask, furthest_unobstructed_boundary_point_vis


    def get_noodle_action(self, image, masks, categories, log_path = None):
        
        noodle_mask = None
        for (category, mask) in zip(categories, masks):
            if category == 'noodles':
                noodle_mask = mask[0]

        filling_centroids = []
        filling_masks = []
        filling_push_start = None
        filling_push_end = None

        for i, (category, mask) in enumerate(zip(categories, masks)):
            if category in ['meat/seafood', 'vegetable']:
                for item_mask in mask:
                    centroid = detect_centroid(item_mask)
                    filling_centroids.append(centroid)
                    filling_masks.append(item_mask)

        if len(filling_masks) > 0:
            filling_masks, filling_centroids = detect_fillings_in_mask(filling_masks, filling_centroids, noodle_mask)
        
        obstructions_masks_combined = np.zeros_like(noodle_mask)
        for mask in filling_masks:
            obstruction_mask = cv2.dilate(mask, np.ones((20,20), np.uint8), iterations=1)
            obstructions_masks_combined = cv2.bitwise_or(obstructions_masks_combined, obstruction_mask)
        
        densest, heatmap = new_detect_densest(noodle_mask, obstructions_masks_combined)
        
        # Detect twirl angle
        sparsest, sparsest_candidates = detect_sparsest(noodle_mask, densest)
        twirl_angle, _, _, minispanet_vis = self.run_minispanet_inference(densest[0], sparsest[1], image)
        
        furthest_unobstructed_boundary_point, furthest_unobstructed_boundary_point_vis = expanded_detect_furthest_unobstructed_boundary_point(densest, noodle_mask, obstructions_masks_combined)

        heatmap = cv2.bitwise_and(heatmap, noodle_mask)
        density = calculate_heatmap_density(heatmap)
        entropy = calculate_heatmap_entropy(heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)

        H,W,C = image.shape
        vis = np.zeros((H,W))
        
        if not len(filling_masks):
            valid_actions = ['Acquire', 'Group']
            action_vis_mask = np.zeros((H,W), dtype=np.uint8)
            action_vis_mask = cv2.cvtColor(action_vis_mask, cv2.COLOR_GRAY2RGB)
            action_vis_mask = visualize_push(action_vis_mask, sparsest, densest, color=(0,255,0))
            action_vis_mask = visualize_keypoints(action_vis_mask, [densest], radius=30, color=(0,0,255))
            action_vis_mask = cv2.putText(action_vis_mask, 'Valid Actions: %s'%valid_actions, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        else:
            twirl_mask = visualize_keypoints(vis.copy(), [densest], radius=30).astype(np.uint8)

            valid_actions = ['Push Filling']
            if not (np.any(cv2.bitwise_and(obstructions_masks_combined, twirl_mask))):
                print('Acquire is valid')
                valid_actions.append('Acquire')

            if furthest_unobstructed_boundary_point is not None:
                valid_actions.append('Group')
                sparsest = tuple(furthest_unobstructed_boundary_point)

            action_vis_mask = np.zeros((H,W), dtype=np.uint8)
            action_vis_mask = cv2.cvtColor(action_vis_mask, cv2.COLOR_GRAY2RGB)
            action_vis_mask[obstructions_masks_combined > 0] = (100,100,100)
            action_vis_mask = visualize_push(action_vis_mask, sparsest, densest, color=(0,255,0))
            action_vis_mask = visualize_keypoints(action_vis_mask, [densest], radius=30, color=(0,0,255))
            action_vis_mask = cv2.putText(action_vis_mask, 'Valid Actions: %s'%valid_actions, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            filling_push_start, filling_push_end = detect_filling_push_noodles(densest, sparsest, filling_centroids, sparsest_candidates)
        
        action = self.determine_action(density, entropy, valid_actions)
        
        heatmap = cv2.putText(heatmap, 'Density Score: %.2f'%(density), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
        heatmap = cv2.putText(heatmap, 'Entropy Score: %.2f'%(entropy), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
        heatmap = cv2.putText(heatmap, 'Action: %s'%(action), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

        if action == 'Acquire':
            heatmap = visualize_keypoints(heatmap, [densest], radius=5, color=(0,0,255))
        elif action == 'Group':
            heatmap = visualize_push(heatmap, sparsest, densest, radius=2, color=(0,255,0))
            heatmap = visualize_keypoints(heatmap, [densest], radius=5, color=(0,0,255))
            heatmap = visualize_keypoints(heatmap, [sparsest], radius=5, color=(255,0,0))
        elif action == 'Push Filling':
            heatmap = visualize_push(heatmap, filling_push_start, filling_push_end, radius=2, color=(0,255,0))
            heatmap = visualize_keypoints(heatmap, [filling_push_start], radius=5, color=(0,0,255))
            heatmap = visualize_keypoints(heatmap, [filling_push_end], radius=5, color=(255,0,0))

        if log_path is not None:
            cv2.imwrite(log_path + "_noodle_mask.png", noodle_mask)
            cv2.imwrite(log_path + "_noodle_heatmap.png", heatmap)
            cv2.imwrite(log_path + "_noodle_valid_actions.png", action_vis_mask)
        
        return densest, sparsest, twirl_angle, filling_push_start, filling_push_end, valid_actions, action_vis_mask, heatmap, action, noodle_mask, furthest_unobstructed_boundary_point_vis
    
    def get_cut_action(self, masks, color_image, log_path = None):

        cut_length = 85
         # pixels, depends on height of the camera with respect to the plate

        for mask in masks:
            bbox = detect_angular_bbox(mask)

            if bbox is None:
                kk = input("bbox is None")

            print("Length of sides: ", np.linalg.norm(np.array(bbox[0]) - np.array(bbox[1])), np.linalg.norm(np.array(bbox[1]) - np.array(bbox[2])))
            max_length = np.max([np.linalg.norm(np.array(bbox[0]) - np.array(bbox[1])), np.linalg.norm(np.array(bbox[1]) - np.array(bbox[2]))])

            color_image_vis = color_image.copy()
            color_image_vis = cv2.putText(color_image_vis, 'Max Length: %.2f'%(max_length), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            if np.linalg.norm(np.array(bbox[0]) - np.array(bbox[1])) < cut_length and np.linalg.norm(np.array(bbox[1]) - np.array(bbox[2])) < cut_length:
                # return False, mask, None
                return False, mask, None, color_image_vis

        center = np.array([bbox[0][0], bbox[0][1]]) + np.array([bbox[1][0], bbox[1][1]]) + np.array([bbox[2][0], bbox[2][1]]) + np.array([bbox[3][0], bbox[3][1]])
        center = center / 4
        center = center.astype(int)
        
        if np.linalg.norm(np.array(bbox[0]) - np.array(bbox[1])) > np.linalg.norm(np.array(bbox[1]) - np.array(bbox[2])):
            one_end = np.array(bbox[1] + bbox[2]) / 2
            other_end = np.array(bbox[0] + bbox[3]) / 2
        else:
            one_end = np.array(bbox[0] + bbox[1]) / 2
            other_end = np.array(bbox[2] + bbox[3]) / 2

        one_end = one_end.astype(int)
        other_end = other_end.astype(int)

        # Select the end point with the highest y value
        if one_end[1] > other_end[1]:
            cut_end = one_end
        else:
            cut_end = other_end

        # Take delta length of 10 pixels in the direction of the center from the cut end
        cut_point = cut_end + cut_length * (center - cut_end) / np.linalg.norm(center - cut_end)
        cut_point = cut_point.astype(int)

        cut_angle = math.atan2(cut_point[1] - cut_end[1], cut_point[0] - cut_end[0])

        color_image_vis = color_image.copy()
        cv2.line(color_image_vis, tuple(bbox[0]), tuple(bbox[1]), (0,0,255), 2)
        cv2.line(color_image_vis, tuple(bbox[1]), tuple(bbox[2]), (0,0,255), 2)
        cv2.line(color_image_vis, tuple(bbox[2]), tuple(bbox[3]), (0,0,255), 2)
        cv2.line(color_image_vis, tuple(bbox[3]), tuple(bbox[0]), (0,0,255), 2)
        cv2.circle(color_image_vis, tuple(center), 5, (0,0,255), -1)
        cv2.circle(color_image_vis, tuple(one_end), 5, (0,0,255), -1)
        cv2.circle(color_image_vis, tuple(other_end), 5, (0,0,255), -1)
        cv2.circle(color_image_vis, tuple(cut_point), 5, (0,255,0), -1)
        
        # visualize axis of cutting
        pt = cmath.rect(20, np.pi/2-cut_angle)
        x2 = int(pt.real)
        y2 = int(pt.imag)
        cv2.line(color_image_vis, (cut_point[0]-x2,cut_point[1]+y2), (cut_point[0]+x2,cut_point[1]-y2), (255,0,0), 2)

        color_image_vis = cv2.putText(color_image_vis, 'Max Length: %.2f'%(max_length), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        # cv2.imshow('img', color_image_vis)
        # cv2.waitKey(0)

        # return True, cut_point, cut_angle
        return True, cut_point, cut_angle, color_image_vis
    
    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    def get_skewer_action(self, mask):

        bbox = detect_angular_bbox(mask)

        center = np.array([bbox[0][0], bbox[0][1]]) + np.array([bbox[1][0], bbox[1][1]]) + np.array([bbox[2][0], bbox[2][1]]) + np.array([bbox[3][0], bbox[3][1]])
        center = center / 4
        center = center.astype(int)

        if np.linalg.norm(np.array(bbox[0]) - np.array(bbox[1])) > np.linalg.norm(np.array(bbox[1]) - np.array(bbox[2])):
            skewering_angle = math.atan2(bbox[0][1] - bbox[1][1], bbox[0][0] - bbox[1][0])
        else:
            skewering_angle = math.atan2(bbox[1][1] - bbox[2][1], bbox[1][0] - bbox[2][0])
        if skewering_angle >= np.pi/2:
            skewering_angle -= np.pi
        return center, skewering_angle
    
    def get_dip_action(self, mask):

        bbox = detect_angular_bbox(mask)

        center = np.array([bbox[0][0], bbox[0][1]]) + np.array([bbox[1][0], bbox[1][1]]) + np.array([bbox[2][0], bbox[2][1]]) + np.array([bbox[3][0], bbox[3][1]])
        center = center / 4
        center = center.astype(int)

        return center

    def food_on_fork(self, image, visualize=False, log_path = None):
        image = image[:300, 600:900, :]
        print("Image shape: ", image.shape)
        if visualize:
            cv2.imshow('img', image)
            cv2.waitKey(0)
        if log_path is not None:
            cv2.imwrite(log_path + "_food_on_fork.png", image)
        prompt = "Is there a food item on the fork? Answer Yes or No with no additional explanation."
        response = self.gpt4v_client.prompt_zero_shot(image, prompt).strip()
        if 'Yes' in response:
            print("Food on fork: True")
            return True
        else:
            print("Food on fork: False")
            return False
        
    def get_annotated_image(self, image, detections, labels):
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image
    
    def check_valid_plate_depth(self, image, depth):
        plate_mask = detect_plate(image)
        ys, xs = np.where(plate_mask > 0)
        vals = depth[ys, xs]
        min_depth, max_depth = np.amin(vals), np.amax(vals)
        print('Checking min/max plate depth', min_depth, max_depth)
        if max_depth > 590 or max_depth < 570:
            print('----!!!---\nInvalid plate depth!')
            return False
        return True
    
    def crop_plate(self, image):

        plate_mask = detect_plate(image)
        plate_mask_vis = np.repeat(plate_mask[:,:,np.newaxis], 3, axis=2)

        # get bounding box of the plate
        non_zero_points = cv2.findNonZero(plate_mask)
        x, y, w, h = cv2.boundingRect(non_zero_points)
        plate_bounds = [x, y, w, h]

        cropped_image = image.copy()[y:y+h, x:x+w]

        return cropped_image

    def detect_items(self, image, log_path = None):

        assert FOOD_MODELS_IMPORTS, "Food detection imports (Grounding DINO, Segment Anything, Depth Anything) required to run this function"
        assert self.FOOD_CLASSES is not None, "Food classes not initialized"
        assert self.FOOD_CATEGORIES is not None, "Food categories not initialized"
        
        plate_mask = detect_plate(image, multiplier=2.0)
        plate_mask_vis = np.repeat(plate_mask[:,:,np.newaxis], 3, axis=2)

        # get bounding box of the plate
        non_zero_points = cv2.findNonZero(plate_mask)
        x, y, w, h = cv2.boundingRect(non_zero_points)
        plate_bounds = [x, y, w, h]

        cropped_image = image.copy()[y:y+h, x:x+w]
        # print("Cropped Image Shape: ", cropped_image.shape)

        # cropped_image = image.copy()[0:440, 550:990]

        # cv2.imshow('img', cropped_image)
        # cv2.waitKey(0)

        # k = input("Visualizing cropped image, is it correct?")
        # if k == 'n':
        #     cv2.destroyAllWindows()
        #     exit(1)

        # self.FOOD_CLASSES = [f.replace('fettuccine', 'noodles') for f in self.FOOD_CLASSES]
        # self.FOOD_CLASSES = [f.replace('spaghetti', 'noodles') for f in self.FOOD_CLASSES]
        # self.FOOD_CLASSES.append('blue plate')

        food_classes_being_detected = []
        replacement_dict = {}
        for i, category in enumerate(self.FOOD_CATEGORIES):
            if category == 'solid': # append "piece to solid items"
                # if "steak" in self.FOOD_CLASSES[i]:
                #     replacement_dict[self.FOOD_CLASSES[i]] = "small cut brown steak piece"
                #     food_classes_being_detected.append("small cut brown steak piece")
                if "potato" in self.FOOD_CLASSES[i]:
                    replacement_dict[self.FOOD_CLASSES[i]] = "yellow potato wedge piece"
                    food_classes_being_detected.append("yellow potato wedge piece")
                elif "chicken" in self.FOOD_CLASSES[i]:
                    replacement_dict[self.FOOD_CLASSES[i]] = "white chicken breast strip piece"
                    food_classes_being_detected.append("white chicken breast strip piece")
                elif "hash" in self.FOOD_CLASSES[i]:
                    replacement_dict[self.FOOD_CLASSES[i]] = "round hash brown piece"
                    food_classes_being_detected.append("round hash brown piece")
                elif "broccoli" in self.FOOD_CLASSES[i]:
                    replacement_dict[self.FOOD_CLASSES[i]] = "green broccoli floret piece"
                    food_classes_being_detected.append("green broccoli floret piece")
                elif "apple" in self.FOOD_CLASSES[i]:
                    replacement_dict[self.FOOD_CLASSES[i]] = "cut white apple slice piece"
                    food_classes_being_detected.append("cut white apple slice piece")
                # elif "strawberry" in self.FOOD_CLASSES[i]:
                #     replacement_dict[self.FOOD_CLASSES[i]] = "red strawberry individual piece"
                #     food_classes_being_detected.append("red strawberry individual piece")
                # elif "cantaloupe" in self.FOOD_CLASSES[i]:
                #     replacement_dict[self.FOOD_CLASSES[i]] = "cut square orange cantaloupe individual piece"
                #     food_classes_being_detected.append("cut square orange cantaloupe individual piece")
                else:
                    replacement_dict[self.FOOD_CLASSES[i]] = "small cut up " + self.FOOD_CLASSES[i] + " piece"
                    food_classes_being_detected.append("small cut up " + self.FOOD_CLASSES[i] + " piece")
            else: # append "dip" to dip items
                replacement_dict[self.FOOD_CLASSES[i]] = self.FOOD_CLASSES[i] + " dip"
                food_classes_being_detected.append(self.FOOD_CLASSES[i] + " dip")

        # # add blue plate to the list of classes so that we can remove it later
        # food_classes_being_detected.append('blue plate')
        print("Food Classes being detected: ", food_classes_being_detected)

        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=cropped_image,
            classes=food_classes_being_detected,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD
        )
        
        # if IS_NOODLE:
        #     filtered_idxs = []
        #     for i in range(len(detections)):
        #         box = detections.xyxy[i]
        #         area = (box[2] - box[0])*(box[3] - box[1])
        #         if area < 50000 and area > 300:
        #             filtered_idxs.append(i)
        #     detections = sv.Detections(xyxy=detections.xyxy[filtered_idxs], class_id=np.array(detections.class_id[filtered_idxs]), confidence=detections.confidence[filtered_idxs])

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{food_classes_being_detected[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _
            in detections]
        annotated_frame = box_annotator.annotate(scene=cropped_image.copy(), detections=detections, labels=labels)

        # NMS post process
        #print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            self.NMS_THRESHOLD
        ).numpy().tolist()

        # remove boxes which are union of two boxes
        
        detections.xyxy = detections.xyxy[nms_idx]

        # recover detections in original image
        detections.xyxy[:,0] += plate_bounds[0]
        detections.xyxy[:,1] += plate_bounds[1]
        detections.xyxy[:,2] += plate_bounds[0]
        detections.xyxy[:,3] += plate_bounds[1]

        # detections.xyxy[:,0] += 550
        # detections.xyxy[:,1] += 0
        # detections.xyxy[:,2] += 550
        # detections.xyxy[:,3] += 0


        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        #print(f"After NMS: {len(detections.xyxy)} boxes")

        if self.use_efficient_sam:
            # collect segment results from EfficientSAM
            result_masks = []
            for box in detections.xyxy:
                mask = efficient_sam_box_prompt_segment(image, box, self.efficientsam)
                result_masks.append(mask)
            
            detections.mask = np.array(result_masks)
        else:
            # Prompting SAM with detected boxes
            def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
                sam_predictor.set_image(image)
                result_masks = []
                for box in xyxy:
                    masks, scores, logits = sam_predictor.predict(
                        box=box,
                        multimask_output=True
                    )
                    index = np.argmax(scores)
                    result_masks.append(masks[index])
                return np.array(result_masks)


            # convert detections to masks
            detections.mask = segment(
                sam_predictor=self.sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
        
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{food_classes_being_detected[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _
            in detections]

        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Clean up to merge multiple detected noodle/semisolid masks
        max_prob = 0
        max_prob_idx = None
        to_remove_idxs = []
        for i in range(len(labels)):
            label = labels[i]
            if 'noodle' in label or 'mashed' in label or 'oatmeal' in label:
                to_remove_idxs.append(i)
                prob = float(label[-4:].strip())
                if prob > max_prob:
                    max_prob_idx = i
                    max_prob = prob
        if len(to_remove_idxs) > 1:
            to_remove_idxs.remove(max_prob_idx)
            idxs = [i for i in range(len(detections)) if not i in to_remove_idxs]
        else:
            idxs = list(range(len(detections)))
        
        noodle_semisolid_idx = None

        # Calculate the maximum area threshold dynamically
        H, W, C = image.shape
        MAX_AREA_THRESHOLD = (H * W) / 25

        individual_masks = []
        refined_labels = []
        for i in range(len(detections)):

            if 'blue plate' in labels[i]:
                continue
            mask_annotator = sv.MaskAnnotator(color=sv.Color.white())
            H,W,C = image.shape
            mask = np.zeros_like(image).astype(np.uint8)
            d = sv.Detections(xyxy=detections.xyxy[i].reshape(1,4), \
                              mask=detections.mask[i].reshape((1,H,W)), \
                              class_id = np.array(detections.class_id[i]).reshape((1,)))
            mask = mask_annotator.annotate(scene=mask, detections=d)
            binary_mask = np.zeros((H,W)).astype(np.uint8)
            if 'noodle' in labels[i] or 'mashed' in labels[i] or 'oatmeal' in labels[i]:
                if noodle_semisolid_idx is None:
                    noodle_semisolid_idx = i
                else:
                    binary_mask = individual_masks[noodle_semisolid_idx]
            ys,xs,_ = np.where(mask > (0,0,0))
            binary_mask[ys,xs] = 255

            # Calculate the area of the binary mask
            area = np.sum(binary_mask > 0)
            
            # Only add the mask if its area is below the threshold
            # print(f"Area of {labels[i]}: {area}, Max Area Threshold: {MAX_AREA_THRESHOLD}")

            # visualize the mask
            # cv2.imshow('mask', binary_mask)
            # cv2.waitKey(0)
            # input("Visualizing mask, press enter to continue")

            if area <= MAX_AREA_THRESHOLD:
                if i in idxs:
                    individual_masks.append(binary_mask)
                    refined_labels.append(labels[i])
            else:
                print(f"Ignoring {labels[i]} due to large area: {area} with max area threshold: {MAX_AREA_THRESHOLD}")

        labels = refined_labels

        # Detect the plate, detect blue color
        # plate_mask = detect_plate(image)
        # individual_masks.append(plate_mask)
        # labels.append('blue plate')
        # individual_masks.append(blue_mask)
        # labels.append('blue')

        # Clean up masks
        refined_masks = []
        portion_weights = []
        for i in range(len(individual_masks)):
            mask = individual_masks[i]
            label = labels[i]

            clean_mask = cleanup_mask(mask)
            refined_masks.append(clean_mask)
            portion_weights.append(1) # each mask is a single portion for bites and dips

        print('Labels before replacement: ', labels)
        for key, value in replacement_dict.items():
            labels = [l.replace(value, key) for l in labels]
        print('Labels after replacement: ', labels)

        return annotated_image, detections, refined_masks, portion_weights, labels, plate_bounds

    def clean_labels(self, labels):
        clean_labels = []
        instance_count = {}
        for label in labels:
            label = label[:-4].strip()
            clean_labels.append(label)
            if label in instance_count:
                instance_count[label] += 1
            else:
                instance_count[label] = 1
        return clean_labels, instance_count

    def categorize_items(self, labels, sim=True):
        categories = []

        if sim:
            for label in labels:
                if 'noodle' in label or 'fettuccine' in label:
                    categories.append('noodles')
                elif 'mashed' in label or 'oatmeal' in label:
                    categories.append('semisolid')
                elif 'banana' in label or 'strawberry' in label or 'watermelon' in label or 'grape' in label or 'celery' in label or 'baby carrot' in label or 'cantaloupe' in label or 'apple' in label:
                    categories.append('fruit')
                elif 'broccoli' in label:
                    categories.append('vegetable')
                elif 'blue' in label:
                    categories.append('plate')
                elif 'sausage' in label or 'meatball' in label or 'meat' in label or 'chicken' in label:
                    categories.append('meat/seafood')
                elif 'brownie' in label:
                    categories.append('brownie')
                elif 'ranch dressing' in label or 'ketchup' in label or 'caramel' in label or 'chocolate sauce' in label:
                    categories.append('dip')
                else:
                    raise KeyError(f"Label {label} not recognized")

        else:
            prompt = """
                    Acceptable outputs: ['noodles', 'meat/seafood', 'vegetable', 'brownie', 'dip', 'fruit', 'semisolid']

                    Input: 'noodles 0.69'
                    Output: 'noodles'

                    Input: 'shrimp 0.26'
                    Output: 'meat/seafood'

                    Input: 'meat 0.46'
                    Output: 'meat/seafood'

                    Input: 'broccoli 0.42'
                    Output: 'vegetable'

                    Input: 'celery 0.69'
                    Output: 'vegetable'

                    Input: 'baby carrot 0.47'
                    Output: 'vegetable'

                    Input: 'chicken 0.27'
                    Output: 'meat/seafood'

                    Input: 'brownie 0.47'
                    Output: 'brownie'

                    Input: 'ketchup 0.47'
                    Output: 'dip'

                    Input: 'ranch 0.24'
                    Output: 'dip'

                    Input: 'mashed potato 0.43'
                    Output: 'semisolid'

                    Input: 'mashed potato 0.30'
                    Output: 'semisolid'

                    Input: 'risotto 0.40'
                    Output: 'semisolid'

                    Input: 'oatmeal 0.43'
                    Output: 'semisolid'

                    Input: 'caramel 0.28'
                    Output: 'dip'

                    Input: 'chocolate sauce 0.24'
                    Output: 'dip'

                    Input: 'strawberry 0.57'
                    Output: 'fruit'

                    Input: 'watermelon 0.47'
                    Output: 'fruit'

                    Input: 'oatmeal 0.43'
                    Output: 'semisolid'

                    Input: 'blueberry 0.87'
                    Output: 'fruit'

                    Input: '%s'
                    Output:
                    """
            for label in labels:
                predicted_category = self.chat_with_openai(prompt%label).strip().replace("'",'')
                categories.append(predicted_category)

        return categories

    def get_manual_action(self, annotated_image, image, masks, categories, labels, portions, preference, history, continue_food_label = None, log_path = None):
        print("Categories: ", categories)
        idx = int(input('Which category (index) do you want to pick up?'))
        if categories[idx] == 'noodles':
            densest, sparsest, twirl_angle, filling_push_start, filling_push_end, valid_actions, valid_actions_vis, heatmap, action = self.get_noodle_action(image, masks, categories)
            action = input('Twirl (t) or Push (p)')
            if action == 't':
                metadata = (idx, 'Twirl', {'point':densest, 'twirl_angle':twirl_angle})
            else:
                type_push = input('Filling (f) or noodles (n)?')
                if type_push == 'f':
                    metadata = (idx, 'Push', {'start':filling_push_start, 'end':filling_push_end})
                elif type_push == 'n':
                    metadata = (idx, 'Push', {'start':sparsest, 'end':densest})
            return metadata, None
        elif categories[idx] == 'semisolid':
            densest, sparsest, filling_push_start, filling_push_end, valid_actions, valid_actions_vis, heatmap, action, start_px, end_px = self.get_scoop_action(image, masks, categories, log_path)
            action = input('Scoop (s) or Push (p)')
            if action == 's':
                return (idx, 'Scoop', {'start':start_px, 'end':end_px}), None
            else:
                return (idx, 'Push', {'start':filling_push_start, 'end':filling_push_end}), None
        elif categories[idx] in ['meat/seafood', 'vegetable', 'fruit', 'brownie']:
            action = input('Cut (c) or Skewer (s)?')
            should_dip = input('Dip (d) or plain (p)?') == 'd'
            dip_action = None
            if should_dip:
                available_dips = labels[categories.index('dip')]
                print('Dips: %s'%str(available_dips))
                dip_idx = int(input('Which dip (index) do you want?'))
                dip_point = self.get_dip_action(masks[categories.index('dip')][dip_idx])
                dip_action = (dip_idx, 'Dip', {'point': dip_point, 'label':labels[categories.index('dip')][dip_idx]})
            if action == 'c':
                requires_cut, cut_point, cut_angle = self.get_cut_action(masks[idx][0], image)
                return (idx, 'Cut', {'point': cut_point, 'cut_angle': cut_angle}), None
            else:
                noodle_or_semisolid_mask = None
                if 'noodles' in categories:
                    noodle_or_semisolid_mask = masks[categories.index('noodles')][0]
                elif 'semisolid' in categories:
                    noodle_or_semisolid_mask = masks[categories.index('semisolid')][0]
                # skewer_mask = self.detect_most_obstructing_filling(masks[idx], noodle_or_semisolid_mask)
                
                # randomly select the idx of the mask to skewer among the possible masks
                id_to_skewer = random.choice(range(len(masks[idx])))
                skewer_mask = masks[idx][id_to_skewer]
                skewer_point, skewer_angle = self.get_skewer_action(skewer_mask)
                metadata = (idx, 'Skewer', {'point': skewer_point, 'skewer_angle': skewer_angle})
                vis = visualize_skewer(image, skewer_point, skewer_angle)
                if log_path is not None:
                    cv2.imwrite(log_path + "_skewer_vis.png", vis)
                return metadata, dip_action
        elif categories[idx] == 'dip':
            raise NotImplementedError # Cannot dip without a skewer

    def detect_most_obstructing_filling(self, filling_masks, noodle_or_semisolid_mask):
        if noodle_or_semisolid_mask is None:
            
            # return mask with smallest area (the one which def doesn't require cutting)
            min_area = 1000000
            min_area_idx = 0
            for i in range(len(filling_masks)):
                area = np.count_nonzero(filling_masks[i])
                if area < min_area:
                    min_area = area
                    min_area_idx = i
            return filling_masks[min_area_idx].astype(np.uint8), min_area_idx
        
        filled_noodle_or_semisolid_mask = fill_enclosing_polygon(noodle_or_semisolid_mask)
        #cv2.imshow('img', filled_noodle_or_semisolid_mask)
        #cv2.waitKey(0)
        max_occlusion = 0
        max_occluding_mask = filling_masks[0]
        max_occluding_mask_idx = 0

        for i in range(len(filling_masks)):
            mask = filling_masks[i]
            occlusion_amount = np.count_nonzero(cv2.bitwise_and(mask, filled_noodle_or_semisolid_mask))
            if occlusion_amount > max_occlusion:
                max_occlusion = occlusion_amount
                max_occluding_mask = mask
                max_occluding_mask_idx = i
        return max_occluding_mask.astype(np.uint8), max_occluding_mask_idx
    
    def get_autonomous_action(self, image, masks, categories, labels, portions, preference, history):
        
        skewer_actions = []
        dip_actions = []
        efficiency_scores = []        
        bite_mask_idx = None

        for idx in range(len(categories)):
            if categories[idx] == "solid":
                id_to_skewer = random.choice(range(len(masks[idx])))
                skewer_mask = masks[idx][id_to_skewer]
                bite_mask_idx = id_to_skewer
                skewer_point, skewer_angle = self.get_skewer_action(skewer_mask)
                skewer_actions.append((idx, 'Skewer', {'point': skewer_point, 'skewer_angle': skewer_angle}))
                efficiency_scores.append(1.0)
            elif categories[idx] == "dip":
                dip_point = self.get_dip_action(masks[idx][0])
                dip_actions.append((idx, 'Dip', {'point': dip_point}))
            else:
                print("Category not recognized")

        skewer_labels = []
        skewer_portions_rounded = []
        dip_labels = []
        for idx in range(len(labels)):
            if categories[idx] != 'dip':
                skewer_labels.append(labels[idx])
                skewer_portions_rounded.append(round(portions[idx]))
            else:
                dip_labels.append(labels[idx])

        next_bite, response = self.preference_planner.plan(skewer_labels, skewer_portions_rounded, efficiency_scores, preference, dip_labels, history, mode=self.mode)

        print('Next bite', next_bite)
        print('skewer_labels', skewer_labels)
        print('Next actions', skewer_actions)

        if len(next_bite) == 1 and next_bite[0] in labels:
            print(skewer_labels, next_bite[0])
            idx = skewer_labels.index(next_bite[0])
            return skewer_actions[idx], None, bite_mask_idx
        elif len(next_bite) == 2:
            acquire_idx = skewer_labels.index(next_bite[0])
            dip_idx = dip_labels.index(next_bite[1])
            return skewer_actions[acquire_idx], dip_actions[dip_idx], bite_mask_idx
        else: 
            return None, None, None
                      
    # def get_autonomous_action(self, annotated_image, image, masks, categories, labels, portions, preference, history, continue_food_label = None, continue_dip_label = None, log_path = None):
    #     vis = image.copy()

    #     if continue_food_label is not None:
    #         food_to_consider = [i for i in range(len(labels)) if labels[i] == continue_food_label]
    #         if continue_dip_label is not None:
    #             dip_idx = labels.index(continue_dip_label)
    #             food_to_consider.append(dip_idx)
    #     else:
    #         food_to_consider = range(len(categories))

    #     print('Food to consider: ', food_to_consider)

    #     next_actions = []
    #     dip_actions = []
    #     efficiency_scores = []        
    #     bite_mask_idx = None

    #     print(categories, food_to_consider)
    #     for idx in food_to_consider:
    #         if categories[idx] == 'noodles':
    #             densest, sparsest, twirl_angle, filling_push_start, filling_push_end, valid_actions, valid_actions_vis, heatmap, action = self.get_noodle_action(image, masks, categories)
    #             print(valid_actions)
    #             if action == 'Acquire':
    #                 efficiency_scores.append(1)
    #                 next_actions.append((idx, 'Twirl', {'point':densest, 'twirl_angle':twirl_angle}))
    #             elif action == 'Push Filling':
    #                 efficiency_scores.append(2) 
    #                 next_actions.append((idx, 'Push', {'start':filling_push_start, 'end':filling_push_end}))
    #             else:
    #                 efficiency_scores.append(2.5) # Should this be even higher?
    #                 next_actions.append((idx, 'Group', {'start':sparsest, 'end':densest}))
    #         elif categories[idx] == 'semisolid':
    #             densest, furthest_unobstructed_boundary_point, filling_push_start, filling_push_end, valid_actions, action_vis_mask, heatmap, action, start_px, end_px, color_image_vis, semisolid_mask, furthest_unobstructed_boundary_point_vis  = self.get_scoop_action(image, masks, categories, log_path)
    #             # densest, sparsest, filling_push_start, filling_push_end, valid_actions, valid_actions_vis, heatmap, action, start_px, end_px = self.get_scoop_action(image, masks, categories, log_path)
    #             if action == 'Acquire':
    #                 efficiency_scores.append(1)
    #                 next_actions.append((idx, 'Scoop', {'start':start_px, 'end':end_px}))
    #             else:
    #                 efficiency_scores.append(2)
    #                 next_actions.append((idx, 'Push', {'start':filling_push_start, 'end':filling_push_end}))
    #         elif categories[idx] in ['solid', 'meat/seafood', 'vegetable', 'fruit', 'brownie']:
    #             # if categories[idx] != 'brownie':
    #             #     requires_cut, cut_point, cut_angle = self.get_cut_action(masks[idx], image)
    #             #     if requires_cut:
    #             #         efficiency_scores.append(2)
    #             #         next_actions.append((idx, 'Cut', {'point': cut_point, 'cut_angle': cut_angle}))
    #             #         continue
    #             #     else:
    #             #         print('No cut required')
    #             noodle_or_semisolid_mask = None
    #             if 'noodles' in categories:
    #                 noodle_or_semisolid_mask = masks[categories.index('noodles')][0]
    #             elif 'semisolid' in categories:
    #                 noodle_or_semisolid_mask = masks[categories.index('semisolid')][0]
    #             if self.mode == 'preference':
    #                 id_to_skewer = random.choice(range(len(masks[idx])))
    #                 skewer_mask = masks[idx][id_to_skewer]
    #                 bite_mask_idx = id_to_skewer
    #             else:
    #                 skewer_mask, bite_mask_idx = self.detect_most_obstructing_filling(masks[idx], noodle_or_semisolid_mask)
    #             efficiency_scores.append(0.9)
    #             #skewer_point, skewer_angle = self.get_skewer_action(masks[idx][0])
    #             skewer_point, skewer_angle = self.get_skewer_action(skewer_mask)
    #             print("Adding skewer action for label: ", labels[idx])
    #             next_actions.append((idx, 'Skewer', {'point': skewer_point, 'skewer_angle': skewer_angle}))
    #             vis = visualize_skewer(image, skewer_point, skewer_angle)
    #             if log_path is not None:
    #                 cv2.imwrite(log_path + "_skewer_vis.png", vis)
    #         elif categories[idx] == 'dip':
    #             print('Adding dip action for label: ', labels[idx])
    #             dip_point = self.get_dip_action(masks[idx][0])
    #             dip_actions.append((idx, 'Dip', {'point': dip_point}))
    #         else:
    #             raise ValueError(f"Category {categories[idx]} not recognized")
        
    #     print('Length of next actions: ', len(next_actions))
    #     # if len(next_actions) == 1: # Only one item left or if we are continuing to eat the same item
    #     #     return next_actions[0]
        
    #     print('Candidate actions: ', next_actions)

    #     if self.mode == 'efficiency':
    #         return next_actions[np.argmin(efficiency_scores)], None, bite_mask_idx
        
    #     # round efficiency scores to nearest integer
    #     efficiency_scores = [round(score) for score in efficiency_scores]

    #     # take reciprocal of efficiency scores and multiply with LCM
    #     print('Efficiency scores before reciprocal: ', efficiency_scores)
    #     efficiency_scores = np.array([1/score for score in efficiency_scores]) * int(np.lcm.reduce(efficiency_scores))
    #     efficiency_scores = efficiency_scores.astype(int).tolist()

    #     non_dip_labels = []
    #     non_dip_portions_rounded = []
    #     dip_labels = []
    #     for idx in range(len(labels)):
    #         if categories[idx] != 'dip':
    #             non_dip_labels.append(labels[idx])
    #             non_dip_portions_rounded.append(round(portions[idx]))
    #         else:
    #             dip_labels.append(labels[idx])

    #     if continue_food_label is not None:
    #         if continue_dip_label is not None:
    #             next_bite = [continue_food_label, continue_dip_label]
    #         else:
    #             next_bite = [continue_food_label]
    #     else:

    #         print('Non dip labels: ', non_dip_labels)
    #         print('Efficiency scores: ', efficiency_scores)
    #         print('Bite portions: ', non_dip_portions_rounded)
    #         print('Preference: ', preference)

    #         # k = input("Press [n] to exit or otherwise I will query bite sequencing planner...")
    #         # if k == 'n':
    #             # return None, None, None

    #         next_bite, response = self.preference_planner.plan(non_dip_labels, non_dip_portions_rounded, efficiency_scores, preference, dip_labels, history, mode=self.mode)
        
    #     print('Next bite', next_bite)
    #     print('non_dip_labels', non_dip_labels)
    #     print('Next actions', next_actions)

    #     if len(next_bite) == 1 and next_bite[0] in labels:
    #         print(non_dip_labels, next_bite[0])
    #         idx = non_dip_labels.index(next_bite[0])
    #         return next_actions[idx], None, bite_mask_idx
    #     elif len(next_bite) == 2:
    #         acquire_idx = non_dip_labels.index(next_bite[0])
    #         dip_idx = dip_labels.index(next_bite[1])
    #         return next_actions[acquire_idx], dip_actions[dip_idx], bite_mask_idx
    #     else: 
    #         return None, None, None