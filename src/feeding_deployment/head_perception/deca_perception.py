import argparse
import math
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Rajat ToDo: pip install DECA
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from DECA.decalib.datasets import datasets, detectors
from DECA.decalib.deca import DECA
from DECA.decalib.utils import util
from DECA.decalib.utils.config import cfg as deca_cfg
from DECA.decalib.utils.tensor_cropper import transform_points
from scipy.spatial.transform import Rotation
from skimage.io import imread, imsave
from skimage.transform import (
    SimilarityTransform,
    estimate_transform,
    rescale,
    resize,
    warp,
)
from tqdm import tqdm

plt.switch_backend("agg")

# ignore warnings
import warnings
from copy import deepcopy
from threading import Lock

warnings.filterwarnings("ignore")

# UTENSIL = "fork"
UTENSIL = "spoon"
# 1: fork
# 2: spoon

RECORD_GOAL_POSE = False
USER = "shunyang"

INSIDE_MOUTH_DISTANCE = 0.015
MOUTH_OFFSET = -0.005

import sys
import warnings
from collections import deque
from collections.abc import Mapping, Set
from numbers import Number

warnings.filterwarnings("ignore")


class HeadPerception:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # run DECA
        deca_cfg.model.use_tex = False
        deca_cfg.rasterizer_type = "pytorch3d"
        self.deca = DECA(config=deca_cfg, device=self.device)

        self.face_detector_model = detectors.FAN()
        # self.face_detector_model = detectors.FaceBoxes()

        sparse_FLAME_vertices = {
            "nose_line_center": [3526, 3501, 3548, 3518, 3553, 3704],
            "nose_line_left": [3610, 3633, 3606, 3648, 2195, 3684],
            "nose_line_right": [3820, 3834, 3816, 3842, 704, 3851],
            "forehead_line_top": [3729, 3773, 3735, 3786, 3878, 3899, 3874],
            "forehead_line_centre": [3157, 335, 3705, 2176, 671, 3863, 16, 2138],
            "forehead_line_bottom": [2566, 338, 3712, 2179, 674, 3868, 27, 1429],
            "beside_left_eye_center": [2180],
            "beside_left_eye_top": [2573, 1982],
            "beside_left_eye_bottom": [2317],
            "beside_right_eye_center": [679],
            "beside_right_eye_top": [1436, 569],
            "beside_right_eye_bottom": [928],
            "bottom_right_cheek": [1600, 2080, 1575],
            "middle_right_cheek": [2101, 688],
            "top_right_cheek": [2060, 2094, 2106, 3898],
            "beside_nose_right": [2050, 623, 500, 3891],
            "bottom_left_cheek": [2736, 3115, 2711],
            "middle_left_cheek": [3128, 2185],
            "top_left_cheek": [3095, 3121, 3133, 3770],
            "beside_nose_left": [3085, 2086, 1754, 3759],
        }

        self.required_indicies = []
        for key in sparse_FLAME_vertices.keys():
            self.required_indicies += sparse_FLAME_vertices[key]

        self.FLAME_neck_index = 2151

        roll_threshold = np.pi / 7
        pitch_threshold = np.pi / 7
        yaw_threshold = np.pi / 7

        self.max_rotation_threshold = np.array(
            [roll_threshold, pitch_threshold, yaw_threshold]
        )
        self.max_distance_threshold = np.array([0.4, 0.4, 0.4])  # m

        self.min_rotation_threshold = np.array([np.pi / 18, np.pi / 18, np.pi / 18])
        self.min_distance_threshold = np.array([0.01, 0.01, 0.01])

        self.last_trans = None
        self.last_neck_frame = None

        self.last_forque_target_pose = None
        self.last_landmark2d = None
        self.last_landmarks3d = None
        self.last_viz_image = None

        self.trans_perception_buffer = []
        self.neck_frame_perception_buffer = []

        current_filepath = os.path.dirname(os.path.abspath(__file__))
        self.fixed_model_head_points = np.load(
            current_filepath + "/assets/fixed_model_head_points.npy"
        )
        if not RECORD_GOAL_POSE:
            self.reference_forque_pose = np.load(
                current_filepath + "/assets/fork_forque_pose.npy"
            )
            self.reference_head_points = np.load(
                current_filepath + "/assets/fork_head_points.npy"
            )
            self.reference_neck_frame = np.load(
                current_filepath + "/assets/fork_reference_neck_frame.npy"
            )

    def run_deca(
        self,
        image,
        camera_info_msg,
        depth_image,
        base_to_camera,
        debug_print=False,
        visualize=False,
    ):

        viz_image = None

        if visualize:
            viz_image = deepcopy(image)

        b, g, r = cv2.split(image)
        image = np.asarray(cv2.merge([r, g, b]))

        def get_deca_codedict(image):

            input_image = self.getInputImage(image, iscrop=True)

            images = input_image["image"].to(self.device)[None, ...]
            tform = input_image["tform"][None, ...]
            tform = torch.inverse(tform).transpose(1, 2).to(self.device)

            with torch.no_grad():
                codedict = self.deca.encode(images, use_detail=False)

            return codedict, tform, images, input_image["src_pts"]

        codedict, tform, images, src_pts = get_deca_codedict(image)

        if codedict is None:
            return None, None

        # original_images = torch.tensor(image/255).float().to(self.device)[None,...]

        with torch.no_grad():
            orig_opdict, orig_visdict = self.deca.minimal_decode(codedict, tform=tform)

        h = 720
        w = 1280

        landmarks = orig_opdict["landmarks2d_image"].detach().cpu().numpy()
        landmark = landmarks[0]
        landmark[..., 0] = landmark[..., 0] * w / 2 + w / 2
        landmark[..., 1] = landmark[..., 1] * h / 2 + h / 2

        landmarks_selected = landmark[self.required_indicies]

        landmarks_model = orig_opdict["verts"].detach().cpu().numpy()

        # print("Landmarks World [0] Shape: ", landmarks_model[0].shape)
        landmarks_selected_model = landmarks_model[0][self.required_indicies]
        neck_landmark_model = landmarks_model[0][self.FLAME_neck_index]

        face_landmark_model = orig_opdict["landmarks3d_world"].detach().cpu().numpy()
        face_landmark_model = face_landmark_model[0]

        valid_landmarks_selected_model = []
        valid_landmarks_selected_world = []
        for i in range(landmarks_selected.shape[0]):
            validity, point = self.pixel2World(
                camera_info_msg,
                landmarks_selected[i, 0].astype(int),
                landmarks_selected[i, 1].astype(int),
                depth_image,
            )
            if validity:
                valid_landmarks_selected_model.append(landmarks_selected_model[i])
                valid_landmarks_selected_world.append(point)

        if len(valid_landmarks_selected_world) < 4:
            print("Not enough landmarks to fit model.")
            return None, None, None

        valid_landmarks_selected_model = np.array(valid_landmarks_selected_model)
        valid_landmarks_selected_world = np.array(valid_landmarks_selected_world)

        # scale_fixed = 0.993134671601228
        scale_fixed = 1.0
        s, ret_R, ret_t = self.kabschUmeyamaGivenScale(
            valid_landmarks_selected_world, valid_landmarks_selected_model, scale_fixed
        )

        landmarks2d = orig_opdict["landmarks2d"].detach().cpu().numpy()

        landmark2d = landmarks2d[0]

        landmark2d[..., 0] = landmark2d[..., 0] * w / 2 + w / 2
        landmark2d[..., 1] = landmark2d[..., 1] * h / 2 + h / 2

        mouth_state, dist = self.is_mouth_open(landmark2d)

        if visualize:

            kpts = landmark2d.copy()
            radius = max(int(min(viz_image.shape[0], viz_image.shape[1]) / 200), 1)
            for i in range(kpts.shape[0]):
                st = kpts[i, :2]
                c = (0, 0, 255)
                viz_image = cv2.circle(
                    viz_image, (int(st[0]), int(st[1])), radius, c, radius * 2
                )

            viz_image = cv2.rectangle(
                viz_image,
                (int(src_pts[0][0]), int(src_pts[0][1])),
                (int(src_pts[2][0]), int(src_pts[1][1])),
                (0, 255, 0),
                radius * 2,
            )

            # allow window size to be adjusted
            # cv2.namedWindow("viz_image", cv2.WINDOW_NORMAL)
            # cv2.imshow("viz_image",viz_image)
            # cv2.waitKey(10)

        if RECORD_GOAL_POSE:

            forque_pose = np.array(
                [
                    [-0.99991264, -0.01268631, -0.00371056, 0.04003387],
                    [0.01267759, -0.99991685, 0.00236201, 0.06337766],
                    [-0.00374022, 0.00231477, 0.99999033, 0.31434989],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            np.save(
                "/home/rkjenamani/bite_transfer_ws/src/head_perception/"
                + UTENSIL
                + "_forque_pose.npy",
                forque_pose,
            )

            print(
                "landmarks_selected_model[:,:,np.newaxis].shape: ",
                landmarks_selected_model[:, :, np.newaxis].shape,
            )
            print("ret_R.shape: ", ret_R.shape)
            landmarks_selected_model_camera_frame = ret_t.reshape(3, 1) + s * (
                ret_R @ landmarks_selected_model[:, :, np.newaxis]
            )
            landmarks_selected_model_camera_frame = np.squeeze(
                landmarks_selected_model_camera_frame
            )

            np.save(
                "/home/rkjenamani/bite_transfer_ws/src/head_perception/"
                + UTENSIL
                + "_head_points.npy",
                landmarks_selected_model_camera_frame,
            )

            neck_point_model_camera_frame = ret_t.reshape(3, 1) + s * (
                ret_R @ neck_landmark_model.reshape(3, 1)
            )

            landmarks_selected_model_world_frame = np.squeeze(
                base_to_camera[:3, 3].reshape(3, 1)
                + base_to_camera[:3, :3]
                @ landmarks_selected_model_camera_frame[:, :, np.newaxis]
            )
            neck_point_model_world_frame = (
                base_to_camera[:3, 3].reshape(3, 1)
                + base_to_camera[:3, :3] @ neck_point_model_camera_frame
            ).reshape(1, 3)

            print(
                "landmarks_selected_model_world_frame.shape: ",
                landmarks_selected_model_world_frame.shape,
            )
            print(
                "neck_point_model_world_frame.shape: ",
                neck_point_model_world_frame.shape,
            )

            visualization_points_world_frame = np.concatenate(
                (landmarks_selected_model_world_frame, neck_point_model_world_frame),
                axis=0,
            )

            filtering_points_world_frame = (
                landmarks_selected_model_world_frame - neck_point_model_world_frame
            )

            # np.save('/home/emprise/feeding_ws/src/head_perception/filtering_points.npy',filtering_points_world_frame)

            R, rmsd = Rotation.align_vectors(
                filtering_points_world_frame, self.fixed_model_head_points
            )
            print("Rotation estimation error: ", rmsd)

            neck_frame = np.zeros((4, 4))
            neck_frame[:3, :3] = R.as_matrix()
            neck_frame[:3, 3] = neck_point_model_world_frame.reshape(1, 3)
            neck_frame[3, 3] = 1

            np.save(
                "/home/rkjenamani/bite_transfer_ws/src/head_perception/"
                + UTENSIL
                + "_reference_neck_frame.npy",
                neck_frame,
            )

            print("Press [ENTER] to update saved data... ")
            lol = input()

            landmark2d = None
            landmarks3d = None

            return (
                landmark2d,
                landmarks3d,
                viz_image,
                mouth_state,
                average_head_point,
                forque_target_pose,
                neck_frame,
                neck_frame,
            )

        else:

            current_head_points = ret_t.reshape(3, 1) + s * (
                ret_R @ landmarks_selected_model[:, :, np.newaxis]
            )
            current_head_points = np.squeeze(current_head_points)

            average_head_point = np.mean(current_head_points, axis=0)

            current_neck_point = ret_t.reshape(3, 1) + s * (
                ret_R @ neck_landmark_model.reshape(3, 1)
            )

            current_face_points = ret_t.reshape(3, 1) + s * (
                ret_R @ face_landmark_model[:, :, np.newaxis]
            )
            current_face_points = np.squeeze(current_face_points)
            # print("current_face_points.shape: ",current_face_points.shape) # (68, 3)

            s, R, t = self.kabschUmeyamaGivenScale(
                current_head_points, self.reference_head_points, 1.0
            )

            trans = np.zeros((4, 4))
            trans[:3, :3] = R
            trans[:3, 3] = t.reshape(1, 3)
            trans[3, 3] = 1

            landmarks3d = current_face_points

            current_head_points_world_frame = np.squeeze(
                base_to_camera[:3, 3].reshape(3, 1)
                + base_to_camera[:3, :3] @ current_head_points[:, :, np.newaxis]
            )
            current_neck_point_world_frame = (
                base_to_camera[:3, 3].reshape(3, 1)
                + base_to_camera[:3, :3] @ current_neck_point
            ).reshape(1, 3)
            current_face_points_world_frame = np.squeeze(
                base_to_camera[:3, 3].reshape(3, 1)
                + base_to_camera[:3, :3] @ current_face_points[:, :, np.newaxis]
            )

            filtering_points_world_frame = (
                current_head_points_world_frame - current_neck_point_world_frame
            )

            R, rmsd = Rotation.align_vectors(
                filtering_points_world_frame, self.fixed_model_head_points
            )
            if debug_print:
                print("Rotation estimation error: ", rmsd)

            neck_frame = np.zeros((4, 4))
            neck_frame[:3, :3] = R.as_matrix()
            neck_frame[:3, 3] = current_neck_point_world_frame.reshape(1, 3)
            neck_frame[3, 3] = 1

            neck_flexion, neck_rotation, neck_lateral_flexion = R.from_matrix(
                neck_frame[:3, :3]
            ).as_euler("xyz")
            (
                reference_neck_flexion,
                reference_neck_rotation,
                reference_neck_lateral_flexion,
            ) = R.from_matrix(self.reference_neck_frame[:3, :3]).as_euler("xyz")

            translation_from_reference = (
                neck_frame[:3, 3] - self.reference_neck_frame[:3, 3]
            ).reshape(
                3,
            )
            rotation_from_reference = np.array(
                [
                    neck_flexion - reference_neck_flexion,
                    neck_rotation - reference_neck_rotation,
                    neck_lateral_flexion - reference_neck_lateral_flexion,
                ]
            )

            # Rajat ToDo: Bring noise search back
            # if self.last_trans is not None:
            #     if (np.any(np.abs(translation_from_reference) > self.max_distance_threshold)
            #          or np.any( np.abs(rotation_from_reference) > self.max_rotation_threshold)):
            #         # print("Noisy reading!")

            #         self.last_landmark2d = landmark2d
            #         self.last_landmarks3d = landmarks3d
            #         self.last_viz_image = viz_image

            #         return landmark2d, landmarks3d, viz_image, mouth_state, average_head_point, forque_target_pose, self.reference_neck_frame, neck_frame

            if self.last_trans is not None:
                if debug_print:
                    print("Unfiltered Movement: ")
                    print(
                        "[x, y, z]: ",
                        (trans[:3, 3] - self.last_trans[:3, 3]).reshape(
                            3,
                        ),
                    )

            self.trans_perception_buffer.append(trans)
            self.neck_frame_perception_buffer.append(neck_frame)

            if len(self.trans_perception_buffer) > 10:
                self.trans_perception_buffer.pop(0)
                self.neck_frame_perception_buffer.pop(0)
                # print("Deleting..")

            trans_std = np.std(np.array(self.trans_perception_buffer), axis=0)
            if debug_print:
                print("trans_std: ", trans_std)

            if (
                np.abs(trans_std[0, 3]) < 0.005
                and np.abs(trans_std[1, 3]) < 0.005
                and np.abs(trans_std[2, 3]) < 0.005
            ):
                # if trans_std < 0.003:
                trans = np.mean(np.array(self.trans_perception_buffer), axis=0)
                neck_frame = np.mean(
                    np.array(self.neck_frame_perception_buffer), axis=0
                )

            if self.last_trans is not None:
                if debug_print:
                    print("Filtered Movement: ")
                    print(
                        "[x, y, z]: ",
                        (trans[:3, 3] - self.last_trans[:3, 3]).reshape(
                            3,
                        ),
                    )

            if self.last_trans is not None:

                if (
                    np.abs(trans[0, 3] - self.last_trans[0, 3]) < 0.005
                    and np.abs(trans[1, 3] - self.last_trans[1, 3]) < 0.005
                    and np.abs(trans[2, 3] - self.last_trans[2, 3]) < 0.005
                ):
                    # print("Not updating forque target pose")
                    self.last_landmark2d = landmark2d
                    self.last_landmarks3d = landmarks3d
                    self.last_viz_image = viz_image

                    return (
                        landmark2d,
                        landmarks3d,
                        viz_image,
                        mouth_state,
                        average_head_point,
                        self.last_forque_target_pose,
                        self.visualization_points_world_frame,
                        self.reference_neck_frame,
                        self.last_neck_frame,
                    )

            self.last_trans = trans
            self.last_neck_frame = neck_frame

            forque_target_pose = trans @ self.reference_forque_pose

            forque_target_pose = base_to_camera @ forque_target_pose

            z_vector = np.array([0, 0, 1]).reshape(1, 3) @ forque_target_pose[:3, :3]
            z_vector = z_vector.reshape(
                3,
            )
            # print("z_vector: ", z_vector)
            z_angle = np.arctan2(z_vector[2], np.linalg.norm(z_vector[:2]))

            # rotate mouth frame by z_angle along x axis
            forque_target_pose = np.matmul(
                forque_target_pose,
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(z_angle), -np.sin(z_angle), 0],
                        [0, np.sin(z_angle), np.cos(z_angle), 0],
                        [0, 0, 0, 1],
                    ]
                ),
            )

            visualization_points_world_frame = np.concatenate(
                (
                    current_head_points_world_frame,
                    current_neck_point_world_frame,
                    current_face_points_world_frame,
                ),
                axis=0,
            )

            self.last_forque_target_pose = forque_target_pose
            self.last_landmark2d = landmark2d
            self.last_landmarks3d = landmarks3d
            self.last_viz_image = viz_image
            self.visualization_points_world_frame = visualization_points_world_frame

            return (
                landmark2d,
                landmarks3d,
                viz_image,
                mouth_state,
                average_head_point,
                forque_target_pose,
                visualization_points_world_frame,
                self.reference_neck_frame,
                neck_frame,
            )

    def is_mouth_open(self, keypoints):

        if len(keypoints) != 68:
            print("Not enough keypoints : ", np.array(keypoints).shape[0])
            return

        # print("Top Lower Lip: ", keypoints[66])
        # print("Bottom Lower Lip: ", keypoints[57])

        # print("Top Upper Lip: ", keypoints[51])
        # print("Bottom Upper Lip: ", keypoints[62])

        lipDist = np.sqrt(
            (keypoints[66][0] - keypoints[62][0]) ** 2
            + (keypoints[66][1] - keypoints[62][1]) ** 2
        )

        lipThickness = float(
            np.sqrt(
                (keypoints[51][0] - keypoints[62][0]) ** 2
                + (keypoints[51][1] - keypoints[62][1]) ** 2
            )
            / 2
        ) + float(
            np.sqrt(
                (keypoints[57][0] - keypoints[66][0]) ** 2
                + (keypoints[57][1] - keypoints[66][1]) ** 2
            )
            / 2
        )

        # print("lipDist: ",lipDist)
        # print("lipThickness: ",lipThickness)

        if lipDist >= 1.5 * lipThickness:
            return True, lipDist / lipThickness
        else:
            return False, lipDist / lipThickness

    def bbox2Point(self, left, right, top, bottom, type="bbox"):
        """Bbox from detector and landmarks are different."""
        if type == "kpt68":
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array(
                [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0]
            )
        elif type == "bbox":
            old_size = (right - left + bottom - top) / 2
            center = np.array(
                [
                    right - (right - left) / 2.0,
                    bottom - (bottom - top) / 2.0 + old_size * 0.12,
                ]
            )
        else:
            raise NotImplementedError
        return old_size, center

    def getInputImage(
        self, image, iscrop=True, crop_size=224, scale=1.25, debug_print=False
    ):

        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape
        if iscrop:
            bbox, bbox_type = self.face_detector_model.run(image)
            if len(bbox) < 4:
                print("no face detected! run original image")
                left = 0
                right = h - 1
                top = 0
                bottom = w - 1
            else:
                print("face detected!")
                left = bbox[0]
                right = bbox[2]
                top = bbox[1]
                bottom = bbox[3]

            old_size, center = self.bbox2Point(left, right, top, bottom, type=bbox_type)
            size = int(old_size * scale)
            src_pts = np.array(
                [
                    [center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2],
                ]
            )
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])

        tform = estimate_transform("similarity", src_pts, DST_PTS)

        image = image / 255.0
        dst_image = warp(image, tform.inverse, output_shape=(224, 224))
        dst_image = dst_image.transpose(2, 0, 1)
        # print('img shape', img.shape)

        return {
            "image": torch.tensor(dst_image).float(),
            "tform": torch.tensor(tform.params).float(),
            "original_image": torch.tensor(image.transpose(2, 0, 1)).float(),
            "src_pts": torch.tensor(src_pts).float(),
        }

    def kabschUmeyamaGivenScale(self, A, B, scale):

        assert A.shape == B.shape

        # Calculate scaled B
        scaled_B = B * scale

        # Calculate translation using centroids
        A_centered = A - np.mean(A, axis=0)
        B_centered = scaled_B - np.mean(scaled_B, axis=0)

        # Calculate rotation using scipy

        R, rmsd = Rotation.align_vectors(A_centered, B_centered)

        t = np.mean(A, axis=0) - R.as_matrix() @ np.mean(scaled_B, axis=0)

        return scale, R.as_matrix(), t

    def kabschUmeyama(self, A, B):
        assert A.shape == B.shape
        n, m = A.shape

        EA = np.mean(A, axis=0)
        EB = np.mean(B, axis=0)
        VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

        H = ((A - EA).T @ (B - EB)) / n
        U, D, VT = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
        S = np.diag([1] * (m - 1) + [d])

        R = U @ S @ VT
        c = VarA / np.trace(np.diag(D) @ S)
        t = EA - c * R @ EB

        return c, R, t

    def pixel2World(self, camera_info, image_x, image_y, depth_image):

        # print("(image_y,image_x): ",image_y,image_x)
        # print("depth image: ", depth_image.shape[0], depth_image.shape[1])

        if image_y >= depth_image.shape[0] or image_x >= depth_image.shape[1]:
            return False, None

        depth = depth_image[image_y, image_x]

        if math.isnan(depth) or depth < 0.05 or depth > 1.0:

            depth = []
            for i in range(-2, 2):
                for j in range(-2, 2):
                    if (
                        image_y + i >= depth_image.shape[0]
                        or image_x + j >= depth_image.shape[1]
                    ):
                        return False, None
                    pixel_depth = depth_image[image_y + i, image_x + j]
                    if not (
                        math.isnan(pixel_depth)
                        or pixel_depth < 50
                        or pixel_depth > 1000
                    ):
                        depth += [pixel_depth]

            if len(depth) == 0:
                return False, None

            depth = np.mean(np.array(depth))

        depth = depth / 1000.0  # Convert from mm to m

        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]

        # Convert to world space
        world_x = (depth / fx) * (image_x - cx)
        world_y = (depth / fy) * (image_y - cy)
        world_z = depth

        return True, (world_x, world_y, world_z)

    def world2Pixel(self, camera_info, world_x, world_y, world_z):

        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]

        image_x = world_x * (fx / world_z) + cx
        image_y = world_y * (fy / world_z) + cy

        return image_x, image_y


if __name__ == "__main__":

    head_perception = HeadPerception()

    time.sleep(2)  # Allow buffers to fillup
