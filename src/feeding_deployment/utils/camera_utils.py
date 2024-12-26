import numpy as np
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class CustomCameraInfo:
    fx: float
    fy: float
    cx: float
    cy: float

def angle_between_pixels(source_px, target_px, image_width, image_height, orientation_symmetry = True):
    def angle_between(p1, p2):
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    if orientation_symmetry and source_px[1] > target_px[1]:
        source_px, target_px = target_px, source_px
    source_px_cartesian = np.array([source_px[0], image_height-source_px[1]])
    target_px_cartesian = np.array([target_px[0], image_height-target_px[1]])
    angle = angle_between(np.array([-image_width,0]), source_px_cartesian-target_px_cartesian)
    robot_angle_offset = -90
    return angle + robot_angle_offset


def pixel2World(camera_info, image_x, image_y, depth_image, box_width = 2):

    # print("(image_y,image_x): ",image_y,image_x)
    # print("depth image: ", depth_image.shape[0], depth_image.shape[1])

    if image_y >= depth_image.shape[0] or image_x >= depth_image.shape[1]:
        return False, None

    depth = depth_image[image_y, image_x]

    if math.isnan(depth) or depth < 0.05 or depth > 1.0:

        depth = []
        for i in range(-box_width,box_width):
            for j in range(-box_width,box_width):
                if image_y+i >= depth_image.shape[0] or image_x+j >= depth_image.shape[1]:
                    return False, None
                pixel_depth = depth_image[image_y+i, image_x+j]
                if not (math.isnan(pixel_depth) or pixel_depth < 50 or pixel_depth > 1000):
                    depth += [pixel_depth]

        if len(depth) == 0:
            return False, None

        depth = np.mean(np.array(depth))

    depth = depth/1000.0 # Convert from mm to m

    fx = camera_info.fx
    fy = camera_info.fy
    cx = camera_info.cx
    cy = camera_info.cy 

    # Convert to world space
    world_x = (depth / fx) * (image_x - cx)
    world_y = (depth / fy) * (image_y - cy)
    world_z = depth

    return True, np.array([world_x, world_y, world_z])

def world2Pixel(camera_info, world_x, world_y, world_z):

    fx = camera_info.fx
    fy = camera_info.fy
    cx = camera_info.cx
    cy = camera_info.cy  

    image_x = world_x * (fx / world_z) + cx
    image_y = world_y * (fy / world_z) + cy

    return image_x, image_y