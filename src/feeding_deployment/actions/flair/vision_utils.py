import cv2
import numpy as np
import math
import cmath

from scipy.spatial import Delaunay

import torch
from torchvision.transforms import ToTensor

from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

# IS_NOODLE = True


def resize_to_square(image, output_size=None):
    if image is None:
        raise ValueError("Could not read the image.")
    height, width = image.shape[:2]
    # Determine the smaller dimension
    min_dim = min(width, height)
    # Calculate crop coordinates
    start_x = width // 2 - min_dim // 2
    start_y = height // 2 - min_dim // 2
    # Crop to a square
    square_image = image[start_y:start_y + min_dim, start_x:start_x + min_dim]
    # Resize if an output size is specified
    if output_size is not None:
        square_image = cv2.resize(square_image, (output_size, output_size))
    return square_image

def calculate_heatmap_density(heatmap):
    return heatmap.max()/255

def calculate_heatmap_entropy(heatmap):
    # Normalize the heatmap to sum to 1 (like a probability distribution)
    heatmap_normalized = heatmap / np.sum(heatmap)
    # Flatten the heatmap
    heatmap_flattened = heatmap_normalized.flatten()
    # Calculate the entropy
    heatmap_entropy = entropy(heatmap_flattened)
    return heatmap_entropy

def efficient_sam_box_prompt_segment(image, pts_sampled, model):
    bbox = torch.reshape(torch.tensor(pts_sampled), [1, 1, 2, 2])
    bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].cuda(),
        bbox.cuda(),
        bbox_labels.cuda(),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
            curr_predicted_iou > max_predicted_iou
            or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou

def outpaint_masks(target_mask, other_masks):
    for mask in other_masks:
        target_mask = cv2.bitwise_and(cv2.bitwise_not(mask), target_mask)
    #    ys,xs = np.where(mask > 0)
    #    target_mask[ys,xs] = 0
    return target_mask

def detect_plate(img, multiplier = 1.7):
    H,W,C = img.shape
    print("Detected plate H,W,C", H,W,C)
    img_orig = img.copy()
    img = cv2.resize(img, (W//2, H//2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred,
                       cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                   param2 = 30, minRadius = 50, maxRadius = 200)
    plate_mask = np.zeros((H,W)).astype(np.uint8)
    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            plate_small_mask = plate_mask.copy()
            cv2.circle(plate_small_mask, (a*2, b*2), int(r*multiplier), (255,255,255), -1)
            break
        return plate_small_mask

def detect_blue(image):
    lower_blue = np.array([45,30,15]) 
    upper_blue = np.array([255,150,80]) 
    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask = cv2.inRange(image, lower_blue, upper_blue)
    return mask

def nearest_neighbor(points, target_point):
    points = np.array(points)
    neigh = NearestNeighbors()
    neigh.fit(points)
    dists, idxs = neigh.kneighbors(np.array(target_point).reshape(1,-1), 1, return_distance=True)
    return points[idxs.squeeze()]

def nearest_point_to_mask(point, mask):

    print("Point: ", point)
    hull = detect_convex_hull(mask)
    num_samples = 100
    dense_hull_points_2d = np.vstack([np.linspace(hull[i], hull[i+1], num_samples) for i in range(len(hull)-1)])
    
    filling_to_push_target_2d = None
    dists = np.linalg.norm(dense_hull_points_2d - point, axis=1)
    filling_to_push_target_2d = dense_hull_points_2d[np.argmin(dists)]

    return filling_to_push_target_2d

def expanded_detect_furthest_unobstructed_boundary_point(point, mask, obstructions_masks_combined):

    # negation of mask
    negated_mask = cv2.bitwise_not(mask)
    
    # find furthest point to point that is not obstructed
    hull = detect_convex_hull(mask)
    num_samples = 100
    dense_hull_points_2d = np.vstack([np.linspace(hull[i], hull[i+1], num_samples) for i in range(len(hull)-1)])

    furthest = None
    dists = np.linalg.norm(dense_hull_points_2d - point, axis=1)

    rgb_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    rgb_vis[obstructions_masks_combined > 0] = (100,100,100)
    rgb_vis = visualize_keypoints(rgb_vis, [point], color=(0,0,255), radius=5)

    # in order of increasing distance, find the first point that is not obstructed (line connecting point and that point does not intersect with any obstruction mask)
    for i in np.argsort(dists)[::-1]:
        furthest = dense_hull_points_2d[i]
        line = np.linspace(point, furthest, 100)
        line = line.astype(int)
        line_mask = np.zeros_like(mask)
        for j in range(len(line)):
            line_mask[line[j][1], line[j][0]] = 1
        
        # intersection with any obstruction mask
        line_filling_mask = cv2.bitwise_and(line_mask, obstructions_masks_combined)

        # intersection with holes in mask
        line_holes_mask = cv2.bitwise_and(line_mask, negated_mask)

        if np.count_nonzero(line_filling_mask) == 0:
            collision_free_line_found = True
            furthest = furthest.astype(int)
            rgb_vis = visualize_push(rgb_vis, furthest, point, radius=3, color=(0,255,0))
            rgb_vis = visualize_keypoints(rgb_vis, [furthest], color=(255,0,0), radius=5)
            rgb_vis = cv2.putText(rgb_vis, "Free furthest line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            return furthest, rgb_vis

    rgb_vis = cv2.putText(rgb_vis, "No free furthest line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return None, rgb_vis

def detect_furthest_unobstructed_boundary_point(point, mask, obstruction_masks, continuous=False):

    # negation of mask
    negated_mask = cv2.bitwise_not(mask)

    # combined visualization of obstruction_masks
    obstruction_masks_combined = np.zeros_like(mask)
    for obstruction_mask in obstruction_masks:
        obstruction_masks_combined = cv2.bitwise_or(obstruction_masks_combined, obstruction_mask)

    # enlarge obstruction masks by 50 pixels
    kernel = np.ones((20,20), np.uint8)
    obstruction_masks_combined = cv2.dilate(obstruction_masks_combined, kernel, iterations=1)
    
    # find furthest point to point that is not obstructed
    hull = detect_convex_hull(mask)
    num_samples = 100
    dense_hull_points_2d = np.vstack([np.linspace(hull[i], hull[i+1], num_samples) for i in range(len(hull)-1)])

    furthest = None
    actual_furthest = None
    dists = np.linalg.norm(dense_hull_points_2d - point, axis=1)

    best_furthest_yet = None
    best_furthest_count = np.inf
    collision_free_line_found = False

    # in order of increasing distance, find the first point that is not obstructed (line connecting point and that point does not intersect with any obstruction mask)
    for i in np.argsort(dists)[::-1]:
        furthest = dense_hull_points_2d[i]
        line = np.linspace(point, furthest, 100)
        line = line.astype(int)
        line_mask = np.zeros_like(mask)
        for j in range(len(line)):
            line_mask[line[j][1], line[j][0]] = 1
        
        # intersection with any obstruction mask
        line_filling_mask = cv2.bitwise_and(line_mask, obstruction_masks_combined)

        # intersection with holes in mask
        line_holes_mask = cv2.bitwise_and(line_mask, negated_mask)

        if np.count_nonzero(line_filling_mask) == 0:
            collision_free_line_found = True
            if continuous:
                if np.count_nonzero(line_holes_mask) == 0:
                    best_furthest_yet = None
                    break
                else:
                    if best_furthest_yet is None:
                        best_furthest_yet = furthest
                        best_furthest_count = np.count_nonzero(line_holes_mask)
                    elif np.count_nonzero(line_holes_mask) < best_furthest_count:
                        best_furthest_yet = furthest
                        best_furthest_count = np.count_nonzero(line_holes_mask)
            else:
                actual_furthest = furthest
                break

    if actual_furthest is None:
        # if no unobstructed point found, return furthest point
        actual_furthest = dense_hull_points_2d[np.argsort(dists)[::-1][0]]
    actual_furthest = actual_furthest.astype(int)
    
    if best_furthest_yet is not None:
        furthest = best_furthest_yet

    rgb_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # for obstruction_mask in obstruction_masks:
    #     rgb_vis[obstruction_mask > 0] = (100,100,100)
    rgb_vis[obstruction_masks_combined > 0] = (100,100,100)
    rgb_vis = visualize_push(rgb_vis, actual_furthest, point, radius=3, color=(0,255,0))
    rgb_vis = visualize_keypoints(rgb_vis, [point], color=(0,0,255), radius=5)
    rgb_vis = visualize_keypoints(rgb_vis, [actual_furthest], color=(255,0,0), radius=5)
    if collision_free_line_found:
        rgb_vis = cv2.putText(rgb_vis, "Free grouping line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        rgb_vis = cv2.putText(rgb_vis, "Grouping line in collision", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return actual_furthest, rgb_vis, collision_free_line_found

# def nearest_point_to_mask(points, mask):
#     ys, xs = np.where(mask > 0)
#     if not len(ys):
#         return px
#     mask_pixels = np.vstack((xs,ys)).T
#     neigh = NearestNeighbors()
#     neigh.fit(mask_pixels)
#     dists, idxs = neigh.kneighbors(np.array(points), 1, return_distance=True)
#     min_dist_idx = dists.argmin()
#     return points[min_dist_idx]

def proj_pix2mask(px, mask):
    ys, xs = np.where(mask > 0)
    if not len(ys):
        return px
    mask_pixels = np.vstack((xs,ys)).T
    neigh = NearestNeighbors()
    neigh.fit(mask_pixels)
    dists, idxs = neigh.kneighbors(np.array(px).reshape(1,-1), 1, return_distance=True)
    projected_px = mask_pixels[idxs.squeeze()]
    return projected_px

def get_density_heatmap(mask, kernel, visualize=False):
    # Find desired pixel in densest_masked_noodles
    kernel = np.ones(kernel, np.float32)/kernel[0]**2
    dst = cv2.filter2D(mask, -1, kernel)
    if visualize:
        vis = cv2.applyColorMap(dst, cv2.COLORMAP_JET)
        cv2.imshow('vis', vis)
        cv2.waitKey(0)
    return dst

def detect_densest(mask, kernel=(60,60), is_noodle=False):
    # if is_noodle:
    #     heatmap = get_density_heatmap(mask, (60,60))
    #     heatmap_for_densest = get_density_heatmap(mask, (30,30), visualize=False)
    #     pred_y, pred_x = np.unravel_index(heatmap_for_densest.argmax(), heatmap_for_densest.shape)
    #     densest = proj_pix2mask((pred_x, pred_y), mask)
    #     densest = (int(densest[0]), int(densest[1]))
    #     return densest, heatmap
    # else:
    #     kernel = (120,120)
    #     heatmap = get_density_heatmap(mask, kernel)
    #     pred_y, pred_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    #     densest = proj_pix2mask((pred_x, pred_y), mask)
    #     densest = (int(densest[0]), int(densest[1]))
    #     return densest, heatmap
    kernel = (120,120)
    heatmap = get_density_heatmap(mask, kernel)
    pred_y, pred_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    densest = proj_pix2mask((pred_x, pred_y), mask)
    densest = (int(densest[0]), int(densest[1]))
    return densest, heatmap

def new_detect_densest(mask, obstructions_masks_combined):  
    kernel = (120,120)
    heatmap = get_density_heatmap(mask, kernel)
    densest_mask = cv2.bitwise_and(mask, cv2.bitwise_not(obstructions_masks_combined))
    heatmap_for_densest = get_density_heatmap(densest_mask, kernel)
    pred_y, pred_x = np.unravel_index(heatmap_for_densest.argmax(), heatmap_for_densest.shape)
    densest = proj_pix2mask((pred_x, pred_y), densest_mask)
    densest = (int(densest[0]), int(densest[1]))
    return densest, heatmap
    

def fill_enclosing_polygon(binary_mask):
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a new mask with the same dimensions as the input
    filled_mask = np.zeros_like(binary_mask)
    if len(contours) > 0:
        # Get the largest contour (assuming it's the enclosing polygon)
        largest_contour = max(contours, key=cv2.contourArea)
        # Fit a polygon to the largest contour
        epsilon = 0.03 * cv2.arcLength(largest_contour, True)
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        # Fill the polygon with white
        cv2.fillPoly(filled_mask, [approx_polygon], 255)
    return filled_mask

def detect_convex_hull(mask):
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    #cont = np.vstack(([c for c in contours if cv2.contourArea(c) > 100]))
    cont = np.vstack((contours))
    hull = cv2.convexHull(cont)
    hull = hull.reshape(len(hull), 2).astype(int)
    return hull

def detect_sparsest(mask, densest):
    hull = detect_convex_hull(mask)
    neigh = NearestNeighbors()
    neigh.fit(hull)
    dists, idxs = neigh.kneighbors(np.array(densest).reshape(1,-1), len(hull), return_distance=True)
    furthest = hull[idxs.squeeze()[-1]]
    furthest = proj_pix2mask(np.array(furthest), mask)
    furthest = (int(furthest[0]), int(furthest[1]))
    return furthest, hull

def detect_fillings_in_mask(fillings, filling_centroids, mask):
    hull = detect_convex_hull(mask)
    delaunay = Delaunay(hull)
    fillings_in_hull_idx = delaunay.find_simplex(filling_centroids) >= 0
    fillings_in_hull_centroids = np.array(filling_centroids)[fillings_in_hull_idx]
    fillings_in_hull = np.array(fillings)[fillings_in_hull_idx]
    return fillings_in_hull, fillings_in_hull_centroids

def detect_filling_push_noodles(dense_center, furthest, fillings, hull, vis = None):
    dense_center = np.array(dense_center)
    furthest = np.array(furthest)
    delaunay = Delaunay(hull)
    fillings_in_hull = delaunay.find_simplex(fillings) >= 0
    fillings_in_hull = np.array(fillings)[fillings_in_hull]

    if len(fillings_in_hull) == 0:
        print("No fillings in hull. Returning...")
        return None, None

    # print('fillings_in_hull', fillings_in_hull)

    filling_to_push = None
    min_dist = float('inf')
    num_samples = 100
    dense_line_segment_2d = np.linspace(dense_center, furthest, num_samples)
    for filling in fillings_in_hull:
        dists = np.linalg.norm(dense_line_segment_2d - filling, axis=1)
        if np.min(dists) < min_dist:
            min_dist = np.min(dists)
            filling_to_push = filling

    num_samples = 100
    dense_hull_points_2d = np.vstack([np.linspace(hull[i], hull[i+1], num_samples) for i in range(len(hull)-1)])

    cos_angles = []
    for dense_hull_point_2d in dense_hull_points_2d:
        line_1 = furthest - dense_center
        line_2 = filling_to_push - dense_hull_point_2d
        cos_angle = np.dot(line_1, line_2) / (np.linalg.norm(line_1) * np.linalg.norm(line_2))
        cos_angles.append(cos_angle)
    cos_angles = np.array(cos_angles)
    cos_angles = np.abs(cos_angles)
    print('cos_angles', cos_angles)
    # select the points with cos less than 0.1
    dense_hull_points_2d = dense_hull_points_2d[cos_angles < 0.1]
    # find nearest point to filling_to_push
    neigh = NearestNeighbors()
    neigh.fit(dense_hull_points_2d)
    match_idxs = neigh.kneighbors([filling_to_push], len(dense_hull_points_2d), return_distance=False)
    filling_to_push_target_2d = dense_hull_points_2d[match_idxs.squeeze().tolist()[0]]
    filling_to_push_target_2d = filling_to_push_target_2d.astype(int)

    # visualize filling_to_push_target_2d and filling_to_push on mask
    if vis is not None:

        # visualize dense_hull_points_2d
        for dense_hull_point_2d in dense_hull_points_2d:
            vis = cv2.circle(vis, tuple(dense_hull_point_2d.astype(int)), 5, (255, 255, 0), -1)

        vis = cv2.circle(vis, tuple(filling_to_push_target_2d), 5, (0, 255, 0), -1)
        vis = cv2.circle(vis, tuple(filling_to_push), 5, (0, 0, 255), -1)
        vis = cv2.circle(vis, tuple(dense_center), 5, (255, 0, 0), -1)
        vis = cv2.circle(vis, tuple(furthest), 5, (255, 0, 0), -1)
        cv2.imshow('vis', vis)
        cv2.waitKey(0)
        input("Press enter to continue")
        cv2.destroyAllWindows()

    # OFFSET CORRECTION
    offset = filling_to_push_target_2d - filling_to_push
    offset = 35*(offset / np.linalg.norm(offset))
    filling_to_push = np.array(filling_to_push - offset).astype(int)

    target_offset = 50*(offset / np.linalg.norm(offset))
    
    filling_to_push_target_2d = np.array(filling_to_push_target_2d + target_offset).astype(int)

    print("filling_to_push_target_2d", filling_to_push_target_2d)
    return filling_to_push, filling_to_push_target_2d

def detect_filling_push_semisolid(fillings, hull, vis = None):
    
    delaunay = Delaunay(hull)
    fillings_in_hull = delaunay.find_simplex(fillings) >= 0
    fillings_in_hull = np.array(fillings)[fillings_in_hull]

    if len(fillings_in_hull) == 0:
        print("No fillings in hull. Returning...")
        return None, None, None

    num_samples = 100
    dense_hull_points_2d = np.vstack([np.linspace(hull[i], hull[i+1], num_samples) for i in range(len(hull)-1)])

    filling_to_push = None # Find the filling that is closest to the boundary of the hull
    filling_to_push_target_2d = None

    min_dist = float('inf')
    for filling in fillings_in_hull:
        dists = np.linalg.norm(dense_hull_points_2d - filling, axis=1)
        if np.min(dists) < min_dist:
            min_dist = np.min(dists)
            filling_to_push = filling
            filling_to_push_target_2d = dense_hull_points_2d[np.argmin(dists)]

    if vis is not None:
        # visualize dense_hull_points_2d
        for dense_hull_point_2d in dense_hull_points_2d:
            vis = cv2.circle(vis, tuple(dense_hull_point_2d.astype(int)), 5, (255, 255, 0), -1)
        
        vis = cv2.circle(vis, tuple(filling_to_push_target_2d.astype(int)), 5, (0, 255, 0), -1)
        vis = cv2.circle(vis, tuple(filling_to_push.astype(int)), 5, (0, 0, 255), -1)

    filling_pos = filling_to_push.copy()

    # OFFSET CORRECTION
    offset = filling_to_push_target_2d - filling_to_push
    offset = 35*(offset / np.linalg.norm(offset))
    filling_to_push = np.array(filling_to_push - offset).astype(int)

    target_offset = 80*(offset / np.linalg.norm(offset))
    filling_to_push_target_2d = np.array(filling_to_push_target_2d + target_offset).astype(int)
    print("filling_to_push_target_2d", filling_to_push_target_2d)

    if vis is not None:
        vis = cv2.circle(vis, tuple(filling_to_push_target_2d), 5, (0, 255, 0), -1)
        vis = cv2.circle(vis, tuple(filling_to_push), 5, (0, 0, 255), -1)
        cv2.imshow('vis', vis)
        cv2.waitKey(0)
        input("Press enter to continue")
        cv2.destroyAllWindows()

    return filling_pos, filling_to_push, filling_to_push_target_2d

def detect_centroid(mask):
    cX, cY = 0, 0
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
    centroid = proj_pix2mask(np.array([cX, cY]), mask)
    centroid = (int(centroid[0]), int(centroid[1]))
    return centroid

def detect_angular_bbox(mask):
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box
    else:
        return None

def mask_weight(mask):
    H,W = mask.shape
    return np.count_nonzero(mask)/(W*H)

def cleanup_mask(mask, blur_kernel_size=(5, 5), threshold=127, erosion_size=3):
    """
    Applies low-pass filter, thresholds, and erodes an image mask.

    :param image: Input image mask in grayscale.
    :param blur_kernel_size: Size of the Gaussian blur kernel.
    :param threshold: Threshold value for binary thresholding.
    :param erosion_size: Size of the kernel for erosion.
    :return: Processed image.
    """
    # Apply Gaussian Blur for low-pass filtering
    blurred = cv2.GaussianBlur(mask, blur_kernel_size, 0)
    # Apply thresholding
    _, thresholded = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    # Create erosion kernel
    erosion_kernel = np.ones((erosion_size, erosion_size), np.uint8)
    # Apply erosion
    eroded = cv2.erode(thresholded, erosion_kernel, iterations=1)
    return eroded

def visualize_push(image, start, end, radius=10, color=(255,255,255)):
    vis = cv2.arrowedLine(image.copy(), tuple(start), tuple(end), color, radius)
    return vis

def visualize_skewer(image, center, angle):
    vis = cv2.circle(image.copy(), tuple(center), 5, (255, 0, 0), -1)
    pt = cmath.rect(20, np.pi/2-angle)
    x2 = int(pt.real)
    y2 = int(pt.imag)
    vis = cv2.line(vis, (center[0]-x2,center[1]+y2), (center[0]+x2,center[1]-y2), (255,0,0), 2)
    return vis


def detect_mask_area(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not (len(contours)):
        return 0
    largest_contour = max(contours, key=cv2.contourArea)
    mask_area = cv2.contourArea(largest_contour)
    H,W = mask.shape
    return mask_area/H*W

def visualize_keypoints(image, keypoints, color=(255,255,255), radius=8):
    for k in keypoints:
        cv2.circle(image, tuple(k), radius, color, -1)
    return image

if __name__ == '__main__':
    mask = np.zeros((480,640)).astype(np.uint8)
    mask[255:280] = 255
    cv2.imshow('img', mask)
    cv2.waitKey(0)
    detect_angular_bbox(mask)
