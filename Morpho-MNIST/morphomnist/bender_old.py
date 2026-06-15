import numpy as np
from matplotlib import pyplot as plt
import glob
from skimage.io import imread, imsave
from skimage.draw import polygon_perimeter, polygon
from skimage.morphology import label, medial_axis, skeletonize
from scipy.ndimage.measurements import center_of_mass
from skimage.draw import bezier_curve
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import warp, SimilarityTransform
from skimage.graph import route_through_array
import cv2
import warnings
import csv
from skimage.util import invert
from skimage import data as skdata
import copy

import contextlib

@contextlib.contextmanager
def set_temp_seed(seed): # set random seed temporarily for a particual section of the code
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class ChooseSkeletonEndpointsException(Exception):
    pass

class BadSkeletonBendException(Exception):
    pass


def closest_point(node, nodes):
    """
        Returns the closest point from `nodes` to `node`
    """
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return nodes[np.argmin(dist_2),:].tolist()


def perpendicular_angle(end1, end2):
    """
        Perpendicular angle between y-axis and the diff of coordinates of endpoints
    """
    angle = np.arctan2((np.array(end1) - np.array(end2))[0], (np.array(end1) - np.array(end2))[1])
    return [angle + np.pi/2.0, angle - np.pi/2.0]


def random_endpts_angle(end1, end2, fixed_angle_seed, debug=False):
    if fixed_angle_seed is None:
        fixed_angle_seed = np.random.randint(1,1000)
    with set_temp_seed(fixed_angle_seed):
        angle = np.random.choice(perpendicular_angle(end1, end2), 1)
    if debug:
        print(f"angle={angle}, fixed_angle_seed={fixed_angle_seed}")
    return angle[0]


def shift_midpoint(end1, end2, angle, dist_shift, size, debug=False):
    init_mid = (np.array(end1) + np.array(end2))/2.0
    rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-1*np.sin(angle), np.cos(angle)]]) # clockwise rotation
    new_mid = np.matmul(rot_mat, np.array([[0], [dist_shift]])) + np.array([[init_mid[0]], [init_mid[1]]])
    new_mid = [int(new_mid[0][0]), int(new_mid[1][0])]
    if debug:
        plt.plot(init_mid[0], init_mid[1], color='g',marker='*', label="original mid-point")
        plt.plot(new_mid[0], new_mid[1], color='r',marker='o', label="shifted-mid-point")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.legend()
        plt.title("Shifting of mid-point")
        plt.show()
    if new_mid[0] < 0:
        new_mid[0] = 0
    if new_mid[0] > size[0]:
        new_mid[0] = size[0]
    if new_mid[1] < 0:
        new_mid[1] = 0
    if new_mid[1] > size[1]:
        new_mid[1] = size[1]
    return new_mid


def remove_path(image, orig_skel, new_skel, debug=False):
    imnew = image.copy()
    path = np.column_stack(np.where(difference(orig_skel, new_skel) > 0)).tolist()
    skel_pts = np.column_stack(np.where(orig_skel > 0)).tolist()
    deleted_wt = 0.
    for pt in np.column_stack(np.where(image > 0)).tolist():
        if closest_point(pt, skel_pts) in path:
            imnew[pt[0], pt[1]] = 0
            deleted_wt += 1
    if debug:
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(imnew)
        plt.title("remove_path")
        plt.show()
        print(f"Num pixels removed for bend: {deleted_wt}")
    
    return imnew
    
def choose_endpts(skeleton, distance, start_endpoints=None):
    assert False, "Use `choose_endpts_using_midpoint` function"
    skel_pts = np.where(skeleton > 0)
    possible_pts = []
    N = len(start_endpoints) if start_endpoints is not None else 100
    for attempt in range(N):
        if start_endpoints is None:
            rand_index = np.random.choice(range(len(skel_pts[0])), 1)
            end1 = [skel_pts[0][rand_index][0], skel_pts[1][rand_index][0]]
        else:
            end1 = start_endpoints[attempt]
        costs = invert(skeleton).astype(np.float) + 1 # use neg costs?
        for i in range(len(skel_pts[0])):
            pt = [skel_pts[0][i], skel_pts[1][i]]
            _, dist = route_through_array(costs, end1, pt, geometric=False)
            if dist == distance:
                possible_pts = possible_pts + [pt]
        if len(possible_pts) > 0:
            break
    if len(possible_pts) == 0:
        raise ChooseSkeletonEndpointsException("Length of `possible_pts` is 0, most likely due to a bad skeleton")
    end2 = possible_pts[np.random.choice(range(len(possible_pts)), 1)[0]]
    return end1, end2


def aux_choose_endpts(skeleton, distance, end1, costs):
    skel_pts = np.where(skeleton > 0)
    rand_idxs = np.random.permutation(len(skel_pts[0]))
    possible_pts = []
    for i in rand_idxs:
        pt = [skel_pts[0][i], skel_pts[1][i]]
        _, dist = route_through_array(costs, end1, pt, geometric=False)
        if dist == distance:
            possible_pts = possible_pts + [pt]
    return possible_pts
    
def choose_endpts_using_midpoint(skeleton, distance, mid_points=None, debug=False):
    """
        Choose two endpoints at a distance of `distance`/2 from one of the mid_points
    """
    skel_pts = np.where(skeleton > 0)
    possible_pts = []
    N = len(mid_points) if mid_points is not None else 100
    for attempt in range(N): # can optimize
        if mid_points is None:
            rand_index = np.random.choice(range(len(skel_pts[0])), 1)
            end1 = [skel_pts[0][rand_index][0], skel_pts[1][rand_index][0]]
        else:
            end1 = mid_points[attempt]
        costs = invert(skeleton).astype(np.float) + 1 # use neg costs?
        ends = aux_choose_endpts(skeleton, distance//2, end1, costs)
        ends.sort()
        if len(ends)==2:
            if debug:
                plt.subplot(1,2,1)
                plt.imshow(skeleton)
                plt.title(f"`choose_endpts_using_midpoint`")
                print(f"{len(ends)} endpoints found at distance {distance//2} from the center(i.e. *)")
                plt.subplot(1,2,2)
                plt.imshow(skeleton)
                plt.plot(end1[1], end1[0], 'r*')
                for pt in ends:
                    plt.plot(pt[1], pt[0], 'g.', alpha=0.8)
                plt.show()
             
            return ends[0], ends[1]
    raise ChooseSkeletonEndpointsException("Unable to find suitable endpoints for any of the sampled midpoints") 


def bend_skel_with_endpts(skel, end1, end2, dist_shift, fixed_angle_seed=None, debug=False):
    angle = random_endpts_angle(end1, end2, fixed_angle_seed, debug=debug)
    mid = shift_midpoint(end1, end2, angle, dist_shift, size=[skel.shape[0], skel.shape[1]], debug=debug)
    # delete the points from the path
    path, _ = route_through_array(1 - skel, end1, end2) # cost+1 ?
    skelnew = skel.copy()
    for pt in path:
        skelnew[pt] = 0
    if debug:
        plt.subplot(1,2,1)
        plt.imshow(skel)
        plt.subplot(1,2,2)
        plt.imshow(skelnew)
        plt.title("bend_skel_with_endpts (before adding the bend)")
        plt.show()
    rr, cc = bezier_curve(end1[0], end1[1], mid[0], mid[1],
                        end2[0], end2[1], 2)
    skelnew[rr, cc] = 1
    
    return skelnew, angle


def bend_skel(skel, distance, dist_shift, mid_points=None, fixed_angle_seed=None, debug=False):
    
#     end1, end2 = choose_endpts(skel, distance, start_endpoints=start_endpoints)
    end1, end2 = choose_endpts_using_midpoint(skel, distance, mid_points=mid_points, debug=debug)
    bent, _ = bend_skel_with_endpts(skel, end1, end2, dist_shift, fixed_angle_seed=fixed_angle_seed, debug=debug)
    if debug:
        plt.subplot(1,2,1)
        plt.imshow(skel)
        plt.subplot(1,2,2)
        plt.imshow(bent)
        plt.title("bend_skel")
        plt.show()
    return bent

def get_skeletonizer_func(which="medial_axis"):
    """
        Pick skeltonization method (medial_axis or skeletonize)
    """
    if which == "medial_axis":
        return medial_axis
    elif which == "skeletonize":
        return skeletonize
    else:
        raise ValueError(f"Wrong value for `which `({which})")

def difference(ima, imb):
    img = ima.copy()
    img[np.where(imb > 0)] = 0
    return img


def create_rectangle(center, height, width, angle, filled, size):
    """
        Creates a rectangular patch around the center
    """
    rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-1*np.sin(angle), np.cos(angle)]])
    corners = np.array([[height/2.0, -1*height/2.0, -1*height/2.0, height/2.0],
                       [width/2.0, width/2.0, -1*width/2.0, -1*width/2.0]])
    corners =  np.transpose(np.matmul(rot_mat, corners) + np.array([[center[0]], [center[1]]])).astype(np.int32)
#     if True:
#         c = np.array([[height/2.0, -1*height/2.0, -1*height/2.0, height/2.0],
#                        [width/2.0, width/2.0, -1*width/2.0, -1*width/2.0]])
#         corners0 =  np.transpose(c + np.array([[center[0]], [center[1]]])).astype(np.int32)
#         print(corners0==corners)
    rect = np.zeros(size, dtype=np.uint8)
    if filled:
        rr, cc = polygon(corners[:,0], corners[:,1], shape=rect.shape)
    else:
        rr, cc = polygon_perimeter(corners[:,0], corners[:,1], shape=rect.shape, clip=True)
    rect[rr, cc] = 1
    return rect

def add_bend(image, orig_skel, bent_skel, skeletonizer, debug=False):
    """
        Adds a bend to the `image`
        `image` - original image
        `orig_skel` - skeleton of the original image
        `bent_skel` - skeleton of the bent image (used to create a bend on the original image)
    """
    imnew = remove_path(image, orig_skel, bent_skel, debug=debug)
    new_section = np.column_stack(np.where(difference(bent_skel, orig_skel) > 0)).tolist()
    old_section = np.column_stack(np.where(difference(orig_skel, bent_skel) > 0)).tolist()
    if debug:
        plt.subplot(1,2,1)
        dummy_img = np.zeros((orig_skel.shape[0], orig_skel.shape[1]))
        dummy_img[np.where(difference(orig_skel, bent_skel) > 0)]=1
        plt.imshow(dummy_img)
        plt.subplot(1,2,2)
        dummy_img = np.zeros((orig_skel.shape[0], orig_skel.shape[1]))
        dummy_img[np.where(difference(bent_skel, orig_skel) > 0)]=1
        plt.imshow(dummy_img)
        plt.title("add_bend (old_section, new_section of skeleton)")
        plt.show() 
    if len(old_section)==0:
        raise BadSkeletonBendException("Unable to create a bend in the skeleton. Rerun with different random seed should fix this")
    skeletonize_func = get_skeletonizer_func(which=skeletonizer)
    _, dist = skeletonize_func(image, return_distance=True)
    if debug:
        half = int(len(new_section)//2)
        plt.subplot(1,3,1)
        plt.imshow(imnew)
    added_region = np.zeros((orig_skel.shape[0], orig_skel.shape[1])) # for debugging purpose
    for i,pt in enumerate(new_section):
        if debug and (i+1)==half:
            plt.subplot(1,3,2)
            plt.imshow(imnew)
        old_pt = closest_point(pt, old_section)
        size = int(dist[old_pt[0], old_pt[1]]) + 1
        size1 = size
        dh = np.random.rand()
        height = size
        width = size
        rectangle = create_rectangle(pt, height, width, 0, filled=True, size=(orig_skel.shape[0], orig_skel.shape[1]))
        if debug:
            added_region += rectangle
            #print(rectangle.sum(), len(np.where(rectangle)[0]))
        imnew = imnew + rectangle
    if debug:
        plt.subplot(1,3,3)
        plt.imshow(imnew)
        plt.title("add_bend")
        plt.show()
        new_wt = len(np.where(added_region > 0)[0])
        print(f"Num pixels added for bend: {new_wt, added_region.sum()}")
    imnew[np.where(imnew > 0)] = 1
    return imnew

def bend_image(image, distance, dist_shift, mid_points=None, skeletonizer=None, fixed_angle_seed=None, debug=False):
    """
        Bend `image` between two random points of the characters separated by `distance`, with a bend depth of `dist_shift`
        `debug` - Plots inputs/outputs at various stages for debugging
    """
    skeletonize_func = get_skeletonizer_func(which=skeletonizer)
    skel = skeletonize_func(image)
    bent_skel = bend_skel(skel, distance, dist_shift, mid_points=mid_points, fixed_angle_seed=fixed_angle_seed, debug=debug)
    imnew = add_bend(image, skel, bent_skel, skeletonizer=skeletonizer, debug=debug)
    if debug:
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(imnew)
        plt.title("main_function")
        plt.show()
        plt.subplot(1,2,1)
        plt.imshow(image, cmap="gray")
        plt.subplot(1,2,2)
        plt.imshow(imnew, cmap="gray")
        plt.title("main_function")
        plt.show()
        print(f"Input image shape: {image.shape}")
    return imnew

def bender(image, distance, dist_shift, mid_points=None, skeletonizer=None, angle_seed=None, debug=False):
    """
       Bend `image` between two random points of the characters separated by `distance`, with a bend depth of `dist_shift`
        `debug` - Plots inputs/outputs at various stages for debugging
        `skeletonizer` - skeletonization method (should be either `medial_aixs` or `skeletonize`)
    """
    new_image = bend_image(image, distance, dist_shift, mid_points=mid_points, skeletonizer=skeletonizer, 
                           fixed_angle_seed=angle_seed, debug=debug)
    bin_image = new_image.astype(np.float32) # invert method wants the inputs to be float array
    return bin_image

# def bender(image, distance, dist_shift, endpts=None, debug=False):
#     """
#        Bend `image` between two random points of the characters separated by `distance`, with a bend depth of `dist_shift`
#         `debug` - Plots inputs/outputs at various stages for debugging 
#     """
#     new_image = bend_image(image, distance, dist_shift, endpts=endpts, debug=debug)
#     return new_image