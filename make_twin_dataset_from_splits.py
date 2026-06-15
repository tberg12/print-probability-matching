#
#  Make twin dataset from alignment directory
import traceback
import skimage
import shlex
import matplotlib.pyplot as plt
from skimage.util import invert
import copy
import os
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import re
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import math
import random
from pathlib import Path
from torchvision import transforms
from torchvision.transforms.functional import affine
import json
import time
import sys
from collections import defaultdict
sys.path.append('Morpho-MNIST')
sys.path.append('Morpho-MNIST/morphomnist')
from morphomnist import io, morpho, perturb
from perturb_settings import TwinSettings
from character_bender import BadSkeletonBendException, ChooseSkeletonEndpointsException
from argparse import ArgumentParser
import datetime
from scipy.stats import truncnorm
from joblib import Parallel, delayed
from tqdm import tqdm
from perturber_args import get_char_settings
import torch
import torch.nn as nn
import torch.nn.functional as F


class Binarizer:
    """
        Binarize using `threshold_minimum` thrsholding scheme, uses `threshold_otsu` as fallback if minimum fails
    """
    def __init__(self, output_dtype=None):
        self.output_dtype = output_dtype # only needed for np input

    def __call__(self, img):
        np_img = np.array(img)
        no_white_mask = np_img < 255
        try:
            threshold = skimage.filters.threshold_otsu(np_img[no_white_mask].reshape(-1, 1))
            # bin_img = (img >threshold).astype(self.output_dtype)
        except Exception as err:
            print(err, 'using threshold_minimum')
            try:
                threshold = skimage.filters.threshold_minimum(np_img[no_white_mask].reshape(-1, 1))
            except Exception as err:
                print(err, 'backing off to thresholding at 127')
                threshold = 127
                Image.fromarray(np_img).save('threshold_minimum_failed.png')
        
        binary_img = img.point(lambda x: 255 if x > threshold else 0).convert('1')#.convert('L')
#         np_img = np.array(binary_img)
#         print(np_img.min(), np_img.max(), set(np_img.reshape(-1).tolist()))
        return binary_img


class SquarePad:
    """ from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10 """
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.Pad(padding, fill=255, padding_mode='constant')(image)



def obj2str(perturbation):
    """
        Returns a string representing the type of perturbation
    """
    if isinstance(perturbation, perturb.Identity):
        return "identity"
    elif isinstance(perturbation, perturb.Thinning):
        return "thin"
    elif isinstance(perturbation, perturb.Thickening):
        return "thick"
    elif isinstance(perturbation, perturb.Swelling):
        return "swell"
    elif isinstance(perturbation, (perturb.SingleBend, perturb.PairBend)):
        return "bend"
    elif isinstance(perturbation, (perturb.OpenFracture, perturb.PairOpenFracture)):
        return "fracture"
    else:
        raise ValueError("Invalid type of perturbation")
        

def warp_img(img):
    """ Currently not being used. Instead, we do this random warping via a transform during training.
    """
    random_rotation_angle = 3.0 * random.normalvariate(0.0, 0.3)
    random_translate = (
        int((random.normalvariate(0.0, 0.2) * 8. * 64) / img.size[0]), 
        int((random.normalvariate(0.0, 0.2) * 8. * 64) / img.size[1])
    )
    random_scale = random.uniform(0.95, 1.05)
    random_shear = (
        random.normalvariate(0.0, 0.2) * 3.5, 
        random.normalvariate(0.0, 0.2) * 3.5
    )
#     print(random_translate)
    return affine(img, 
                  angle=random_rotation_angle, 
                  translate=random_translate, 
                  scale=random_scale, 
                  shear=random_shear, 
                  resample=False, 
                  fillcolor=0)

                    
def apply_random_perturbations1(img, percent_apply_both_damage, damage=True, warp=False, src_img_path=None):
    if src_img_path is not None:
        char = src_img_path.name.split('_')[7]
        char_settings = get_char_settings(char)
        twin_settings = TwinSettings(char_settings)

    perturb_str = ''  # put the perturb labels in here in case we want to append to new file name
    make_morph_object = lambda x: morpho.ImageMorphology(x, scale=1)
    inverted_img = invert(np.array(img).astype(bool))

    # save both img and inverted_img to files with their respective names
    if isinstance(img, Image.Image):
        img.save('img.png')
    elif isinstance(img, np.ndarray):
        Image.fromarray(inverted_img).save('inverted_img.png')
    # Image.fromarray(inverted_img.astype(bool)).convert('1').save('inverted_img.tif')
    # exit(0)


    # apply perturbations
    if damage:
        img_morphology = make_morph_object(inverted_img)
        if random.random() < percent_apply_both_damage:
            damaged_img, (x_nn, y_nn) = (twin_settings.damage_perturbations_single[0]()(img_morphology))
            img_morphology = make_morph_object(damaged_img)
            result_img, (x_nn, y_nn) = twin_settings.damage_perturbations_single[1]()(img_morphology)
            perturb_str += '_damage_' + obj2str(twin_settings.damage_perturbations_single[0]()) + '_' + obj2str(twin_settings.damage_perturbations_single[1]())
        else:
            damage_perturbation = random.choice(twin_settings.damage_perturbations_single)
            result_img, (x_nn, y_nn) = damage_perturbation()(img_morphology)
            perturb_str += '_damage_' + obj2str(damage_perturbation())
    else:
        img_morphology = make_morph_object(inverted_img)
        global_inking_perturbation = random.choices(twin_settings.global_inking_perturbations, 
                                                    weights=twin_settings.global_inking_perturbations_probs)[0]
        global_inking_perturbed_img = global_inking_perturbation()(img_morphology)
        perturb_str += '_globalink_' + obj2str(global_inking_perturbation())
        # swell
        img_morphology = make_morph_object(global_inking_perturbed_img)
        result_img = twin_settings.local_inking_perturbation()(img_morphology)
        perturb_str += '_localink_' + obj2str(twin_settings.local_inking_perturbation())
    # warp image
    if warp:
#         print('before', np.unique(result_img), damage, perturb_str)
        result_img = np.asarray(warp_img(Image.fromarray(result_img)))
#         print('after', np.unique(result_img), damage, perturb_str)
    if damage:
        return invert(result_img), perturb_str, (x_nn, y_nn)
    return invert(result_img), perturb_str


def apply_random_perturbations2(img1, img2, percent_apply_both_damage, damage=True, warp=False, global_skeleton=None):
    imgs = [img1, img2]
    perturb_str1 = ''  # put the perturb labels in here in case we want to append to new file name
    perturb_str2 = ''
    make_morph_object = lambda x: morpho.ImageMorphology(x, scale=1)
    inverted_imgs = [invert(np.array(img).astype(bool)) for img in imgs]

    # apply perturbations
    if damage:
        img_morphologys = [make_morph_object(inverted_img) for inverted_img in inverted_imgs]
        if random.random() < percent_apply_both_damage:
            damaged_imgs, (x_nn, y_nn) = (twin_settings.damage_perturbations[0]()(*img_morphologys, perturb.MergeImagePair('union')(*img_morphologys), global_skeleton))
            img_morphologys = [make_morph_object(damaged_img) for damaged_img in damaged_imgs]
            result_imgs, (x_nn, y_nn) = twin_settings.damage_perturbations[1]()(*img_morphologys, perturb.MergeImagePair('union')(*img_morphologys), global_skeleton)
            damage_str = '_damage_' + obj2str(twin_settings.damage_perturbations[0]()) + '_' + obj2str(twin_settings.damage_perturbations[1]())
            perturb_str1 += damage_str
            perturb_str2 += damage_str
        else:
            damage_perturbation = random.choice(twin_settings.damage_perturbations)
            new_img1, new_img2, (x_nn, y_nn) = damage_perturbation()(*img_morphologys, perturb.MergeImagePair('union')(*img_morphologys), global_skeleton)
            result_imgs = (new_img1, new_img2)
            damage_str = '_damage_' + obj2str(damage_perturbation())
            perturb_str1 += damage_str
            perturb_str2 += damage_str
    else:
        img_morphologys = [make_morph_object(inverted_img) for inverted_img in inverted_imgs]
        global_inking_perturbations = random.choices(twin_settings.global_inking_perturbations, 
                                                    weights=twin_settings.global_inking_perturbations_probs, k=2)
        global_inking_perturbed_imgs = [global_inking_perturbations[i]()(img_morphologys[i]) for i in range(2)]
        perturb_str1 += '_globalink_' + obj2str(global_inking_perturbations[0]())
        perturb_str2 += '_globalink_' + obj2str(global_inking_perturbations[1]())
        # swell
        img_morphologys = [make_morph_object(img) for img in global_inking_perturbed_imgs]
        local_inking_perturbations = [twin_settings.local_inking_perturbation() for _ in range(2)]
        result_imgs = [local_inking_perturbations[i](img_morphologys[i]) for i in range(2)]
        perturb_str1 += '_localink_' + obj2str(local_inking_perturbations[0])
        perturb_str2 += '_localink_' + obj2str(local_inking_perturbations[1])
    # warp image
    if warp:
        result_imgs = [np.asarray(warp_img(Image.fromarray(result_img))) for result_img in result_imgs]
        
    if global_skeleton is None:
        return invert(result_imgs[0]), invert(result_imgs[1]), perturb_str1, perturb_str2
    else:  # return coords on global skeleton as final element in tuple
        return invert(result_imgs[0]), invert(result_imgs[1]), perturb_str1, perturb_str2, (x_nn, y_nn)


def process_img1_singleton(src_img_path, new_img_path, row, percent_apply_both_damage, transform):
    try:
        img = transform(Image.open(str(src_img_path)))
    except PIL.UnidentifiedImageError as e:
        print(f'{traceback.format_exc()}\n{e}: {src_img_path}')
        return
    # Image.open(str(src_img_path)).save('src_img.png')
    # Image.fromarray(np.array(img)).save('img.png')
    # import ipdb; ipdb.set_trace()
    # perturb img
    
    try:
        damaged_img, perturb_str, (x_nn, y_nn) = apply_random_perturbations1(img, percent_apply_both_damage, damage=True, warp=False, src_img_path=src_img_path)
    except Exception as e:
        print(f'{traceback.format_exc()}\n{e}: {src_img_path}')
        return

    img1 = damaged_img

    # make copies of damaged character
    # img1, img2 = copy.deepcopy(damaged_img), copy.deepcopy(damaged_img)
    # perturb img1, img2 with different random global inking


    # try:
    # inking perturbations dont need locations
    # TODO: maybe swelling should?
        # img1, perturb_str1 = apply_random_perturbations1(img1, percent_apply_both_damage, damage=False, warp=False)  # set warp=True if not doing warping during training time
        # img2, perturb_str2 = apply_random_perturbations1(img2, percent_apply_both_damage, damage=False, warp=False)  # set warp=True if not doing warping during training time
        # perturb_ink_strs = [perturb_str1, perturb_str2]
    # except Exception as e:
    #     print(f'{traceback.format_exc()}\n{e}: {src_img_path}')
    #     return


#     print('Pre-bin:', np.unique(img1), np.unique(img2), np.all(img1 < 1), np.all(img2 < 1))
    img1[img1 < 0.5] = 0.
    # img2[img2 < 0.5] = 0.
    img1[img1 >= 0.5] = 1.
    # img2[img2 >= 0.5] = 1.
#     print('Post-bin:', np.unique(img1), np.unique(img2))
#     print('Post-bin sum:', np.sum(img1), np.sum(img2))
    new_rows = []
    # append perturbation info to this img_path
    row_copy = copy.deepcopy(row)
    new_suffix = f'{perturb_str}.tif'
    row_copy["original_file_path"] = str(src_img_path)
    row_copy["file_path"] = str(new_img_path) + new_suffix
    row_copy["fname"] = str(new_img_path.name) + new_suffix
    row_copy["x"] = str(x_nn)
    row_copy["y"] = str(y_nn)

    result = Image.fromarray(img1.astype(bool)).convert('1')
    assert result.mode == '1'
    # save perturbed img to new img path
    try:
        result.save(row_copy["file_path"])
    except SystemError as e:
        print(f'{traceback.format_exc()}\n{e}: {new_img_path}')
        return
    new_rows.append(row_copy)
    # NOTE:
    # save damaged_img (without warping) to test warping amounts
#         damaged_img[damaged_img < 0.5] = 0.
#         damaged_img[damaged_img >= 0.5] = 1.
#         Image.fromarray(perturbed_img.astype(bool)).convert('1').save(str(new_img_path) + perturb_str + '.tif')
    return new_rows


def process_img1(src_img_path, new_img_paths, rows, percent_apply_both_damage, transform):
    img = transform(Image.open(str(src_img_path)))
    # Image.open(str(src_img_path)).save('src_img.png')
    # Image.fromarray(np.array(img)).save('img.png')
    # import ipdb; ipdb.set_trace()
    # perturb img
    
    try:
        damaged_img, perturb_str,  (x_nn, y_nn) = apply_random_perturbations1(img, percent_apply_both_damage, damage=True, warp=False)
    except Exception as e:
        print(f'{traceback.format_exc()}\n{e}: {src_img_path}')
        return


    # make copies of damaged character
    img1, img2 = copy.deepcopy(damaged_img), copy.deepcopy(damaged_img)
    # perturb img1, img2 with different random global inking


    try:
    # inking perturbations dont need locations
    # TODO: maybe swelling should?
        img1, perturb_str1 = apply_random_perturbations1(img1, percent_apply_both_damage, damage=False, warp=False)  # set warp=True if not doing warping during training time
        img2, perturb_str2 = apply_random_perturbations1(img2, percent_apply_both_damage, damage=False, warp=False)  # set warp=True if not doing warping during training time
        perturb_ink_strs = [perturb_str1, perturb_str2]
    except Exception as e:
        print(f'{traceback.format_exc()}\n{e}: {src_img_path}')
        return


#     print('Pre-bin:', np.unique(img1), np.unique(img2), np.all(img1 < 1), np.all(img2 < 1))
    img1[img1 < 0.5] = 0.
    img2[img2 < 0.5] = 0.
    img1[img1 >= 0.5] = 1.
    img2[img2 >= 0.5] = 1.
#     print('Post-bin:', np.unique(img1), np.unique(img2))
#     print('Post-bin sum:', np.sum(img1), np.sum(img2))
    new_rows = []
    # append perturbation info to this img_path
    for j, perturbed_img in enumerate([img1, img2]):
        row_copy = copy.deepcopy(rows[j])
        new_suffix = f'{perturb_str}{perturb_ink_strs[j]}_{j+1}.tif'
        row_copy["original_file_path"] = str(src_img_path)
        row_copy["file_path"] = str(new_img_paths[j]) + new_suffix
        row_copy["fname"] = str(new_img_paths[j].name) + new_suffix
        row_copy["x"] = str(x_nn)
        row_copy["y"] = str(y_nn)

        result = Image.fromarray(perturbed_img.astype(bool)).convert('1')
        assert result.mode == '1'
        # save perturbed img to new img path
        result.save(row_copy["file_path"])
        new_rows.append(row_copy)
        # NOTE:
        # save damaged_img (without warping) to test warping amounts
#         damaged_img[damaged_img < 0.5] = 0.
#         damaged_img[damaged_img >= 0.5] = 1.
#         Image.fromarray(perturbed_img.astype(bool)).convert('1').save(str(new_img_path) + perturb_str + '.tif')
    return new_rows


def process_img2(img_paths, new_img_paths, rows, percent_apply_both_damage, transform):
    imgs = (
        transform(Image.open(str(img_paths[0]))), 
        transform(Image.open(str(img_paths[1])))
    )
    # perturb img
    try:
        # NOTE: perturb_str should be the same for both imgs here cause they're damaged the same
        img1, img2, perturb_str, _, (x_nn, y_nn) = apply_random_perturbations2(imgs[0], imgs[1], percent_apply_both_damage, damage=True, warp=False, global_skeleton=global_skeleton)
    except Exception as e:
        print(f'{traceback.format_exc()}\n{e}: {img_paths}', file=sys.stderr)
        return
    # perturb img1, img2 with different random global inking
    try:
        img1, img2, perturb_str1, perturb_str2 = apply_random_perturbations2(img1, img2, percent_apply_both_damage, damage=False, warp=False)  # set warp=True if not doing warping during training time
        perturb_ink_strs = [perturb_str1, perturb_str2]
    except Exception as e:
        print(f'{traceback.format_exc()}\n{e}: {img_paths}', file=sys.stderr)
        return

#     print('Pre-bin:', np.unique(img1), np.unique(img2), np.all(img1 < 1), np.all(img2 < 1))
    img1[img1 < 0.5] = 0.
    img2[img2 < 0.5] = 0.
    img1[img1 >= 0.5] = 1.
    img2[img2 >= 0.5] = 1.
#     print('Post-bin:', np.unique(img1), np.unique(img2))
#     print('Post-bin sum:', np.sum(img1), np.sum(img2))

    new_rows = []
    # append perturbation info to this img_path
    for j, perturbed_img in enumerate([img1, img2]):
        row_copy = copy.deepcopy(rows[j])
        # log damage type, location, and inking type to file name
        new_suffix = f'{perturb_str}_x{x_nn}_y{y_nn}{perturb_ink_strs[j]}_{j+1}.tif'
        row_copy["file_path"] = str(new_img_paths[j]) + new_suffix
        row_copy["fname"] = str(new_img_paths[j].name) + new_suffix
        row_copy["x"] = str(x_nn)
        row_copy["y"] = str(y_nn)

        result = Image.fromarray(perturbed_img.astype(bool)).convert('1')
        assert result.mode == '1'
        # save perturbed img to new img path
        result.save(row_copy["file_path"])
        new_rows.append(row_copy)

        # NOTE:
        # save damaged_img (without warping) to test warping amounts
#         damaged_img[damaged_img < 0.5] = 0.
#         damaged_img[damaged_img >= 0.5] = 1.
#         Image.fromarray(perturbed_img.astype(bool)).convert('1').save(str(new_img_path) + perturb_str + '.tif')
    return new_rows


def make_dataset_diff_base_images(dataset, dataset_df, dataset_idxs, num_random_idxs, new_dataframe_save_path, dest_dir, num_jobs, num_repetitions, percent_apply_both_damage, transform):
    """ Create and save a fake twin dataset using different base images for matched pairs.
    Args:
        dataset_df: Our UCSDDataset_Residue object's Dataframe containing normals to sample from
        dataset_idxs: A list of dataset indices that we want to sample from to perturb
        num_random_idxs: The number of samples we want from dataset_idxs.
            If we want to generate new damaged anomalies, this should be equal to len(dataset.anomaly_labels)
    """
    global_skeleton = compute_global_skeleton(dataset)
    
    perturbed_df_new_rows = []
        
    # generate a dict mapping book names to all dataset indices from different books
    bookname2valididxs = dict()
    for bookname in dataset_df.book_name.unique():
        bookname2valididxs[bookname] = dataset_df.loc[dataset_df.book_name != bookname].index.tolist()

    for n in tqdm(list(range(num_random_idxs * num_repetitions // num_jobs))):
#         if n >= 10:
#             break
        idxs1 = np.random.choice(dataset_idxs, replace=False, size=num_jobs)
        idxs2 = []
        for i in idxs1:
            idxs2.append(np.random.choice(bookname2valididxs[dataset_df.iloc[i].book_name], size=1)[0])
        pair_idxs = [idxs1, np.array(idxs2)]
        
        pair_rows = [[dataset_df.iloc[i].copy(deep=True) 
                      for i in idxs]
                     for idxs in pair_idxs
                    ]
        
        timestamp = f"{int(time.time())}"
#         j = 0
#         out = process_img2((pair_rows[j][0].file_path, pair_rows[j][1].file_path), 
#                     (
#                         dest_dir/(str(pair_rows[j][0].file_path.stem) + f"_perturb{n+j}_anomaly_{timestamp}{j}"),
#                         dest_dir/(str(pair_rows[j][1].file_path.stem) + f"_perturb{n+j}_anomaly_{timestamp}{j}")
#                     ), 
#                     (pair_rows[j][0], pair_rows[j][1]),
#                     global_skeleton,
#                     percent_apply_both_damage,
#                     transform
#                 )
        out = Parallel(n_jobs=num_jobs)(
                delayed(process_img2)(
                    (pair_rows[0][j].file_path, pair_rows[1][j].file_path), 
                    (
                        dest_dir/(str(pair_rows[0][j].file_path.stem) + f"_perturb{n+j}_anomaly_{timestamp}{j}"),
                        dest_dir/(str(pair_rows[1][j].file_path.stem) + f"_perturb{n+j}_anomaly_{timestamp}{j}")
                    ), 
                    (pair_rows[0][j], pair_rows[1][j]),
                    global_skeleton, 
                    percent_apply_both_damage,
                    transform
                )
            for j in range(num_jobs)
        )
        
        for new_row_list in out:
            if new_row_list is not None:
                for new_row in new_row_list:
                    perturbed_df_new_rows.append(new_row)
                    
    perturbed_df = pd.DataFrame(perturbed_df_new_rows, columns=dataset_df.columns.values.tolist() + ['x', 'y'])
    print(perturbed_df.shape)
    perturbed_df.to_csv(str(new_dataframe_save_path))

def make_dataset_same_base_image_singleton(
        file_list, 
        new_dataframe_save_path, 
        dest_dir, 
        num_jobs, 
        num_repetitions, 
        percent_apply_both_damage, 
        transform,
        stop_after=None,
        debug=False,
    ):
    """ Create and save a fake twin dataset using same base images for matched pairs.
    """
    perturbed_df_new_rows = []
    random.shuffle(file_list)
    # create num_jobs sized batches of image indices to perturb
    idxs = list(range(len(file_list))) * num_repetitions
    if debug:
        idxs = sorted(idxs)
    batches = [idxs[i:i+num_jobs] for i in range(0, len(idxs), num_jobs)]

    for n in tqdm(list(range(len(batches)))):
        # check if there are already stop_after perturbed images containing "bend" in the filename to exit early
        if stop_after is not None and len(perturbed_df_new_rows) >= stop_after:
            break
        # create batch of image indices to perturb
        idxs1 = batches[n]
        singleton_rows = [
            [
                {
                    'file_path': file_list[i],
                    'fname': file_list[i].name,
                    'x': None,
                    'y': None
                }
                for i in idxs1
            ],
        ]
        
        timestamp = f"{int(time.time())}"

        out = Parallel(n_jobs=num_jobs)(
                delayed(process_img1_singleton)(
                    singleton_rows[0][j]['file_path'], 
                    (
                        dest_dir/(str(singleton_rows[0][j]['file_path'].stem) + f"_perturb{n+j}_anomaly_{timestamp}{j}")
                    ), 
                    (
                        singleton_rows[0][j]
                    ),
                    percent_apply_both_damage,
                    transform
                )
            for j in range(num_jobs)
        )
        
        for new_row_list in out:
            if new_row_list is not None:
                for new_row in new_row_list:
                    perturbed_df_new_rows.append(new_row)
                    
    perturbed_df = pd.DataFrame(perturbed_df_new_rows)
    print(perturbed_df.shape)
    perturbed_df.to_csv(str(new_dataframe_save_path))


def make_dataset_same_base_images(
        file_list, 
        new_dataframe_save_path, 
        dest_dir, 
        num_jobs, 
        num_repetitions, 
        percent_apply_both_damage, 
        transform,
        stop_after=None,
        debug=False,
    ):
    """ Create and save a fake twin dataset using same base images for matched pairs.
    """
    perturbed_df_new_rows = []
    random.shuffle(file_list)
    # create num_jobs sized batches of image indices to perturb
    idxs = list(range(len(file_list))) * num_repetitions
    if debug:
        idxs = sorted(idxs)
    batches = [idxs[i:i+num_jobs] for i in range(0, len(idxs), num_jobs)]

    for n in tqdm(list(range(len(batches)))):
        # check if there are already stop_after perturbed images containing "bend" in the filename to exit early
        if stop_after is not None and len(perturbed_df_new_rows) >= stop_after:
            break
        # create batch of image indices to perturb
        idxs1 = batches[n]
        idxs2 = copy.deepcopy(idxs1)
        pair_idxs = [idxs1, idxs2]
        pair_rows = [
            [
                {
                    'file_path': file_list[i],
                    'fname': file_list[i].name,
                    'x': None,
                    'y': None
                }
                for i in idxs1
            ],
            [
                {
                    'file_path': file_list[i],
                    'fname': file_list[i].name,
                    'x': None,
                    'y': None
                }
                for i in idxs2
            ]
        ]
        
        timestamp = f"{int(time.time())}"
        # j = 0
        # # import ipdb; ipdb.set_trace()
        # out = process_img1(
        #     pair_rows[1][j]['file_path'], 
        #     (
        #         dest_dir/(str(pair_rows[0][j]['file_path'].stem) + f"_perturb{n+j}_anomaly_{timestamp}{j}"),
        #         dest_dir/(str(pair_rows[1][j]['file_path'].stem) + f"_perturb{n+j}_anomaly_{timestamp}{j}")
        #     ), 
        #     (
        #         pair_rows[0][j], 
        #         pair_rows[1][j]
        #     ),
        #     percent_apply_both_damage,
        #     transform
        # )

        out = Parallel(n_jobs=num_jobs)(
                delayed(process_img1)(
                    pair_rows[1][j]['file_path'], 
                    (
                        dest_dir/(str(pair_rows[0][j]['file_path'].stem) + f"_perturb{n+j}_anomaly_{timestamp}{j}"),
                        dest_dir/(str(pair_rows[1][j]['file_path'].stem) + f"_perturb{n+j}_anomaly_{timestamp}{j}")
                    ), 
                    (
                        pair_rows[0][j], 
                        pair_rows[1][j]
                    ),
                    percent_apply_both_damage,
                    transform
                )
            for j in range(num_jobs)
        )
        
        for new_row_list in out:
            if new_row_list is not None:
                for new_row in new_row_list:
                    perturbed_df_new_rows.append(new_row)
                    
    perturbed_df = pd.DataFrame(perturbed_df_new_rows)
    print(perturbed_df.shape)
    perturbed_df.to_csv(str(new_dataframe_save_path))


def make_dataset_same_base_image_old(dataset, dataset_df, dataset_idxs, num_random_idxs, new_dataframe_save_path, dest_dir, num_jobs, num_repetitions, percent_apply_both_damage):
    """ Create and save a fake twin dataset using the same base image for matched pairs.
    Args:
        dataset_df: Our UCSDDataset_Residue object's Dataframe containing normals to sample from
        dataset_idxs: A list of dataset indices that we want to sample from to perturb
        num_random_idxs: The number of samples we want from dataset_idxs.
            If we want to generate new damaged anomalies, this should be equal to len(dataset.anomaly_labels)
    """
    random_idxs_to_perturb = np.random.choice(dataset_idxs, num_random_idxs)
    perturbed_df_new_rows = []

    for iteration, i in enumerate(tqdm(random_idxs_to_perturb)):
        (img, class_label), img_path = dataset[i]
        row = dataset_df.iloc[i].copy(deep=True)
        row_copies = [copy.deepcopy(row) for _ in range(num_repetitions)]

        timestamp = f"{int(time.time())}"
        out = Parallel(n_jobs=num_jobs)(delayed(process_img1)(img, img_path, dest_dir/(str(img_path.stem) + f"_perturb{r}_anomaly_{timestamp}{r}"), row_copies[r], percent_apply_both_damage) for r in range(num_repetitions))
        for new_row_list in out:
            if new_row_list is not None:
                for new_row in new_row_list:
                    perturbed_df_new_rows.append(new_row)
        
    perturbed_df = pd.DataFrame(perturbed_df_new_rows, columns=dataset_df.columns)
    print(perturbed_df.shape)
    perturbed_df.to_csv(str(new_dataframe_save_path))


def make_twin_perturbation_dataset(
        file_list, 
        new_dataframe_save_path, 
        dest_dir, 
        num_jobs, 
        num_repetitions, 
        percent_apply_both_damage, 
        transform, 
        use_diff_base_images=False,
        stop_after=None,
        debug=False
    ):
    # func = make_dataset_diff_base_images if use_diff_base_images else make_dataset_same_base_images
    # return func(file_list, new_dataframe_save_path, dest_dir, num_jobs, num_repetitions, percent_apply_both_damage, transform)
    # return make_dataset_same_base_images(file_list, new_dataframe_save_path, dest_dir, num_jobs, num_repetitions, percent_apply_both_damage, transform, stop_after=stop_after, debug=debug) if not use_diff_base_images else make_dataset_diff_base_images(file_list, new_dataframe_save_path, dest_dir, num_jobs, num_repetitions, percent_apply_both_damage, transform)
    return make_dataset_same_base_image_singleton(file_list, new_dataframe_save_path, dest_dir, num_jobs, num_repetitions, percent_apply_both_damage, transform, stop_after=stop_after, debug=debug) if not use_diff_base_images else make_dataset_diff_base_images(file_list, new_dataframe_save_path, dest_dir, num_jobs, num_repetitions, percent_apply_both_damage, transform)


def compute_global_skeleton(dataset):
    avg_image = None
    for i, x in enumerate(dataset):
        import ipdb; ipdb.set_trace()
        (img, class_label), img_path = x
        img = np.array(img).astype(float)
        if i == 0:  # set to img size
            avg_image = np.zeros_like(img)
        avg_image += img
    avg_image /= len(dataset)
    avg_image = (avg_image * 255).astype(np.uint8)
#     print(np.unique(avg_image))

    avg_image_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            Binarizer(),
        ]
    )

    return morpho.ImageMorphology(invert(np.array(avg_image_transform(Image.fromarray(avg_image))))).skeleton


def extract_filename_fields(filename):
    parts = filename.split('_')
    printer = parts[0]
    estc = parts[1]
    title_page = parts[4].split('-')
    title = title_page[0]
    page = title_page[1]
    return printer, estc, title, page


def split_data_by_book(filelist, train_frac=0.7, held_out_frac=0.1):
    # Group files by (printer, estc, title) to keep books separate
    file_groups = defaultdict(list)
    for file in sorted(filelist):
        printer, estc, title, page = extract_filename_fields(file.name)
        # if title ends with capital letter, then remove it
        # NOTE: this is to make sure books that are separated into parts are grouped together!
        if title[-1].isupper():
            title = title[:-1]
        file_groups[(printer, estc, title)].append(file)

    # Shuffle and split groups into train, valid, test sets
    group_keys = list(file_groups.keys())
    random.shuffle(group_keys)

    train_split = int(train_frac * len(group_keys))
    valid_split = int((train_frac + held_out_frac) * len(group_keys))

    train_keys = group_keys[:train_split]
    valid_keys = group_keys[train_split:valid_split]
    test_keys = group_keys[valid_split:]

    # Collect files for each set
    train_files = [file for key in train_keys for file in file_groups[key]]
    valid_files = [file for key in valid_keys for file in file_groups[key]]
    test_files = [file for key in test_keys for file in file_groups[key]]

    # Output the splits
    print("Train Files:", f"({len(train_files)} files)")
    print("Valid Files:", f"({len(valid_files)} files)")
    print("Test Files:", f"({len(test_files)} files)")

    # print('Train keys:', train_keys)
    # print('\n\n')
    # print('Valid keys:', valid_keys)
    # print('\n\n')
    # print('Test keys:', test_keys)
    # print('\n\n')
    # exit(0)

    return train_files, valid_files, test_files


def parse_args():
    parser = ArgumentParser('Creates a new twin dataset of train/valid/test images from random normal images. First, we sample an image, then we damage it, make two copies, and apply different inking perturbations to the copies. Loads perturbation settings from perturber_args.')
    parser.add_argument('--char', type=str, default=None)
    parser.add_argument('--binarize_image', action='store_true')
    parser.add_argument('--num_repetitions', default=5, type=int, help='Number of random perturbations to apply to a single image')
    parser.add_argument('--percent_apply_both_damage', type=float, default=0.0, help='Frequency to apply both bends and fractures to the same image.')  # was 0.03
    # parser.add_argument('--splits_dir', type=str, default=f"/home/kishore/data/anomaly-detection/aligned-dataset/M/mixed_book_splits", help='Path to directory containing dataset splits')
    # parser.add_argument('--src_dataframe_path', type=str, default=None) # Path("/home/kishore/data/anomaly-detection/datasetv3"))
    # parser.add_argument('--dest_dir', type=str, default="/home/kishore/data/anomaly-detection/aligned-dataset/M")
    # parser.add_argument('--dest_dataset_name', type=str, default=f"M_aligned_perturbed20_twin_05_15")
    parser.add_argument('--num_jobs', default=4, type=int)
    parser.add_argument('--use_diff_base_images', action='store_true', help='Whether to generate twin damage pairs from aligning the skeletons of different base images, instead of using the same base image.')
    parser.add_argument('--merge_method', type=str, choices=['union', 'intersection'], default='union', help='Merge method for aligning twin images with different base images')
    parser.add_argument('--splits', type=str, choices=['train', 'valid', 'test'], default=['train', 'valid', 'test'], nargs='+')
    # -------------------
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_dir', type=str, default="/graft2/code/nvog/git/matching/data/redo_top_5pct_logprob", help='Path to raw images')
    parser.add_argument('--output_dir', type=str, default="/graft2/code/nvog/git/matching/data/synthetic/redo_top_5pct_logprob_singleton_5_2024-11-04", help='Path to output dir')
    parser.add_argument('--stop_after', type=int, default=1000, help='Stop after this many images have been perturbed')
    parser.add_argument('--unpadded_image_size', type=int, default=112, help='Size of output images')
    parser.add_argument('--padded_image_size', type=int, default=224, help='Size of output images')
    parser.add_argument('--train_size', type=int, default=10000, help='Number of training pairs to generate')
    parser.add_argument('--valid_size', type=int, default=1000, help='Number of validation pairs to generate')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of test pairs to generate')
    parser.add_argument('--split_data_by_book_ratios', type=float, nargs=3, default=[0.9, 0.05, 0.05], help='Train, valid, test ratios')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args


def main():
    dt = datetime.datetime.today()
    args = parse_args()
    print(args)
    
    # x = transform(Image.open(
    #     '/trunk2/nvog/shared/char_images3_aligned/shakespeare/shakespeare-G-out/char_G_uc/amaxwell_R153_uk_4_englishschoolmaster1673-0002_page1rline14_char29_G_uc.tif'
    # ))
    # x.save('test.png')
    # import ipdb; ipdb.set_trace()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.char and args.char not in args.input_dir:
        print(f"Warning: char {args.char} not in input_dir {args.input_dir}")
        exit(1)

    data = dict()
    
    data['train_files'], data['valid_files'], data['test_files'] = split_data_by_book(
        list(Path(args.input_dir).glob('*uc.tif')), 
        train_frac=args.split_data_by_book_ratios[0],
        held_out_frac=args.split_data_by_book_ratios[1]
    )

    global twin_settings
    if args.char:
        char_settings = get_char_settings(args.char)
        twin_settings = TwinSettings(char_settings)

    output_dir = Path(args.output_dir)
    print(f"Output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: change binarizer
    if args.binarize_image:
        transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((args.unpadded_image_size, args.unpadded_image_size)),
            transforms.Grayscale(num_output_channels=1),
            Binarizer(),
            transforms.Pad((args.padded_image_size - args.unpadded_image_size) // 2, fill=255, padding_mode='constant'),
        ])
    else:
        raise NotImplementedError()

    for split in args.splits: 
        dest_dataset_path = output_dir/split
        # dest_dataframe_path = dest_dir/split/"twin_images.csv"
        dest_dataframe_path = output_dir/split/"pairs.csv"
        dest_dataset_path.mkdir(parents=True, exist_ok=True)
        if split == 'train':
            stop_after = args.train_size
        elif split == 'valid':
            stop_after = args.valid_size
        elif split == 'test':
            stop_after = args.test_size

        make_twin_perturbation_dataset(
            data[f"{split}_files"],
            dest_dataframe_path, 
            dest_dataset_path,
            args.num_jobs,
            args.num_repetitions, 
            args.percent_apply_both_damage, 
            transform, 
            use_diff_base_images=args.use_diff_base_images,
            stop_after=stop_after,
            debug=args.debug
        )


if __name__ == '__main__':
    main()
