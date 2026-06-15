import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import pandas as pd
import re
import functools
from pathlib import Path
import copy
import shutil
import collections
import torch
from torchvision import transforms
import sys
import utils
from skimage.util import invert
from collections import defaultdict
import torchvision
import torchvision.transforms.functional as F
import random
import struct
import imagesize
import math
from skimage import morphology

sys.path.append("Morpho-MNIST/morphomnist")
import morpho
import perturb


class TypeAugmenter:
    def __init__(self, args, debug=False):
        self.args = args
        self.debug = debug

    def __call__(self, x):
        global_inking = transforms.RandomApply(
            random.choices(
                [
                    lambda x: torch.tensor(
                        morphology.erosion(
                            x.squeeze().detach().numpy(),
                            morphology.disk(np.random.randint(1, 2)),
                        )
                    )
                    .unsqueeze(0)
                    .requires_grad_(False),
                    lambda x: torch.tensor(
                        morphology.dilation(
                            x.squeeze().detach().numpy(),
                            morphology.disk(np.random.randint(1, 3)),
                        )
                    )
                    .unsqueeze(0)
                    .requires_grad_(False),
                ]
            ),
            p=1.0,
        )
        # jitter = self.jitter_transform()
        # result = jitter(global_inking(x))
        result = global_inking(x)
        return result

    def jitter_transform(
        self,
        x_translate_range=(-0.5, 0.5),
        y_translate_range=(-0.5, 0.5),
        rotation_angle_range=(-10, 10),
    ):
        return transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: transforms.functional.affine(
                        img,
                        angle=random.uniform(*rotation_angle_range),
                        translate=(
                            int((random.uniform(*x_translate_range) * 10 * 64) / 64),
                            int((random.uniform(*y_translate_range) * 10 * 64) / 64),
                        ),
                        scale=1.0,
                        shear=0.0,
                        resample=False,
                        fill=0.0,
                    )
                ),
            ]
        )


def make_jitter_transform(apply_random_inking=False, apply_random_pepper=False, apply_position_jitter=True):
    transform_list = []
    if apply_random_inking:
        """
            lambda: perturb.Thinning(amount=np.random.uniform(0.1, 
                                                              0.35)),
            lambda: perturb.Thickening(amount=np.random.uniform(0.075, 
                                                                0.235)),
        """
        global_inking = transforms.RandomApply(
            random.choices(
                [
                    lambda x: torch.tensor(
                        morphology.erosion(
                            x.squeeze().detach().numpy(),
                            morphology.disk(np.random.randint(1, 2)),
                        )
                    )
                    .unsqueeze(0)
                    .requires_grad_(False),
                    lambda x: torch.tensor(
                        morphology.dilation(
                            x.squeeze().detach().numpy(),
                            morphology.disk(np.random.randint(1, 3)),
                        )
                    )
                    .unsqueeze(0)
                    .requires_grad_(False),
                ]
            ),
            p=1.0,
        )
        transform_list.append(global_inking)
    
    if apply_random_pepper:
        raise NotImplementedError
        pass

    if apply_position_jitter:
        # x_translate = random.uniform(-0.5, 0.5)
        # y_translate = random.uniform(-0.5, 0.5)
        # rotation_angle = random.uniform(-45, 45)
        x_translate = random.uniform(-0.25, 0.25)
        y_translate = random.uniform(-0.25, 0.25)
        rotation_angle = random.uniform(-10, 10)
        transform_list += [
            # transforms.ToPILImage(mode='L'),
            transforms.Lambda(
                lambda img: transforms.functional.affine(
                    img,
                    angle=rotation_angle,
                    translate=(
                        int((x_translate * 10 * 64) / 64),
                        int((y_translate * 10 * 64) / 64),
                    ),
                    scale=1.0,
                    shear=0.0,
                    fill=0.0,
                )
            ),
            # transforms.ToTensor(),
            # and finally, take the borders and threshold them to 1 (fillcolor doesnt do this correctly for some reason)
            # transforms.Lambda(
            #     lambda img: torch.where(img > 0, torch.tensor(1.0), img)
            # )
        ]
    
    return transforms.Compose(transform_list)



def add_salt_and_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    """
    Add salt and pepper noise to a PIL image.
    
    Parameters:
        image (PIL.Image): Input image.
        amount (float): Proportion of image pixels to alter.
        salt_vs_pepper (float): Proportion of salt vs. pepper noise (0.5 means equal amounts).
        
    Returns:
        PIL.Image: Image with salt and pepper noise added.
    """
    # Convert image to NumPy array
    np_img = np.array(image)

    # Determine the number of pixels to alter
    total_pixels = np_img.shape[0] * np_img.shape[1]
    num_salt = int(amount * total_pixels * salt_vs_pepper)
    num_pepper = int(amount * total_pixels * (1.0 - salt_vs_pepper))

    # Add salt noise (white pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in np_img.shape[:2]]
    if np_img.ndim == 2:  # grayscale
        np_img[coords[0], coords[1]] = 255
    else:  # RGB or RGBA
        np_img[coords[0], coords[1]] = [255] * np_img.shape[2]

    # Add pepper noise (black pixels)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in np_img.shape[:2]]
    if np_img.ndim == 2:
        np_img[coords[0], coords[1]] = 0
    else:
        np_img[coords[0], coords[1]] = [0] * np_img.shape[2]

    # Convert back to PIL image
    return Image.fromarray(np_img)


class ImagePadder:
    """pad input image `img` with a constant value `pad_value` to size up to H x W"""

    def __init__(self, H, W, pad_value):
        self.H = H
        self.W = W
        self.pad_value = pad_value

    def __call__(self, img):
        w, h = img.size
        dh = self.H - h
        dw = self.W - w
        pad_w_left = dw // 2
        pad_w_right = dw - pad_w_left
        pad_h_bottom = dh // 2
        pad_h_top = dh - pad_h_bottom
        return transforms.Pad(
            padding=(pad_w_left, pad_h_top, pad_w_right, pad_h_bottom),
            fill=self.pad_value,
            padding_mode="constant",
        )(img)


class FixedHeightResizeAndPad:
    def __init__(self, size, pad_value):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, img):
        w, h = img.size
        aspect_ratio = float(h) / float(w)
        new_w = math.ceil(self.size / aspect_ratio)
        resized = transforms.Resize((self.size, new_w))(img)
        return ImagePadder(self.size, self.size, 255)(resized)


class DamageDataset(Dataset):
    def __init__(
        self,
        args,
        split,
        transform,
        csv_path_or_filelist,
        pregenerate_fixed_jitter=False,
        jitter=False,
        limit_train_data=sys.maxsize,
        is_pair_data=False
    ):
        self.args = args
        self.split = split
        self.is_pair_data = is_pair_data
        self.char_regex = re.compile(r"([A-Za-z])_[ul]c")
        self.damage_loc_regex = re.compile(r"_x([\d]+)_y([\d]+)")
        if isinstance(csv_path_or_filelist, dict):
            self.path2label = csv_path_or_filelist
        else:
            if self.is_pair_data:
                df = pd.read_csv(csv_path_or_filelist)
                # pando_data_dir = Path('/graft2/code/nvog/git/matching/data/synthetic_data')
                # only grab columns ["file_path", "preproc_original_file_path"] and set as list of pairs
                # self.damages = [str(pando_data_dir/f'{self.char_regex.findall(f)[0]}_twin_samebase_noink_sz164_5_2025-03-16'/split/Path(f).name) for f in df.file_path.to_list()]
                # self.normals = [str(pando_data_dir/f'{self.char_regex.findall(f)[0]}_twin_samebase_noink_sz164_5_2025-03-16'/split/Path(f).name) for f in df.preproc_original_file_path.to_list()]
                self.damages = [str(f) for f in df.file_path.to_list()]
                self.normals = [str(f) for f in df.preproc_original_file_path.to_list()]
                self.normal2damage = {
                    self.normals[i]: self.damages[i] for i in range(len(self.normals))
                }
                self.damage2normal = {
                    self.damages[i]: self.normals[i] for i in range(len(self.damages))
                }
                self.pairs = list(zip(self.normals, self.damages))
                # construct path2labels from normals and damages
                self.path2label = {self.damages[i]: 1 for i in range(len(self.damages))}
                self.path2label.update({self.normals[i]: 0 for i in range(len(self.normals))})
            else:
                rows = open(csv_path_or_filelist).readlines()
                if 'label' in rows[0] or 'path' in rows[0]:
                    rows = rows[1:]
                self.path2label = {
                    line.strip().split(",")[0]: (
                        int(line.strip().split(",")[1])
                        if len(line.strip().split(",")) > 1
                        else -1
                    )
                    for line in rows
                }

        # if the values are type string then they are the pairs
        # if the values are type int then they are the labels

        # anon_12539984_62954_55height_leviathanornamentsG-0391_page1rline13_char6_G_uc_perturb99_anomaly_162823437692_damage_bend_x36_y19_globalink_identity_localink_identity_1.tif
        # x, y = path.split('_')[14:16]
        self.transform = transform
        self.paths = []
        self.labels = []
        self.char_classes = []  # string representation of char classes
        self.char_classes_labels = []  # int encoded representation of char classes
        self.char2i = defaultdict(lambda: len(self.char2i))
        # self.i2char = {}
        self.char2paths = defaultdict(list)
        self.path2char = dict()
        self.pregenerate_fixed_jitter = pregenerate_fixed_jitter
        self.fixed_jitter_transforms = []
        self.fixed_jitter_transforms_damage = []
        self.fixed_jitter_transforms_original = []
        self.limit_train_data = limit_train_data

        if split != "test":
            self.balance_classes()
        for path, label in sorted(
            self.path2label.items(), key=lambda x: self.char_regex.findall(x[0])[0]
        ):
            # for path, label in self.path2label.items(): #sorted(self.path2label.items(), key=lambda x: self.char_regex.findall(x[0])[0]):
            self.paths.append(path)
            self.labels.append(label)
            # print(path, label)
            char = self.char_regex.findall(path)[0]
            self.char_classes_labels.append(self.char2i[char])
            self.char_classes.append(char)
            self.char2paths[char].append(path)
            self.path2char[path] = char
            if self.pregenerate_fixed_jitter:
                self.fixed_jitter_transforms.append(make_jitter_transform(apply_random_inking=self.args.apply_random_inking, apply_position_jitter=self.args.apply_position_jitter))
                self.fixed_jitter_transforms_damage.append(make_jitter_transform(apply_random_inking=self.args.apply_random_inking, apply_position_jitter=self.args.apply_position_jitter))
                self.fixed_jitter_transforms_original.append(make_jitter_transform(apply_random_inking=self.args.apply_random_inking, apply_position_jitter=self.args.apply_position_jitter))
        self.avg_templates = self.compute_avg_templates(limit_to_first_k=100)
        if not ('resnet' in self.args.encoder_type or 'vit' in self.args.encoder_type or 'hf_hub' in self.args.encoder_type):
            self.find_max_img_size()
        self.jitter = jitter

    # def pad_images(self, height=64, width=64):
    #     # TODO
    #     for char, paths in self.char2paths.items():
    #         images = [self.transform(Image.open(path)) for path in paths]
    #             max_height = max([img.size(1) for img in images])
    #             max_width = max([img.size(2) for img in images])
    #             padded_images = [

    #                 for img in images
    #         ]

    def balance_classes(self):
        balanced_path2label = dict()
        label2count = defaultdict(int)
        for path, label in self.path2label.items():
            label2count[label] += 1
        max_label_amount = min(min(label2count.values()), self.limit_train_data // 2)
        label2count = defaultdict(int)
        for path, label in self.path2label.items():
            if label2count[label] < max_label_amount:
                balanced_path2label[path] = label
                label2count[label] += 1
        self.path2label = balanced_path2label

    def get_num_channels(self):
        return 3 if Image.open(self.paths[0]).mode == "RGB" else 1

    def find_max_img_size(self):
        self.max_width = 0
        self.max_height = 0
        for char, paths in self.char2paths.items():
            for path in paths:
                try:
                    width, height = imagesize.get(path)
                except struct.error:
                    continue
                if width > self.max_width:
                    self.max_width = width
                if height > self.max_height:
                    self.max_height = height
        print("Max width/height:", self.max_width, self.max_height)
        # TODO:
        self.max_width = 140
        self.max_height = 140
        print("Forcing max width/height to be equal:", self.max_width, self.max_height)

    def compute_avg_templates(self, limit_to_first_k=None):
        avg_templates = dict()
        for char, paths in self.char2paths.items():
            images = []
            if limit_to_first_k is not None:
                random.shuffle(paths)
                paths = paths[:limit_to_first_k]
            # for path in tqdm(paths, desc=f'Building avg image for {char}'):
            for path in paths:
                try:
                    images.append(self.transform(Image.open(path)))
                except (PIL.UnidentifiedImageError, TypeError) as e:
                    print(f"Skipping {path} due to error reading image", e)
            max_height = max([img.size(1) for img in images])
            min_height = min([img.size(1) for img in images])
            max_width = max([img.size(2) for img in images])
            min_width = min([img.size(2) for img in images])
            padded_images = [
                F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])
                for img in images
            ]
            # mask_batch = [
            #     F.pad(mask, [0, max_width - mask.size(1), 0, max_height - mask.size(0)])
            #     for mask in mask_batch
            # ]
            avg_templates[char] = torch.mean(torch.stack(padded_images), dim=0)
        # print((min_height, max_height), (min_width, max_width))
        return avg_templates

    def get_avg_template(self, char):
        return self.avg_templates[char].unsqueeze(0)

    def save_avg_templates(self, output_dir, filename_prefix=""):
        for c, tensor_img in self.avg_templates.items():
            torchvision.transforms.ToPILImage()(tensor_img).save(
                Path(output_dir) / f"{filename_prefix}_template.png"
            )

    def _get_damage_loc(self, path):
        xy = self.damage_loc_regex.findall(path)
        if xy:
            return [int(xy[0][0]), int(xy[0][1])]
        return [-1, -1]

    def _create_circular_mask(self, h, w, center=None, radius=None):
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if (
            radius is None
        ):  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= radius
        return mask

    def _sample_non_damage_loc_xy(
        self, im, damage_loc_xy, min_distance_from_damage_loc=25
    ):
        """Sample a location on the character min_distance_from_damage_loc
        pixels from damage_loc_xy and return (x,y) coordinate
        """
        dam_x, dam_y = damage_loc_xy
        # create im mask with min_distance_from_damage_loc pixels from damage_loc_xy masked out
        im = np.array(im)
        h, w = im.shape[-2:]
        mask = ~self._create_circular_mask(
            h, w, center=damage_loc_xy, radius=min_distance_from_damage_loc
        )
        mask = mask & ~im.astype(bool)
        # get nonzero indices of mask for sampling candidates
        x_cands, y_cands = np.nonzero(mask.squeeze())
        # sample one index for x, y
        i = np.random.choice(len(x_cands))
        non_dam_loc_xy = (x_cands[i], y_cands[i])
        return non_dam_loc_xy

    def __getitem__(self, i):
        # image = Image.open(self.paths[i])
        if self.is_pair_data:
            image = Image.open(self.damages[i])
            original_image = Image.open(self.normals[i])
        else:
            image = Image.open(self.paths[i])
        if 'resnet' in self.args.encoder_type or 'vit' in self.args.encoder_type or 'hf_hub' in self.args.encoder_type:
            image = self.transform(image)
            if self.is_pair_data:
                original_image = self.transform(original_image)
        else:
            if image.mode == "RGB":
                color_transform = transforms.Compose(
                    [
                        ImagePadder(self.max_height, self.max_width, 0),
                        transforms.ToTensor(),
                    ]
                )
                im = color_transform(image)
            else:
                aligned_transform = transforms.Compose(
                    [
                        ImagePadder(self.max_height, self.max_width, 0),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.max() - x),
                    ]
                )
                self.transform = aligned_transform
                im = self.transform(image)
        char = self.path2char[self.paths[i]]
        avg_template = self.avg_templates[char]
        x, y = (
            self._get_damage_loc(self.paths[i])[0],
            self._get_damage_loc(self.paths[i])[1],
        )

        if self.pregenerate_fixed_jitter:
            image = self.fixed_jitter_transforms_damage[i](image)
            if self.is_pair_data:
                original_image = self.fixed_jitter_transforms_original[i](original_image)
        elif self.jitter:
            jitter_transform_damage = make_jitter_transform(apply_random_inking=self.args.apply_random_inking, apply_position_jitter=self.args.apply_position_jitter)
            jitter_transform_original = make_jitter_transform(apply_random_inking=self.args.apply_random_inking, apply_position_jitter=self.args.apply_position_jitter)
            if False and self.labels[i]:
                import ipdb; ipdb.set_trace()
                xy_mask = torch.ones_like(image)
                xy_mask[0, x, y] = 0.0
                xy_prejitter = (xy_mask[0] == 0.0).nonzero(as_tuple=True)
                xy_mask_jittered = jitter_transform(xy_mask)
                xy_postjitter = (xy_mask_jittered[0] == 0.0).nonzero(as_tuple=True)
                print(xy_postjitter[0], xy_postjitter[1])
                import ipdb; ipdb.set_trace()
                x, y = xy_postjitter[0].item(), xy_postjitter[1].item()
            image = jitter_transform_damage(image)
            if self.is_pair_data:
                original_image = jitter_transform_original(original_image)
            # TODO: avg_template jitter
            # avg_template = jitter_transform(avg_template)
            # if self.labels[i]:
            #     import ipdb; ipdb.set_trace()

        return {
            "image": image,
            "original_image": original_image if self.is_pair_data else image,
            "label": self.labels[i],
            "original_label": 0,
            "path": self.paths[i],
            "damage_loc_xy": [x, y],
            "non_damage_loc_xy": self._sample_non_damage_loc_xy(image, (x, y))
            if x != -1
            else (-1, -1),
            "avg_template": avg_template,
            "char": char,
            "char_idx": self.char2i[char],
        }

    def get_labels(self):
        return self.labels

    def get_char_classes_labels(self):
        return self.char_classes_labels

    def __len__(self):
        return len(self.damages) if self.is_pair_data else len(self.paths)

class DamageDatasetFilelist(Dataset):
    def __init__(
        self,
        args,
        split,
        transform,
        filelist,
        pregenerate_fixed_jitter=False,
        jitter=False,
        limit_train_data=sys.maxsize,
    ):
        self.args = args
        self.split = split
        if isinstance(filelist, list):
            self.paths = filelist
        else:
            self.paths = [
                line.strip()
                for line in open(filelist).readlines()
            ]

        self.transform = transform
        self.pregenerate_fixed_jitter = pregenerate_fixed_jitter
        self.fixed_jitter_transforms = []
        self.limit_train_data = limit_train_data

        if split != "test":
            self.balance_classes()
        for path in self.paths:
            if self.pregenerate_fixed_jitter:
                self.fixed_jitter_transforms.append(make_jitter_transform(apply_random_inking=self.args.apply_random_inking, apply_position_jitter=self.args.apply_position_jitter))
        if not ('resnet' in self.args.encoder_type or 'vit' in self.args.encoder_type or 'hf_hub' in self.args.encoder_type):
            self.find_max_img_size()
        self.jitter = jitter

    def get_num_channels(self):
        return 3 if Image.open(self.paths[0]).mode == "RGB" else 1

    def find_max_img_size(self):
        self.max_width = 0
        self.max_height = 0
        for char, paths in self.char2paths.items():
            for path in paths:
                try:
                    width, height = imagesize.get(path)
                except struct.error:
                    continue
                if width > self.max_width:
                    self.max_width = width
                if height > self.max_height:
                    self.max_height = height
        print("Max width/height:", self.max_width, self.max_height)
        # TODO:
        self.max_width = 140
        self.max_height = 140
        print("Forcing max width/height to be equal:", self.max_width, self.max_height)

    def save_avg_templates(self, output_dir, filename_prefix=""):
        for c, tensor_img in self.avg_templates.items():
            torchvision.transforms.ToPILImage()(tensor_img).save(
                Path(output_dir) / f"{filename_prefix}_template.png"
            )

    def __getitem__(self, i):
        image = Image.open(self.paths[i])
        if 'resnet' in self.args.encoder_type or 'vit' in self.args.encoder_type or 'hf_hub' in self.args.encoder_type:
            im = self.transform(image)
        else:
            if image.mode == "RGB":
                color_transform = transforms.Compose(
                    [
                        ImagePadder(self.max_height, self.max_width, 0),
                        transforms.ToTensor(),
                    ]
                )
                im = color_transform(image)
            else:
                aligned_transform = transforms.Compose(
                    [
                        ImagePadder(self.max_height, self.max_width, 0),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.max() - x),
                    ]
                )
                self.transform = aligned_transform
                im = self.transform(image)

        if self.pregenerate_fixed_jitter:
            image = self.fixed_jitter_transforms[i](im)
        elif self.jitter:
            jitter_transform = make_jitter_transform(apply_random_inking=self.args.apply_random_inking, apply_position_jitter=self.args.apply_position_jitter)
            if False and self.labels[i]:
                import ipdb; ipdb.set_trace()
                xy_mask = torch.ones_like(im)
                xy_mask[0, x, y] = 0.0
                xy_prejitter = (xy_mask[0] == 0.0).nonzero(as_tuple=True)
                xy_mask_jittered = jitter_transform(xy_mask)
                xy_postjitter = (xy_mask_jittered[0] == 0.0).nonzero(as_tuple=True)
                print(xy_postjitter[0], xy_postjitter[1])
                import ipdb; ipdb.set_trace()
                x, y = xy_postjitter[0].item(), xy_postjitter[1].item()
            image = jitter_transform(im)
            avg_template = jitter_transform(avg_template)
            # if self.labels[i]:
            #     import ipdb; ipdb.set_trace()
        else:
            image = im

        return {
            "image": im,
            "path": self.paths[i],
        }

    def __len__(self):
        return len(self.paths)


class ImagePathDataset(Dataset):
    """loads a list of files (or is passed a list of them)"""

    def __init__(self, transform, image_paths_file=None, image_paths=None):
        assert image_paths_file is not None or image_paths is not None, (
            "Either image_paths_file or image_paths must be provided."
        )
        self.image_paths = (
            image_paths
            if isinstance(image_paths, list)
            else [p.strip() for p in open(image_paths_file).readlines()]
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # TODO: error checking?
        # try:
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        # except (PIL.UnidentifiedImageError, FileNotFoundError) as e:
        #     print(f"Error: {e}. Skipping {p}.")
        #     return None, None
        return image_tensor, image_path


class MultiCharTwinDataset(Dataset):
    """
    Holds multiple TwinDataset objects and combines them
    to enable multiple character training. To do so, it combines the
    TwinDatasets' pairs attribute, while keeping track of
    each pair's character class label and the associated character img transform.
    """

    def __init__(self, args, split, datasets_dict=None):
        self.split = split
        self.pair_datasets_dict = dict()
        self.damage_datasets_dict = dict()
        self.pairs = []
        self.args = args
        self.pairs_labels = []
        self.damage_images = []
        self.normal_images = []
        self.damage_images_labels = []
        self.char2transform = dict()
        self.pair_idx2og_idx = []
        self.damage_idx2og_idx = []
        # self.avg_image = dict()
        if datasets_dict is not None:
            for char, dataset in self.pair_datasets_dict.items():
                self.add_char_dataset_pair(char, dataset)
        self.char2idx = (
            dict()
        )  # {char: i for i, char in enumerate(self.datasets_dict.keys())}
        self.idx2char = dict()  # {i: char for char, i in self.char2idx.items()}
        # self.avg_image_tensor = torch.zeros((1, 64, 64), dtype=torch.float)
        # for char, avg_image in self.avg_image.items():
        #     self.avg_image_tensor[self.char2idx[char]] = avg_image

    def add_char_dataset_pair(self, char, dataset, limit_dataset_size):
        print(
            f"Adding {char} dataset ({len(dataset.pairs[:limit_dataset_size])} pairs)."
        )
        assert char not in self.pair_datasets_dict, (
            f"Character {char} already in pair dataset."
        )
        self.pair_datasets_dict.update({char: dataset})
        self.char2transform[char] = dataset.transform
        # self.avg_image[char] = dataset.avg_image
        for og_idx, test_pair in enumerate(dataset.pairs[:limit_dataset_size]):
            self.pairs.append(test_pair)
            self.pairs_labels.append(char)
            self.pair_idx2og_idx.append(og_idx)
        if char not in self.char2idx:
            cur_len = len(self.char2idx)
            self.char2idx[char] = cur_len
            self.idx2char[cur_len] = char
            # if cur_len == 0:
            #     self.pair_avg_image_tensor = dataset.avg_image.unsqueeze(0).unsqueeze(1)
            # else:
            #     self.pair_avg_image_tensor = torch.cat([self.pair_avg_image_tensor, dataset.avg_image.unsqueeze(0).unsqueeze(1)], dim=0)

    def get_char_classes_labels(self):
        """return int encoded labels of char class for each pair in entire dataset"""
        return [self.char2idx[self.pairs_labels[i]] for i in range(len(self.pairs))]

    def add_char_dataset_damage(self, char, dataset):
        # if limit_dataset_size is None:
        #     limit_dataset_size = len(dataset)
        # NOTE: normal images from this dataset are just sampled so we don't need to keep track of idxs
        print(f"Adding {char} dataset: {dataset}")
        assert char not in self.damage_datasets_dict, (
            f"Character {char} already in damage dataset."
        )
        self.damage_datasets_dict.update({char: dataset})
        for og_idx in range(len(dataset)):
            #     self.damage_images.append(dataset.damage_images[og_idx])
            #     self.normal_images.append(dataset.normal_images[og_idx])
            self.damage_images_labels.append(char)
            self.damage_idx2og_idx.append(og_idx)
        if char not in self.char2idx:
            cur_len = len(self.char2idx)
            self.char2idx[char] = cur_len
            self.idx2char[cur_len] = char

    # def get_avg_image(self, char_idx):
    #     return self.pair_avg_image_tensor[char_idx]

    def indices_for_char(self, char):
        """Returns the list of data indices with the `char` character label"""
        indices = []
        for i in range(len(self.pairs_labels)):
            if self.pairs_labels[i] == char:
                indices.append(i)
        return indices

    def pair_idx_to_char(self, idx):
        return self.pairs_labels[idx]

    def damage_idx_to_char(self, idx):
        return self.damage_images_labels[idx]

    def __getitem__(self, index):
        # dispatches TwinDataset __getitem__ method, where it uses its transform to load the image
        item = self.pair_datasets_dict[
            self.pair_idx_to_char(index)
        ].__getitem__(self.pair_idx2og_idx[index])
        # (normal_img, damage_img) = self.damage_datasets_dict[self.damage_idx_to_char(index)].__getitem__(self.damage_idx2og_idx[index])
        # damage_dataset = self.damage_datasets_dict[self.damage_idx_to_char(index)]
        # if self.args.damage_detection:
        #     return (img1, img2), self.pair_idx_to_char(index), (damage_dataset.sample_normal_image(), damage_dataset.sample_damage_image())
        # else:
        if self.args.debug:
            print(f"Pair index: {index}, Char: {self.pair_idx_to_char(index)}")
            # import ipdb; ipdb.set_trace()
        item.update({'char': self.pair_idx_to_char(index)})
        return item

    @staticmethod
    def prepare_image(img_path, transform):
        pil_img = Image.open(img_path)
        # print(pil_img.size)
        img = transform(pil_img)
        return img

    def __len__(self):
        return len(self.pairs)


def compute_avg_image(img_paths, transform=None):
    """Use optional transform argument if you want to Binarize the avg_image"""
    avg_image = torch.zeros(
        np.array(Image.open(img_paths[0]), dtype=np.float64).shape, dtype=torch.float
    )
    for img_path in img_paths:
        avg_image += (
            transforms.ToTensor()(np.array(Image.open(img_path), dtype=np.float64))
            .to(torch.float)
            .squeeze()
        )
    avg_image /= len(img_paths)

    if transform is None:
        return avg_image
    else:
        return transform(Image.fromarray(avg_image.cpu().numpy()))
    return avg_image


class PairedBalancedDamageDetectionDataset(Dataset):
    """
    PAIRED Dataset for balanced binary classification of normal/damaged characters (anomalies)
    """

    def __init__(self, root_dir, transform, shuffle=True):
        self.normal_dir = Path(root_dir) / "normal"
        self.damage_dir = Path(root_dir) / "anomaly"
        self.normal_images = list(self.normal_dir.glob("*.tif"))
        self.damage_images = list(self.damage_dir.glob("*.tif"))
        np.random.shuffle(self.normal_images)
        np.random.shuffle(self.damage_images)
        self.transform = transform

    def __getitem__(self, index):
        normal_img = self.transform(Image.open(self.normal_images[index]))
        damage_img = self.transform(Image.open(self.damage_images[index]))
        return normal_img, damage_img

    def __len__(self):
        return min(len(self.normal_images), len(self.damage_images))

    def __str__(self):
        return f"DamageDetectionDataset({len(self.normal_images)} normal images, {len(self.damage_images)} damaged images)"


class DamageDetectionDataset(Dataset):
    """
    Dataset for binary classification of normal/damaged characters (anomalies)
    """

    def __init__(self, root_dir, transform, shuffle=True):
        self.NORMAL = 0
        self.DAMAGE = 1
        self.normal_dir = Path(root_dir) / "normal"
        self.damage_dir = Path(root_dir) / "anomaly"
        self.normal_images = list(self.normal_dir.glob("*.tif"))
        self.damage_images = list(self.damage_dir.glob("*.tif"))
        self.normal_images_with_labels = list(
            zip(self.normal_images, len(self.normal_images) * [self.NORMAL])
        )
        self.damage_images_with_labels = list(
            zip(self.damage_images, len(self.damage_images) * [self.DAMAGE])
        )
        self.all_images_with_labels = (
            self.normal_images_with_labels + self.damage_images_with_labels
        )
        if shuffle:
            np.random.shuffle(self.all_images_with_labels)
        self.all_images_normal_indices = [
            i
            for i in range(len(self.all_images_with_labels))
            if self.all_images_with_labels[i][1] == self.NORMAL
        ]
        self.all_images_damage_indices = [
            i
            for i in range(len(self.all_images_with_labels))
            if self.all_images_with_labels[i][1] == self.DAMAGE
        ]
        self.transform = transform

    def __getitem__(self, index):
        img_path, y = self.all_images_with_labels[index]
        img = self.transform(Image.open(img_path))
        return img, y

    def __len__(self):
        return len(self.all_images_with_labels)

    def sample_normal_image(self):
        return self.__getitem__(np.random.choice(self.all_images_normal_indices))[
            0
        ]  # return just the image

    def sample_damage_image(self):
        return self.__getitem__(np.random.choice(self.all_images_damage_indices))[
            0
        ]  # return just the image

    def get_num_normal_images(self):
        return len(self.normal_images)

    def get_num_damage_images(self):
        return len(self.damage_images)


def save_img(img, name, dir="scratch"):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif isinstance(img, torch.Tensor):
        if img.max() == 1:  # if it's binary, convert to uint8
            img = img.mul(255).byte()
        img = Image.fromarray(img.squeeze().numpy())
    dir = Path(dir)
    dir.mkdir(exist_ok=True, parents=True)
    img.save(str(dir / f"{name}.png"))


def get_random_global_inking_transform(low=0.0, high=0.3):
    amt = np.random.uniform(low, high)
    op = np.random.choice([perturb.BinaryThinning(amt), perturb.BinaryThickening(amt)])
    return op


class TwinDataset(Dataset):
    """Contains paired examples for training matcher"""

    def __init__(self, args, split_name, **kwargs):
        self.setup_data(args, split_name)
        self.args = args

    def setup_data(self, args, split_name):
        self.data_path = Path(args["data_path"])
        self.transform = args["transform"]
        self.num_samples = args["num_samples"]
        self.random_seed = (
            args["random_seed"] if args["random_seed"] is not None else 42
        )
        self.random_state = np.random.RandomState(self.random_seed)
        img_paths = [str(p) for p in self.data_path.glob("*.tif")]
        self.normal_img_paths = sorted([str(self.data_path / Path(p).name) for p in img_paths if '_original' in Path(p).name])
        self.img_paths = sorted([str(self.data_path / Path(p).name) for p in img_paths if '_original' not in Path(p).name])
        # set the filename as index
        df = pd.DataFrame(self.img_paths, columns=["file_path"])
        df["fname"] = df["file_path"].apply(lambda x: x.split("/")[-1])
        df.set_index("fname", inplace=True)

        assert len(self.img_paths) % 2 == 0, "One or more image(s) is missing a twin!"
        print(
            f"{split_name}: {len(self.img_paths) // 2} image pairs, ({len(self.img_paths)} images in total)"
        )
        # self.img_name_pattern = re.compile(r"(.*)_[12].tif$") # old one where there was no damage settings
        # self.pair_id_pattern = re.compile(r".*_([12]).tif$")
        # self.img_name_pattern = re.compile(r"(.*?)_damage.*?_[12].tif$") # works for same base image
        self.img_name_pattern = re.compile(
            r".*?_anomaly_([0-9]+)_damage.*?_[12].tif$"
        )  # should work for both same/diff base images
        self.pair_id_pattern = re.compile(r".*?_damage.*?_([12]).tif$")

        df["image_class"] = df["file_path"].apply(
            functools.partial(
                self.extract_image_name_pairid, cpattern=self.img_name_pattern
            )
        )
        df["pair_id"] = df["file_path"].apply(
            functools.partial(
                self.extract_image_name_pairid, cpattern=self.pair_id_pattern
            )
        )
        df["image_class_id"] = df.groupby(
            "image_class"
        ).ngroup()  # assigns same number to each pair
        labels = df["image_class_id"].values.tolist()
        self.labels = labels  # use with self.img_paths
        self.df = df

        if self.num_samples is not None:
            all_pairs = list(set(self.labels))
            self.random_state.shuffle(all_pairs)
            pairs_subset = all_pairs[: self.num_samples]
            subset_img_paths = []
            subset_labels = []
            labels_copy = copy.deepcopy(self.labels)
            for pair_id in pairs_subset:
                idx1 = labels_copy.index(pair_id)
                labels_copy[idx1] = -1
                idx2 = labels_copy.index(pair_id)
                subset_labels.extend([pair_id, pair_id])
                img_path1 = self.img_paths[idx1]
                img_path2 = self.img_paths[idx2]
                subset_img_paths.extend([img_path1, img_path2])

            assert len(subset_img_paths) == len(subset_labels) == 2 * self.num_samples, (
                "data length does not match `num_samples`"
            )
            self.img_paths = subset_img_paths
            self.labels = subset_labels
            print(
                f"Shrunk dataset to {len(self.img_paths) // 2} image pairs, ({len(self.img_paths)} images in total)"
            )
        self.pattern = r"(?P<char_stamp>.*?_perturb)[0-9]+_(?P<class>[a-z]+)_(?P<timestamp>[0-9]+)_damage_(?P<damage>.+)_globalink_(?P<global_inking>[a-z]+)_localink_(?P<local_inking>[a-z]+)_[12]\.tif$"
        self.pattern_normal = r"(?P<char_stamp>.*?_perturb)[0-9]+_(?P<class>[a-z]+)_(?P<timestamp>[0-9]+)_original\.tif$"
        self.cpattern = re.compile(self.pattern)
        self.cpattern_normal = re.compile(self.pattern_normal)
        self.negative_groups = collections.defaultdict(set)
        self.out_dist = collections.defaultdict(int)
        self.groups = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )  # dict of dict of list
        self.img_details = dict()
        self.idx2img = dict()
        self.split_name = split_name
        self.transform = args["transform"]
        self.cstamps = collections.defaultdict(int)
        self.same_stamp_negative_groups = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        self.damage2normal = dict()

        for idx, img_path in enumerate(self.img_paths):
            img_name = Path(img_path).name
            parts = self.get_fname_parts(img_name)
            char_stamp = parts["char_stamp"]
            timestamp = parts["timestamp"]
            damage = parts["damage"]
            global_ink = parts["global_inking"]
            local_ink = parts["local_inking"]
            img_data = (idx, img_name)
            self.groups["char_stamp"][char_stamp].add(img_data)
            self.cstamps[char_stamp] += 1
            self.groups["timestamp"][timestamp].add(img_data)
            self.groups["damage"][damage].add(img_data)
            self.groups["global_ink"][global_ink].add(img_data)
            self.groups["local_ink"][local_ink].add(img_data)
            self.same_stamp_negative_groups[char_stamp][damage].add(img_data)
            self.img_details[img_name] = parts
            self.idx2img[idx] = img_name
            self.damage2normal[img_path] = self.normal_img_paths[idx]
        
        for idx, img_path in enumerate(self.normal_img_paths):
            img_name = Path(img_path).name
            parts = self.get_fname_parts(img_name)
            self.img_details[img_name] = parts

        self.labels_set = set(self.labels)
        ls = np.array(self.labels)
        self.label_to_indices = {
            label: np.where(ls == label)[0] for label in self.labels_set
        }

        positive_pairs = []
        for label in sorted(self.labels_set):
            idxs = self.label_to_indices[label]
            if len(idxs) == 2:  # using same/diff base without original image for reference
                img1_name, img2_name = self.idx2img[idxs[0]], self.idx2img[idxs[1]]
            elif len(idxs) == 1:  # using same with original image for reference
                img1_name = self.idx2img[idxs[0]]
                img2_name = '_'.join(img1_name.split('_')[:-7]) + '_original.tif'

            assert img1_name != img2_name
            assert (
                self.img_details[img1_name]["timestamp"]
                == self.img_details[img2_name]["timestamp"]
            )
            # NOTE: duplicated pairs are added if using same with original image for reference 
            positive_pair = (idxs[0], idxs[1], 1) if len(idxs) == 2 else (idxs[0], idxs[0], 1)
            positive_pairs.append(positive_pair)

        # negative_pairs = []
        # neg_percent = (1.0 * len(negative_pairs) / len(positive_pairs)) * 100
        # print(f"Num of synthetic pairs: {len(positive_pairs)}")#, num negative pairs: {len(negative_pairs)} ({neg_percent:.2f} % of positive pairs)")
        # print(split_name, {k:len(v) for k,v in self.negative_groups.items()})
        self.pairs = positive_pairs  # + negative_pairs
        # self.avg_image = compute_avg_image(self.img_paths, transform=None)
        self.gen_histogram()

    @staticmethod
    def extract_image_name_pairid(fpath, cpattern):
        fname = Path(fpath).name
        match = re.findall(cpattern, fname)[0]
        return match

    def __len__(self):
        return len(self.img_paths)

    def plot_gray_img(self, img):
        if img.shape[0] == 1:
            img = img.squeeze(0)
        plt.imshow(img, cmap="gray")

    def gen_histogram(self):
        f = plt.figure()
        xs = []
        data = {k: len(v) for k, v in self.negative_groups.items()}
        sorted_data = dict(sorted(data.items(), key=lambda x: x[1]))
        plt.barh(list(sorted_data.keys()), list(sorted_data.values()))
        plt.grid(True, alpha=0.7)
        f.savefig(f"negative-hist-{self.split_name}.png", bbox_inches="tight")
        plt.close()

    def get_fname_parts(self, s):
        perturbation = dict()
        s = str(s).split("/")[-1]
        if "_original" in s:
            match = self.cpattern_normal.match(s)
            assert match is not None, f"Name {s} (normal image) has no matching pattern!"
            perturbation["char_stamp"] = match.group("char_stamp")
            perturbation["timestamp"] = match.group("timestamp")
            return perturbation

        match = self.cpattern.match(s)
        assert match is not None, f"Name {s} (synthetically damaged file) has no matching pattern!"
        if "_" in match.group("damage"):
            perturbation["damage"] = np.random.choice(
                tuple(sorted(match.group("damage").split("_")))
            )
        else:
            perturbation["damage"] = match.group("damage")
        perturbation["global_inking"] = match.group("global_inking")
        perturbation["local_inking"] = match.group("local_inking")
        perturbation["char_stamp"] = match.group("char_stamp")
        perturbation["timestamp"] = match.group("timestamp")
        return perturbation


    def __getitem__(self, index):
        # print(f"index={index}, {self.split_name}")
        # try:
        img1_idx, img2_idx, target = self.pairs[index]
        img1_path = self.img_paths[img1_idx]
        img1 = Image.open(img1_path)
        img2_path = self.img_paths[img2_idx]
        img2 = Image.open(img2_path)
        normal_img_path = self.damage2normal[img1_path]
        normal_img = Image.open(normal_img_path)

        img1_raw = img1
        img2_raw = img2

        # default true
        if self.split_name == "train" and self.args.add_global_inking:
            make_morph_object = lambda x: morpho.ImageMorphology(x, scale=1)
            inv_img1_raw = invert(np.array(img1_raw).astype(bool))
            inv_img2_raw = invert(np.array(img2_raw).astype(bool))
            img_morph1 = make_morph_object(inv_img1_raw)
            img_morph2 = make_morph_object(inv_img2_raw)
            ops = [
                get_random_global_inking_transform(),
                get_random_global_inking_transform(),
            ]
            if self.args.debug:
                save_img(img1_raw, "pre_global_ink_img1")
                save_img(img2_raw, "pre_global_ink_img2")
                print(f"Saved pre_global_ink images for index {index}")
            try:
                img1_raw = Image.fromarray(invert(ops[0](img_morph1)))
                img2_raw = Image.fromarray(invert(ops[1](img_morph2)))
            except ValueError:
                img1_raw = img1
                img2_raw = img2


            if self.args.debug:
                save_img(img1_raw, "post_global_ink_img1")
                save_img(img2_raw, "post_global_ink_img2")
                print(f"Saved post_global_ink images for index {index}")

        img1 = self.transform(img1_raw)
        img2 = self.transform(img2_raw)
        normal_img = self.transform(normal_img)

        if self.split_name == "train":  #and self.args.add_salt_pepper_noise:
            # apply salt and pepper noise to img1_raw and img2_raw
            img1 = transforms.ToTensor()(add_salt_and_pepper_noise(transforms.ToPILImage()(img1), amount=random.uniform(0.01, 0.1)))
            img2 = transforms.ToTensor()(add_salt_and_pepper_noise(transforms.ToPILImage()(img2), amount=random.uniform(0.01, 0.1)))
            normal_img = transforms.ToTensor()(add_salt_and_pepper_noise(transforms.ToPILImage()(normal_img), amount=random.uniform(0.01, 0.1)))


        if self.args.debug:
            save_img(img1, f"{str(index).zfill(5)}_post_transform_img1")
            save_img(img2, f"{str(index).zfill(5)}_post_transform_img2")
            save_img(normal_img, f"{str(index).zfill(5)}_post_transform_normal_img")
            print(f"Saved post_transform images for index {index}")


        # import ipdb; ipdb.set_trace()
        # sample = (img1, img2), target
        # return sample
        return {
            "damage1": img1,
            "damage2": img2,
            "label": target,
            "normal": normal_img,
        }

    @staticmethod
    def prepare_image(img_path, transform):
        pil_img = Image.open(img_path)
        # print(pil_img.size)
        img = transform(pil_img)
        return img

    def __len__(self):
        return len(self.pairs)

    def plot(self, img1, img2, label=None):
        ax = plt.subplot(1, 2, 1)
        plt.imshow(img1[0].cpu().numpy())
        ax.set_title("image1")
        plt.axis("off")
        ax = plt.subplot(1, 2, 2)
        plt.imshow(img2[0].cpu().numpy())
        ax.set_title("image2")
        plt.axis("off")
        if label is not None:
            plt.gcf().suptitle(f"Label: {label}")
        plt.show()


class GroundTruthMatchesDataset(Dataset):
    def __init__(
        self, csv_path, transform, limit_num_matches_to=sys.maxsize
    ):  # default is don't limit
        self.csv_path = csv_path
        self.limit_num_matches_to = limit_num_matches_to
        self.match_dir = Path(csv_path).parent
        # get all pairs dict
        self.match_img_paths = utils.get_gold_matches(csv_path)
        assert len(self.match_img_paths) > 0, f"No matches found in {csv_path}"
        self.match_img_names = set(
            [
                Path(p).name
                for i, p in enumerate(self.match_img_paths.keys())
                if i < self.limit_num_matches_to
            ]
        )
        self.match_img_items_in_dict = []
        self.img_paths = set()
        for k, vs in self.match_img_paths.items():
            try:
                k_path = self.match_dir / k
                for v in vs:
                    v_path = self.match_dir / v
                    if k_path.exists() and v_path.exists():
                        self.match_img_items_in_dict.append((k_path, v_path))
                        self.img_paths.add(k_path)
                        self.img_paths.add(v_path)
            except KeyError:
                continue
        self.match_img_items_in_dict = self.match_img_items_in_dict[
            : self.limit_num_matches_to
        ]
        self.img_paths = list(self.img_paths)
        # self.match_img_items_in_dict = [(self.match_dir/k, self.match_dir/v.pop()) for k, v in self.match_img_paths.items()][:self.limit_num_matches_to]
        self.pairs = self.match_img_items_in_dict
        self.transform = transform
        # self.avg_image = compute_avg_image(self.img_paths, transform=None)

    def __getitem__(self, index):
        img1_path, img2_path = self.match_img_items_in_dict[index]
        img1 = self.transform(Image.open(img1_path))
        img2 = self.transform(Image.open(img2_path))
        return (img1, img2), 1  # 1 indicates that this pair is a match (target)

    def __len__(self):
        return len(self.match_img_items_in_dict)

    @staticmethod
    def prepare_image(img_path, transform):
        pil_img = Image.open(img_path)
        img = transform(pil_img)
        return img

    @classmethod
    def prepare_images(cls, img_path1, img_path2, transform):
        img1 = cls.prepare_image(img_path1, transform)
        img2 = cls.prepare_image(img_path2, transform)
        img12 = torch.cat([img1, img2], dim=0)  # 2HW
        img12 = img12.unsqueeze(0)
        return img12


def create_datasets(
    args,
    splits_dir,
    dataset_args,
    limit_num_matches_to=sys.maxsize,
    ds_class=TwinDataset,
    splits=("train", "valid", "test"),
):
    """
    Creates train, valid and test twin matching datasets
    """
    datasets = dict()
    for split in splits:
        split_dir = splits_dir / split
        dataset_args = copy.deepcopy(dataset_args)
        dataset_args["data_path"] = split_dir
        ds = None
        if ds_class == GroundTruthMatchesDataset:
            matches_csv_path = split_dir / "matches.csv"
            ds = ds_class(
                matches_csv_path, dataset_args["transform"], limit_num_matches_to
            )
        else:
            ds = ds_class(
                dataset_args,
                split_name=split,
            )
        ds.args = args
        datasets[split] = ds
    return datasets


def generate_matches_csv(file_list, dest_dir, n_pairs=None, sort=False):
    """Creates a matches.csv file of matching pairs from a list of file names used for matching.
    Also copies the files over to the dest_dir.

    Example
    -------
    file_list = ['ab_1_1', 'ab_1_2', 'ab_2_1', 'ab_2_2', 'bb_1_1', 'bb_1_2', 'bb_2_1', 'bb_2_2']
    random.shuffle(file_list)
    generate_matches_csv(file_list, "/home/kishore/data/anomaly-detection/gold-test-sets/fake-test-sets/G/valid", 10)

    """
    # sort to get matches as neighbors
    if sort:
        sorted_file_list = sorted(file_list)
    else:
        sorted_file_list = file_list
    matches = []
    for p1, p2 in zip(sorted_file_list[0::2], sorted_file_list[1::2]):
        matches.append((Path(p1), Path(p2)))
    # randomly subselect n matches
    if n_pairs is not None:
        idxs = np.random.choice(list(range(len(matches))), size=n_pairs, replace=False)
        new_matches = []
        for idx in idxs:
            new_matches.append(matches[idx])
        matches = new_matches
    for p1, p2 in matches:
        shutil.copy2(p1, dest_dir)
        shutil.copy2(p2, dest_dir)
    with open(Path(dest_dir) / "matches.csv", "w") as f:
        f.write("\n".join([f"{match[0].name},{match[1].name}" for match in matches]))
    return matches
