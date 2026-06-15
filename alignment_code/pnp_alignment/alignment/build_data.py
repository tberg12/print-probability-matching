import argparse
import os
import torch
import torch.nn.functional as F
import math
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from random import sample
import traceback
import skimage.filters
import matplotlib.pyplot as plt
import csv
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import pad
from torchvision.transforms.functional import to_grayscale


def build_data(args, image_data):
    maxh = 0
    maxw = 0
    images = []
    print()
    print(f"File list has {len(image_data)} images")
    zero_hw_filenames = []
    for d in image_data:
        filename = d["inp_path"]
        try:
            imp = to_grayscale(Image.open(filename))
        except Exception as e:
            print(f"{traceback.print_exc()}\n{e}: skipping the image: " + str(filename))
            # import ipdb; ipdb.set_trace()
        else:
            width, height = imp.size
            if ((width > args.image_size) or (height > args.image_size)) and (
                width > 0 and height > 0
            ):
                maxres = max([width, height])
                ratio = args.image_size * 1.0 / maxres
                width = math.floor(width * ratio)
                height = math.floor(height * ratio)
                if width == 0 or height == 0:
                    zero_hw_filenames.append(filename)
                else:
                    imp = imp.resize(size=(width, height))
            if width > maxw:
                maxw = width
            if height > maxh:
                maxh = height
    maxh = args.image_size
    maxw = args.image_size
    im_id = -1
    print(f"File list has {len(zero_hw_filenames)} zero h/w files:")
    for filename in zero_hw_filenames:
        print(filename)
        image_data.remove(filename)
    print(f"File list has {len(image_data)} images after removing the zero h/w files")
    for filename in image_data:
        try:
            imp = to_grayscale(Image.open(filename))
        except:
            print("skipping the image: " + str(filename))
        else:
            im_id += 1
            width, height = imp.size
            orig_w = width
            orig_h = height
            if (width > args.image_size) or (height > args.image_size):
                maxres = max([width, height])
                ratio = args.image_size * 1.0 / maxres
                width = math.floor(width * ratio)
                height = math.floor(height * ratio)
                imp = imp.resize(size=(width, height))
            wdiff = maxw - width
            r = math.floor(wdiff / 2.0)
            l = math.ceil(wdiff / 2.0)
            hdiff = maxh - height
            b = math.floor(hdiff / 2.0)
            t = math.ceil(hdiff / 2.0)
            wfrac = width * 1.1 / maxw
            hfrac = height * 1.1 / maxh
            try:
                if args.binarize_method == 'sauvola':
                    imp = skimage.filters.threshold_sauvola(imp, window_size=25)
                elif args.binarize_method == 'otsu':
                    imp = skimage.filters.threshold_otsu(imp)
                elif args.binarize_method == 'manual':
                    imp = 
                    # imp = binarizer(
                    #     pad(imp, (l, t, r, b), fill=255, padding_mode="constant"),
                    #     thresh=binthresh[filename],
                    # )
            except:
                print("couldn't binarize: " + str(filename))
            else:
                imp = pad(
                    imp, (l, t, r, b), fill=255, padding_mode="constant"
                ).convert("1")
                imp = ToTensor()(imp)  # here imp changes to H,W
                imp = imp.squeeze()  # .to(device)
                images.append(imp)
    # sample 1000 images and average them
    init_imps = [im.inp_img for im in sample(images, min(1000, len(images)))]
    template = torch.stack(
        [
            F.interpolate(
                imp.unsqueeze(0).unsqueeze(0), (args.image_size, args.image_size), mode="area"
            )
            .squeeze()
            .reshape(args.image_size * args.image_size)
            for imp in init_imps
        ]
    ).mean(dim=0, keepdim=True)  # 1, image_size*image_size
    save_image(template.view(1, 1, args.image_size, args.image_size), "average_image.png", normalize=True)
    return images, template


def rank_images_by_distance_from_template(images, template):
    dists = []
    for im in images:
        dists.append(torch.dist(im.inp_img.view(-1), template.view(-1)))
    # sort and return images and distances as two separate lists
    dists, images = zip(*sorted(zip(dists, images)))
    return images, dists


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("aligncsv", type=str)
    parser.add_argument("--image_size", type=int, default=64, help="size of the image")
    parser.add_argument("--binarize_method", type=str, default='sauvola', choices=['sauvola', 'otsu', 'manual', 'pre'], help="binarization method")
    """
    python build_data.py --image_size 64
    """
    args = parser.parse_args()

    image_data = []
    with open(args.aligncsv, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            image_data.append(
                {
                    "inp_path": row[0],
                    "out_path": row[1],
                }
            )
            if len(row) > 2:
                image_data[-1]["binthresh"] = row[2]

    images, template = build_data(args, image_data)
    images, dists = rank_images_by_distance_from_template(images, template)
    # plot dists
    plt.plot(dists)
    plt.show()

    print(f"images: {len(images)}")
    print(f"template: {template.shape}")
    print(f"template: {template}")
