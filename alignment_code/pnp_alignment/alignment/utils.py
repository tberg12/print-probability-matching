import torch
import torch.nn as nn
import time
import sys
import math
import os
from torch import cuda

# import torch_dct
from random import shuffle
from Optim import Optim
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.nn import Parameter
import torch.utils.data as data
from collections import defaultdict
from collections import Counter
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torchvision.transforms import ToPILImage
from torchvision.transforms import RandomAffine
from torchvision.transforms.functional import affine
from torchvision.transforms.functional import pad
from torchvision.transforms.functional import to_grayscale
from torchvision.utils import save_image

# import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
from PIL import Image, ImageOps
import cv2 as cv
import numpy as np
import random
import skimage.filters


class Binarizer:
    def __init__(self, output_dtype=None):
        self.output_dtype = output_dtype  # only needed for np input

    def __call__(self, img, thresh=-1.0):
        np_img = np.array(img)
        if thresh > 0.0:
            threshold = thresh
        else:
            try:
                threshold = skimage.filters.threshold_minimum(np_img)
            except RuntimeError as rerr:
                threshold = skimage.filters.threshold_otsu(np_img)

        binary_img = img.point(lambda x: x > threshold and 255).convert(
            "1"
        )  # .convert('L')
        return binary_img


def gen_warp(imp):
    # print(6.0*random.normalvariate(0.0,0.5))
    rot = 3.0 * random.normalvariate(0.0, 0.3)
    trans = (
        int((random.normalvariate(0.0, 0.2) * 6 / 56.0) * imp.size[0]),
        int(random.normalvariate(0.0, 0.2) * 6 / 56.0 * imp.size[1]),
    )
    scale = random.uniform(0.9, 1.1)
    shear = (random.normalvariate(0.0, 0.2) * 3.5, random.normalvariate(0.0, 0.2) * 3.5)
    mod_imp = ImageOps.invert(
        affine(
            imp,
            angle=rot,
            translate=trans,
            scale=scale,
            shear=shear,
            resample=False,
            fillcolor=256,
        )
    )
    cv_imp = np.array(mod_imp)
    x_ker = random.randrange(1, 5)
    y_ker = random.randrange(1, 5)
    # print(x_ker,y_ker)
    kernel = np.ones((x_ker, y_ker), np.uint8)  # keep vals from 1-4
    erosion = cv.erode(cv_imp, kernel, iterations=1)
    dilation = cv.dilate(cv_imp, kernel, iterations=1)
    opening = cv.morphologyEx(cv_imp, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(cv_imp, cv.MORPH_CLOSE, kernel)
    PILerode = Image.fromarray(erosion)
    PILdilate = Image.fromarray(dilation)
    PILopen = Image.fromarray(opening)
    PILclose = Image.fromarray(closing)
    ret_img = ImageOps.invert(random.choice([PILerode, PILdilate, PILopen, PILclose]))
    return ret_img
    # display(ImageOps.invert(ret_img))


class Statistics(object):
    def __init__(self, loss=0, n_inst=0, n_corr=0, BCE=0.0, KLD=0.0):
        self.loss = loss
        self.BCE = BCE
        self.KLD = KLD
        self.n_inst = n_inst
        self.ncorr = n_corr
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_inst += stat.n_inst
        self.ncorr += stat.ncorr
        self.BCE += stat.BCE
        self.KLD += stat.KLD

    def ppl(self):
        return math.exp(min(self.loss / self.n_inst, 100))

    def get_loss(self):
        return self.loss * 1.0 / self.n_inst

    def get_BCE(self):
        return self.BCE * 1.0 / self.n_inst

    def get_KLD(self):
        return self.KLD * 1.0 / self.n_inst

    def get_acc(self):
        return self.ncorr * 1.0 / self.n_inst

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        t = self.elapsed_time()
        print(
            (
                "Epoch %2d, %5d/%5d; loss: %6.9f; ppl: %6.2f; acc: %6.2f; BCE: %6.2f; KLD %6.2f;"
                + "%3.0f tok/s; %6.0f s elapsed"
            )
            % (
                epoch,
                batch,
                n_batches,
                self.get_loss(),
                self.ppl(),
                self.get_acc(),
                self.get_BCE(),
                self.get_KLD(),
                self.n_inst / (t + 1e-5),
                time.time() - start,
            )
        )
        sys.stdout.flush()


'''def image_dct(image):
  """Does a type-II DCT (aka "The DCT") on axes 1 and 2 of a rank-3 tensor."""
  image = torch.as_tensor(image)
  dct_y = torch.transpose(torch_dct.dct(image, norm='ortho'), 1, 2)
  dct_x = torch.transpose(torch_dct.dct(dct_y, norm='ortho'), 1, 2)
  return dct_x'''


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError("index {} is out of range".format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


def pad_sequence(sequences, batch_first=False, padding_value=0):
    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
            raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels

    def forward(self, input):
        size = int((input.size(1) // self.n_channels) ** 0.5)
        # print("size",size)
        return input.view(input.size(0), self.n_channels, size, size)


def pad_hmm_sequence(sequences, max_len, batch_first=False, padding_value=0):
    msize = sequences[0].size()
    trailing_dims = msize[1:]

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def one2many(preds, targs):
    cts = defaultdict(list)
    ncorr = 0
    for p, t in zip(preds, targs):
        cts[int(p)].append(int(t))
    for k in cts:
        ncorr += Counter(cts[k]).most_common(1)[0][1]
    return ncorr


def make_img_data_iter(data, opt, device, batchable=True):
    batches = []
    cur_len = 0
    x = []
    y = []
    if len(data) == 0:
        return batches
    nbatches = opt.batch_size if (opt.batch_size <= len(data)) else len(data)
    bsize = len(data) / nbatches
    for i in range(len(data)):
        x += [torch.FloatTensor(a) for a in data[i][0]]
        y += data[i][1]  # torch.LongTensor([a for a in data[i][1]])
        cur_len += len(data[i][1])
        if (cur_len < bsize) and (batchable) and not (i == len(data) - 1):
            continue
        else:
            y = torch.LongTensor(y)
            if opt.mnist:
                x = torch.stack(x)

            if opt.mnist:
                x = x.to(device)
            else:
                x = [a.to(device) for a in x]
            y = y.to(device)
            if opt.mnist:
                batches.append((x, y))
            else:
                batches.append((x, y))
            x = []
            y = []
            cur_len = 0
    return batches


def make_exemplars(data, opt):
    batches = [[] for i in range(opt.nc)]
    x = []
    y = []
    for i in range(len(data)):
        y = data[i][1][0]
        batches[int(y)].append(torch.FloatTensor(data[i][0][0]))
    batches = [(torch.stack(x)).to(device) for x in batches]
    return batches


def reparametrize(mu, logvar, device):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(*(std.size())).to(device)
    return eps.mul(std).add_(mu)


class img_dat(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index][0][0]), self.data[index][1]


def build_dataset(opt):
    un_tr, sup_tr, dev, valid, start, transition = make_hmm_data(
        bmp=True,
        ypix=opt.y_pix,
        mnist=opt.mnist,
        sup_frac=opt.sup_frac,
        bernoulli=opt.bernoulli,
    )
    unsup_train = img_dat(un_tr)
    sup_train = img_dat(sup_tr)
    dev = img_dat(dev)
    valid = img_dat(valid)
    return (
        data.DataLoader(unsup_train, batch_size=opt.batch_size, shuffle=True),
        data.DataLoader(sup_train, batch_size=opt.batch_size, shuffle=True),
        data.DataLoader(dev, batch_size=opt.batch_size, shuffle=True),
        data.DataLoader(valid, batch_size=opt.batch_size, shuffle=True),
        start,
        transition,
    )


def main():
    (
        unsup_train_dataset,
        sup_train_dataset,
        dev,
        valid_dataset,
        start,
        transition,
    ) = build_dataset()
    unsup_train = img_dat(unsup_train_dataset[0:5])
    # print unsup_train[1], len(unsup_train)
    data_loader = data.DataLoader(unsup_train, batch_size=3, shuffle=True)
    for i in range(2):
        data_iter = iter(data_loader)
        for bid, dat in enumerate(data_iter):
            continue
            # print bid, dat, dat[0].size()


if __name__ == "__main__":
    main()
