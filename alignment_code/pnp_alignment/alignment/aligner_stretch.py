#!/usr/bin/env python
import traceback
import argparse
import random
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import pad
from torchvision.transforms.functional import to_grayscale
import torch
import torch.nn as nn
import time
import sys
import math
import os
from torch import cuda
from random import shuffle
from Optim import Optim
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.nn import Parameter
from pathlib import Path
from PIL import Image, ImageOps
import os
import itertools
from torchvision.transforms import ToTensor
from collections import defaultdict
from collections import Counter
import opts
import matplotlib
import matplotlib.pyplot as plt
from random import sample
import gc
import tqdm

# import GPUtil
import torchvision.transforms.functional as TF
from utils import Binarizer as Binarizer
import csv
import json

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
# torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser(
    description="aligner.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
opts.add_md_help_argument(parser)
opts.train_opts(parser)
opt = parser.parse_args()
print("\n===============================================")
print(opt)
print("===============================================\n")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
device_str = "cuda:" + str(opt.gpu) if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
print(device)
binarizer = Binarizer()


class mixture_model(nn.Module):
    def __init__(self, ypix, nclasses=1, max_off=5, initializer=None, npts=0, rot=True, no_shear=False):
        super(mixture_model, self).__init__()
        self.ypix = ypix
        self.max_off = max_off
        self.nc = nclasses
        self.no_shear = no_shear  # Add the no_shear flag
        self.mix_prob = nn.Embedding(nclasses, 1)
        self.base_img = nn.Embedding(nclasses, ypix * ypix)
        self.lsm = nn.LogSoftmax()
        self.rot_adj = rot
        if npts > 0:
            self.offsets = nn.Embedding(nclasses * npts, 2)
            self.width = nn.Embedding(nclasses * npts, 2)
            self.rotn = nn.Embedding(nclasses * npts, 1)
            self.shear = nn.Embedding(nclasses * npts, 2)
            if self.no_shear:
                # set embedding to zero if no_shear is True
                self.shear.weight.data.fill_(0.0)
                self.shear.weight.requires_grad = False

        self.padder = nn.ConstantPad2d(max_off, 100.0)
        if initializer is not None:
            self.base_img.weight = Parameter(initializer)

    def map(
        self,
        offset,
        width_var,
        rotn_var,
        shear_var,
        ink_var=0.2,
        xsz=opt.y_pix,
        ysz=opt.y_pix,
        pr=None,
        batch_mode=False,
    ):
        # all the images are resized such that max size corresponds to ypix. and padded to ypix,ypix. So fac_x,fac_y=1. If this is not the case modify the code.
        # offset is the offset in the template space. But since everything is padded to the same size the pad_ratios and width variables will take care of resizing.
        # wx<1 = zooming in on the template and pr is orig/ypix which actually requires zooming out. so divide py pr.
        # dist_x is the attention map. should not scale it so that it corresponds to the actual image(template in this case)
        # assuming rotationa nd shear are not affected by scaling. Confirm this.
        pad_sz = float(self.ypix)
        fac_x = pad_sz / xsz
        fac_y = pad_sz / ysz
        # assert fac_x==1.0
        # assert fac_y==1.0
        b_inc = 0
        if batch_mode:
            b_inc = 1
        radian = (rotn_var / 180.0) * 3.142
        shear = (shear_var / 180.0) * 3.142
        if not (self.rot_adj):
            radian *= 0.0
        if self.no_shear:
            shear *= 0.0
        cos = torch.cos(radian)  # b,1
        sin = torch.sin(radian)  # b,1

        if batch_mode:
            tanx = torch.tan(1.5 * shear_var[:, 0]).unsqueeze(1) if batch_mode else torch.tan(1.5 * shear_var[0])
            tany = torch.tan(1.0 * shear_var[:, 1]).unsqueeze(1) if batch_mode else torch.tan(1.0 * shear_var[1])
            
            wx = width_var[:, 0] / pr[:, 0] if batch_mode else width_var[0] / pr[0]
            wy = width_var[:, 1] / pr[:, 1] if batch_mode else width_var[1] / pr[1]

            mean_x = (
                torch.Tensor([range(-math.floor(xsz / 2.0), math.ceil(xsz / 2.0))])
                .to(device)  # b,xsz
                - ((offset[:, 0] / fac_x).unsqueeze(1))  # b,1
            ) * wx.unsqueeze(1)  # b,range
            mean_y = (
                torch.Tensor([range(-math.floor(ysz / 2.0), math.ceil(ysz / 2.0))])
                .to(device)
                - ((offset[:, 1] / fac_y).unsqueeze(1))
            ) * wy.unsqueeze(1)  # b,range
        rot_x = (
            (mean_x * (cos + tanx * sin + tany * (-sin + tanx * cos))).unsqueeze(
                1 + b_inc
            )
        ) + (mean_y * (-sin + tanx * cos)).unsqueeze(
            0 + b_inc
        )  # b,range,range
        rot_y = ((mean_x * (sin + tany * cos)).unsqueeze(1 + b_inc)) + (
            mean_y * cos
        ).unsqueeze(0 + b_inc)

        # dist_x = (torch.Tensor([range(-int(self.max_off),self.ypix+int(self.max_off))]).to(device))*wx.unsqueeze(0+b_inc) #b,1,side
        # dist_y = (torch.Tensor([range(-int(self.max_off),self.ypix+int(self.max_off))]).to(device))*wy.unsqueeze(0+b_inc) #b,1,side
        # dist_x = (torch.Tensor([range(-int(self.max_off),self.ypix+int(self.max_off))]).to(device)).unsqueeze(0+b_inc) #b,1,side
        # dist_y = (torch.Tensor([range(-int(self.max_off),self.ypix+int(self.max_off))]).to(device)).unsqueeze(0+b_inc) #b,1,side
        # rot_x = torch.clamp(rot_x, -opt.max_off*1.0, self.ypix+opt.max_off*1.0)
        # rot_y = torch.clamp(rot_y, -opt.max_off*1.0, self.ypix+opt.max_off*1.0)
        dist_x = (
            torch.Tensor(
                [
                    range(
                        -(int(self.max_off) + math.floor(self.ypix / 2.0)),
                        math.ceil(self.ypix / 2.0) + int(self.max_off),
                    )
                ]
            ).to(device)
        ).unsqueeze(
            0 + b_inc
        )  # b,1,side
        dist_y = (
            torch.Tensor(
                [
                    range(
                        -(int(self.max_off) + math.floor(self.ypix / 2.0)),
                        math.ceil(self.ypix / 2.0) + int(self.max_off),
                    )
                ]
            ).to(device)
        ).unsqueeze(
            0 + b_inc
        )  # b,1,side
        rot_x = torch.clamp(
            rot_x,
            -(opt.max_off * 1.0 + math.floor(self.ypix / 2.0)),
            math.ceil(self.ypix / 2.0) + opt.max_off * 1.0,
        )
        rot_y = torch.clamp(
            rot_y,
            -(opt.max_off * 1.0 + math.floor(self.ypix / 2.0)),
            math.ceil(self.ypix / 2.0) + opt.max_off * 1.0,
        )
        dist_x = (dist_x.unsqueeze(0 + b_inc) - rot_x.unsqueeze(2 + b_inc)).unsqueeze(
            3 + b_inc
        )  # r,r,s,1
        dist_y = (dist_y.unsqueeze(0 + b_inc) - rot_y.unsqueeze(2 + b_inc)).unsqueeze(
            2 + b_inc
        )  # r,r,1,s
        rot_x = torch.exp(-1.0 * torch.pow(dist_x, 2) / ink_var)
        rot_y = torch.exp(-1.0 * torch.pow(dist_y, 2) / ink_var)
        return rot_x * rot_y  # r,r,s,s

    def heat_map(self, dists):
        heatmap = torch.matmul(dists[0].unsqueeze(1), dists[1])
        return heatmap

    def combine(self, heatmap, base_img, batch_mode=False):
        b_inc = 0
        if batch_mode:
            b_inc = 1
        return (heatmap * base_img).sum(3 + b_inc).sum(2 + b_inc)

    def continuous_offset(self, img, img_id, pad_ratio, cls, batch_mode=False):
        b_inc = 0
        if batch_mode:
            b_inc = 1

        max_w = 0.9  # 0.4 originally in kartik's code
        max_rot = 7.0
        max_shear = 3.5

        if not batch_mode:
            bsz = 1
            basic_img = self.padder(
                self.base_img(torch.Tensor([int(cls)]).to(dtype=torch.long, device=device))
                .squeeze()
                .view(self.ypix, self.ypix)
            )
            offset = self.max_off * torch.tanh(self.offsets(torch.Tensor([int(opt.nc) * int(img_id) + int(cls)]).to(dtype=torch.long, device=device)).squeeze())
            width_var = 1.0 - max_w * torch.tanh(self.width(torch.Tensor([int(opt.nc) * int(img_id) + int(cls)]).to(dtype=torch.long, device=device)).squeeze())
            rotn_var = max_rot * torch.tanh(self.rotn(torch.Tensor([int(opt.nc) * int(img_id) + int(cls)]).to(dtype=torch.long, device=device)).squeeze())
            shear_var = max_shear * torch.tanh(self.shear(torch.Tensor([int(opt.nc) * int(img_id) + int(cls)]).to(dtype=torch.long, device=device)).squeeze())
        else:
            bsz = img.size(0)
            basic_img = self.padder(
                self.base_img(torch.Tensor([int(cls)]).to(dtype=torch.long, device=device))
                .view(1, self.ypix, self.ypix)
            )
            offset = self.max_off * torch.tanh(self.offsets((int(opt.nc) * img_id) + cls))
            width_var = 1.0 - max_w * torch.tanh(self.width((int(opt.nc) * img_id) + cls))
            rotn_var = max_rot * torch.tanh(self.rotn((int(opt.nc) * img_id) + cls))
            shear_var = max_shear * torch.tanh(self.shear((int(opt.nc) * img_id) + cls))

        prev_bce = 0.0
        flag = True
        ret_offset = None
        ret_width = None
        ret_img = None
        ret_rotn = None
        ret_shear = None
        BCEs = []
        for ink_var in [opt.ink_var]:  # , 0.02, 0.2, 2.0]:
            xsz = img.size(0 + b_inc)
            ysz = img.size(1 + b_inc)
            cand_img = self.combine(
                (
                    self.map(
                        offset,
                        width_var,
                        rotn_var,
                        shear_var,
                        ink_var,
                        xsz=xsz,
                        ysz=ysz,
                        pr=pad_ratio,
                        batch_mode=batch_mode,
                    )
                ),
                basic_img,
                batch_mode,
            )  # b,r,r
            BCE = (
                -1.0
                * (
                    img * (F.logsigmoid(cand_img))
                    + (1.0 - img) * torch.log(1.0 - torch.sigmoid(cand_img) + 1e-28)
                ).sum()
            )  # change for batching
            BCE /= bsz
            ret_img = cand_img
            ret_offset = offset
            ret_width = width_var
            ret_rotn = rotn_var
            ret_shear = shear_var
            if flag:
                prev_bce = BCE.item()
                flag = False
            else:
                cur_bce = BCE.item()
                if cur_bce < prev_bce:
                    prev_bce = cur_bce
        # ret_offset is not a tuple but a b,2 tensor now handle appropriately
        # xoff = ret_offset[0].item()
        # yoff = ret_offset[1].item()
        # ret_offset= (xoff,yoff)
        return (ret_img, ret_offset, prev_bce, ret_width, ret_rotn, ret_shear)

    def forward(self, img, img_id, padratio, batch_mode=False):
        lp_mix = self.lsm(self.mix_prob.weight.squeeze(1))
        emits = []
        bsz = 1
        b_inc = 0
        if batch_mode:
            bsz = img.size(0)
            b_inc = 1
            lp_mix = lp_mix.unsqueeze(1)
        for i in range(self.nc):
            (
                cand_img,
                cand_offset,
                cand_bce,
                width_var,
                rotn_var,
                shear_var,
            ) = self.continuous_offset(img, img_id, padratio, i, batch_mode)

            BCE = (
                (
                    img * (F.logsigmoid(cand_img))
                    + (1.0 - img) * torch.log(1.0 - torch.sigmoid(cand_img) + 1e-28)
                )
                .sum(b_inc + 1)
                .sum(b_inc + 0)
            )
            emits.append(BCE)
            del cand_img, cand_offset, cand_bce, width_var, rotn_var, shear_var
        emit = torch.stack(emits)  # nc,b
        batch_loss = -1.0 * (torch.logsumexp(lp_mix + emit, dim=0)).sum() / bsz
        # reg = opt.lbd*(torch.pow(torch.norm(self.base_img.weight),2))  #/(self.ypix*self.ypix)
        reg = opt.lbd * (
            torch.pow(torch.norm(self.offsets.weight), 2)
            + torch.pow(torch.norm(self.width.weight), 2)
            + torch.pow(torch.norm(self.rotn.weight), 2)
            + torch.pow(torch.norm(self.shear.weight), 2)
        )
        return batch_loss, reg

    def decode(self, img, img_id, padratio, batch_mode=False):
        with torch.no_grad():
            lp_mix = self.lsm(self.mix_prob.weight.squeeze(1))
            bsz = 1
            b_inc = 0
            if batch_mode:
                bsz = img.size(0)
                b_inc = 1
                lp_mix = lp_mix.unsqueeze(1)
            emits = []
            offsets = []
            widths = []
            rotns = []
            shears = []
            for i in range(self.nc):
                (
                    cand_img,
                    cand_offset,
                    cand_bce,
                    width_var,
                    rotn_var,
                    shear_var,
                ) = self.continuous_offset(img, img_id, padratio, i, batch_mode)

                BCE = (
                    (
                        img * (F.logsigmoid(cand_img))
                        + (1.0 - img) * torch.log(1.0 - torch.sigmoid(cand_img) + 1e-28)
                    )
                    .sum(b_inc + 1)
                    .sum(b_inc + 0)
                )
                emits.append(BCE)
                offsets.append(cand_offset)
                widths.append(width_var)
                rotns.append(rotn_var)
                shears.append(shear_var)
                # del cand_img, cand_offset, cand_bce, ink_var
            emit = torch.stack(emits)  # nc,b

            # val,ind = torch.topk((lp_mix+emit).squeeze(),1,dim=0)
            val, ind = torch.topk((lp_mix + emit), 1, dim=0)
            # make batch index based selections ugh
            # ind = (ind.squeeze()).to_list()
            if len(ind.size()) > 1:
                ind = ind.squeeze(0)
            if len(val.size()) > 1:
                val = val.squeeze(0)

            ind = torch.unbind(ind)
            offset = []
            width = []
            rotn = []
            shear = []
            for i, index in enumerate(ind):
                index = index.item()
                # print(index,i)
                offset.append(offsets[index][i])
                width.append(widths[index][i])
                rotn.append(rotns[index][i])
                shear.append(shears[index][i])
            offset = torch.stack(offset)
            width = torch.stack(width)
            rotn = torch.stack(rotn)
            shear = torch.stack(shear)
            return offset, width, rotn, shear, ind, val


def build_optim(params, learning_rate, optim, checkpoint=None, refresh=False):
    if (opt.train_from) and (refresh == False):
        print("Loading optimizer from checkpoint.", flush=True)
        optim = checkpoint["optim"]
        optim.optimizer.load_state_dict(checkpoint["optim"].optimizer.state_dict())
        print("Loaded optim:", optim, flush=True)
    else:
        # what members of opt does Optim need?
        optim = Optim(
            optim,
            learning_rate,
            opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            opt=opt,
        )
    # optim.set_parameters(model.parameters(), offset=True)
    optim.set_parameters(params, offset=False)
    return optim


class BinImg:
    def __init__(
        self, img, ind, fname, pad_ratio, orig_dim, scale, offset, shear, rotation
    ):
        self.inp_img = img
        self.ind = ind
        self.fname = fname
        self.pad_ratio = pad_ratio
        self.orig_dim = orig_dim
        self.scale = scale
        self.offset = offset
        self.shear = shear
        self.rotation = rotation
        self.score = -100.0

    def to_dict(self):
        return {
            "name": self.fname,
            "idx": self.ind,
            "pr": self.pad_ratio,
            "score": self.score,
            "offset": self.offset,
            "og": self.orig_dim,
            "shear": self.shear,
            "rotn": self.rotation,
        }


def build_data(infile_list, binthresh, init_dir=None):
    maxh = 0
    maxw = 0
    images = []
    # import ipdb; ipdb.set_trace()
    print(f"File list has {len(infile_list)} images")
    zero_hw_filenames = []
    for filename in infile_list:
        try:
            imp = to_grayscale(Image.open(filename))
        except Exception as e:
            print(f"{traceback.print_exc()}\n{e}: skipping the image: " + str(filename))
            # import ipdb; ipdb.set_trace()
        else:
            width, height = imp.size
            # TODO: is this a bug? shouldn't width be greater than x pix or 0?
            if ((width > opt.y_pix) or (height > opt.y_pix)) and (
                width > 0 and height > 0
            ):
                # if ((width>0) or (height >0)):
                maxres = max([width, height])
                ratio = opt.y_pix * 1.0 / maxres
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
    maxh = opt.y_pix
    maxw = opt.y_pix
    im_id = -1
    print(f"File list has {len(zero_hw_filenames)} zero h/w files:")
    for filename in zero_hw_filenames:
        print(filename)
        infile_list.remove(filename)
    print(f"File list has {len(infile_list)} images after removing the zero h/w files")
    for filename in infile_list:
        try:
            imp = to_grayscale(Image.open(filename))
        except:
            print("skipping the image: " + str(filename))
        else:
            im_id += 1
            width, height = imp.size
            orig_w = width
            orig_h = height
            # if ((width>0) or (height >0)):
            if (width > opt.y_pix) or (height > opt.y_pix):
                maxres = max([width, height])
                ratio = opt.y_pix * 1.0 / maxres
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
                if opt.no_binarize:  # already binarized so just pad
                    imp = pad(
                        imp, (l, t, r, b), fill=255, padding_mode="constant"
                    ).convert("1")
                else:
                    imp = binarizer(
                        pad(imp, (l, t, r, b), fill=255, padding_mode="constant"),
                        thresh=binthresh[filename],
                    )

            except:
                print("couldn't binarize: " + str(filename))
            else:
                imp = ToTensor()(imp)  # here imp changes to H,W
                imp = imp.squeeze()  # .to(device)
                pad_ratio = [wfrac, hfrac]
                orig_dim = [orig_w, orig_h]
                scale = [1.0, 1.0]
                offset = [0.0, 0.0]
                shear = [0.0, 0.0]
                rotation = [0.0, 0.0]
                bin_img = BinImg(
                    imp,
                    im_id,
                    filename,
                    pad_ratio,
                    orig_dim,
                    scale,
                    offset,
                    shear,
                    rotation,
                )
                images.append(bin_img)
    initializers = None
    if (opt.init_rand) and (init_dir is None) and (len(images) >= opt.nc):
        init_imps = [im.inp_img for im in sample(images, opt.nc)]
        # init_imps = [images[0],images[1]]
        initializers = torch.stack(
            [
                F.interpolate(
                    imp.unsqueeze(0).unsqueeze(0), (opt.y_pix, opt.y_pix), mode="area"
                )
                .squeeze()
                .reshape(opt.y_pix * opt.y_pix)
                for imp in init_imps
            ]
        )
        initializers = 0.1 + initializers * 0.8
        initializers = torch.log(initializers) - torch.log(1.0 - initializers)
    elif opt.init_avg:
        assert opt.nc == 1, "Only one cluster can be used with init_avg"
        # sample 1000 images and average them
        init_imps = [im.inp_img for im in sample(images, min(1000, len(images)))]
        initializers = torch.stack(
            [
                F.interpolate(
                    imp.unsqueeze(0).unsqueeze(0), (opt.y_pix, opt.y_pix), mode="area"
                )
                .squeeze()
                .reshape(opt.y_pix * opt.y_pix)
                for imp in init_imps
            ]
        ).mean(dim=0, keepdim=True)  # 1, y_pix*y_pix
        # save the average image
        # save_image(initializers.view(1, 1, opt.y_pix, opt.y_pix), "average_image.png", normalize=True)
        initializers = 0.1 + initializers * 0.8
        # compute logits: log(x/(1-x))
        initializers = torch.log(initializers) - torch.log(1.0 - initializers)
    else:
        raise ValueError("No initializer specified")
    print("Padding images to max width / height:", maxw, "/", maxh)
    return images, initializers


def reconstruct(
    img, offset, width, rotn, shear, pr, ink_var=0.2, max_val=0.999, min_val=0.001
):
    # all the images are resized such that max size corresponds to ypix. and padded to ypix,ypix. So fac_x,fac_y=1. If this is not the case modify the code.
    # offset is the offset in the template space. But since everything is padded to the same size the pad_ratios and width variables will take care of resizing.
    # wx<1 = zooming in on the template and pr is orig/ypix which actually requires zooming out. so divide py pr.
    # dist_x is the attention map. should not scale it so that it corresponds to the actual image(template in this case)
    # assuming rotationa nd shear are not affected by scaling. Confirm this.
    b_inc = 1
    xsz = img.size(0 + b_inc)
    ysz = img.size(1 + b_inc)
    pad_sz = float(opt.y_pix)
    fac_x = pad_sz / xsz
    fac_y = pad_sz / ysz
    # assert fac_x==1.0
    # assert fac_y==1.0
    radian = (rotn / 180.0) * 3.142
    if opt.no_rotn:
        radian *= 0.0
    shear = (shear / 180.0) * 3.142
    if opt.no_shear:
        shear *= 0.0
    wx = (width[:, 0] / pr[:, 0]).unsqueeze(1)  # b,1
    wy = (width[:, 1] / pr[:, 1]).unsqueeze(1)  # b,1
    cos = torch.cos(-radian)
    sin = torch.sin(-radian)
    tanx = torch.tan(-1.5 * shear[:, 0]).unsqueeze(1)
    tany = torch.tan(-1.0 * shear[:, 1]).unsqueeze(1)
    mean_x = (
        torch.Tensor([range(-math.floor(opt.y_pix / 2.0), math.ceil(opt.y_pix / 2.0))])
        .squeeze()
        .to(device)
    ) / (wx) + (
        offset[:, 0].unsqueeze(1)
    )  # /fac_x
    mean_y = (
        torch.Tensor([range(-math.floor(opt.y_pix / 2.0), math.ceil(opt.y_pix / 2.0))])
        .squeeze()
        .to(device)
    ) / (wy) + (
        offset[:, 1].unsqueeze(1)
    )  # /fac_y
    dist_x = (
        (
            torch.Tensor(
                [
                    range(
                        -(math.ceil(opt.max_off / fac_x) + math.floor(xsz / 2.0)),
                        math.ceil(xsz / 2.0) + math.ceil(opt.max_off / fac_x),
                    )
                ]
            ).to(device)
        )
    ).unsqueeze(0 + b_inc)
    dist_y = (
        (
            torch.Tensor(
                [
                    range(
                        -(math.ceil(opt.max_off / fac_y) + math.floor(ysz / 2.0)),
                        math.ceil(ysz / 2.0) + math.ceil(opt.max_off / fac_y),
                    )
                ]
            ).to(device)
        )
    ).unsqueeze(0 + b_inc)
    # print(mean_x.size(), mean_y.size(),dist_x.size(), dist_y.size())
    rot_x = (
        (mean_x * (cos + tanx * sin + tany * (-sin + tanx * cos))).unsqueeze(1 + b_inc)
    ) + (mean_y * (-sin + tanx * cos)).unsqueeze(
        0 + b_inc
    )  # b,range,range
    rot_y = ((mean_x * (sin + tany * cos)).unsqueeze(1 + b_inc)) + (
        mean_y * cos
    ).unsqueeze(0 + b_inc)
    rot_x = torch.clamp(
        rot_x,
        -(math.ceil(opt.max_off / fac_x) * 1.0 + math.floor(xsz / 2.0)),
        math.ceil(xsz / 2.0) + math.ceil(opt.max_off / fac_x) * 1.0,
    )
    rot_y = torch.clamp(
        rot_y,
        -(math.ceil(opt.max_off / fac_y) * 1.0 + math.floor(ysz / 2.0)),
        math.ceil(ysz / 2.0) + math.ceil(opt.max_off / fac_y) * 1.0,
    )
    dist_x = (dist_x.unsqueeze(0 + b_inc) - rot_x.unsqueeze(2 + b_inc)).unsqueeze(
        3 + b_inc
    )
    dist_y = (dist_y.unsqueeze(0 + b_inc) - rot_y.unsqueeze(2 + b_inc)).unsqueeze(
        2 + b_inc
    )
    # print(dist_x.size(),dist_y.size(),rot_x.size(),rot_y.size())
    rot_x = torch.exp(-1.0 * torch.pow(dist_x, 2) / ink_var)
    rot_y = torch.exp(-1.0 * torch.pow(dist_y, 2) / ink_var)

    heatmap = rot_x * rot_y

    heatmap = heatmap / heatmap.sum(3 + b_inc, keepdim=True).sum(
        2 + b_inc, keepdim=True
    )
    padder = nn.ConstantPad2d(
        (
            math.ceil(opt.max_off / fac_y),
            math.ceil(opt.max_off / fac_y),
            math.ceil(opt.max_off / fac_x),
            math.ceil(opt.max_off / fac_x),
        ),
        10.0,
    )
    base_img = padder(
        torch.log(min_val + img * (max_val - min_val))
        - torch.log(1.0 - (min_val + img * (max_val - min_val)))
    )
    # print(heatmap.size(), base_img.size())
    return (
        (heatmap * (base_img.unsqueeze(1).unsqueeze(2))).sum(3 + b_inc).sum(2 + b_inc)
    )


def showTensor(aTensor):
    plt.figure()
    plt.imshow(aTensor.numpy(), cmap=plt.cm.gray)
    plt.show()


def visualize_template(template, epoch, cl=1, template_dir="./templates"):
    with torch.no_grad():
        tmp = template.to("cpu").reshape(1, opt.y_pix, opt.y_pix)
        nptmp = torch.sigmoid(tmp)

        im = ToPILImage()(nptmp)
        im.save(template_dir + "/base_" + str(epoch) + "_" + str(cl) + ".tif", "tiff")


def visualize_img(img, fname="img", template_dir="./reconstruction"):
    with torch.no_grad():
        tmp = img.to("cpu").reshape(1, img.size(0), img.size(1))
        tmp = torch.sigmoid(tmp)

        im = ToPILImage()(tmp)
        im.save(template_dir + "/" + os.path.split(fname)[1] + ".tif", "tiff")


def visualize_real_img(img, fname="real_img", template_dir="./reconstruction"):
    with torch.no_grad():
        tmp = img.to("cpu").reshape(1, img.size(0), img.size(1))
        # nptmp = torch.sigmoid(tmp)

        im = ToPILImage()(tmp)
        im.save(template_dir + "/" + os.path.split(fname)[1] + ".tif", "tiff")


def reconstruct_lib(
    img, offset, width, rotn, ink_var=0.2, max_val=0.999, min_val=0.001
):
    xsz = img.size(0)
    ysz = img.size(1)
    pad_sz = float(opt.y_pix)
    fac_x = pad_sz / xsz
    fac_y = pad_sz / ysz
    radian = (rotn / 180.0) * 3.142


def sanity_reconstruct(img, offset, max_val=0.999, min_val=0.001):
    base_img = torch.log(min_val + img * (max_val - min_val)) - torch.log(
        1.0 - (min_val + img * (max_val - min_val))
    )
    return base_img


def make_batches(train_pts, bsize):
    # not vectorized -- just batches the BinImg objects
    npts = len(train_pts)
    nbatches = (int(npts) / int(bsize)) + 1
    batches = []
    for i in range(int(nbatches)):
        if i * bsize == npts:
            break
        bpts = train_pts[i * bsize : (i + 1) * bsize]
        # inds = torch.LongTensor(range(i*bsize,(i*bsize)+len(bpts)))
        # bpts = torch.stack(bpts)
        batches.append(bpts)
    return batches


def shuffle_batches(batches, bsize):
    pts = []
    for batch in batches:
        pts.extend(batch)
    # ids= list(range(len(pts)))
    shuffle(pts)
    nbatches = int(math.ceil((1.0 * len(pts) / int(bsize))))
    newbatches = []
    for i in range(int(nbatches)):
        newbatches.append(pts[i * bsize : (i + 1) * bsize])
    return newbatches


def save_model(epoch, loss, model, dirname="./saved_models"):
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "loss": loss},
        str(dirname) + "/model_" + str(epoch) + ".ckpt",
    )


def data():
    init_dir = None
    # cl_id = False
    if not (opt.init_dir == ""):
        init_dir = opt.init_dir
    cl_id = opt.cluster
    train_pts, initializer = build_data(opt.data, init_dir, cl_id)
    initializer = initializer.to(device)
    npts = len(train_pts)
    print(npts)


def main():
    # opt.data dir is a temp directory obtained by pulling relelvant fiules from the tarfiles in a previous stage.
    # TODO:Populate BinImg's other attributed and store them in a csv/xml
    init_dir = None
    if not (opt.init_dir == ""):
        init_dir = opt.init_dir
    train_pts = []
    initializers = []
    batches = []
    alignfiles = {}
    binthresh = {}
    # with open(opt.bincsv, newline='') as csvfile:
    #  csveader = csv.reader(csvfile, delimiter=',')
    #  for row in csvreader:
    with open(os.path.join(opt.data, opt.aligncsv), newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            alignfiles[os.path.join(opt.data, row[0])] = os.path.join(opt.data, row[1])
            binthresh[os.path.join(opt.data, row[0])] = float(row[2])
    print(f"alignfiles contains {len(alignfiles)} files.")
    print(f"here are the first five:")
    print("\n".join([str(f) for f in alignfiles][:5]))
    print()
    infile_list = []
    for fn in alignfiles:
        infile_list.append(fn)
    train_pts, initializer = build_data(infile_list, binthresh, init_dir)
    bsize = int(opt.batch_size)
    batches = make_batches(train_pts, bsize)
    initializer = initializer.to(device)
    npts = len(train_pts)
    print(npts)
    if opt.data_size < npts:
        train_pts = train_pts[: opt.data_size]
        npts = opt.data_size
    template_dir = os.path.join(opt.data, opt.output)
    reconstruct_dir = os.path.join(opt.data, opt.rec)
    save_dir = os.path.join(opt.data, opt.save_dir)
    # print(len(batches))
    try:
        os.mkdir(template_dir)
    except:
        print("The output directory already exists")
        pass
    try:
        os.mkdir(reconstruct_dir)
    except:
        print("The output directory already exists")
        pass
    try:
        os.mkdir(save_dir)
    except:
        print("The output directory already exists")
        pass

    model = mixture_model(
        opt.y_pix, opt.nc, opt.max_off, initializer, npts, rot=not (opt.no_rotn), no_shear=opt.no_shear
    )
    # print(model.shear.weight.size())
    loaded_epoch = 0
    loaded_loss = 0.0
    if not (opt.train_from == ""):
        ckpt = torch.load(
            str(opt.train_from), map_location=device
        )  # if device_str == 'cpu' else torch.load(str(opt.train_from))
        # dummy_model = mixture_model(opt.y_pix,opt.nc, opt.max_off, initializer, 1, rot = not(opt.no_rotn))
        # dummy_model.load_state_dict(ckpt['model_state_dict'])
        my_dict = model.state_dict()
        model_dict = {}
        for k in ckpt["model_state_dict"].keys():
            if ("base_img" in k) or ("mix_prob" in k):
                model_dict[k] = ckpt["model_state_dict"][k]
            my_dict.update(model_dict)
        model.load_state_dict(my_dict)
        # model.base_img = dummy_model.base_img
        # model.mix_prob = dummy_model.mix_prob
        loaded_epoch = ckpt["epoch"]
        loaded_loss = ckpt["loss"]
        # del dummy_model
    model = model.to(device)
    template_params = (
        [] if opt.freeze_template_params else [p for p in model.base_img.parameters()]
    )
    # NOTE: not used for this lbd mixture model code
    optim_model = build_optim(
        template_params + [p for p in model.mix_prob.parameters()],
        opt.learning_rate,
        opt.optim,
        refresh=True,
    )
    optim_adj = build_optim(
        [p for p in model.offsets.parameters()]
        + [p for p in model.width.parameters()]
        + [p for p in model.rotn.parameters()]
        + [p for p in model.shear.parameters()] if opt.no_shear else [],
        opt.adj_learning_rate,
        opt.adj_optim,
        refresh=True,
    )
    model.train()

    for j in range(opt.nc):
        visualize_template(
            model.base_img(torch.LongTensor([j]).to(device)), 0, j, template_dir
        )

    for epoch in range(opt.epochs):
        print(f"Epoch {epoch} started...", flush=True)
        total_loss = 0.0
        total_lp = 0.0
        total_reg = 0.0
        template_grad_norm = 0.0
        batches = shuffle_batches(batches, bsize)
        json_objs = {}
        for bno, batch in enumerate(tqdm.tqdm(batches)):
            # train_pt,cl,i,fn,pr = batch
            # print(bno, len(batch))
            train_pt = torch.stack([pt.inp_img for pt in batch])
            i = torch.LongTensor([pt.ind for pt in batch])
            pr = torch.Tensor([pt.pad_ratio for pt in batch])
            fn = [pt.fname for pt in batch]
            # fname.extend(fn)
            train_pt = train_pt.to(device)
            i = i.to(device)
            pr = pr.to(device)
            # GPUtil.showUtilization()
            if bsize > 1:
                for msteps in range(1):
                    loss, reg = model(train_pt, i, pr, batch_mode=True)
                    (loss + reg).backward()
                    optim_model.step()
                    optim_model.zero_grad()
                    optim_adj.zero_grad()
                    del loss
                    del reg
            for osteps in range(4):
                loss, reg = model(train_pt, i, pr, batch_mode=True)
                (loss + reg).backward()
                optim_adj.step()
                optim_model.zero_grad()
                optim_adj.zero_grad()
                del loss
                del reg

            loss, reg = model(train_pt, i, pr, batch_mode=True)
            (loss + reg).backward()
            template_grad_norm += torch.norm(model.base_img.weight.grad)
            optim_model.step()
            loss, reg = model(train_pt, i, pr, batch_mode=True)
            (loss + reg).backward()
            template_grad_norm += torch.norm(model.base_img.weight.grad)
            optim_model.step()
            optim_adj.step()
            optim_model.zero_grad()
            optim_adj.zero_grad()
            total_lp += loss.item()
            total_reg += reg.item()
            total_loss += (loss + reg).item()
            # print(f"Epoch {epoch}, \t Loss: {loss.item():.4f}")
            del loss
            del reg

            if (epoch + 1) % 1 == 0:
                # scale, offset, shearr, rotation
                offsets, widths, rotn, shear, ind, val = model.decode(
                    train_pt, i, pr, batch_mode=True
                )
                rec_img = reconstruct(train_pt, offsets, widths, rotn, shear, pr)
                rec_imgs = torch.unbind(rec_img)
                img_ids = torch.unbind(i)
                osets = torch.unbind(offsets)
                ws = torch.unbind(widths)
                rts = torch.unbind(rotn)
                shs = torch.unbind(shear)
                vals = torch.unbind(val)
                ctr = 0
                # TODO: make JSON object here
                for rec, real_id, dec_id in zip(rec_imgs, img_ids, ind):
                    cluster_id = int(dec_id.item())
                    real_id = real_id.item()
                    batch[ctr].scale = ws[ctr].tolist()
                    batch[ctr].offset = osets[ctr].tolist()
                    batch[ctr].shear = shs[ctr].tolist()
                    batch[ctr].rotation = rts[ctr].tolist()
                    batch[ctr].score = float(vals[ctr].item())
                    # visualize_real_img(train_pt[ctr],fn[ctr][:-4]+"_"+"real", reconstruct_dir)
                    visualize_real_img(
                        train_pt[ctr], batch[ctr].fname[:-4], reconstruct_dir
                    )
                    visualize_img(
                        rec, alignfiles[batch[ctr].fname][:-4], reconstruct_dir
                    )
                    json_objs[real_id] = batch[ctr].to_dict()
                    ctr += 1
        with torch.no_grad():
            print(torch.exp(model.lsm(model.mix_prob.weight.squeeze())))
        if (epoch % 1 == 0) and (opt.visualize):
            for j in range(opt.nc):
                visualize_template(
                    model.base_img(torch.LongTensor([j]).to(device)),
                    epoch + 1,
                    j,
                    template_dir,
                )
        if ((epoch + 1) % 10 == 0) and (epoch > 0):
            save_model(epoch, total_loss * bsize / len(train_pts), model, save_dir)
        if ((epoch + 1) % 5 == 0) and (epoch > 0):
            jfp = open(
                os.path.join(opt.data, opt.save_dir + ".json"), "w", encoding="utf8"
            )
            json.dump(json_objs, jfp)
            jfp.close()
        print(
            f"Epoch {epoch}, \t Average Loss: {total_loss*bsize/len(train_pts):.4f}, \t \
      \t Average NegLogprob: {total_lp*bsize/(len(train_pts)):.4f}, \
      \t Average Reg: {total_reg/(opt.lbd*len(train_pts)+1):.4f}, \t Average grad norm: {template_grad_norm/len(train_pts):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
