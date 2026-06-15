from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import torch
from torch import nn
import itertools
import shutil
import json
from tqdm import tqdm_notebook as tqdm
import pickle
import random

from IPython.display import display, Image, clear_output
from PIL import Image as PILImage
# from pigeonXT import annotate



def show_imgs_sidebyside(imgs, captions, grayscale_imgs=None, title=None, figsize=(12,4)):
    """ Assume images already loaded """
    NUM_ROWS = 1
    IMGS_PER_ROW = len(imgs)
    f, ax = plt.subplots(NUM_ROWS, IMGS_PER_ROW, figsize=figsize,)# constrained_layout=True)
    
    for i, (img, caption) in enumerate(zip(imgs, captions)):
        ax[i].imshow(img, cmap='gray')
        if isinstance(caption, str):
            ax[i].set_title(caption)
        else:
            ax[i].set_title(**caption)
        ax[i].axes.xaxis.set_ticks([])
        ax[i].axes.yaxis.set_ticks([])
        if title is not None and i == 0:            
            ax[i].set_ylabel(**title, rotation=0, fontsize=12, labelpad=20)
    plt.tight_layout(pad=0.0)
        
    if grayscale_imgs is not None:
        assert len(grayscale_imgs) == len(imgs)
        f1, ax1 = plt.subplots(NUM_ROWS, IMGS_PER_ROW, figsize=(16,6))
    if grayscale_imgs is not None:
        for i, img in enumerate(grayscale_imgs):
            ax1[i].imshow(img)  # cmap='gray'
            if isinstance(caption, str):
                ax[i].set_title(caption)
            else:
                ax[i].set_title(**caption)
            ax1[i].axes.xaxis.set_ticks([])
            ax1[i].axes.yaxis.set_ticks([])
            if title is not None and i == 0:            
                ax1[i].set_ylabel(**title, rotation=0, fontsize=12, labelpad=20)
        plt.tight_layout(pad=0.0)
    plt.show()
    

def find_grayscale_filepath(bw_path):
    name = Path(bw_path).name
    if name in all_grayscale_filepaths:
        return all_grayscale_filepaths[name]
    else:
#         print(bw_path, name, sep='\n')
        return bw_path

def get_book_name(path_str):
    p = Path(path_str).name.split('-')[0].split('_')[-1]
    if p[-1].isupper():
        p = p[:-1]
    return p

def get_char_location(path_str):
    info = Path(path_str).name.split('-')[1].split('_')
    pg = info[0]
    line = info[1].replace('page1rline', '')
    pos = info[2].replace('char', '')
    return f'{pg}_page1rline{line}_char{pos}'

def convert_aligned_to_color(path_str, color_dir):
    return Path(color_dir)/f"char_{Path(path_str).name.replace('_uc_aligned.tif', '')[-1]}_uc"/Path(path_str).name.replace('_aligned', '')

