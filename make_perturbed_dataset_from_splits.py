#
#  Make perturbed dataset for provided train/valid/test splits
#
import skimage
import matplotlib.pyplot as plt
from skimage.util import invert
import copy

from tqdm import tqdm
from PIL import Image
import re
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import math
import random
from pathlib import Path
import torchvision
from torchvision import transforms
import json
import time

import random

import sys
sys.path.append("../..")
from classifier_code import *


from classifier_code import random_splits_from_folder
from pathlib import Path
from anom_dataset import UCSDDataset_Residue
sys.path.append('Morpho-MNIST')
sys.path.append('Morpho-MNIST/morphomnist')
from morphomnist import io, morpho, perturb
from bender import BadSkeletonBendException, ChooseSkeletonEndpointsException
from perturber_args import *
from argparse import ArgumentParser
import datetime
from scipy.stats import truncnorm
from joblib import Parallel, delayed
from perturb_settings import AnomalyDetectionSettings


dt = datetime.datetime.today()
# todaysdate_str = '{dt.month:02d}_{dt.day:02d}'

parser = ArgumentParser('Creates a new perturbed dataset of train/valid/test images from random normal images. Loads perturbation settings from perturber_args.')
parser.add_argument('--char', default='M')
parser.add_argument('--binarize_image', action='store_true')
parser.add_argument('--num_repetitions', default=20, help='Number of random perturbations to apply to a single image')
parser.add_argument('--percent_apply_both_damage', default=0.03, help='Frequency to apply both bends and fractures to the same image.')
parser.add_argument('--splits_dir', default=Path(f"/home/kishore/data/anomaly-detection/aligned-dataset/M/mixed_book_splits", help='Path to directory containing dataset splits'))
parser.add_argument('--src_dataframe_path', default=Path("/home/kishore/data/anomaly-detection/datasetv3"))
parser.add_argument('--dest_dir', default=Path("/home/kishore/data/anomaly-detection/aligned-dataset/M"))
parser.add_argument('--dest_dataset_name', default=f"M_aligned_perturbed20_05_10")
parser.add_argument('--num_jobs', default=4, type=int)
args = parser.parse_args()
print(args)

                    
char_settings = get_char_settings(args.char)
anom_detect_settings = AnomalyDetectionSettings(char_settings)

                    
def apply_random_perturbations(img, damage=True):
    """ Sometimes applies multiple damages to same character.
    """
    make_morph_object = lambda x: morpho.ImageMorphology(x, scale=1)
    inverted_img = invert(np.array(img).astype(np.bool))
    assert np.array_equal(inverted_img, inverted_img.astype(np.bool)) if not args.binarize_image else True, f'{inverted_img} not binary.{np.unique(inverted_img)}'
    img_morphology = make_morph_object(inverted_img)
    assert np.array_equal(img_morphology.binary_image, img_morphology.binary_image.astype(np.bool)) if not args.binarize_image else True, f'{img_morphology.binary_image} not binary.{np.unique(img_morphology.binary_image)}'
    
    # get random perturbations
    if damage:
        damage_perturbation = random.choice(anom_detect_settings.damage_perturbations)
    # apply perturbations (sometimes multiple)
    if damage:
        if random.random() < args.percent_apply_both_damage:
            damaged_img = (anom_detect_settings.damage_perturbations[0]()(img_morphology))
            assert np.array_equal(damaged_img, damaged_img.astype(np.bool)) if not args.binarize_image else True, f'{damaged_img} not binary.{np.unique(damaged_img)}'
            img_morphology = make_morph_object(damaged_img)
            damaged_img = anom_detect_settings.damage_perturbations[1]()(img_morphology)
            assert np.array_equal(damaged_img, damaged_img.astype(np.bool)) if not args.binarize_image else True, f'{damaged_img} not binary.{np.unique(damaged_img)}'
        else:
            damaged_img = damage_perturbation()(img_morphology)
    else:
        damaged_img = inverted_img

    assert np.array_equal(damaged_img, damaged_img.astype(np.bool)) if not args.binarize_image else True, f'{damaged_img} not binary.{np.unique(damaged_img)}'
                    
    img_morphology = make_morph_object(damaged_img)
    global_inking_perturbation = random.choices(anom_detect_settings.global_inking_perturbations,
                                                weights=anom_detect_settings.global_inking_perturbations_probs)[0]
    global_inking_perturbed_img = global_inking_perturbation()(img_morphology)
    assert np.array_equal(global_inking_perturbed_img, global_inking_perturbed_img.astype(np.bool)) if not args.binarize_image else True, f'{global_inking_perturbed_img} not binary.{np.unique(global_inking_perturbed_img)}'
    img_morphology = make_morph_object(global_inking_perturbed_img)
    swelled_img = anom_detect_settings.local_inking_perturbation()(img_morphology)
    #print('Pre-Bin:', np.unique(swelled_img))
    swelled_img[swelled_img < 1] = 0
    #print('Post-Bin:', np.unique(swelled_img))
    assert np.array_equal(swelled_img, swelled_img.astype(np.bool)) if not args.binarize_image else True, f'{swelled_img} not binary. {np.unique(swelled_img)}'
    return invert(swelled_img)


def process_img(img, img_path, new_img_path, damage, row):
    # perturb img
    try:
        perturbed_img = apply_random_perturbations(img, damage=damage)
    except (ValueError, ChooseSkeletonEndpointsException, BadSkeletonBendException) as e:
        print(f'{e}: {img_path}')
        return
     
    row["fname"] = new_img_path.name
    row["file_path"] = new_img_path
    # append perturbation info to this img_path
    # save perturbed img to new img path
    result = Image.fromarray(perturbed_img)
    assert result.mode == '1' if not args.binarize_image else True, f'Image is mode={result.mode}. It should be binary.'
    result.save(new_img_path)
    return row
    

def make_perturbation_dataset(perturbed_df_new_rows, dataset_df, dataset_idxs, num_random_idxs, n_repetitions, dest_dataset_path, dest_dataframe_path, damage=True):
    """ Create and save a fake perturbed dataset
    Args:
        dataset_df: Our UCSDDataset_Residue object's Dataframe containing normals to sample from
        dataset_idxs: A list of dataset indices that we want to sample from to perturb
        num_random_idxs: The number of samples we want from dataset_idxs.
            If we want to generate new damaged anomalies, this should be equal to len(dataset.anomaly_labels)
        damage: whether or not to apply damage perturbations like Fracture and Bend
    """
    random_idxs_to_perturb = np.random.choice(dataset_idxs, num_random_idxs)
#     perturbed_df_new_rows = []

    for iteration, i in enumerate(tqdm(random_idxs_to_perturb)):
        (img, class_label), img_path = dataset[i]
        row = dataset_df.iloc[i].copy(deep=True)
        row_copies = [copy.deepcopy(row) for _ in range(n_repetitions)]
        
#         [process_img(img, img_path, dest_dataset_path/(img_path.stem + f"_perturb{r}.tif"), damage, row_copies[r], perturbed_df_new_rows) for r in n_repetitions]
        anomaly_str = "anomaly"
        normal_str = "normal"
        out = Parallel(n_jobs=args.num_jobs)(delayed(process_img)(img, img_path, dest_dataset_path/(img_path.stem + f"_perturb{r}_{anomaly_str if damage else normal_str}_{int(time.time())}{r}.tif"), damage, row_copies[r]) for r in range(n_repetitions))
        for new_row in out:
            if new_row is not None:
                perturbed_df_new_rows.append(new_row)

    perturbed_df = pd.DataFrame(perturbed_df_new_rows, columns=dataset_df.columns)
    print(perturbed_df.shape)
    perturbed_df.to_csv(str(dest_dataframe_path))

    return perturbed_df_new_rows

    

# define destination paths for dataset
splits_dir = Path(args.splits_dir)
# splits_dir = Path("/trunk/kishore/data/anomaly-detection")/"ucsd_dataset"/f"{args.char}_no_italics"/"mixed_book_splits"
# aligned data below
# splits_dir = Path("/home/kishore/data/anomaly-detection")/"aligned-dataset"/"G"/"mixed_book_splits"

dataframe_path = args.src_dataframe_path
dest_dir = Path(args.dest_dir)

with open(dataframe_path/"book_names.json", 'r') as f:
    books = json.load(f)

if args.binarize_image:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            Binarizer(),
        ]
    )
else:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
        ]
    )


dataset_args = {
    "normal_df_path": dataframe_path/"normal_images.csv",
    "anomaly_df_path": dataframe_path/"anomaly_images.csv",
    "normal_data_path": splits_dir/".."/"normal",
    "anomaly_data_path": splits_dir/".."/"anomaly",
    "transform": transform, 
    "char_filters": [args.char],
    "book_filters": books,
    "return_residual": False,
    "debug": True
}


(train_dl, valid_dl, test_dl), wts, ds, (train_ds, valid_ds, test_ds) = random_splits_from_folder(
    dataset_args,
    train_dir = splits_dir/"train",
    valid_dir = splits_dir/"valid",
    test_dir = splits_dir/"test",
    batch_size = 16,
    num_workers = 1
)

for subset, dataset in [('train', train_ds), ('valid', valid_ds), ('test', test_ds)]:
    # damage=False if we want to make more NORMALS, damage=True for making more ANOMALIES
    for damage in [True, False]:
        if damage:
            class_name = 'anomaly'
        else:
            class_name = 'normal'
        
        dest_dataset_path = dest_dir/args.dest_dataset_name/subset/class_name
        dest_dataframe_path = dest_dir/args.dest_dataset_name/subset/f"{class_name}_images.csv"
        print(dest_dataset_path)
        os.makedirs(dest_dataset_path, exist_ok=False)

        normal_df = dataset.filtered_normal
                                                                                                    
        normal_df["class"] = "normal"
        anomaly_df = dataset.filtered_anomaly
        anomaly_df["class"] = "anomaly"
        normal_df.shape, anomaly_df.shape
        normal_df.head()
        normal_df = normal_df.reset_index()
        pd.set_option('display.max_colwidth', -1)
        normal_idxs = list(range(len(dataset.normal_labels)))
        num_random_idxs = len(dataset.anomaly_labels) if damage else len(dataset.normal_labels)
        print(len(normal_df))
        
        perturbed_df_new_rows = []
        make_perturbation_dataset(perturbed_df_new_rows, normal_df, normal_idxs, num_random_idxs, args.num_repetitions, dest_dataset_path, dest_dataframe_path, damage=damage)
