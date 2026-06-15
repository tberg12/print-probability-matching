import re
import argparse

import torch
from torchvision import transforms
import numpy as np

import os
import copy
from PIL import Image

from pathlib import Path
import pandas as pd

import glob

import pickle

from collections import OrderedDict

import numpy as np

from datasets import UCSDSiameseDataset
import sys


sys.path.append("/home/kishore/projects/print-probability/anomaly-detection")
from anom_dataset import Binarizer, AnomalyPad
import model_args

from tqdm import tqdm as tqdm



def get_knn(dist_vecs, vec, k):
    vec = vec.unsqueeze(0)
    diffs = dist_vecs - vec
    distances = torch.norm(diffs, p="fro", dim=1)
    knn = distances.topk(k=k, largest=False, sorted=True)
    return knn


def get_printer_book_name(p):
    s = str(p)
    pattern = r"(?P<printer_name>[a-zA-Z0-9]+)_[0-9a-zA-Z]+_[0-9a-zA-Z]+_[0-9a-zA-Z]+_(?P<book_name>[a-zA-Z0-9]+)[-_].*?\.tif$"
    cpattern = re.compile(pattern)
    match = cpattern.match(s)
    if match is None:
        print(f"{p} has no match for printer and/or book name")
    return match.group("printer_name"), match.group("book_name")

def list_images_in_dirs(dirs):
    img_paths = []
    for dirpath in dirs:
        fpaths = list(glob.glob(f"{dirpath}/*.tif"))
        if len(fpaths) == 0:
            print(f"WARNING: {dirpath} does not have any files!")
        img_paths += fpaths
        print(f"Dir {dirpath} has {len(fpaths)} images")
    return img_paths

def get_gold_matches(gold_matches_fpath):
    # NOTE: changed to exclude from gold matches the images we dont have! 
    # This should change performance on our 
    # real valid/test sets (it should go up)
    img_root_path = Path(gold_matches_fpath).parent
    with open(gold_matches_fpath, 'r') as f:
        gold_matches = OrderedDict()
        for line in f:
            if line.strip()[0] == '#': # skip lines starting with '#'
                continue
            fnames = [fname.strip() for fname in line.split(',') if (img_root_path/fname.strip()).exists()]
            if len(fnames) >= 2:  # otherwise there's no match on the line and the fnames cant be considered "gold"
                match_set = set(fnames)
                # assert len(fnames) == len(match_set)
                for fname in match_set:
                    its_matches = copy.deepcopy(match_set)
                    its_matches.remove(fname)
                    gold_matches[fname] = its_matches
    return gold_matches

def get_knn_matches(anom_distances, img_paths, match_img_names, k):
        knns = dict()
        pbar = tqdm(zip(anom_distances, img_paths), total=len(anom_distances), desc="Indentifying nearest matches")
        problems = set()
        for i, (vec, img_path) in enumerate(pbar):
            if Path(img_path).name not in match_img_names: # cal distances FROM only the golds
                continue
            fname = Path(img_path).name
            img_path = str(img_path)
            knn = get_knn(anom_distances, vec, k)
            k_knn = anom_distances[knn.indices]
            k_knn = k_knn.detach().cpu().numpy()
            k_dists = knn.values.detach().cpu().numpy()
            knn_paths = img_paths[knn.indices.detach().cpu().numpy()]
            knn_fnames = [Path(p).name for p in knn_paths]
            printer_names = []

            book_names = []
            for p in knn_fnames:
                printer, book = get_printer_book_name(p)
                printer_names.append(printer)
                book_names.append(book)
            # res = filtered_df.loc[knn_fnames, ["book_name", "printer_name"]]
            # book_names, printer_names = res["book_name"].values.tolist(), res["printer_name"].values.tolist()
            entry = {"knn":k_knn, "distances": k_dists, 
                              "knn_paths":knn_paths, 
                              "book_names":book_names, "printer_names":printer_names}
            knns[img_path] = entry
        pbar.close()
        return knns
        

def recall_at_k_metric(max_threshold, k, knns, gold_matches, debug=False): 
        files_at_5 = dict()
        at_threshold = dict()
        at_threshold_drs = dict()
        print(f"\n\tRecall@k metric:")
        extra_info = dict()
        for threshold in range(1, max_threshold+1):
            assert threshold <= k
            found = 0
            total  = 0
            min_dists = []
            max_dists = []
            ranges = []
            for fpath, knn in knns.items():
                fname = Path(fpath).name
                if fname in gold_matches:
                    matches = set(gold_matches[fname])
                    assert len(matches & set(Path(fpath).name)) == 0, "The query img should not be in the match set!"
                    hits = matches & set(knn['knn_fnames'][:threshold])
                    found += len(hits)
                    # NOTE: should get 100% recall if it's threshold=1 and it gets at least 1 match
                    # i.e., shouldnt be penalized for only getting 1 match at threshold=1 when there's 3 total
                    total += min([len(matches), threshold])
                    dists = knn["distances"][:threshold]
                    min_dist, max_dist = dists[0], dists[-1]
                    min_dists.append(min_dist)
                    max_dists.append(max_dist)
                    ranges.append(max_dist-min_dist)
            at_threshold[threshold] = found
            at_threshold_drs[threshold] = total
            extra_info[threshold]= {
                "avg_min_dist": np.mean(min_dists), 
                "avg_max_dist": np.mean(max_dists), 
                "avg_dist_range": np.mean(ranges)
            }
            print(f"Found matches for {found} out of {total} in the top-{threshold}")
        st = ""
        for t in [1, 3, 5, 10]:
            st += f"{at_threshold[t]}/{at_threshold_drs[t]} & "
        #print(st)
        #print(files_at_5)
        #import pickle
        #with open("aligned.txt", 'wb') as f:
        #    pickle.dump(files_at_5, f)
        if debug:
            return at_threshold, at_threshold_drs, st, extra_info
        else:
            return at_threshold, at_threshold_drs, st
        
        
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate models for matches at different difficulty levels")
    parser.add_argument("--baseline_setting", action="store", required=True, type=str)
    parser.add_argument("--eval", action="store", choices=["valid", "test"], required=True, type=str)
    parser.add_argument("--char", action="store", choices=['G', 'D', 'M', 'F'], required=True, type=str)
    parser.add_argument("--fake_gold", action="store_true") # defaults to real-gold
    parser.add_argument("--all_gold", action="store_true") # defaults to using only valid/test split gold
    parser.add_argument("--save_path", default=None, type=str) # defaults to None (does not save the results)
    parser.add_argument("--use_train_fake_gold_matches", action="store_true") # defaults to not using train fake matches, to check for model overfit
    
    args = parser.parse_args()
    
    print(args)
    
    transform_image = False
    k = 11 # for KNN
    max_threshold = k
    NORMALIZATION_CONST = 255.
    
    
    # possible settings = {"normals-anomalies", "predicted-anomalies", "true-anomalies", "pure"}
    baseline_setting = args.baseline_setting
    char = args.char
    
    if args.fake_gold is True:
        assert args.baseline_setting == "pure"
    
    assert baseline_setting in {"normals-anomalies", "predicted-anomalies", "true-anomalies", "pure"}
    
    print(f"\nchar: {char}\n k-for-KNN: {k}\n transform-image: {transform_image}\nBaseline-setting: {baseline_setting}")
    
    model_settings = model_args.get_char_settings(char)
    data_root = Path("/home/kishore/data/anomaly-detection")
    dataframe_root = Path("/home/kishore/data/anomaly-detection/")
    dataframe_path = dataframe_root/"datasetv3"
    aligned_dataset_root = dataframe_root/"aligned-dataset"
    dataset_path = aligned_dataset = aligned_dataset_root/char
    splits_dir = aligned_dataset/"mixed_book_splits"
    valid_split_dir = splits_dir/"valid"
    test_split_dir = splits_dir/"test"
    
    if args.eval == "valid":
        selected_split_dir = valid_split_dir
    else:
        selected_split_dir = test_split_dir

    which_split = "all" if args.all_gold is True else args.eval
    if args.fake_gold is False:
        matches_dir = data_root/"gold-test-sets"/char/which_split
    else:
        if args.use_train_fake_gold_matches:
            matches_dir = dataframe_root/"gold-test-sets"/"fake-test-sets-07-24"/char/"train"
            assert args.baseline_setting == "pure", "Using train fake matches to test overfit, cannot use other settings"
        else:
            # matches_dir = data_root/"gold-test-sets"/"fake-test-sets"/char/which_split
            # matches_dir = data_root/"gold-test-sets"/"fake-test-sets-07-03"/char/which_split
            matches_dir = data_root/"gold-test-sets"/"fake-test-sets-07-24"/char/which_split
#     matches_dir = data_root/"gold-test-sets"/char/args.eval
    # color_dir = matches_root_dir.parent/"areo_story_7_17_19"
    print(f"\nGold matches dir: {matches_dir}\n")
    
    
    
    if baseline_setting == "normals-anomalies": # all valid/test normals and anomalies + gold matches
        other_img_dirs = [selected_split_dir/"normal", selected_split_dir/"anomaly"]
    
    elif baseline_setting == "predicted-anomalies": # predicted valid/test anomalies + gold matches
        other_img_dirs = [selected_split_dir/"anomaly_predictions"]
    
    elif baseline_setting == "true-anomalies": # true valid/test anomalies + gold matches
        other_img_dirs = [selected_split_dir/"anomaly"]
    
    else: # only gold matches
        other_img_dirs = []
        
    
    print(f"Other image dirs: {other_img_dirs}")
    


    char_args = model_args.get_char_settings(char)
    transform = transforms.Compose(
            [
#                 transforms.Grayscale(num_output_channels=1),
#                 Binarizer(),
                AnomalyPad(150,150, pad_value=1),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[char_args.NormalizationArgs.mean__aligned_full.value],
                #                     std=[char_args.NormalizationArgs.std__aligned_full.value])
            ]
        )


    gold_matches_fpath = matches_dir/"matches.csv"

    gold_matches = get_gold_matches(gold_matches_fpath)
    print(f"Number of gold matches: {len(gold_matches)}")


    match_img_paths = sorted([x for x in glob.glob(f"{matches_dir}/*.tif")], key = lambda x: Path(x).name)

    anom_distances = []
    img_paths = []

    other_img_paths = list_images_in_dirs(other_img_dirs)
    all_img_paths = match_img_paths + other_img_paths

    processed = set()
    pbar = tqdm(all_img_paths, total=len(all_img_paths), desc="Generating distance vectors")
    for img_path in pbar:
        img_name = Path(img_path).name
        if img_name in processed: 
            continue
        processed.add(img_name)
        if transform_image:
            tfm_img = UCSDSiameseDataset.prepare_image(img_path, transform)
            vec = tfm_img.view(1,-1)
        else:
            pil_img = Image.open(img_path)
            vec = torch.tensor(np.array(pil_img)).float().view(1,-1)
            vec = vec / NORMALIZATION_CONST
            pil_img.close()
        anom_distances.append(vec)   
        img_paths.append(img_path)
    anom_distances = torch.cat(anom_distances)
    img_paths = np.array(img_paths)
    pbar.close()


    match_img_names = set([Path(p).name for p in match_img_paths])

    knns = get_knn_matches(anom_distances, img_paths, match_img_names, k)
        
    res = recall_at_k_metric(max_threshold, k, knns, gold_matches)
    print(res)
    
    if args.save_path is not None:
        data = {"knns":knns, "gold_matches":gold_matches, "k":k,
                "aligned_dataset":aligned_dataset, "matches_dir": matches_dir}

        with open(args.save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved result to file: {args.save_path}")


    
    
    