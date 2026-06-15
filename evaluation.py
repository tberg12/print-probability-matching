import torch
from torchvision import transforms
import numpy as np
from pathlib import Path
import pickle
import time
from tqdm import tqdm
import importlib
from glob import glob
import random
from torch.distributions.categorical import Categorical
import wandb
import random
import copy
import re
from collections import defaultdict
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, precision_recall_curve, f1_score
import utils
import sys
import pandas as pd
from PIL import Image
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_best_precision_recall_f1_score(tgts, predicted_probas):
    precision, recall, thresholds = precision_recall_curve(tgts, predicted_probas)
    f1_scores = 2 * precision * recall / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # replace nans from divide by zero
    f1_argmax = np.argmax(f1_scores)
    # best_threshold = thresholds[f1_argmax]
    return {'precision': precision[f1_argmax], 'recall': recall[f1_argmax], 'f1': f1_scores[f1_argmax]}


def damage_detection_random_baseline(damage_dataloader, p=0.5):
    num_positive = damage_dataloader.dataset.get_num_damage_images()
    num_total = len(damage_dataloader.dataset)
    hyps = np.random.binomial(1, p=p, size=num_total)
    tgts = [y for batch in damage_dataloader for y in batch[1].tolist()]
    precision, recall, f1, _ = precision_recall_fscore_support(tgts, hyps, average='binary')
    return {'precision': precision, 'recall': recall, 'f1': f1}


def damage_detection_eval(model, damage_dataloader):
    # TODO: need dataloader on separate damagedetectiondataset for this to work because the length is defined on test_pairs
    metrics = defaultdict(list)
    hyps = []
    pred_probas = []
    tgts = []
    for batch in damage_dataloader:
        with torch.no_grad():
            input_batch, tgts_batch = batch
            damage_convnet_feats = model.convnet(input_batch.to(device))
            damage_logits = model.damage_mlp(damage_convnet_feats)  # (B, 1)
            damage_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(damage_logits, tgts_batch.float().unsqueeze(1).to(damage_logits.device))
            tgts_batch_list = tgts_batch.squeeze().tolist()
            pred_probas_batch_list = torch.sigmoid(damage_logits).squeeze().tolist()
            tgts.extend(tgts_batch_list)
            pred_probas.extend(pred_probas_batch_list)
            mean_damage_loss = damage_loss.mean()
        metrics['loss'].append(mean_damage_loss.item())
    for k, v in metrics.items():
        metrics[k] = np.mean(v)
    prf_metrics = compute_best_precision_recall_f1_score(tgts, pred_probas)
    metrics.update(prf_metrics)
    return metrics


def get_gt_matches(gt_matches_fpath, img_root_path=None):
    # NOTE: changed to exclude from gt matches the images we dont have! 
    # This should change performance on our 
    # real valid/test sets (it should go up)
    img_root_path = Path(gt_matches_fpath).parent if img_root_path is None else img_root_path
    gt_matches = defaultdict(set)
    with open(gt_matches_fpath, 'r') as f:
        for line in f:
            if line.strip()[0] == '#': # skip lines starting with '#'
                continue
            # fnames = [fname.strip() for fname in line.split(',') if (img_root_path/fname.strip()).exists()]
            # fnames = [fname.strip() for fname in line.split(',') if (fname.strip()).exists()]
            fnames = []
            for fname in line.strip().split(','):
                fnames.append(Path(fname.strip()).name)
                # if Path(fname).exists():
                #     fnames.append(fname.strip())
                # elif (img_root_path/fname.strip()).exists():
                #     fnames.append(fname.strip())
                # elif 'anomaly-detection' in fname:
                #     # NOTE: replace savernake path with pando path
                #     fnames.append(fname.strip().replace('/home/nvog/projects/git/anomaly-detection/matching', '/graft2/code/nvog/git/matching/data'))
            # fnames = [Path(f).name for f in fnames]
            if len(fnames) >= 2:  # otherwise there's no match on the line and the fnames cant be considered "gt"
                match_set = set(fnames)
                # assert len(fnames) == len(match_set)
                for fname in match_set:
                    its_matches = copy.deepcopy(match_set)
                    its_matches.remove(fname)
                    gt_matches[fname].update(its_matches)
    assert len(gt_matches) > 0, f"No matches found in {gt_matches_fpath}"
    return gt_matches


def recall_at_k_metric(max_threshold, k, knns, gt_matches, debug=False): 
    files_at_5 = dict()
    at_threshold = dict()
    at_threshold_drs = dict()
    # print(f"\nRecall@k metric:")
    extra_info = dict()

    # first check to make sure that each key in gt_matches does not have an empty set
    assert all([len(v) > 0 for v in gt_matches.values()]), "Some keys in gt_matches have empty sets"

    for threshold in range(1, max_threshold+1):
        assert threshold <= k
        found = 0
        total = 0
        min_dists = []
        max_dists = []
        dist_ranges = []
        min_sims = []
        max_sims = []
        sim_ranges = []
        hit_positions = []
        total_potential_matches = []
        # for each query image
        for fpath, knn in knns.items():
            fname = Path(fpath).name
            # if the query image is in the ground truth matches, we want to compute recall for it
            if fname in gt_matches:
                ranked_candidates = knn['knn_fnames'][:threshold]
                matches = set(gt_matches[fname])
                # ensure that the query image is not in the matches set
                assert len(matches & set(fname)) == 0, "The query img should not be in the match set!"
                # hits is the intersection of the ranked candidates and the ground truth matches
                hits = matches & set(ranked_candidates)
                found += len(hits)
                this_hit_positions = []
                # TODO: if the true answer is in the top k candidates, record its position
                if len(hits) > 0:
                    this_hit_positions = [pos for pos in range(len(ranked_candidates)) if ranked_candidates[pos] in matches]
                hit_positions.append(this_hit_positions)
                total_potential_matches.append(min([len(matches), threshold]))
                # NOTE: should get 100% recall if it's threshold=1 and it gets at least 1 match
                # i.e., shouldnt be penalized for only getting 1 match at threshold=1 when there's 3 total
                total += min([len(matches), threshold])
                dists = knn["distances"][:threshold]
                sims = knn["similarities"][:threshold]
                min_dist, max_dist = dists[0], dists[-1]
                min_dists.append(min_dist)
                max_dists.append(max_dist)
                dist_ranges.append(max_dist - min_dist)
                min_sim, max_sim = sims[0], sims[-1]
                min_sims.append(min_sim)
                max_sims.append(max_sim)
                sim_ranges.append(max_sim - min_sim)
                
        at_threshold[threshold] = found
        at_threshold_drs[threshold] = total
        extra_info[threshold]= {
            "avg_min_dist": np.mean(min_dists), 
            "avg_max_dist": np.mean(max_dists), 
            "avg_dist_range": np.mean(dist_ranges),
            "avg_min_sim": np.mean(min_sims),
            "avg_max_sim": np.mean(max_sims),
            "avg_sim_range": np.mean(sim_ranges),
            # records list of hit positions in ranked candidates for each query 
            # if the ranked candidates have any. Otherwise, empty list.
            "hit_positions": hit_positions,
            "total_potential_matches": total_potential_matches,
        }
        # if args.debug:
        #     print(f"Found matches for {found} out of {total} in the top-{threshold}")
    st = ""
    for t in [1, 3, 5, 10]:
        st += f"k={t}:{at_threshold[t] / at_threshold_drs[t] * 100:0.1f} & "
    # if debug:
    return at_threshold, at_threshold_drs, st, extra_info
    # else:
    #     return at_threshold, at_threshold_drs, st


class KNN:
    def __init__(self):
        self.indices = None
        self.values = None
        self.attention_weights = None
        self.pair_embedding_norms = None
        self.pair_paths = None
        

def get_knn(args, model, ds_class, transform, img_path, img_paths, k):
    distances = []
    model = model.to(device)
    img_pairs = []
    pair_embedding_norms = []
    knn = KNN()
    for img_path2 in img_paths:
        assert Path(img_path).name != Path(img_path2).name, "Should not be comparing same images here!"
        img_path2 = Path(img_path2)
        img12 = ds_class.prepare_images(img_path, img_path2, transform)
        img_pairs.append(img12)
    img_pairs = torch.cat(img_pairs, dim=0).to(device)
    num_channels_per_image = img_pairs.shape[1] // 2
    img1 = img_pairs[:, :num_channels_per_image]
    img2 = img_pairs[:, num_channels_per_image:]
    if args.baseline_type == 'L2':
        distances = []
        im1 = img1[0].view(-1)
        for i in range(img2.shape[0]):
            im2 = img2[i].view(-1)
            distances.append((im1 - im2).pow(2).sum())
        distances = torch.tensor(distances)
    elif args.baseline_type == 'random':
        distances = torch.randperm(img2.shape[0])
    else:
        if args.model_type == 'Attention':
            # L2 Embedding loss so we want embeddings out of the model instead of just a distance
            (embed1, embed2), attn_entropy, attn_weights = model(img1, img2)
            distances = (embed1 - embed2).pow(2).sum(1)
            # distances = model.pair_distance(img_pairs)
            pair_embedding_norms = torch.stack((embed1.norm(dim=-1), embed2.norm(dim=-1))).detach().cpu().numpy()  # (2, len(img_paths))
            pair_paths = np.array([[img_path, img_path2] for img_path2 in img_paths]).T  # (2, len(img_paths))
        else:
            distances = model.pair_distance(img_pairs)
            
    distances = distances.flatten().detach().cpu().numpy()
       
    if args.rerank_with_l2:
        l2_distances = []
        im1 = img1[0].view(-1)
        for i in range(img2.shape[0]):
            im2 = img2[i].view(-1)
            l2_distances.append((im1 - im2).pow(2).sum())
        l2_distances = torch.tensor(l2_distances)
        l2_distances = l2_distances.flatten().detach().cpu().numpy()
        # TODO: scale distances first to same range
        distances += l2_distances
            
    if args.model_type == 'StackedBCE':
        # 1 is pos class, 0 is negative so sorting needs to be flipped
        distances *= -1
    min_indices = np.argsort(distances)
    knn.indices = min_indices[:k].tolist()
    knn.values = distances[knn.indices]
    if args.baseline_type is None and args.model_type == 'Attention':
        knn.attention_weights = attn_weights[knn.indices]  # (k, 16*16)
        knn.pair_embedding_norms = pair_embedding_norms
        knn.pair_paths = pair_paths
    return knn


def get_printer_book_name(p):
    s = str(p)
    pattern = r"(?P<printer_name>[a-zA-Z0-9]+)_[0-9a-zA-Z]+_[0-9a-zA-Z]+_[0-9a-zA-Z]+_(?P<book_name>[a-zA-Z0-9]+)[-_].*?\.tif$"
    cpattern = re.compile(pattern)
    match = cpattern.match(s)
    if match is None:
        print(f"{p} has no match for printer and/or book name")
    return match.group("printer_name"), match.group("book_name")



def load_recall_match_eval(matches_dir, background_set_filelist=None, limit_background_amount=None, limit_matches_amount=None):
    gt_matches_fpath = matches_dir/"matches.csv"
    gt_matches = get_gt_matches(gt_matches_fpath)  # should be filenames only, not paths
    gt_matches_img_paths = list(gt_matches.keys())
    # ensure that gt_matches_img_paths are the full paths and not just filenames.
    # if they're just filenames, add the matches_dir to the front
    if not all([Path(x).exists() for x in gt_matches_img_paths]):
        gt_matches_img_paths = [x.replace('/home/nvog/projects/git/anomaly-detection/matching', '/graft2/code/nvog/git/matching/data') if 'anomaly-detection' in x else str(matches_dir/x) for x in gt_matches_img_paths]
        # gt_matches_img_paths = [str(matches_dir/x) for x in gt_matches_img_paths]
    background_set_img_paths = []
    # import ipdb; ipdb.set_trace()
    if background_set_filelist:
        with open(background_set_filelist) as f:
            for line in f:
                path = line.strip()
                if 'anomaly-detection' in path:
                    # NOTE: replace savernake path with pando path
                    path = path.replace('/home/nvog/projects/git/anomaly-detection/matching', '/graft2/code/nvog/git/matching/data')
                if Path(path).exists():
                    background_set_img_paths.append(path)
            assert len(background_set_img_paths) > 0, f"No images found in {background_set_filelist}"
            random.seed(42)
            random.shuffle(background_set_img_paths)
            if limit_background_amount is not None:
                background_set_img_paths = background_set_img_paths[:limit_background_amount]
    if limit_matches_amount is not None:
        matches_dir_images = list(glob(f"{matches_dir}/*.tif"))[:limit_matches_amount]
        combined_path_list = matches_dir_images + gt_matches_img_paths + background_set_img_paths
        combined_name_set = set([Path(x).name for x in combined_path_list])
        match_img_paths = sorted(
            [x for x in combined_path_list if Path(x).name in combined_name_set],
            key=lambda x: Path(x).name
        )
    else:
        matches_dir_images = list(glob(f"{matches_dir}/*.tif"))
        combined_path_list = matches_dir_images + gt_matches_img_paths + background_set_img_paths
        match_img_paths = sorted(
            combined_path_list,
            key=lambda x: Path(x).name
        )
    anom_distances = []
    img_paths = []
    processed = set()
    pbar = tqdm(match_img_paths, total=len(match_img_paths), desc="Loading matches from CSV")
    for img_path in pbar:
        img_name = Path(img_path).name
        if img_name in processed: 
            continue
        processed.add(img_name)
        if 'anomaly-detection' in img_path:
            # NOTE: replace savernake path with pando path
            img_path = img_path.replace('/home/nvog/projects/git/anomaly-detection/matching', '/graft2/code/nvog/git/matching/data')
        img_paths.append(img_path)
    img_paths = np.array(img_paths)
    pbar.close()
    assert len(img_paths) > 0, "No images found in the matches directory"
    assert len(gt_matches) > 0, f"No matches found in {gt_matches_fpath}"
    return img_paths, gt_matches


def match_eval(args, model, match_img_paths, gt_matches, transform, topk=10):
    match_img_paths = match_img_paths.tolist() if isinstance(match_img_paths, np.ndarray) else match_img_paths
    knns = {
        'gt_matches': gt_matches,
        'results': dict(),  # stores topk matches for each image
    }
    match_img_names = [Path(p).name for p in match_img_paths]
    gt_matches_key_list = list(gt_matches.keys())
    # create gt_matches_paths by going through keys and getting the full path from the match_img_paths
    # NOTE: this fails if the match_img_names is not in the list (because we filtered it out with damage score filter)
    gt_matches_paths = [match_img_paths[match_img_names.index(k)] for k in gt_matches_key_list if k in match_img_names]
    # also get indices of gt_matches_paths in match_img_paths
    gt_matches_indices = [match_img_paths.index(p) for p in gt_matches_paths]
    model.eval()
    scores = None
    # NOTE: all new models use similarities, old use distances
    scores_are_distances = False

    # import ipdb; ipdb.set_trace()
    query_tensor = torch.cat([
        transform(Image.open(p)).unsqueeze(0) 
        for p in gt_matches_paths
    ], dim=0).to(device)
    candidate_tensor = torch.cat([transform(Image.open(img_path)).unsqueeze(0) for img_path in match_img_paths]).to(device)
    split_size = candidate_tensor.shape[0]  # split_size is the number of candidates to compare to each query
    float_dtype = torch.float32
    scores = torch.ones((len(query_tensor), len(candidate_tensor)), dtype=float_dtype) * torch.finfo(float_dtype).min
    q = 0
    for query in tqdm(query_tensor, total=len(query_tensor), desc=f"Matching (split_size=|C|={split_size})"):
        img2s_split = torch.split(candidate_tensor, split_size)
        for s, img2s in enumerate(img2s_split):
            img1s = query.unsqueeze(0).repeat_interleave(img2s.shape[0], dim=0)
            if args.model_type == 'DualEncoder':
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=args.amp):  # Enable AMP if args.amp is True
                        emb1s = model(img1s).squeeze().cpu()
                        emb2s = model(img2s).squeeze().cpu()
                if args.similarity_metric == 'dot':
                    sims = torch.mm(emb1s, emb2s.T)
                elif args.similarity_metric == 'cosine':
                    sims = torch.nn.functional.cosine_similarity(emb1s, emb2s)
                else:
                    raise ValueError(f"Unknown similarity metric {args.similarity_metric}")
                scores[q, s * split_size: s * split_size + min(split_size, len(sims))] = sims
            elif args.model_type == 'CrossEncoder':
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=args.amp):  # Enable AMP if args.amp is True
                        sims = model(img1s, img2s).squeeze().cpu()
                scores[q, s * split_size: s * split_size + min(split_size, len(sims))] = sims
            else:
                raise NotImplementedError(f"Model type {args.model_type} not supported")
        q += 1

            # if args.model_type == 'DualEncoder':
            # load/transform images
            # imgs = torch.cat([
            #     transform(Image.open(p)).unsqueeze(0) 
            #     for p in match_img_paths
            # ], dim=0)
            # get embeddings for all images as a batch
            # out = model(imgs.to(device))
            # # compute similarity here via dot product
            # if args.similarity_metric == 'dot':
            #     scores = torch.mm(out, out.T)  # [total_queries_and_candidates, total_queries_and_candidates]
            # elif args.similarity_metric == 'cosine':
            #     scores = torch.nn.functional.cosine_similarity(out, out) # [total_queries_and_candidates, total_queries_and_candidates]
            # else:
            #     raise ValueError(f"Unknown similarity metric {args.similarity_metric}")

            #
            #
            #
            # Create all pairs of query and candidate tensors in a single batch
            # num_queries = query_tensor.size(0)
            # num_candidates = candidate_tensor.size(0)
            # # Expand tensors to create all pairs
            # query_expanded = query_tensor.unsqueeze(1).expand(-1, num_candidates, -1, -1, -1)
            # candidate_expanded = candidate_tensor.unsqueeze(0).expand(num_queries, -1, -1, -1, -1)
            # # Reshape to batch all pairs
            # query_flat = query_expanded.reshape(-1, *query_tensor.shape[1:])
            # candidate_flat = candidate_expanded.reshape(-1, *candidate_tensor.shape[1:])
            # # Compute similarities in one pass
            # with torch.no_grad():
            #     with torch.amp.autocast('cuda', enabled=args.amp):  # Enable AMP if args.amp is True
            #         scores_flat = model(query_flat, candidate_flat).squeeze().cpu()
            # # Reshape scores to matrix format
            # scores = scores_flat.view(num_queries, num_candidates)


            #
            #
            # Hybridize both above approaches
            # Initialize scores matrix
            # num_queries = query_tensor.size(0)
            # num_candidates = candidate_tensor.size(0)
            # scores = torch.ones((num_queries, num_candidates), dtype=torch.float32) * torch.finfo(torch.float32).min
            # # Split candidates into manageable batches
            # candidate_batch_size = 64  # Adjust based on GPU memory
            # candidate_batches = torch.split(candidate_tensor, candidate_batch_size)
            # # Iterate over queries and process in chunks
            # query_batch_size = 1  # Adjust based on GPU memory
            # query_batches = torch.split(query_tensor, query_batch_size)

            # for q_start, query_batch in tqdm(enumerate(query_batches), total=len(query_batches), desc="Processing Queries"):
            #     # Iterate over candidate batches
            #     for c_start, candidate_batch in enumerate(candidate_batches):
            #         # Expand queries and candidates for pairwise comparison
            #         query_expanded = query_batch.unsqueeze(1).expand(-1, candidate_batch.size(0), -1, -1, -1)
            #         candidate_expanded = candidate_batch.unsqueeze(0).expand(query_batch.size(0), -1, -1, -1, -1)
            #         # Reshape for batched model input
            #         query_flat = query_expanded.reshape(-1, *query_tensor.shape[1:])
            #         candidate_flat = candidate_expanded.reshape(-1, *candidate_tensor.shape[1:])
            #         with torch.no_grad():
            #             with torch.amp.autocast('cuda', enabled=args.amp):  # Enable AMP if args.amp is True
            #                 sims_flat = model(query_flat, candidate_flat).squeeze().cpu()
            #         # Reshape scores and store in the main matrix
            #         sims_matrix = sims_flat.view(query_batch.size(0), candidate_batch.size(0))
            #         q_idx = slice(q_start * query_batch_size, q_start * query_batch_size + query_batch.size(0))
            #         c_idx = slice(c_start * candidate_batch_size, c_start * candidate_batch_size + candidate_batch.size(0))
            #         scores[q_idx, c_idx] = sims_matrix

        # else:
        #     raise NotImplementedError(f"Model type {args.model_type} not supported")

    # set gt_matches_indices to -1000 so they are not considered in the topk
    self_score = torch.finfo(float_dtype).max if scores_are_distances else torch.finfo(float_dtype).min
    for i in range(scores.shape[0]):
        scores[i, gt_matches_indices[i]] = self_score

    mintopk = min(topk, scores.shape[1])
    print(f"Computing top-{mintopk} matches for {len(gt_matches)} queries and {len(match_img_paths)} candidates")
    topk_vals, topk_inds = torch.topk(scores, mintopk, largest=not scores_are_distances)
    topk_vals = topk_vals.cpu().numpy()
    topk_inds = topk_inds.cpu().numpy()
    # TODO: this should only be run over the gt_matches_paths and not all match_img_paths
    # since we only want to evaluate the ground truth queries as queries, not all candidates as queries
    for i in range(scores.shape[0]):
        vals_i = topk_vals[i]
        inds_i = topk_inds[i]
        # TODO: if the true answer is in the mintopk candidates, record its position
        # import ipdb; ipdb.set_trace()
        knns['results'][gt_matches_paths[i]] = {
            "distances": vals_i if scores_are_distances else -vals_i,
            "similarities": vals_i if not scores_are_distances else -vals_i,
            "knn_paths": [match_img_paths[j] for j in inds_i],
            "knn_fnames": [Path(p).name for p in [match_img_paths[j] for j in inds_i]],
            "book_names": [get_printer_book_name(Path(p).name)[1] for p in [match_img_paths[j] for j in inds_i]],
            "printer_names": [get_printer_book_name(Path(p).name)[0] for p in [match_img_paths[j] for j in inds_i]],
            # TODO: for attention model:
            "attention_weights": None,  # TODO
            "pair_embedding_norms": None,  # TODO
            "pair_paths": None  # TODO
        }
        # if mintopk <= 20:
        #     assert gt_matches_paths[i] not in knns['results'][gt_matches_paths[i]]['knn_paths'], "Query image should not be in knn_paths (masked out from topk)"
        # else:
        if gt_matches_paths[i] in knns['results'][gt_matches_paths[i]]['knn_paths']:
            print(f"WARNING: Query image {gt_matches_paths[i]} in knn_paths (masked out from topk) for k={mintopk}")
    return knns


def evaluate_model(args, model, char, epoch, exp_name, root_result_dir, matches_dict_by_char, matches_ds_by_char, background_sets, topk=10, eval_keys=[]):
    wandb_log_dict = defaultdict(float)
    # init the recall_results/knns dicts with our different evaluation data settings so we can later fill them
    recall_results = dict()  # {'syn_train': {'G': {}} }
    knns = dict()  # {'syn_train': {'G': {}} }
    for data_type in ['syn', 'gt']:
        for subset in ['train', 'valid', 'test']:
            if data_type == 'gt' and subset == 'test':
                for background_set in background_sets:
                    recall_results[f'{data_type}_{subset}_{background_set}'] = dict()
                    knns[f'{data_type}_{subset}_{background_set}'] = dict()
            elif data_type == 'syn' and subset in ['train', 'valid']:
                recall_results[f'{data_type}_{subset}'] = dict()
                knns[f'{data_type}_{subset}'] = dict()

    # if args.input_residual or args.conv_template_residual:
    #     model.avg_image = val_loader.dataset.get_avg_image(val_loader.dataset.char2idx[char]).unsqueeze(0)

    print('---', char, '---')
    # SYNTHETIC top-5 recall eval
    for subset in ['train', 'valid']:  #'test']:
        key = f'syn_{subset}'
        if eval_keys and key not in eval_keys:
            continue
        match_img_paths = matches_dict_by_char[char][f"syn_{subset}_img_paths"]
        gt_matches = matches_dict_by_char[char][f"syn_{subset}_gt_pairs"]
        transform = matches_ds_by_char[char]['syn'][subset].transform
        start_time = time.time()
        knns[key][char] = match_eval(
            args, model, match_img_paths, gt_matches, transform, topk
        )
        elapsed_time = time.time() - start_time
        print(f"Eval Time: syn_{subset}_{char} top-{topk} matches took {int(elapsed_time)}s with {len(gt_matches)} queries & {len(match_img_paths)} candidates.")
        recall_results[key][char] = recall_at_k_metric(topk, topk, knns[f'syn_{subset}'][char]['results'], gt_matches)
        # print(f"syn_{subset} Recall@k results: {recall_results[f'syn_{subset}'][char]}")
    # REAL GT top-5 recall eval
    if any([k.startswith('gt') for k in matches_dict_by_char[char].keys()]):
        for subset in ['test']:  # NOTE: no valid
            for background_set in background_sets:
                key = f'gt_{subset}_{background_set}'
                if eval_keys and key not in eval_keys:
                    continue
                # some chars don't have gt matches in some sets (Q in leviathan for example)
                if key + "_img_paths" not in matches_dict_by_char[char]:
                    continue
                if background_set == 'areo':
                    ds_name = f"gt_{subset}_areo"
                elif background_set == 'lockespinoza_mix_neg':
                    ds_name = f"gt_{subset}_lockespinoza"
                else:
                    ds_name = f"gt_{subset}_pos"
                match_img_paths = matches_dict_by_char[char][key + "_img_paths"]
                if args.apply_damage_score_filter:
                    damage_df = pd.read_csv(args.apply_damage_score_filter[0], header=None)
                    damage_df.columns = ["path", "damage"]
                    print(f"Damage df shape: {damage_df.shape}")
                    damage_df["path"] = damage_df["path"].apply(lambda x: os.path.basename(x))
                    damage_df = damage_df.set_index("path")["damage"]
                    # TODO: change
                    match_img_paths = [p for p in match_img_paths if os.path.basename(p) not in damage_df or damage_df[os.path.basename(p)] > float(args.apply_damage_score_filter[1])]
                gt_matches = matches_dict_by_char[char][f"{ds_name}_gt_pairs"]
                transform = matches_ds_by_char[char][f"{ds_name}"][subset].transform
                start_time = time.time()
                if len(match_img_paths) == 0 or len(set(gt_matches.keys()).intersection(set([Path(p).name for p in match_img_paths]))) == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Eval Time: {key}_{char} top-{topk} matches took {int(elapsed_time)} seconds with {len(gt_matches)} queries & {len(match_img_paths)} candidates.")
                    print(f"WARNING: No match_img_paths found for {key} {char}. Were they filtered out by damage score > {args.apply_damage_score_filter[1]}?")
                    recall_results[key][char] = [
                        {k: 0 for k in range(1, topk+1)},  # recall at k
                        {k: len(gt_matches) for k in range(1, topk+1)},  # denom
                        {k: 0 for k in range(1, topk+1)},  # extra info
                    ]
                    return recall_results, knns
                else:
                    knns[key][char] = match_eval(
                        args, model, match_img_paths, gt_matches, transform, topk
                    )
                    elapsed_time = time.time() - start_time
                    print(f"Eval Time: {key}_{char} top-{topk} matches took {int(elapsed_time)} seconds with {len(gt_matches)} queries & {len(match_img_paths)} candidates.")
                    recall_results[key][char] = recall_at_k_metric(topk, topk, knns[key][char]['results'], gt_matches)
                # print(f"{key} Recall@k results: {recall_results[key][char]}")
    
    # if we're evaluating and the recall results are not empty (there exist chars for this key)
    if args.evaluate_ckpt and any([char in recall_results[key] for key in recall_results]):
        eval_ckpt_dir = Path(root_result_dir)/args.evaluate_ckpt_dest
        print('> Saving evaluation results to', str(eval_ckpt_dir))
        eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
        split_keys = eval_keys if eval_keys else ['syn_valid'] + ['gt_test_' + bg_set for bg_set in background_sets]
        dict_to_dump = {}
        for key in split_keys:
            dict_to_dump[key+'_knns'] = knns[key]
            dict_to_dump[key+'_recall_results'] = recall_results[key]
        
        # NOTE: otherwise exp name is too long for the file name
        name = '-'.join(exp_name.split('-')[-6:]) + f"_char{char}"
        print(f"> Saving pkl to {eval_ckpt_dir/(name + '.pkl')}")
        with open(eval_ckpt_dir/(name + '.pkl'), 'wb') as f:  # remove "/epoch40.pt" from exp_name for example
            pickle.dump(dict_to_dump, f)

        if args.model_type not in {'DualEncoder', 'CrossEncoder'}:
            raise NotImplementedError(f"Model type {args.model_type} not supported for evaluation due to similarities vs distances definition")
        # also save knn results to a csv, where each row is a query image and the columns are the top k matches followed by their similarity scores
        for k, v in knns.items():
            if not v:
                continue
            knn_results = []
            for query_img, knn_dict in v[char]['results'].items():
                knn_results.append([query_img] + knn_dict['knn_paths'] + knn_dict['similarities'].tolist())
            knn_results = np.array(knn_results)
            np.savetxt(eval_ckpt_dir/(name + f'_EvalChar{char}_{k}_knn_results.csv'), knn_results, fmt='%s', delimiter=',')

        for k in recall_results[split_keys[0]][char][0]:
            for split in split_keys:
                wandb_log_dict.update({
                    f"{split}_recall_{char}.{k}": recall_results[split][char][0][k] if char in recall_results[split] else None,
                    f"{split}_recall_denom_{char}.{k}": recall_results[split][char][1][k] if char in recall_results[split] else None,
                    f"{split}_recall_pct_{char}.{k}": recall_results[split][char][0][k] / recall_results[split][char][1][k] 
                        if char in recall_results[split] else None,
                })

        wandb_log_dict = utils.update_best_metrics(wandb_log_dict, epoch)
        if args.wandb:
            wandb.log(wandb_log_dict)
        else:
            subsets = eval_keys if eval_keys else ['syn_train', 'syn_valid', 'syn_test'] + ['gt_test_' + bg_set for bg_set in background_sets]
            for subset in subsets:
                # print(f"{subset}_recall_numer_{char}.5:", wandb_log_dict[f"{subset}_recall_{char}.5"])
                # print(f"{split}_recall_denom_{char}.{k}", wandb_log_dict[f"{split}_recall_denom_{char}.{k}"])
                print(f"*** {subset}_recall_pct_{char}.5:", wandb_log_dict[f"{subset}_recall_pct_{char}.5"])
                print(f"*** {subset}_recall_pct_{char}.10:", wandb_log_dict[f"{subset}_recall_pct_{char}.10"])

            # print(wandb_log_dict)

    return recall_results, knns
