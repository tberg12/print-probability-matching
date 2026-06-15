import argparse
import PIL
from PIL import Image
from tqdm import tqdm
from losses import TripletDualEncoderLoss, TripletDualEncoderLoss, TripletScoreLoss, CrossEncoderTripletBCELoss, NPairsWithExtraNegatives, ContrastiveLoss
from models import TripletDualEncoderNet, AttentionNet, StackedNet, DualEncoderNet, CrossEncoderNet
from datasets import create_datasets, TwinDataset, GroundTruthMatchesDataset, MultiCharTwinDataset, compute_avg_image, ImagePathDataset
from contextlib import redirect_stdout
import pickle
from pathlib import Path
import random
import time
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import sys
import model_args
from datetime import datetime
import evaluation
import torch
from torchvision import transforms
import numpy as np
from pathlib import Path
#from metrics import get_roc
from glob import glob
from torch.distributions.categorical import Categorical
import wandb
from collections import defaultdict
import copy
import utils
from pytorch_metric_learning.losses import TripletMarginLoss, NPairsLoss, NTXentLoss
from pytorch_metric_learning.reducers import DoNothingReducer, MeanReducer
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from make_twin_dataset_from_splits import Binarizer, SquarePad
import faiss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_matches(args, model, exp_name, matches_ds_by_char, root_result_dir):
    assert len(args.discover_matches) == 2
    query_list_path, candidate_list_path = args.discover_matches
    print(f"Discovering matches for interesting file list:\n{query_list_path}\nusing candidate images from:{candidate_list_path}")
    with open(query_list_path) as f:
        query_list = [Path(line.strip()) for line in f]
    with open(candidate_list_path) as f:
        candidate_list = [Path(line.strip()) for line in f]
    model.eval()
    # import ipdb; ipdb.set_trace()
    ds = matches_ds_by_char[args.char[0]]['gt_test_pos']['test']
    discovery_knns = evaluation.get_knn_matches(
        args, 
        model, 
        ds,
        np.array(query_list + candidate_list), 
        query_list, 
        [p.name for p in query_list], 
        args.k
    ) 
    eval_ckpt_dir = Path(root_result_dir)/args.evaluate_ckpt_dest
    print('> Saving evaluation results to', str(eval_ckpt_dir))
    eval_ckpt_dir.mkdir(parents=True, exist_ok=True)
    dict_to_dump = {'discovery_knns': discovery_knns,}
    name = 'Discover_' + (exp_name if args.baseline_type else str(Path(exp_name).parent.name))
    print(f"> Saving pkl to {eval_ckpt_dir/(name + '.pkl')}")
    with open(eval_ckpt_dir/(name + '.pkl'), 'wb') as f:  # remove "/epoch40.pt" from exp_name for example
        pickle.dump(dict_to_dump, f)
    print('Done!')


def evaluate(args, model, epoch, exp_name, root_result_dir, matches_dict_by_char, matches_ds_by_char, wandb_log_dict):
    start_time = time.time()
    
    for char in sorted(set(args.char).union(set(args.eval_char))):
        recall_results, knns = evaluation.evaluate_model(
            args, model, char, epoch, exp_name, root_result_dir,
            matches_dict_by_char, matches_ds_by_char,
            background_sets=['pos', 'strong_neg', 'weak_neg', 'mix_neg'] + (['areo'] if args.use_areo and char in 'DFGM' else []) + (['lockespinoza_mix_neg'] if char in 'ABCFGHNOPQRT' else []),
            topk=args.k,
            eval_keys=args.limit_eval_keys
        )
        # loop thru and print results
        # for k in recall_results['syn_train'][char][0]:
        for k in [1, 3, 5, 10]:
            # assert char in recall_results['syn_train']
            # assert char in recall_results['syn_valid']
            for split in recall_results.keys():
                if char in recall_results[split]:
                    assert recall_results[split][char][1][k] > 0, f"Denominator is 0 for {char} at k={k} in {split}. Was the dataset empty?"
                if split in recall_results and char in recall_results[split]:
                    wandb_log_dict.update({
                        f"{split}_recall_{char}.{k}": recall_results[split][char][0][k] if char in recall_results[split] else 0,
                        f"{split}_recall_denom_{char}.{k}": recall_results[split][char][1][k] if char in recall_results[split] else 0,
                        f"{split}_recall_pct_{char}.{k}": recall_results[split][char][0][k] / recall_results[split][char][1][k] * 100 if char in recall_results[split] else 0,
                    })
    elapsed_time = time.time() - start_time
    # also add macro average pct over characters for each k value
    dummy_key = args.limit_eval_keys[0] if args.limit_eval_keys else 'syn_train'
    if char in recall_results[dummy_key]:
        for k in recall_results[dummy_key][char][0]:
            if k in [1, 3, 5, 10, 100, 1000]:
                for split in recall_results.keys():
                    wandb_log_dict.update({
                        # macro avg pct
                        f"{split}_recall_pct_allchars.{k}": np.mean(
                            [
                                np.nan_to_num(wandb_log_dict[f"{split}_recall_pct_{char}.{k}"])
                                for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                if f"{split}_recall_pct_{char}.{k}" in wandb_log_dict
                            ]
                        ),
                    })
                    print(f"{split}_recall_pct_allchars.{k}: {wandb_log_dict[f'{split}_recall_pct_allchars.{k}']:0.1f}")
    # import ipdb; ipdb.set_trace()
    wandb_log_dict = utils.update_best_metrics(wandb_log_dict, epoch)
    # wandb.log(wandb_log_dict) if args.wandb else print(wandb_log_dict)
    if args.wandb:
        wandb.log(wandb_log_dict)
    print(f"Total Eval Time: {int(elapsed_time)} seconds ({len(args.char)} characters).")
    return wandb_log_dict

 
def train_model(args, 
                train_loader, 
                model, 
                loss_fn, 
                optimizer, 
                scheduler,
                metrics,
                root_result_dir, 
                exp_name, 
                matches_dict_by_char, 
                matches_ds_by_char, 
                start_epoch=1,
                ):
    """
        Trainer for matching networks
    """
    result_dir = Path(root_result_dir)/exp_name
    print('> Saving results to', str(result_dir))
    if not result_dir.exists():
        result_dir.mkdir()
        
    n_epochs = args.n_epochs
    train_losses = []
    valid_losses = []
    best_score_so_far = 0.0
    best_epoch = 0

    if args.baseline_type or args.evaluate_ckpt:
        train_loss = train_losses_by_char = train_softmax_distribution_entropy = 0
        mean_attn_entropy_pos = mean_attn_entropy_neg = 0
        anchor_embeddings = positive_embeddings = negative_embeddings = hardest_negative_embeddings = 0
        n_epochs = 1
        if args.input_residual or args.conv_template_residual:
            #TODO: make part of model?
            model.avg_image = train_loader.dataset.get_avg_image(train_loader.dataset.char2idx[args.char[0]])[:1]
        if args.discover_matches:
            discover_matches(args, model, exp_name, matches_ds_by_char, root_result_dir)
            return
        
    epoch_result = None        
    wandb_log_dict = defaultdict(float)
    for epoch in range(start_epoch, n_epochs+1):
        #do first eval on randomly initialized
        if False:
        # if epoch == start_epoch:
            # now do matching recall evaluation
            if args.baseline_type or args.evaluate_ckpt or epoch == start_epoch or epoch % args.eval_interval == 0:
                wandb_log_dict = evaluate(args, model, epoch, exp_name, root_result_dir, matches_dict_by_char, matches_ds_by_char, wandb_log_dict)
            if args.evaluate_ckpt:
                return
            if args.baseline_type:
                train_loss = epoch_result['train_loss'] if epoch_result is not None else sys.maxsize
                pct_metric_suffix = f"_recall_pct_{args.char[0]}"
                num_metric_suffix = f"_recall_{args.char[0]}"
                denom_metric_suffix = f"_recall_denom_{args.char[0]}"
                # best_metric_name = f'{args.stopping_metric}_recall_pct_{args.char[0]}.10'
                best_metric_name = f'{args.stopping_metric}_recall_pct_allchars.5'
                if wandb_log_dict[best_metric_name] > best_score_so_far:
                    best_score_so_far = wandb_log_dict[best_metric_name]
                    best_epoch = epoch
                    print(f'*** {args.char} Best Score {best_metric_name} at Epoch {epoch}: {best_score_so_far:0.1f}')
                    for metric_name in ['syn_valid', 'gt_test_pos', 'gt_test_strong_neg', 'gt_test_weak_neg', 'gt_test_mix_neg'] + (['gt_test_areo'] if args.use_areo and args.char[0] in 'DFGM' else []):
                        for metric_key in [metric_name + pct_metric_suffix, metric_name + num_metric_suffix, metric_name + denom_metric_suffix]:
                            for n in [1, 3, 5, 10]:
                                print(f"*** Epoch {epoch} {args.char} {metric_key + f'.{n}'}: {wandb_log_dict[metric_key + f'.{n}']:0.1f}")
                else:
                    print(f'{args.char} did not improve on {best_metric_name} at epoch {epoch}: {wandb_log_dict[best_metric_name]:0.1f}\n')
                    print(f'>>> {args.char} Score {best_metric_name} at Epoch {epoch}: {best_score_so_far:0.1f}\n')
                    for metric_name in ['syn_valid', 'gt_test_pos', 'gt_test_strong_neg', 'gt_test_weak_neg', 'gt_test_mix_neg'] + (['gt_test_areo'] if args.use_areo and args.char[0] in 'DFGM' else []):
                        for metric_key in [metric_name + pct_metric_suffix, metric_name + num_metric_suffix, metric_name + denom_metric_suffix]:
                            for n in [1, 3, 5, 10]:
                                print(f">>> {args.char} {metric_key + f'.{n}'}: {wandb_log_dict[metric_key + f'.{n}']:0.1f}")
                return
            elif args.wandb:
                wandb.log(wandb_log_dict)
            if args.baseline_type is None and args.evaluate_ckpt is False and (epoch == start_epoch or epoch % args.eval_interval == 0):
            # if False:
                train_loss = epoch_result['train_loss'] if epoch_result is not None else sys.maxsize
                pct_metric_suffix = f"_recall_pct_{args.char[0]}"
                num_metric_suffix = f"_recall_{args.char[0]}"
                denom_metric_suffix = f"_recall_denom_{args.char[0]}"
                # best_metric_name = f'{args.stopping_metric}_recall_pct_{args.char[0]}.5'
                best_metric_name = f'{args.stopping_metric}_recall_pct_allchars.5'
                state = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                }
                if wandb_log_dict[best_metric_name] > best_score_so_far:
                    best_score_so_far = wandb_log_dict[best_metric_name]
                    best_epoch = epoch
                    print(f'*** {args.char} Best Score {best_metric_name} at Epoch {epoch}: {best_score_so_far:0.1f}\n')
                    for metric_name in ['syn_valid', 'gt_test_pos', 'gt_test_strong_neg', 'gt_test_weak_neg', 'gt_test_mix_neg'] + (['gt_test_areo'] if args.use_areo and args.char[0] in 'DFGM' else []):
                        for metric_key in [metric_name + pct_metric_suffix, metric_name + num_metric_suffix, metric_name + denom_metric_suffix]:
                            for n in [1, 3, 5, 10]:
                                print(f"*** {args.char} {metric_key + f'.{n}'}: {wandb_log_dict[metric_key + f'.{n}']:0.1f}")
                                state[metric_key] = wandb_log_dict[metric_key + f'.{n}']
                    torch.save(state, result_dir/f"best.pt")
                    # torch.save(state, result_dir/f"epoch{epoch}.pt")
                else:
                    print(f'{args.char} did not improve on {best_metric_name} at epoch {epoch}: {wandb_log_dict[best_metric_name]:0.1f}\n')
                    print(f'>>> {args.char} Score {best_metric_name} at Epoch {epoch}: {best_score_so_far:0.1f}\n')
                    for metric_name in ['syn_valid', 'gt_test_pos', 'gt_test_strong_neg', 'gt_test_weak_neg', 'gt_test_mix_neg'] + (['gt_test_areo'] if args.use_areo and args.char[0] in 'DFGM' else []):
                        for metric_key in [metric_name + pct_metric_suffix, metric_name + num_metric_suffix, metric_name + denom_metric_suffix]:
                            for n in [1, 3, 5, 10]:
                                print(f">>> {args.char} {metric_key + f'.{n}'}: {wandb_log_dict[metric_key + f'.{n}']:0.5f}")
                                state[metric_key] = wandb_log_dict[metric_key + f'.{n}']
                    # TODO: CHECKPOINT EVERY EPOCH!
                    # torch.save(state, result_dir/f"epoch{epoch}.pt")
                    torch.save(state, result_dir/f"best.pt")
                train_losses.append(train_loss)
        
        
        if args.baseline_type is None and args.evaluate_ckpt is False:            
            epoch_result = train_epoch(args, train_loader, model, loss_fn, optimizer, scheduler, epoch)
            message = f"Epoch: {epoch}/{n_epochs}. Train set: Average loss: {epoch_result['train_loss']}"
            print(message)
            wandb_log_dict.update({
                "exp_name": exp_name,
                "train_on_syn_matches": args.train_on_syn_matches,
                "epoch": epoch,
                "train_loss": epoch_result['train_loss'],
                # "softmax_distribution_entropy": epoch_result['train_softmax_distribution_entropy'],
                "mean_attn_entropy_pos": epoch_result['mean_attn_entropy_pos'],
                "mean_attn_entropy_neg": epoch_result['mean_attn_entropy_neg'],
            })

        # now do matching recall evaluation
        if args.baseline_type or args.evaluate_ckpt or epoch % args.eval_interval == 0:
            wandb_log_dict = evaluate(args, model, epoch, exp_name, root_result_dir, matches_dict_by_char, matches_ds_by_char, wandb_log_dict)
        elif args.wandb:
            wandb.log(wandb_log_dict)

        if args.baseline_type is None and args.evaluate_ckpt is False and (epoch == start_epoch or epoch % args.eval_interval == 0):
        # if False:
            train_loss = epoch_result['train_loss'] if epoch_result is not None else sys.maxsize
            best_metric_name = f'{args.stopping_metric}_recall_pct_allchars.5'
            pct_metric_suffix = f"_recall_pct_{args.char[0]}"
            num_metric_suffix = f"_recall_{args.char[0]}"
            denom_metric_suffix = f"_recall_denom_{args.char[0]}"
            # best_metric_name = f'{args.stopping_metric}_recall_pct_{args.char[0]}.5'
            state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
            }
            if wandb_log_dict[best_metric_name] > best_score_so_far:
                best_score_so_far = wandb_log_dict[best_metric_name]
                best_epoch = epoch
                print(f'*** {args.char} Best Score {best_metric_name} at Epoch {epoch}: {best_score_so_far:0.1f}\n')
                for metric_name in ['syn_valid', 'gt_test_pos', 'gt_test_strong_neg', 'gt_test_weak_neg', 'gt_test_mix_neg'] + (['gt_test_areo'] if args.use_areo and args.char[0] in 'DFGM' else []):
                    for metric_key in [metric_name + pct_metric_suffix, metric_name + num_metric_suffix, metric_name + denom_metric_suffix]:
                        for n in [1, 3, 5, 10]:
                            print(f"*** {args.char} {metric_key + f'.{n}'}: {wandb_log_dict[metric_key + f'.{n}']:0.1f}")
                            state[metric_key] = wandb_log_dict[metric_key + f'.{n}']
                torch.save(state, result_dir/"best.pt")
            else:
                print(f'{args.char} did not improve on {best_metric_name} at epoch {epoch}: {wandb_log_dict[best_metric_name]:0.1f}\n')
                print(f'>>> {args.char} Score {best_metric_name} at Epoch {epoch}: {best_score_so_far:0.1f}\n')
                for metric_name in ['syn_valid', 'gt_test_pos', 'gt_test_strong_neg', 'gt_test_weak_neg', 'gt_test_mix_neg'] + (['gt_test_areo'] if args.use_areo and args.char[0] in 'DFGM' else []):
                    for metric_key in [metric_name + pct_metric_suffix, metric_name + num_metric_suffix, metric_name + denom_metric_suffix]:
                        for n in [1, 3, 5, 10]:
                            print(f">>> {args.char} {metric_key + f'.{n}'}: {wandb_log_dict[metric_key + f'.{n}']:0.5f}")
                            state[metric_key] = wandb_log_dict[metric_key + f'.{n}']
                # TODO: CHECKPOINT EVERY EPOCH!
                torch.save(state, result_dir/f"epoch{epoch}.pt")
            train_losses.append(train_loss)
        
    print(f'Finished training {args.char}. Best score {best_metric_name} = {best_score_so_far:0.1f} reached at Epoch {best_epoch}')
    print('Args:\n', args, '\n\nResults:', wandb_log_dict)



def train_epoch(args, train_loader, model, loss_fn, optimizer, scheduler, epoch):
    metrics = None
    losses = []
    entropies = []
    attn_entropies_pos = []
    attn_entropies_neg = []
    anchor_embeddings = []
    positive_embeddings = []
    negative_embeddings = []
    hardest_negative_embeddings = []
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    # make a pbar that updates with loss after each batch
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    for b, batch in pbar:
        start_time = time.time()
        # (img1_batch, img2_batch), char_label_batch, (normal_img_batch, damage_img_batch) = batch
        img1_batch = batch['damage1']
        img2_batch = batch['damage2']
        normal_img_batch = batch['normal']
        char_label_batch = batch['char']


        assert len(set(char_label_batch)) == 1, "Multiple characters in this single batch!"
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=args.amp):
            if 'DualEncoder' in args.model_type:
                anchor = img1_batch.to(device)
                positive = img2_batch.to(device)
                extra_negatives = normal_img_batch.to(device)
                imgs = torch.cat([img1_batch, img2_batch, normal_img_batch], dim=0).to(device)
                # print(f"imgs shape: {imgs.shape}")
                # print(f"labels shape: {labels.shape}")
                # print("labels:", labels)
                # print(f"imgs_w_normal shape: {imgs_w_normal.shape}")
                # breakpoint()
                if args.loss_type == 'triplet':
                    embeds = model(imgs)
                    loss = loss_fn(embeds, labels)
                elif args.loss_type == 'npairs':
                    embeds = model(imgs)
                    # loss = loss_fn(embeds, labels)
                    # uncombine for loss
                    anchor = embeds[:len(img1_batch)]
                    positive = embeds[len(img1_batch):2*len(img1_batch)]
                    extra_negatives = embeds[2*len(img1_batch):]
                    loss = loss_fn(anchor, positive, extra_negatives)
                elif args.loss_type == 'clip':
                    embeds = model(imgs)
                    # uncombine for loss
                    anchor = embeds[:len(img1_batch)]
                    positive = embeds[len(img1_batch):2*len(img1_batch)]
                    # extra_negatives = embeds[2*len(img1_batch):]
                    loss = loss_fn(anchor, positive, None)
                elif args.loss_type == 'clip_extra':
                    embeds = model(imgs)
                    # loss = loss_fn(embeds, labels)
                    # uncombine for loss
                    anchor = embeds[:len(img1_batch)]
                    positive = embeds[len(img1_batch):2*len(img1_batch)]
                    extra_negatives = embeds[2*len(img1_batch):]
                    loss = loss_fn(anchor, positive, extra_negatives)

            elif args.model_type == 'CrossEncoder':
                if args.loss_type == 'triplet_bce':
                    img1_batch = img1_batch.to(device)
                    img2_batch = img2_batch.to(device)
                    anchor_img = img1_batch
                    pos_img = img2_batch
                    if args.negative_mining == 'l2':
                        neg_img = torch.cat([img2_batch[torch.cdist(anchor_img[i].unsqueeze(0), img2_batch).argmax()] for i in range(len(anchor_img))])
                    elif args.negative_mining == 'random':
                        neg_img_idxs = [
                            random.choice(
                                [i for i in range(len(img2_batch)) if i != j]
                            ) 
                            for j in range(len(anchor_img))
                        ]
                        neg_img = img2_batch[neg_img_idxs]
                    # TODO: randomly fill with half random negatives from img2_batch and half from extra_negatives
                    elif args.negative_mining == "random_half_normal":
                        neg_img_list = []
                        for j in range(len(anchor_img)):
                            # randomly choose from img2_batch and normal_img_batch
                            neg_img_list.append(
                                random.choice([
                                    img2_batch[j].unsqueeze(0), 
                                    normal_img_batch[j].unsqueeze(0)
                                ]).to(device)
                            )
                        neg_img = torch.cat(neg_img_list)
                    sim_ap = model(anchor_img, pos_img)
                    sim_an = model(anchor_img, neg_img)
                    loss = loss_fn(sim_ap, sim_an)
                    
            elif args.model_type == 'Attention':
                raise NotImplementedError
                anchor_img = img_pair_batch[:, :num_channels_per_image]
                pos_img = img_pair_batch[:, num_channels_per_image:]
                neg_img = neg_singleton_batch

                (pos_anchor_embedding, positive_embedding), attn_entropy_pos, attn_weights = model(anchor_img, pos_img)
                (neg_anchor_embedding, negative_embedding), attn_entropy_neg, attn_weights = model(anchor_img, neg_img)
                loss = loss_fn(pos_anchor_embedding, positive_embedding, neg_anchor_embedding, negative_embedding)
                attn_entropies_pos.append(attn_entropy_pos)
                attn_entropies_neg.append(attn_entropy_neg)
            elif args.model_type == 'StackedBCE':
                raise NotImplementedError
                pos_probs = model(img_pair_batch)
                neg_probs = model(torch.cat([img_pair_batch[:, :1], neg_singleton_batch], dim=1))
                preds = torch.cat([pos_probs, neg_probs])
                labels = torch.cat([torch.ones_like(pos_probs), torch.zeros_like(neg_probs)])
                loss = loss_fn(preds, labels)

            if isinstance(loss, dict):
                loss = loss['loss']['losses']

            mean_loss = loss.mean()

        scaler.scale(mean_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            # Update learning rate
            scheduler.step()
        if b % 100 == 0:
            print(f"Step {(epoch - 1) * len(train_loader) + b}: LR {scheduler.get_last_lr()[0]:.6f}")
        losses.append(mean_loss.item())
        time_elapsed = time.time() - start_time
        moving_avg_loss = np.mean(losses[-10:]) if len(losses) > 10 else mean_loss.item()
        # start_loss = np.mean(losses[:10]) if len(losses) > 10 else mean_loss.item()
        start_loss = losses[0]
        pbar_postfix_dict = {'start_loss': start_loss, 'mov_avg_loss': moving_avg_loss}
        if hasattr(loss_fn, 'get_temperature'):
            pbar_postfix_dict['clip_temp'] = loss_fn.get_temperature().item()
        pbar.set_postfix(pbar_postfix_dict)

        if args.wandb:
            wandb.log({
                'train_loss': mean_loss.item(),
                'learning_rate': scheduler.get_last_lr()[0] if scheduler else args.lr,
                'clip_temperature': loss_fn.get_temperature().item() if hasattr(loss_fn, 'get_temperature') else None,
                'epoch': epoch,
                'batch': (epoch - 1) * len(train_loader) + b,
            })

    return {
        'train_loss': np.mean(losses), 
        # 'train_loss_by_char': {k: np.mean(v) for k, v in losses_per_char.items()}, 
        'mean_entropy': sum(entropies) / len(entropies) if len(entropies) > 0 else 0., 
        'mean_attn_entropy_pos': sum(attn_entropies_pos) / len(attn_entropies_pos) if len(attn_entropies_pos) > 0 else 0., 
        'mean_attn_entropy_neg': sum(attn_entropies_neg) / len(attn_entropies_neg) if len(attn_entropies_neg) > 0 else 0., 
        'anchor_embeddings': anchor_embeddings, 
        'positive_embeddings': positive_embeddings, 
        'negative_embeddings': negative_embeddings, 
        'hardest_negative_embeddings': hardest_negative_embeddings, 
        'metrics': metrics
    }


def get_loss_fn(args):
    if args.loss_type == 'triplet_bce':
        assert args.model_type == 'CrossEncoder'
        return CrossEncoderTripletBCELoss()
    elif args.loss_type == 'triplet':
        assert args.model_type != 'CrossEncoder'
        return TripletDualEncoderLoss(args.margin, reduction="none", squared=True)
    elif args.loss_type == 'npairs':
        # return NPairsLoss(
        #     reducer=DoNothingReducer(), 
        #     distance=CosineSimilarity() if args.similarity_metric == 'cosine' else DotProductSimilarity(),
        # )
        # return NPairsWithExtraNegatives()
        return ContrastiveLoss(initial_temperature=0.07)
    elif args.loss_type == 'clip':
        assert args.model_type == 'DualEncoder'
        return ContrastiveLoss(initial_temperature=0.07, learnable_temperature=False)
    elif args.loss_type == 'clip_extra':
        assert args.model_type == 'DualEncoder'
        return ContrastiveLoss(initial_temperature=0.07, learnable_temperature=False)
    elif args.loss_type == 'ntxent':
        raise NotImplementedError
        return NTXentLoss(reducer=DoNothingReducer())
    else:
        raise ValueError(f"Illegal loss type: {args.loss_type}")


def run_discover_matches(args, char, exp_name, model):
    result_dir = Path("output")
    assert len(char) == 1
    # char_args = model_args.get_char_settings(args.char[0])
    # mean = char_args.NormalizationArgs.mean__aligned_full.value
    # std = char_args.NormalizationArgs.std__aligned_full.value
    # import ipdb; ipdb.set_trace()
    # TODO: make sure transforms are correct if we're dealing with raw color images
    # transform_list = [transforms.ToTensor(),]
    # transform = transforms.Compose(transform_list)
    transform = get_transform()

    query_list_path, candidate_list_path = args.discover_matches[:2]
    print(f"Discovering matches for interesting file list:\n{query_list_path}\nusing candidate images from: {candidate_list_path}")

    # forty_sermons_paths = sorted(Path('data/forty_sermons').glob('*.tif'))
    # spinoza_theological_paths = sorted(Path('data/spinoza_theological').glob('*.tif'))
    # locke_two_treatises_paths = sorted(Path('data/locke_two_treatises').glob('*.tif'))
    # everingham_english_paths = sorted(Path('data/everingham_english').glob('*.tif'))
    # locke_letter_paths = sorted(Path('data/locke_letter').glob('*.tif'))
    # plutarch_paths = sorted(Path('data/plutarch').glob('*.tif'))
    # king_james_paths = sorted(Path('data/king_james').glob('*.tif'))
    # plot_revived_paths = sorted(Path('data/plot_revived').glob('*.tif'))
    # critical_enquiries_paths = sorted(Path('data/critical_enquiry').glob('*.tif'))

    def filter_char_filelists(char, filelist_path, parts):
        book2part2ranges = {
            'fortysermons1685': {
                'A': [(21, 395), (397, 424), (429, 450)],
                'B/D': [(395, 397), (424, 429), (450, 457), (625, 657)],
                'C': [(465, 625)],
                'E': [(657, 723)],
            },
            'spinozatheologicalpolitical1689': {
                'A': [(1, 274), (386, 485)],
                'B': [(274, 384)],
                'C': [(384, 386)],
            },
            'twotreatisesofgov1690': {
                'A': [(21, 255)],
                'B': [(255, 257)],
                'C': [(257, 464)],
            }
        }
        
        book_names = set()
        lines_are_char_files = 'manual' in filelist_path
        print(f"Lines are char files: {lines_are_char_files}")
        if lines_are_char_files:
            with open(filelist_path) as f:
                char_filepaths = []
                for line in f:
                    #NOTE TODO assumes csv with damage score (of 1.0) in second column
                    fname = line.split('.jpg')[0] + '.jpg'
                    fname_char = utils.get_char(fname)
                    if fname_char == char:
                        # char_filepaths.append(
                        #     Path(f'/graft2/code/nvog/git/matching/char_images3/char_{fname_char}_uc/')/fname.strip()
                        # )
                        char_filepaths.append(
                            Path(f'{fname}')
                        )
                assert len(char_filepaths) > 0, f"No files found before filtering for {char} in {filelist_path}."
                
                # TODO: could have mixed books in the same filelist
                book_name = utils.get_book_name(char_filepaths[0]).replace('REDO', '')
                print(f"Book name: {book_name}")

                # check if the filepath line is valid by book name and part
                filtered_char_list = []
                if parts and book_name in book2part2ranges:
                    for char_fp in char_filepaths:
                        for part in parts:
                            try:
                                page_ranges = book2part2ranges[book_name][part]
                            except KeyError as e:
                                print(f"Error: Part {part} not found for book {book_name}.")
                                exit(1)
                            # TODO: note exclusive upper bound
                            for page_range in page_ranges:
                                if page_range[0] <= utils.get_page_num(char_fp) < page_range[1]:
                                    filtered_char_list.append(char_fp)
                                    book_names.add(book_name)
                                    break

                elif not parts:
                    filtered_char_list.extend(char_filepaths)
                    book_names.add(book_name)
                
                assert len(filtered_char_list) > 0, f"No files found after filtering for {char} in {filelist_path}."
                print(f"{book_name} filtered list has {len(filtered_char_list)} chars (out of {len(char_filepaths)} total).")
        else:
            with open(filelist_path) as f:
                filtered_char_list = []
                for line in f:
                    book_name = line.strip().split('_')[-1]
                    # valid_query_book_name = not args.restrict_query_to or args.restrict_query_to in book_name
                    # valid_query_page_num = not args.restrict_query_page_range or (page_range[0] <= page_num <= page_range[1])
                    # valid_shakespeare4thfolio_part = not args.query_shakespeare_part or utils.get_shakespeare4thfolio_part(line) == args.query_shakespeare_part
                    # if not valid_query_book_name:
                    #     continue

                    # glob for the files with book name at path /graft2/code/nvog/git/matching/char_images3
                    char_filepaths = list(glob(
                        # f'/graft2/code/nvog/git/matching/char_images3/**/*{book_name}*.tif', 
                        f'/graft2/code/nvog/git/matching/char_images3/char_{char}_uc/*{book_name}*.tif', 
                        recursive=True
                    ))

                    # check if the filepath line is valid by book name and part
                    if parts and book_name in book2part2ranges:
                        for char_fp in char_filepaths:
                            for part in parts:
                                try:
                                    page_ranges = book2part2ranges[book_name.replace('REDO', '')][part]
                                except KeyError as e:
                                    print(f"Error: Part {part} not found for book {book_name}.")
                                    exit(1)
                                # TODO: note exclusive upper bound
                                for page_range in page_ranges:
                                    if page_range[0] <= utils.get_page_num(char_fp) < page_range[1]:
                                        filtered_char_list.append(char_fp)
                                        book_names.add(book_name.replace('REDO', ''))
                                        break

                    elif not parts:
                        filtered_char_list.extend(char_filepaths)
                        book_names.add(book_name)
                    
                    print(f"{book_name} filtered list has {len(filtered_char_list)} chars (out of {len(char_filepaths)} total).")

        assert len(filtered_char_list) > 0, f"No files found after filtering for {char} in {filelist_path}."

        return filtered_char_list, book_names


    #
    # Now filter the interesting/candidate lists by book name and part
    print('Filtering query set...')
    query_list, query_book_names = filter_char_filelists(char, query_list_path, args.query_parts)
    print('Filtering candidate set...')
    candidate_list, candidate_book_names = filter_char_filelists(char, candidate_list_path, args.candidate_parts)

    model.eval()    
    k = args.k
    scores = None
    # NOTE: all new models use similarities, old use distances
    scores_are_distances = False

    def load_chars_into_tensor(char_filepaths):
        transformed_imgs = []
        valid_paths = []
        # image_path_dataset = ImagePathDataset(transform, image_paths=[
        #     str(p) for p in char_filepaths
        # ])
        # dataloader = torch.utils.data.DataLoader(
        #     image_path_dataset, batch_size=1, shuffle=False, num_workers=args.eval_num_workers
        # )
        for p in tqdm(char_filepaths, desc='Preprocessing images'):
        # for img, path in tqdm(dataloader, desc='Preprocessing images'):
            try:
                img = Image.open(p)
                transformed_imgs.append(
                    transform(img).unsqueeze(0)
                )
                valid_paths.append(p)
                # transformed_imgs.append(img)
                # valid_paths.append(path)
            except (PIL.UnidentifiedImageError, FileNotFoundError) as e:
                print(f"Error: {e}. Skipping {p}.")
                continue
        return torch.cat(transformed_imgs, dim=0).to(device), valid_paths
    
    # NOTE: we update paths to only include valid paths here too 
    # for query_list/candidate_list
    print('Loading queries...')
    query_tensor, query_list = load_chars_into_tensor(query_list)
    print('Loading candidates...')
    candidate_tensor, candidate_list = load_chars_into_tensor(candidate_list)

    float_dtype = torch.float32
    scores = torch.ones(
        (len(query_tensor), len(candidate_tensor)), 
    dtype=float_dtype) * (
        torch.finfo(float_dtype).max 
        if scores_are_distances 
        else torch.finfo(float_dtype).min
    )
    # split_size = candidate_tensor.shape[0]  # split_size is the number of candidates to compare to each query
    split_size = 1000  # split_size is the number of candidates to compare to each query

    q = 0
    # TODO: loop through dataloader instead
    for query in tqdm(query_tensor, total=len(query_tensor), desc=f"Matching (split_size=|C|={split_size})"):
        img2s_split = torch.split(candidate_tensor, split_size)
        # import ipdb; ipdb.set_trace()
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
    
    # set gt_matches_indices to -1000 so they are not considered in the topk
    # self_score = torch.finfo(float_dtype).max if scores_are_distances else torch.finfo(float_dtype).min
    # for i in range(scores.shape[0]):
    #     scores[i, gt_matches_indices[i]] = self_score

    topk_values, topk_indices = torch.topk(scores, k, largest=not scores_are_distances)
    topk_values = topk_values.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()
    ##########

    if args.exclude_same_book_matches:
        print("=> Excluding same book matches from top-k results.")
        # set columns that have same book name as the row to max_val
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                if utils.get_book_name(str(query_list[i])) == utils.get_book_name(str(candidate_list[j])):
                    scores[i, j] = torch.finfo(float_dtype).max if scores_are_distances else torch.finfo(float_dtype).min

    queries_desc = f"{Path(query_list_path).with_suffix('').name.replace('filelist_', 'scores_')}-{''.join(args.query_parts) + '-' if args.query_parts else ''}"
    candidates_desc = f"{Path(candidate_list_path).with_suffix('').name}{'-' + ''.join(args.candidate_parts) if args.candidate_parts else ''}"
    output_base = Path(args.evaluate_ckpt_dest)/(queries_desc + candidates_desc + f'-topk_{char}')
    output_topk_csv_path = str(output_base) + '.csv'
    # output_topk_pkl_path = str(output_base) + '.pkl'
    Path(args.evaluate_ckpt_dest).mkdir(parents=True, exist_ok=True)
    print(f'Saving top-{k} csv results to', output_topk_csv_path)
    # print(f'Saving top-{k} pkl results to', output_topk_pkl_path)
    knn_dict = {'results': []}
    # with open(output_topk_csv_path, 'w') as f, open(output_topk_pkl_path, 'wb') as p:
    with open(output_topk_csv_path, 'w') as f:
        for i in range(scores.shape[0]):
            vals_i = topk_values[i]
            inds_i = topk_indices[i]
            row_str = str(query_list[i])
            row_str += '|' + '|'.join(str(candidate_list[j.item()]) for j in topk_indices[i])
            row_str += '|' + '|'.join([f'{d:.02f}' for d in topk_values[i]])
            # import ipdb; ipdb.set_trace()
            knn_dict['results'].append({
                "distances": vals_i if scores_are_distances else -vals_i,
                "similarities": vals_i if not scores_are_distances else -vals_i,
                "topk_indices": inds_i
            })
            print(row_str, file=f)
        knn_dict['candidate_paths'] = candidate_list
        knn_dict['query_paths'] = query_list
        # pickle.dump(knn_dict, p)
    
    import networkx as nx
    import plotly.graph_objects as go
    import plotly.io as pio

    # TODO:
    # - do a clustering of the similarity matrix

    # def label_graph(sim_mat):
    #     from sklearn.cluster import SpectralClustering
    #     # Perform spectral clustering
    #     n_clusters = 3  # Specify the number of clusters
    #     clustering = SpectralClustering(
    #         n_clusters=n_clusters, affinity='precomputed', random_state=42
    #     )
    #     labels = clustering.fit_predict(sim_mat)
    #     return labels
    
    def build_graph(sim_mat, labels=None, sim_thresh=0.5):
        G = nx.Graph()
        # Add nodes
        for i in range(sim_mat.shape[0] + sim_mat.shape[1]):
            G.add_node(i, label=labels[i] if labels is not None else None)
        # Add edges based on similarity (above threshold)
        for i in range(sim_mat.shape[0]):
            for j in range(sim_mat.shape[1]):
                if sim_mat[i, j] > sim_thresh:
                    G.add_edge(i, j, weight=sim_mat[i, j])
        return G

    def save_graph_html(G, char_paths, save_path):
        # Example: Positions for nodes
        # G = nx.Graph()
        # for i in range(len(char_paths)):
        #     G.add_node(i)

        # Generate positions for nodes
        pos = nx.spring_layout(G)

        # Create edge traces (same as before)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create node images
        node_size = 0.1  # Scale for image size
        layout_images = []
        for node, (x, y) in pos.items():
            layout_images.append(
                dict(
                    source=Image.open(char_paths[node]),  # Image for the node
                    xref="x",
                    yref="y",
                    x=x - node_size / 2,
                    y=y + node_size / 2,
                    sizex=node_size,
                    sizey=node_size,
                    xanchor="center",
                    yanchor="middle",
                    layer="above"
                )
            )

        # Create the figure
        layout = go.Layout(
            images=layout_images,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )

        fig = go.Figure(data=[edge_trace], layout=layout)
        pio.write_html(fig, file=save_path, full_html=True)
        print(f"Saved graph to {save_path}")

    # labels = label_graph(scores)
    # G = build_graph(scores, labels=None, sim_thresh=0.5)
    # save_graph_html(
    #     G, 
    #     query_list + candidate_list, 
    #     str(output_base) + '.html'
    # )

def init_model(args):
    loss_fn = get_loss_fn(args)
    if args.model_type == 'DualEncoder':
        if args.encoder_type == 'cnn':
            model = TripletDualEncoderNet(args, num_channels=1)
        else:
            model = DualEncoderNet(args)
    elif args.model_type == 'CrossEncoder':
        model = CrossEncoderNet(args)
    elif args.model_type == 'Attention':
        model = AttentionNet(args, num_channels=1)
        loss_fn = TripletDualEncoderLoss(args.margin, reduction="none", squared=True)
    elif args.model_type == 'StackedBCE':
        model = StackedNet(args)
        loss_fn = nn.BCELoss(reduction='none')
    else:
        raise ValueError("Illegal model_type.")
    return model, loss_fn


def main(args, exp_name):
    log_path = Path(args.log_dir)/exp_name
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    log_file = log_path/"log.log"
    
    # TODO
    with open(log_file, 'w') as log, redirect_stdout(utils.TeeFile(log, sys.stdout)):
        root_dir = Path(args.synthetic_data_dir)
        train_ds = MultiCharTwinDataset(args, 'train')
        valid_ds = MultiCharTwinDataset(args, 'valid')
        test_ds = MultiCharTwinDataset(args, 'test')
        matches_dict_by_char = defaultdict(dict)
        matches_ds_by_char = defaultdict(dict)
        # load char eval match data for all characters in the union of char and eval_char
        for char in sorted(set(args.char).union(set(args.eval_char))):
            splits_dir = Path(root_dir).glob(f'{char}*')
            try:
                splits_dir = list(splits_dir)[0]
            except IndexError as e:
                print(f'No synthetic data directory for character {char} in {args.synthetic_data_dir}', file=sys.stderr)
                exit(1)
            recall_eval_csvs_dir = splits_dir/'fake_match_eval_sets'
            # anomaly detection normals
            num_samples = None
            utils.set_random_seed(args.random_seed)
            char_args = model_args.get_char_settings(char)
            mean = char_args.NormalizationArgs.mean__aligned_full.value
            std = char_args.NormalizationArgs.std__aligned_full.value
            # mean = char_args.NormalizationArgs.mean__aligned_residual_global_average.value if args.residual else char_args.NormalizationArgs.mean__aligned_full.value
            # std = char_args.NormalizationArgs.std__aligned_residual_global_average.value if args.residual else char_args.NormalizationArgs.std__aligned_full.value
            if args.augment:
                print("--augment: using utils.make_jitter_transform()")
                train_transform = transforms.Lambda(lambda x: utils.make_jitter_transform()(x))
            else:
                train_transform = transforms.ToTensor()
            eval_transform = transforms.ToTensor()
            # if not args.input_residual:
            #     transform_list.append(transforms.Normalize(mean=[mean], std=[std]))
            # transform = transforms.Compose(transform_list)
            train_dataset_args = {"data_path": None, "transform": train_transform, "num_samples": num_samples, "random_seed": args.random_seed,}
            eval_dataset_args = {"data_path": None, "transform": eval_transform, "num_samples": num_samples, "random_seed": args.random_seed,}
            # create train, valid, test TwinDatasets
            # load the syn matches from the matches.csv for train/valid/test
            matches_dict_by_char[char]["syn_train_img_paths"], matches_dict_by_char[char]["syn_train_gt_pairs"] = evaluation.load_recall_match_eval(recall_eval_csvs_dir/"train", limit_matches_amount=args.limit_synthetic_eval_set_size)
            matches_dict_by_char[char]["syn_valid_img_paths"], matches_dict_by_char[char]["syn_valid_gt_pairs"] = evaluation.load_recall_match_eval(recall_eval_csvs_dir/"valid", limit_matches_amount=args.limit_synthetic_eval_set_size)
            matches_dict_by_char[char]["syn_test_img_paths"], matches_dict_by_char[char]["syn_test_gt_pairs"] = evaluation.load_recall_match_eval(recall_eval_csvs_dir/"test", limit_matches_amount=args.limit_synthetic_eval_set_size)
            matches_ds_by_char[char]['syn'] = create_datasets(args, recall_eval_csvs_dir, train_dataset_args, ds_class=GroundTruthMatchesDataset, splits=('train',))
            matches_ds_by_char[char]['syn'].update(create_datasets(args, recall_eval_csvs_dir, eval_dataset_args, ds_class=GroundTruthMatchesDataset, splits=('valid', 'test')))
            # this creates individual char TwinDataset which we then add to the MultiCharTwinDataset
            datasets = create_datasets(args, splits_dir, train_dataset_args, ds_class=TwinDataset, splits=('train',))
            datasets.update(create_datasets(args, splits_dir, eval_dataset_args, ds_class=TwinDataset, splits=('valid', 'test')))
            # create actual training data (EXCLUDING args.eval_char)
            if char in args.char:
                train_ds.add_char_dataset_pair(char, datasets['train'], limit_dataset_size=args.limit_dataset_size)
            # add valid/test set for char (INCLUDING args.eval_char)
            valid_ds.add_char_dataset_pair(char, datasets['valid'], limit_dataset_size=args.limit_eval_dataset_size)
            test_ds.add_char_dataset_pair(char, datasets['test'], limit_dataset_size=args.limit_eval_dataset_size)
            # load the gt matches from the matches.csv
            # and also throw in background sets for gt leviathan:
            # 1. strong_pos: current leviathan ground truth used for attribution paper
            # X. weak_pos: XXXX
            # 3. strong_neg: Darby books that tricked Noel Malcolm, John Redmayne books
            # 4. weak_neg: books with different fonts like Everingham books
            gt_data_dir = Path(args.gt_data_dir)/char
            if gt_data_dir.exists():
                # import ipdb; ipdb.set_trace()
                matches_dict_by_char[char]["gt_test_pos_img_paths"], matches_dict_by_char[char]["gt_test_pos_gt_pairs"] = evaluation.load_recall_match_eval(gt_data_dir/"test")
                # import ipdb; ipdb.set_trace()
                matches_dict_by_char[char]["gt_test_strong_neg_img_paths"], _ = evaluation.load_recall_match_eval(gt_data_dir/"test", background_set_filelist=gt_data_dir/"test"/"strong_negative_background_set.txt", limit_background_amount=args.limit_gt_eval_background_set_size)
                matches_dict_by_char[char]["gt_test_weak_neg_img_paths"], _ = evaluation.load_recall_match_eval(gt_data_dir/"test", background_set_filelist=gt_data_dir/"test"/"weak_negative_background_set.txt", limit_background_amount=args.limit_gt_eval_background_set_size)
                matches_dict_by_char[char]["gt_test_mix_neg_img_paths"], _ = evaluation.load_recall_match_eval(gt_data_dir/"test", background_set_filelist=gt_data_dir/"test"/"mix_negative_background_set.txt", limit_background_amount=args.limit_gt_eval_background_set_size)
                matches_ds_by_char[char]['gt_test_pos'] = create_datasets(args, gt_data_dir, eval_dataset_args, ds_class=GroundTruthMatchesDataset, splits=('test',))
                assert len(matches_ds_by_char[char]['gt_test_pos']) > 0, f"Dataset empty for {char} in {gt_data_dir}"
                assert len(matches_dict_by_char[char]["gt_test_pos_img_paths"]) > 0, f"img_paths empty for {char} in {gt_data_dir}"
                assert len(matches_dict_by_char[char]["gt_test_pos_gt_pairs"]) > 0, f"gt_pairs empty for {char} in {gt_data_dir}"
            if args.use_areo:
                areo_data_dir = Path(args.areo_data_dir)/char
                if areo_data_dir.exists():
                    matches_ds_by_char[char]['gt_test_areo'] = create_datasets(args, areo_data_dir, eval_dataset_args, ds_class=GroundTruthMatchesDataset, splits=('test',))
                    matches_dict_by_char[char]["gt_test_areo_img_paths"], matches_dict_by_char[char]["gt_test_areo_gt_pairs"] = evaluation.load_recall_match_eval(areo_data_dir/"test")
            # locke spinoza manual matches we currently have
            lockespinoza_data_dir = Path(args.lockespinoza_data_dir)/char
            if lockespinoza_data_dir.exists():
                matches_ds_by_char[char]['gt_test_lockespinoza'] = create_datasets(args, lockespinoza_data_dir, eval_dataset_args, ds_class=GroundTruthMatchesDataset, splits=('test',))
                matches_dict_by_char[char]["gt_test_lockespinoza_img_paths"], matches_dict_by_char[char]["gt_test_lockespinoza_gt_pairs"] = evaluation.load_recall_match_eval(lockespinoza_data_dir/"test")
                matches_dict_by_char[char]["gt_test_lockespinoza_mix_neg_img_paths"], _ = evaluation.load_recall_match_eval(lockespinoza_data_dir/"test", background_set_filelist=lockespinoza_data_dir/"test"/"mix_negative_background_set.txt", limit_background_amount=args.limit_gt_eval_background_set_size)

        # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        # valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        # test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        train_dl = torch.utils.data.DataLoader(train_ds, num_workers=args.train_num_workers, pin_memory=True, batch_sampler=utils.same_char_class_batch_sampler(train_ds.get_char_classes_labels(), args.train_batch_size, shuffle=True, drop_last=True))
        # valid_dl = torch.utils.data.DataLoader(valid_ds, num_workers=args.eval_num_workers, pin_memory=True, batch_sampler=utils.same_char_class_batch_sampler(valid_ds.get_char_classes_labels(), args.eval_batch_size, shuffle=False, drop_last=False))
        # test_dl = torch.utils.data.DataLoader(test_ds, num_workers=args.eval_num_workers, pin_memory=True, batch_sampler=utils.same_char_class_batch_sampler(test_ds.get_char_classes_labels(), args.eval_batch_size, shuffle=False, drop_last=False))
        # initialize model/loss
        model, loss_fn = init_model(args)
        if args.load_model:
            model_path = Path(args.output_dir)/args.load_model
            print("Loading model from", model_path)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)['model_state'])
        print(model)
        utils.get_model_params_and_size(model)
        model = model.to(device)
        if args.optimizer == 'Adam':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        if args.load_optimizer:
            print("Loading optimizer from", args.load_optimizer)
            optimizer.load_state_dict(torch.load(args.load_optimizer)['optimizer_state'])
        if args.scheduler == 'linear_warmup_cosine_decay':
            # Scheduler parameters
            warmup_steps = args.scheduler_warmup_steps
            total_steps = len(train_dl) * args.n_epochs
            scheduler_obj = utils.LinearWarmupCosineDecayScheduler(optimizer, warmup_steps, total_steps, min_lr=0.0)
            scheduler = scheduler_obj.get_scheduler()
            print(f"\nUsing LinearWarmupCosineDecayScheduler with warmup_steps={warmup_steps}, total_steps={total_steps}\n")
        else:
            scheduler = None
            print("No scheduler used.")
        metrics = []  
        if args.loss_type == 'triplet':
            print(f"\n\tMargin={args.margin}\n")
        if args.wandb:
            wandb.watch(model)
        train_model(args, train_dl, model, loss_fn, optimizer, scheduler,
            metrics, Path(args.output_dir), exp_name, matches_dict_by_char, matches_ds_by_char)


def get_transform(unpadded_image_size = 164, padded_image_size = 224):
    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((unpadded_image_size, unpadded_image_size)),
        transforms.Grayscale(num_output_channels=1),
        Binarizer(),
        transforms.Pad((padded_image_size - unpadded_image_size) // 2, fill=255, padding_mode='constant'),
        transforms.ToTensor()
    ])
    return transform


# Function to preprocess images and extract embeddings
def extract_embeddings(image_paths_file, model, transform):
    model.eval()
    image_path_dataset = ImagePathDataset(transform, image_paths_file=image_paths_file)
    dataloader = torch.utils.data.DataLoader(
        image_path_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_num_workers
    )
    embeddings = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Embedding {image_paths_file}")
    for i, (images, image_paths) in pbar:
        images = images.to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=args.amp):
                embeds = model(images).squeeze().cpu().numpy()
        # add batch of embeds to embeddings
        embeddings.extend(embeds)
    return np.array(embeddings)


def build_faiss_index(args, model):
    assert args.model_type == 'DualEncoder', "Only DualEncoder model type is supported for FAISS indexing."
    model = model.to(device).eval()
    transform = get_transform()

    embeddings = extract_embeddings(args.faiss_image_paths_file, model, transform)
    # Normalize embeddings for dot product similarity
    faiss.normalize_L2(embeddings)

    index_dim = embeddings.shape[1]
    if args.faiss_use_approximate:
        # Create an approximate FAISS index using IVF (Inverted File Index)
        quantizer = faiss.IndexFlatIP(index_dim)  # Quantizer for coarse centroids
        index = faiss.IndexIVFFlat(quantizer, index_dim, args.faiss_ivf_nclusters, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)  # Train the index
        index.add(embeddings)
    else:
        # Create an exact FAISS index
        index = faiss.IndexFlatIP(index_dim)  # IP = Inner Product
        index.add(embeddings)

    # Save the index to disk
    faiss.write_index(index, args.faiss_index_path)
    print(f"Saved FAISS index to {args.faiss_index_path}")

    # print(f"Computing similarity matrix (embeddings.shape {embeddings.shape})...")
    # sim_mat = faiss_similarity_matrix(index, embeddings)
    # np.save(args.faiss_sim_mat_path, sim_mat.astype(np.float16))
    # print(f"Saved faiss similarity matrix (shape: {sim_mat.shape}) to {args.faiss_sim_mat_path} as fp16")


def faiss_similarity_matrix(index, embeddings):
    similarity_matrix = index.search(embeddings, embeddings.shape[0])[0]
    return similarity_matrix


def query_faiss_index(args, model):
    assert args.model_type == 'DualEncoder', "Only DualEncoder model type is supported for FAISS indexing."
    model = model.to(device).eval()
    transform = get_transform()

    # load index and candidate paths (image paths used to compute index)
    index = faiss.read_index(args.faiss_index_path)
    candidate_paths = [line.strip() for line in open(args.faiss_image_paths_file)]
    # compute query embeddings
    query_paths = [line.strip() for line in open(args.faiss_query_set_file)]
    query_embedding = extract_embeddings(args.faiss_query_set_file, model, transform)
    # faiss.normalize_L2(query_embedding.reshape(1, -1))
    faiss.normalize_L2(query_embedding)
    # TODO: over estimate k here because it does not filter by character class
    # afterwards we will filter by character class down to the top args.k
    scores, indices = index.search(query_embedding, args.k)
    # Display results
    # print("Top results:")
    rows = []
    for i in range(query_embedding.shape[0]):
        # print(f"Query: {query_paths[i]}")
        row = '|'.join([
                query_paths[i]
            ] + [
                candidate_paths[idx] for idx in indices[i]
            ] + [
                f"{score:.4f}" for score in scores[i]
            ])
        # print(f"Top {args.k} candidates:")
        # for idx, score in zip(indices[i], scores[i]):
        #     print(f"Image: {candidate_paths[idx]}, Score: {score}")
        rows.append(row)
    
    # output_file = Path(args.output_dir)/f"faiss_top_{args.k}_results.csv"
    output_file = Path(args.output_dir)/f"query-{Path(args.faiss_query_set_file).with_suffix('').name}_candidates-{Path(args.faiss_index_path).with_suffix('').name}_top_{args.k}_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # print(rows)
    with open(output_file, 'w') as f:
        for row in rows:
            print(row, file=f)
    print(f"Saved top-{args.k} results to {output_file}")
    return indices, scores



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--stopping_metric", choices=['syn_valid', 'gt_test_pos', 'gt_test_strong_neg', 'gt_test_weak_neg', 'gt_test_mix_neg'], default='syn_valid', help='how to choose best ckpt')
    parser.add_argument("--synthetic_data_dir", type=str, default="data/synthetic_data")  # "/projects/nvog/synthetic/data/synthetic/shakespeare_2021-09-28"
    parser.add_argument("--gt_data_dir", type=str, default="data/leviathan_matching_test_set_preprocessed")  # /home/nvog/projects/git/anomaly-detection/matching/leviathan_matching_test_set
    # NOTE: not using areo data for new preprocessed unaligned data experiments now
    parser.add_argument("--use_areo", action="store_true", help="Use areopagitica matching test set.")
    parser.add_argument("--areo_data_dir", type=str, default="data/areopagitica_matching_test_set")
    parser.add_argument("--lockespinoza_data_dir", type=str, default="data/lockespinoza_matching_test_set_preprocessed")
    parser.add_argument("--char", required=False, type=str, nargs='+')
    parser.add_argument("--eval_char", type=str, nargs='+', default=[])
    parser.add_argument("--output_dir", default="output/matching_results_nov24")  # /trunk/nvog/matching_results_aaai2022_SunAug14
    parser.add_argument("--log_dir", default="output/matching_results_nov24/logs")  # /trunk/nvog/matching_results_aaai2022_SunAug14/logs
    # dataset/dataloader
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--train_num_workers", type=int, default=8)
    parser.add_argument("--eval_num_workers", type=int, default=2)
    # triplet or npairs loss for metric learning
    parser.add_argument("--loss_type", choices=['triplet', 'triplet_bce', 'npairs', 'clip', 'clip_extra'], default='clip_extra', help='triplet/npairs/clip losses')
    # hyperparam for triplet loss
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--negative_mining", choices=['l2', 'random', 'random_half_normal'], default='random_half_normal')
    # projection net options
    # parser.add_argument('--projection_net_in_dim', type=int, default=128)
    parser.add_argument('--projection_net_out_dim', type=int, default=256)
    parser.add_argument('--projection_net_hidden_dim', type=int, default=256)
    parser.add_argument('--projection_net_type', type=str, choices=['mlp', 'linear'], default='linear')
    # optimizer hyperparams
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--scheduler", type=str, choices=['linear_warmup_cosine_decay', 'none'], default='none')
    parser.add_argument("--scheduler_warmup_steps", type=int, default=500)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--train_on_syn_matches", action="store_true", help="for sanity check with loss/recall")
    # model can be a dual encoder or attention cross encoder model
    parser.add_argument("--model_type", choices=['DualEncoder', 'CrossEncoder', 'Attention'], default='Attention')
    parser.add_argument("--encoder_type", choices=[
        'cnn', 
        'vit_b_16', 
        'hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k', 
        'hf_hub:timm/vit_base_patch8_224.dino', 
        'hf_hub:timm/vit_base_patch14_dinov2.lvd142m',
        'hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k', 
        'hf_hub:timm/convnext_base.fb_in22k_ft_in1k'
    ], default='cnn')
    # add similiarity metric options (either dot or cosine)
    parser.add_argument("--similarity_metric", choices=['dot', 'cosine', 'mlp'], default='cosine')
    # cnn options
    parser.add_argument("--channel_input_list", type=str, nargs='+', default=[], choices=['img1', 'img2', 'img1-img2', 'img2-img1', 'img1-avg', 'img2-avg', 'avg'])
    parser.add_argument('--cnn_dim', default=128, type=int)
    parser.add_argument('--num_cnn_blocks', default=2, type=int, help='number of cnn/maxpool blocks')
    parser.add_argument('--num_conv_per_block', default=3, type=int, help='number of conv layers per block')
    # vit options
    parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained model for ViT")
    # other model options
    parser.add_argument("--softmax_temperature", default=1.0, type=float, help="amount to divide logits by prior to softmax")
    parser.add_argument("--mlp_dropout", default=0.0, type=float, help="Dropout for mlp. TODO")
    # dataset options
    parser.add_argument("--limit_dataset_size", type=int, default=sys.maxsize, help='Limit each individual char dataset(s) size to a random subset of paired examples.')
    parser.add_argument("--limit_eval_dataset_size", type=int, default=sys.maxsize, help='Limit each eval (valid/test) individual char dataset(s) size to a random subset of paired examples.')
    parser.add_argument("--limit_synthetic_eval_set_size", type=int, default=100, help='Limit the number of examples for synthetic data evaluation.')
    parser.add_argument("--limit_gt_eval_background_set_size", type=int, default=100, help='Limit the number of background examples for gt data evaluation.')
    # data augmentation options
    parser.add_argument("--augment", action='store_true', help='Performs different jittering/inking variations on the pair images.')
    parser.add_argument("--add_global_inking", action='store_true', help='Apply random thinning/thickening to character pairs during training.')
    parser.add_argument("--jitter_triplet", action='store_true', help='Translates/jitters the triplet the same amount/direction when training to learn invariance to global position changes.')
    # load model/optimizer
    parser.add_argument("--load_optimizer", type=str, default=None, help="Load optimizer weights from file.")
    parser.add_argument("--load_model", type=str, default=None, help="Load trained model weights from file.")
    parser.add_argument("--collapse_attn_operation", default='filter_sum', choices=['filter_sum', 'filter_max', 'hw_sum', 'hw_max'], help='How to reduce the filter or height/width dim in attention')
    parser.add_argument("--use_output_embedding_layer", action='store_true', help='Apply flattening and extra linear layer to output of attention cross-multiplied with conv feature maps to get embedding (similar to original non-attention DualEncoder model).')
    # to run baseline only:
    parser.add_argument("--baseline_type", type=str, default=None, choices=['L2', 'random'], help='Just do L2 or random top-k baseline and report results.')
    parser.add_argument("--random_baseline", action='store_true', help='Just do random top-k baseline')
    parser.add_argument("--input_residual", action='store_true', help="Input L2 residual into DualEncoder or Attention network variant.")
    parser.add_argument("--conv_template_residual", action='store_true', help="Compute L2 conv residual with template before computing distance/computing attention: att( (cnn(img1) - cnn(avg))^2, (cnn(img2) - cnn(avg))^2 ).")
    parser.add_argument("--per_channel_attention", action='store_true')
    # eval args:
    parser.add_argument("--evaluate_ckpt", action='store_true', help='Whether or not to just evaluate a --load_model checkpoint.')
    parser.add_argument("--evaluate_ckpt_dest", type=str, help='Where to save the evaluation knn dicts.')
    parser.add_argument("--discover_matches_split_amount", default=7200, type=int, help='Break up eval candidate matching into splits of this size.')
    parser.add_argument("--k", type=int, default=10, help='number of neighbors to retrieve and save for knn')
    parser.add_argument("--restrict_query_to", type=str, default=None, help='restrict query (rows) to a book name during discovery')
    parser.add_argument("--query_shakespeare_part", type=int, default=0, choices=[0, 1, 2, 3], help='Restrict query to a particular part of shakespeare, and filter out candidates from same set. 0 option corresponds to not doing this.')
    parser.add_argument("--rerank_with_l2", action='store_true', help="Rerank a model result during eval by interpolating with L2 distances.")
    parser.add_argument("--notes", type=str, default='', help='Notes on run to add to experiment name for extra checkpoint identification')
    parser.add_argument("--debug", action='store_true', help='Debug mode')
    parser.add_argument("--discover_matches", type=str, nargs='+', default=None, help="interesting_file_name_list.txt background_file_name_list. (e.g., /home/nvog/output/interesting_Gs.txt /home/nvog/output/G_thresholded_anomaly_without_italics_files.txt) (e.g., /home/nvog/output/interesting_Ms.txt /home/nvog/output/M_thresholded_anomaly_files.txt)")
    parser.add_argument("--exclude_same_book_matches", action='store_true', help="Exclude potential top-k matches from the same book by masking them out during discovery.")
    parser.add_argument("--exclude_book_name_from_candidates", type=str, default=None, help='Exclude a book name from the candidate set to match against')
    parser.add_argument('--query_parts', type=str, nargs='+', default=None, help='Restrict query to a particular part of a book.')
    parser.add_argument('--candidate_parts', type=str, nargs='+', default=None, help='Restrict candidate to a particular part of a book.')
    parser.add_argument("--limit_eval_keys", type=str, nargs='+', default=[], help='Limit the eval keys to a subset of the keys in the matches_dict.')
    # add apply_damage_score_filter which takes two arguments, the first is a string and the second is a float
    parser.add_argument("--apply_damage_score_filter", type=str, nargs='+', default=[], help='Apply a damage score filter to the matches_dict. The first argument is damage predictions csv and the second argument is the threshold.')
    # FAISS Index Creation and Querying
    parser.add_argument("--faiss_build_index", action='store_true', help='Build a FAISS index from a set of image paths.')
    parser.add_argument("--faiss_image_paths_file", help="Path to a file containing image paths, one per line")
    parser.add_argument("--faiss_query_set_file", help="Path to file list of query images for FAISS indexing. Meant to be passed togeter with an index and its file list.")
    parser.add_argument("--faiss_query_image_list", help="Path to the query image list")
    parser.add_argument("--faiss_index_path", help="Path to save the FAISS index")
    parser.add_argument("--faiss_sim_mat_path", help="Path to save the FAISS sim mat")
    parser.add_argument("--faiss_use_approximate", action="store_true", help="Use approximate indexing with IVF")
    parser.add_argument("--faiss_ivf_nclusters", type=int, default=100, help="Number of clusters for IVF indexing (if approximate is used)")
    args = parser.parse_args()

    print(args)

    if args.debug and (args.train_num_workers != 0 or args.eval_num_workers != 0):
        print("Cannot use --debug mode with more than 1 train/eval dataset worker. Set --train_num_workers 0 --eval_num_workers 0")
        exit(1)

    now = datetime.now()
    if args.faiss_build_index or args.faiss_query_set_file or args.faiss_query_image_list:
        pass
    elif args.baseline_type:
        exp_name = f"Char{''.join(sorted(args.char))}_{args.baseline_type}Baseline{'-' + args.notes if args.notes else ''}-{now.strftime('%Y-%m-%d_%H:%M:%S')}"
    else:
        if args.evaluate_ckpt:
            exp_name = str(Path(args.load_model).parent) + f"_eval{'_rerank' if args.rerank_with_l2 else ''}-{now.strftime('%Y-%m-%d_%H:%M:%S')}"
        else:
            cnn_opts = f"_{args.num_cnn_blocks}CNNBlocks_{args.num_conv_per_block}ConvPerBlock" if args.encoder_type == 'cnn' else ''
            vit_opts = f"{'_Pretrained' if args.use_pretrained else ''}" if 'vit' or 'timm' in args.encoder_type else ''
            attn_opts = f"{'_OutputEmbSize'+str(args.embedding_size) if args.model_type == 'Attention' else ''}{'_PerChannelAttn' if args.per_channel_attention else ''}{'OutputEmbeddingLayer' if args.use_output_embedding_layer else ''}{'_CollapseAttn'+str(args.collapse_attn_operation) if args.model_type == 'Attention' else ''}"
            if args.loss_type == 'triplet':
                loss_info = f"TripletMargin{args.margin}-{args.negative_mining}"
            elif args.loss_type == 'triplet_bce':
                loss_info = f"TripletBCE-{args.negative_mining}"
            elif args.loss_type == 'npairs':
                loss_info = "NPairs"
            elif args.loss_type == 'clip':
                loss_info = "CLIP"
            elif args.loss_type == 'clip_extra':
                loss_info = "CLIPExtra"
            exp_name = f"Char{''.join(sorted(args.char))}{'_TempResidual' if args.conv_template_residual else ''}{'_InputResidual' if args.input_residual else ''}_Model{args.model_type}_Encoder{args.encoder_type.lstrip('hf_hub:timm/').capitalize()}_{args.limit_dataset_size}TrainPairs_LimitEval{args.limit_eval_dataset_size}_GlobalInk{args.add_global_inking}{cnn_opts}{vit_opts}{attn_opts}{'Channels=' + ','.join(args.channel_input_list) if args.channel_input_list else ''}{'_Temp'+str(args.softmax_temperature)}{'_Jitter' if args.jitter_triplet else ''}{'_Augment' if args.augment else ''}_{loss_info}_Bsz{args.train_batch_size}_{args.optimizer}LR{args.lr}{'_SchedulerWU' + str(args.scheduler_warmup_steps) if args.scheduler else ''}{'_rerank' if args.rerank_with_l2 else ''}{'-' + args.notes if args.notes else ''}-{now.strftime('%Y-%m-%d_%H:%M:%S')}"
        print(f"\n\n\tExperiment name: {exp_name}\n\n")

    if args.wandb:
        wandb.init(project="matching2025", entity="nvog")
        wandb.run.name = exp_name
        wandb.run.save()

    if args.discover_matches:
        model, _ = init_model(args)
        if args.load_model:
            model_path = args.load_model
            print("Loading model from", model_path)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)['model_state'])
        print(model)
        utils.get_model_params_and_size(model)
        model = model.to(device)
        print('Done constructing model.') 
        for char in args.char:
            print(f'= {char} =')
            try:
                run_discover_matches(args, char, exp_name, model)
            except Exception as e:
                print(f"Error: {e}. Skipping {char}.")
                continue
    elif args.faiss_build_index:
        model, _ = init_model(args)
        if args.load_model:
            model_path = args.load_model
            print("Loading model from", model_path)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)['model_state'])
        print(model)
        utils.get_model_params_and_size(model)
        model = model.to(device).eval()
        print('Done constructing model.')
        build_faiss_index(args, model)
    elif args.faiss_query_set_file:
        model, _ = init_model(args)
        if args.load_model:
            model_path = args.load_model
            print("Loading model from", model_path)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)['model_state'])
        print(model)
        utils.get_model_params_and_size(model)
        model = model.to(device).eval()
        print('Done constructing model.')
        query_faiss_index(args, model)
    else:
        main(args, exp_name)
    print("Done!")
