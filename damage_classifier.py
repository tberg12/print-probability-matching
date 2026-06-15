import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.optim as optim
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
from collections import defaultdict, Counter
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
    average_precision_score,
    accuracy_score,
    precision_recall_curve,
    PrecisionRecallDisplay,
    DetCurveDisplay,
    det_curve,
)
import warnings
import models
from torch.utils.data.sampler import WeightedRandomSampler
from matplotlib import rcParams
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import wandb
import tarfile
from datasets import (
    DamageDataset,
    DamageDatasetFilelist,
    ImagePadder,
    FixedHeightResizeAndPad,
    make_jitter_transform,
    TypeAugmenter,
)
import matcher
import timm
import shutil
from datetime import datetime


warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BAD_EXTRACTION = 2
DAMAGE = 1
NORMAL = 0


def class_imbalance_sampler(labels):
    """
    sampler = class_imbalance_sampler(labels)
    train_loader  = DataLoader(dataset, sampler=sampler)
    """
    class_count = torch.bincount(labels.squeeze())
    class_weighting = 1.0 / class_count
    sample_weights = class_weighting[labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels))
    return sampler


def create_batch(data, bsz):
    l = len(data)
    for ndx in range(0, l, bsz):
        yield data[ndx : min(ndx + bsz, l)]


def same_char_class_batch_sampler(labels, batch_size, shuffle=True, drop_last=False):
    batches = []
    labels = np.array(labels)
    uniq_labels = np.unique(labels)
    for ul in uniq_labels:
        char_idxs = np.flatnonzero(labels == ul)
        if shuffle:
            random.shuffle(char_idxs)
        for batch in create_batch(char_idxs, batch_size):
            if len(batch) < batch_size and drop_last:
                break
            batches.append(batch)
    random.shuffle(batches)
    return np.array(batches)


def prep_datasets(args):
    # TODO: add 'resnet' in args.encoder_type padding to 224,224
    # TODO: add 'resnet' in args.encoder_type duplicate channels from 1 to 3 in case of grayscale input
    img_size = 64
    # is_one_channel = (
    #     args.is_pair_data or
    #     Image.open(pd.read_csv(args.train_csv).values.tolist()[0][0]).mode != "RGB"
    # )

    if 'resnet' in args.encoder_type or 'vit_b_16' in args.encoder_type or 'hf_hub' in args.encoder_type:
        img_size = 224
        channels = 3

    transform_list = []
    if args.size_transform == "pad":
        # transform_list.append(transforms.Lambda(lambda img: F.pad(img, [0, img_size - img.size(2), 0, img_size - img.size(1)])))
        transform_list.append(ImagePadder(img_size, img_size, 255))
        transform_list.append(transforms.ToTensor())
    elif args.size_transform == "resize":
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Resize((img_size, img_size)))
    elif args.size_transform == "fixed_height_resize_and_pad":
        transform_list.append(FixedHeightResizeAndPad(img_size, 255))
        transform_list.append(transforms.ToTensor())
    else:
        transform_list.append(transforms.ToTensor())

    # if is_one_channel:
    #     transform_list.append(transforms.Lambda(lambda x: x.max() - x))
    #     transform_list.append(transforms.Lambda(lambda x: x.repeat(channels, 1, 1)))

    if args.size_transform == "matcher":
        transform = matcher.get_transform()
    else:
        transform = transforms.Compose(transform_list)

    # aligned_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.max() - x)])
    # unaligned_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda img: F.pad(img, [0, 64 - img.size(2), 0, 64 - img.size(1)])), transforms.Lambda(lambda x: x.max() - x)])
    datasets = {
        "train": DamageDataset(
            args,
            "train",
            transform,
            args.train_csv,
            jitter=args.jitter,
            limit_train_data=args.limit_train_data,
            is_pair_data=args.is_pair_data,
        ),
        "valid": DamageDataset(
            args,
            "valid",
            transform,
            args.valid_csv,
            jitter=False,
            limit_train_data=args.limit_train_data,
            is_pair_data=args.is_pair_data,
        ),
        "valid_jitter": DamageDataset(
            args,
            "valid",
            transform,
            args.valid_csv,
            pregenerate_fixed_jitter=True,
            jitter=args.jitter,
            limit_train_data=args.limit_train_data,
            is_pair_data=args.is_pair_data,
        ),
        # NOTE: test is intended to be the real cdt test set
        "test": DamageDataset(
            args,
            "test",
            transform,
            args.test_csv,
            jitter=False,
            limit_train_data=args.limit_train_data,
            is_pair_data=False
        ),
        "test_jitter": DamageDataset(
            args,
            "test",
            transform,
            args.test_csv,
            pregenerate_fixed_jitter=True,
            jitter=args.jitter,
            limit_train_data=args.limit_train_data,
            is_pair_data=False
        ),
    }

    example_dir = Path(args.output_dir) / "examples"
    for p in example_dir.glob("*.png"):
        p.unlink()
    example_dir.mkdir(exist_ok=True, parents=True)
    # save random pre-processed examples
    for split in ["train", "valid", "test"]:
        random_indices = np.random.randint(0, len(datasets[split]), size=50)
        for i in random_indices:
            transforms.ToPILImage()(datasets[split][i]["image"]).save(
                # example_dir / f"train_{i}.png"
                example_dir / f"{split}_{i}_damage.png"
            )
            if split != 'test':
                transforms.ToPILImage()(datasets[split][i]["original_image"]).save(
                    # example_dir / f"train_{i}.png"
                    example_dir / f"{split}_{i}_original.png"
                )

    for name, addl_test_set_csv in eval(args.other_test_csvs).items():
        datasets.update(
            {
                name: DamageDataset(
                    args,
                    name,
                    transform,
                    addl_test_set_csv,
                    jitter=False,
                    limit_train_data=args.limit_train_data,
                )
            }
        )
    for split, ds in datasets.items():
        ds.save_avg_templates(args.output_dir, filename_prefix=split)

    # print dataset info
    for split, ds in datasets.items():
        # assert len(set(ds.get_labels())) == 2, (
        #     f"Expected 2 labels, but found {len(set(ds.get_labels()))} in {split} set."
        # )
        print(
            f"{split} set: {len(ds)} images, {len(set(ds.get_labels()))} unique labels, {len(set(ds.get_char_classes_labels()))} unique char classes"
        )

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            pin_memory=True,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
            batch_size=args.batch_size,
            # batch_sampler=same_char_class_batch_sampler(
            #     datasets["train"].get_char_classes_labels(),
            #     args.batch_size,
            #     shuffle=True,
            #     drop_last=True,
            # ),
        ),  # batch_size=args.batch_size, shuffle=True), #sampler=class_imbalance_sampler(torch.LongTensor(datasets['train'].get_labels()))),  # shuffle=True
        "valid": torch.utils.data.DataLoader(
            datasets["valid"],
            pin_memory=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size * 8,
            shuffle=False,
            # batch_sampler=same_char_class_batch_sampler(
            #     datasets["valid"].get_char_classes_labels(),
            #     args.batch_size,
            #     shuffle=False,
            #     drop_last=False,
            # ),
        ),  # batch_size=args.batch_size, shuffle=False),
        "valid_jitter": torch.utils.data.DataLoader(
            datasets["valid_jitter"],
            pin_memory=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            # batch_sampler=same_char_class_batch_sampler(
            #     datasets["valid_jitter"].get_char_classes_labels(),
            #     args.batch_size,
            #     shuffle=False,
            #     drop_last=False,
            # ),
        ),  # batch_size=args.batch_size, shuffle=False),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            pin_memory=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            # batch_sampler=same_char_class_batch_sampler(
            #     datasets["test"].get_char_classes_labels(),
            #     args.batch_size,
            #     shuffle=False,
            #     drop_last=False,
            # ),
        ),  # batch_size=args.batch_size, shuffle=False),,
        "test_jitter": torch.utils.data.DataLoader(
            datasets["test_jitter"],
            pin_memory=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            # batch_sampler=same_char_class_batch_sampler(
            #     datasets["test_jitter"].get_char_classes_labels(),
            #     args.batch_size,
            #     shuffle=False,
            #     drop_last=False,
            # ),
        ),  # batch_size=args.batch_size, shuffle=False),,
    }
    for name, addl_test_set_csv in eval(args.other_test_csvs).items():
        dataloaders.update(
            {
                name: torch.utils.data.DataLoader(
                    datasets[name],
                    pin_memory=True,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=False,
                    # batch_sampler=same_char_class_batch_sampler(
                    #     datasets[name].get_char_classes_labels(),
                    #     args.batch_size,
                    #     shuffle=False,
                    #     drop_last=False,
                    # ),
                )  # batch_size=args.batch_size, shuffle=False),
            }
        )
    return datasets, dataloaders


def evaluate(args, dataloaders, model, dl):
    model.eval()
    eval_log = defaultdict(list)
    for batch in dl:
        with torch.no_grad():
            image, original_image, label, damage_loc_xy, non_damage_loc_xy = (
                batch["image"],
                batch["original_image"],
                batch["label"],
                batch["damage_loc_xy"],
                batch["non_damage_loc_xy"],
            )
            if dl.dataset.is_pair_data:
                label = torch.cat([torch.ones(image.shape[0]), torch.zeros(image.shape[0])], dim=0).squeeze()
            # image, label, damage_loc_xy = batch['image'], batch['label'], batch['damage_loc_xy']
            # import ipdb; ipdb.set_trace()
            if batch["char"][0] not in dataloaders["train"].dataset.char2i:
                continue
            if 'resnet' in args.encoder_type or 'vit_b_16' in args.encoder_type or 'hf_hub' in args.encoder_type:
                image = models.expand_to_3_channels(image) if image.shape[1] == 1 else image
                if dl.dataset.is_pair_data:
                    original_image = models.expand_to_3_channels(original_image) if original_image.shape[1] == 1 else original_image
                    image_input = torch.cat([image, original_image], dim=0)
                else:
                    image_input = image
                logits = model(image_input.to(device))
                if len(logits.shape) == 2 and logits.shape[1] > 1:
                    # Three-way classification (expects logits of shape [N, 3])
                    loss = F.cross_entropy(logits, target=label.long().to(logits.device))
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits.squeeze(1),
                        target=label.to(logits.device).type_as(logits),
                        reduction="mean",
                    )
                out_dict = {"logits": logits}
            else:
                out_dict = model(
                    batch,
                    augmenter=augmenter,
                    template_image=dataloaders["train"].dataset.get_avg_template(
                        batch["char"][0]
                    )
                    if args.use_template_residual
                    else None,
                )
                loss = model.compute_loss(
                    out_dict,
                    label,
                    damage_loc_xy=damage_loc_xy,
                    non_damage_loc_xy=non_damage_loc_xy,
                )
            assert not torch.isnan(loss)
            eval_log["losses"].append(loss.item())
            if args.loss_type == "sup_con":
                continue

            logits = out_dict["logits"].contiguous() if "logits" in out_dict else None
            if args.loss_type in {"object_ce", "focal"}:
                # xy_targets = []
                # xys = [(x.item(), y.item()) for x, y in zip(*damage_loc_xy)]
                # for b in range(label.shape[0]):
                #     x, y = xys[b]
                #     if x == -1 or y == -1:
                #         xy_targets.append(torch.zeros((1, 1, 16, 16), dtype=torch.float, device=device).view(-1))
                #     else:
                #         xy_targets.append(model.make_target_grid(xys[b], (64, 64), original_res_damage_width=4, output_grid_dim=(16, 16)).view(-1))
                # label = torch.stack(xy_targets)
                # NOTE: make one binary label for entire image based off of whether
                # or not there is a single confident prediction at a location (>0.5)
                # import ipdb; ipdb.set_trace()
                # out = (out.squeeze() > np.log(0.5)).any(dim=1).float()
                logits = logits.max(dim=1)[0]

        # hold pred proba for binary classification; argmax label for 3-way
        eval_log["pred"].extend(
            torch.sigmoid(logits).view(-1).tolist() 
            if logits.shape[1] == 1 
            else logits.argmax(dim=1).tolist()
        )
        eval_log["true"].extend(label.view(-1).tolist())

    # import ipdb; ipdb.set_trace()
    # reduce over batches
    eval_log["loss"] = np.mean(eval_log["losses"])
    if args.loss_type == "sup_con":
        print(f">>> Loss={eval_log['loss']:.2f}")
        return eval_log
    try:
        if len(set(eval_log["true"])) == 2:
            roc_auc = roc_auc_score(eval_log["true"], eval_log["pred"])
            fpr, tpr, thresholds = roc_curve(eval_log["true"], eval_log["pred"])
            display = RocCurveDisplay(
                fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Damage Detection"
            )
            display.plot()
            plt.savefig(Path(args.output_dir) / "latest_roc.png")
            plt.close()
    except ValueError:
        roc_auc = 0.0
        print(
            "Only one class present in y_true. ROC AUC score is not defined in that case. Skipping this epoch..."
        )
    if len(set(eval_log["true"])) == 2:
        prec, rec, thresholds = precision_recall_curve(eval_log["true"], eval_log["pred"])
        disp = PrecisionRecallDisplay(precision=prec, recall=rec)
        disp.plot()
        plt.savefig(Path(args.output_dir) / "latest_prcurve.png")
        plt.close()

        try:
            det_fpr, det_fnr, _ = det_curve(eval_log["true"], eval_log["pred"])
            disp = DetCurveDisplay(fpr=det_fpr, fnr=det_fnr)
            disp.plot()
            plt.savefig(Path(args.output_dir) / "latest_detcurve.png")
            plt.close()
        except ValueError:
            det_fpr, det_fnr = 0.0, 0.0
            print(
                "Only one class present in y_true. Detection error tradeoff curve is not defined in that case. Skipping this epoch..."
            )
        ap = average_precision_score(eval_log["true"], eval_log["pred"])
        median_pred_prob = np.median(eval_log["pred"])
        
        max_acc = max(
            [
                accuracy_score(
                    eval_log["true"], (np.asarray(eval_log["pred"]) > t).astype(int)
                )
                for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, median_pred_prob]
            ]
        )
        print(f">>> Loss={eval_log['loss']:.2f}     ROC-AUC={roc_auc:.2f}      AP={ap:.2f}")
        print(f"Median predicted probability: {median_pred_prob}")
        ps, rs, fs = [], [], []
        for t in [0.25, 0.5, 0.75, median_pred_prob]:
            thresholded_preds = (np.asarray(eval_log["pred"]) > t).astype(int)
            pt, rt, f1t, _ = precision_recall_fscore_support(
                eval_log["true"], thresholded_preds, average="binary"
            )
            print(
                f"Results (@ thresh={t:0.2f}):   P={pt:.2f}     R={rt:.2f}     F1={f1t:.2f}     Acc={max_acc:.2f}"
            )
            ps.append(pt)
            rs.append(rt)
            fs.append(f1t)

        prec, rec, thresholds = precision_recall_curve(eval_log["true"], eval_log["pred"])
        p_dict = {}
        r_dict = {}
        p_interp_dict = {}
        # print(rec)

        # print(f'Precision Recall Curve:\n{prec}\n{rec}\n{thresholds}')
        for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 0.9, 0.95]:
            # i = np.argmin(rec < t)
            i = rec.size - np.searchsorted(rec[::-1], t + 0.01, side="right")
            p_dict[t] = prec[i]
            r_dict[t] = rec[i]
            # https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
            p_interp_dict[t] = np.max(prec[:i])
            print(
                f"{thresholds[i - 1]}: P={p_dict[t]:0.2f} @ R={r_dict[t]:0.2f}      p_interp={p_interp_dict[t]:0.2f}"
            )

        precision_recall_fscore_support

        eval_log = {
            "loss": eval_log["loss"],
            "p": ps[-1],
            "r": rs[-1],
            "f1": fs[-1],
            "info": classification_report(
                eval_log["true"],
                (np.asarray(eval_log["pred"]) > median_pred_prob).astype(int),
            ),
            #'report_info': '\n'.join([f'Threshold={t}\n'+classification_report(eval_log['true'], (np.asarray(eval_log['pred']) > t).astype(int)) for t in [0.25, 0.5, 0.75]]),
            "roc_auc": roc_auc,
            "ap": ap,
            # p, r, f1, _ = precision_recall_fscore_support(eval_log['true'], thresholded_preds, average='binary'),
            "acc": max_acc,
        }
        eval_log.update({f"p@r={k}": p_dict[k] for k in r_dict})
    else:
        ap = np.nan
        roc_auc = np.nan
        p, r, f1, _ = precision_recall_fscore_support(
            eval_log["true"], eval_log["pred"], average="micro", pos_label=DAMAGE
        )
        acc = accuracy_score(eval_log["true"], eval_log["pred"])
        print(f">>> Loss={eval_log['loss']:.2f}     P={p:.2f}   R={r:.2f}   F1={f1:.2f}   Acc={acc:.2f}")
        eval_log = {
            "loss": eval_log["loss"],
            "p": p,
            "r": r,
            "f1": f1,
            "info": classification_report(
                eval_log["true"],
                eval_log["pred"],
                target_names=["normal", "damage", "bad_extraction"],
            ),
            "acc": acc,
        }
    
    # print(eval_log.keys())
    # eval_log.update(p_interp_dict)
    plt.close()
    print(eval_log["info"])
    return eval_log


def evaluate_filelist(args, datasets, model, filelist_name):
    model.eval()
    if filelist_name.endswith(".tar"):
        filelist = dict()
        with tarfile.open(filelist_name, "r") as tar:
            for tarinfo in tar:
                filelist[tarinfo.name] = float("nan")
    else:
        filelist = filelist_name
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda img: F.pad(img, [0, 64 - img.size(2), 0, 64 - img.size(1)])), transforms.Lambda(lambda x: x.max() - x)])
    transform = datasets["train"].transform
    test_ds = DamageDatasetFilelist(args, "test", transform, filelist)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    damage_probs = []
    normal_probs = []
    bad_probs = []
    paths = []
    # print(f"NOTE: computing template image using pixelwise average of TEST images. If test set is small, you might want to change this to use the TRAIN set.")
    for batch in tqdm(test_dl, desc=f"Predicting on {filelist_name}"):
        with torch.no_grad():
            image, path = batch["image"], batch["path"]
            if 'resnet' in args.encoder_type or 'vit_b_16' in args.encoder_type or 'hf_hub' in args.encoder_type:
                image = models.expand_to_3_channels(image) if image.shape[1] == 1 else image
                logits = model(image.to(device))
            else:
                template_ds = (
                    datasets["train"]
                    if batch["char"][0] in datasets["train"].char2i
                    else test_ds
                )
                logits = model(
                    batch, template_image=template_ds.get_avg_template(batch["char"][0])
                )["logits"]
            if len(logits.shape) == 2 and logits.shape[1] > 1:
                # Three-way classification (expects logits of shape [N, 3])
                # NOTE: this is not used for damage prediction
                prob = logits.softmax(dim=1)
                # make it 0.0 if it's not the max
                # damage_prob = torch.where(
                #     damage_prob == damage_prob.max(dim=1, keepdim=True)[0],
                #     damage_prob,
                #     torch.zeros_like(damage_prob),
                # )
                normal_prob = prob[:, NORMAL]
                bad_prob = prob[:, BAD_EXTRACTION]
                damage_prob = prob[:, DAMAGE]
                damage_probs.extend(damage_prob.view(-1).tolist())
                normal_probs.extend(normal_prob.view(-1).tolist())
                bad_probs.extend(bad_prob.view(-1).tolist())
            else:
                damage_prob = (
                    torch.sigmoid(logits).max(dim=1)[0]
                    if args.loss_type in {"object_ce", "focal"}
                    else torch.sigmoid(logits)
                )
                damage_probs.extend(damage_prob.view(-1).tolist())
            try:
                paths.extend(path)
            except TypeError:
                paths.extend([path])
    assert len(paths) == len(damage_probs)
    print(f"Finished predicting on {len(paths)} images.")
    results_csv = (
        Path(args.output_dir)
        / f"{Path(args.evaluate).with_suffix('').name}_damagepredictions.csv"
    )
    damage_thresholds = [
        1e-10,
        1e-5,
        0.01,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.99,
        0.999,
    ]
    #
    # NOTE:
    # if you want to save the images at score thresholds to check them/debug, uncomment the following lines
    # damaged_out_dirs = [
    #     Path(args.output_dir) / "preds" / f"damaged_{dt}" for dt in damage_thresholds
    # ]
    # undamaged_out_dirs = [
    #     Path(args.output_dir) / "preds" / f"undamaged_{dt}" for dt in damage_thresholds
    # ]
    # for p in damaged_out_dirs + undamaged_out_dirs:
    #     p.mkdir(exist_ok=True, parents=True)
    with open(results_csv, "w") as f:
        for path, damage_prob in zip(paths, damage_probs):
        # print("path,normal_prob,damage_prob,bad_prob", file=f)
        # for path, damage_prob, normal_prob, bad_prob in zip(paths, damage_probs, normal_probs, bad_probs):
            print(f"{path},{damage_prob:0.10f}", file=f)
            # print(f"{path},{normal_prob:0.10f},{damage_prob:0.10f},{bad_prob:0.10f}", file=f)
            # for dt, dod, uod in zip(
            #     damage_thresholds, damaged_out_dirs, undamaged_out_dirs
            # ):
            #     shutil.copy2(path, dod) if damage_prob > dt else shutil.copy2(path, uod)
    print(f"Saved results to {results_csv}")
    print("Results summary:")
    damage_probs = np.asarray(damage_probs)
    for t in damage_thresholds:
        print(
            f"  len(damage_probs[damage_probs > {t}]) = {len(damage_probs[damage_probs > t]) / len(damage_probs) * 100:0.1f}%"
        )
    histogram_out = (
        Path(args.output_dir) / f"{Path(args.evaluate).with_suffix('').name}_hist.png"
    )
    plt.figure()
    plt.hist(damage_probs, bins=20)
    plt.savefig(histogram_out)
    plt.close()
    print(f"Saved histogram to {histogram_out}")


def train(args, model, optimizer, dataloaders, augmenter=None):
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    train_log = defaultdict(list)
    train_log["best_valid_score"] = 0.0
    train_log["best_valid_epoch"] = 0
    # eval_splits = [k for k in dataloaders.keys()] #if k != 'train']
    eval_splits = [k for k in dataloaders.keys() if k != "train"]
    for split in eval_splits:
        train_log[f"best_{split}_score"] = 0.0
        train_log[f"best_{split}_epoch"] = 0
    for epoch in range(args.num_epochs if args.num_epochs else args.max_epochs):
        print("Epoch", epoch)

        # evaluate at interval
        if epoch % args.eval_interval == 0:
            for split in eval_splits:
                dl = dataloaders[split]
                print(f"Evaluating on {split} set...")
                eval_log = evaluate(args, dataloaders, model, dl)
                if args.wandb:
                    wandb.log({f"eval_{split}_loss": eval_log["loss"]})
                state = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(
                    state,
                    Path(args.output_dir)
                    / f"{args.loss_type}-{args.sup_con_alpha}-{'embres' if args.use_template_residual else ''}-{'tempchannel' if args.use_template_channel else ''}-{'charemb' if args.use_char_embedding else ''}_last_epoch.pt",
                )
                if args.loss_type == "sup_con":
                    continue
                if split == args.choose_best_checkpoint_on:
                    if (
                        eval_log[args.best_metric]
                        > train_log[f"best_{args.choose_best_checkpoint_on}_score"]
                    ):
                        state = {
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch,
                        }
                        torch.save(state, Path(args.output_dir) / "best.pt")
                        train_log[f"best_{args.choose_best_checkpoint_on}_score"] = (
                            eval_log[args.best_metric]
                        )
                        train_log[f"best_{args.choose_best_checkpoint_on}_epoch"] = epoch
                        train_log[f"best_{args.choose_best_checkpoint_on}_scores_info"] = (
                            eval_log["info"]
                        )
                        train_log[f"best_{args.choose_best_checkpoint_on}_all"] = {
                            k: v for k, v in eval_log.items() if k != "info"
                        }
                    elif (
                        epoch - train_log[f"best_{args.choose_best_checkpoint_on}_epoch"]
                        > args.patience
                        and args.num_epochs is None
                    ):
                        print(
                            f"Patience exceeded at epoch {epoch}. (Last best epoch was at epoch {train_log['best_' + args.choose_best_checkpoint_on + '_epoch']}). Exiting training loop."
                        )
                        return train_log
                else:
                    if eval_log[args.best_metric] > train_log[f"best_{split}_score"]:
                        train_log[f"best_{split}_score"] = eval_log[args.best_metric]
                        train_log[f"best_{split}_epoch"] = epoch
                        train_log[f"best_{split}_scores_info"] = eval_log["info"]
                        train_log[f"best_{split}_all"] = {
                            k: v for k, v in eval_log.items() if k != "info"
                        }

                if args.wandb:
                    wandb.log(
                        {f"eval_{split}_{args.best_metric}": eval_log[args.best_metric]}
                    )


        for b, batch in tqdm(
            enumerate(dataloaders["train"]),
            total=len(dataloaders["train"]),
            desc=f"Epoch {epoch}",
        ):
            model.train()
            optimizer.zero_grad()
            image, original_image, label, damage_loc_xy, non_damage_loc_xy = (
                batch["image"],
                batch["original_image"],
                batch["label"],
                batch["damage_loc_xy"],
                batch["non_damage_loc_xy"],
            )
            if args.is_pair_data:
                label = torch.cat([torch.ones(image.shape[0]), torch.zeros(image.shape[0])], dim=0).squeeze()
            # if args.jitter and args.loss_type != "sup_con":
            #     for batch_id in range(image.shape[0]):
            #         jitter_transform = make_jitter_transform()
            #         image[batch_id] = jitter_transform(image[batch_id])
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if 'resnet' in args.encoder_type or 'vit_b_16' in args.encoder_type or 'hf_hub' in args.encoder_type:
                        image = models.expand_to_3_channels(image) if image.shape[1] == 1 else image
                        if args.is_pair_data:
                            original_image = models.expand_to_3_channels(original_image) if original_image.shape[1] == 1 else original_image
                            image_input = torch.cat([image, original_image], dim=0)
                        else:
                            image_input = image
                        
                        logits = model(image_input.to(device))
                        if len(logits.shape) == 2 and logits.shape[1] > 1:
                            # Three-way classification (expects logits of shape [N, 3])
                            loss = F.cross_entropy(logits, target=label.long().to(logits.device))
                        else:
                            loss = F.binary_cross_entropy_with_logits(
                                logits.squeeze(1),
                                # target=torch.cat([torch.ones_like(logits[:image.shape[0]]), torch.zeros_like(logits[image.shape[0]:])], dim=0).squeeze().to(logits.device),
                                target=label.to(logits.device).type_as(logits),
                                # label.to(logits.device).type_as(logits),
                                reduction="mean",
                            )
                    else:
                        out_dict = model(
                            batch,
                            augmenter=augmenter,
                            update_encoder=(
                                not args.freeze_encoder
                                and epoch > args.sup_con_stage2_warmup_epochs
                            ),
                        )
                        loss = model.compute_loss(
                            out_dict,
                            label,
                            damage_loc_xy=damage_loc_xy,
                            non_damage_loc_xy=non_damage_loc_xy,
                        )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if 'resnet' in args.encoder_type or 'vit_b_16' in args.encoder_type or 'hf_hub' in args.encoder_type:
                    image = models.expand_to_3_channels(image) if image.shape[1] == 1 else image
                    logits = model(image.to(device))
                    if len(logits.shape) == 2 and logits.shape[1] > 1:
                        # Three-way classification (expects logits of shape [N, 3])
                        loss = F.cross_entropy(logits, target=label.long().to(logits.device))
                    else:
                        loss = F.binary_cross_entropy_with_logits(
                            logits.squeeze(1),
                            label.to(logits.device).type_as(logits),
                            reduction="mean",
                        )
                else:
                    out_dict = model(
                        batch,
                        augmenter=augmenter,
                        update_encoder=(
                            not args.freeze_encoder
                            and epoch > args.sup_con_stage2_warmup_epochs
                        ),
                    )
                    loss = model.compute_loss(
                        out_dict,
                        label,
                        damage_loc_xy=damage_loc_xy,
                        non_damage_loc_xy=non_damage_loc_xy,
                    )
                loss.backward()
                optimizer.step()
            train_log["loss"].append(loss.item())
            if args.wandb:
                wandb.log({"loss": loss})
        num_batches = len(dataloaders["train"])
        avg_epoch_loss = np.mean(train_log["loss"][-num_batches:])
        print(f"Completed Epoch {epoch} - Avg Train Loss: {avg_epoch_loss:.4f}")
        print(f"Completed Epoch {epoch} - Avg Train Loss (last 10 batches): {np.mean(train_log['loss'][-10:])}")

    return train_log


def assign_clusters(pred, ground):
    # crude and gredy way to assign tsne clusterings
    assignment = {}
    ratios = {}
    for i, tcl in enumerate(ground):
        ratios[tcl].append(pred[i])
    for key in ratios:
        com = Counter(ratios[key]).most_common(1)[0][0]
        assignment[key] = com
    return assignment


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))


def plot_tsne(args, model, dataloaders):
    model.eval()
    output_file = Path(args.output_dir) / f"tsne_{args.plot_tsne}.png"
    embeddings = []
    damage_labels = []
    char_labels = []
    for batch in dataloaders[args.plot_tsne]:
        with torch.no_grad():
            embedding_batch = F.normalize(
                torch.flatten(model.cnn(batch["image"].to(device)), 1), dim=-1
            )
        embeddings.append(embedding_batch)
        damage_labels.append(batch["label"])
        char_labels.extend(batch["char"])
    # import ipdb; ipdb.set_trace()
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    damage_labels = torch.cat(damage_labels, dim=0).cpu().numpy()

    all_colors = [x for x in rcParams["axes.prop_cycle"]] * 10
    # fig, axs = plt.subplots(2, figsize=(20, 8))
    fig, ax = plt.subplots(figsize=(20, 20))
    pca = PCA(50)
    tsne = TSNE(2)
    tsne_embeds = tsne.fit_transform(pca.fit_transform(embeddings))
    # pred_markers = ['o' for label in labels]
    # get a mapping b/w pred labels and true labels
    # labels_pred = labels_pred_by_sec[s]
    # labels_true = labels_true_by_sec[s]
    # assignment = assign_clusters(labels_pred, labels_true)
    # pred_colors = [all_colors[label] for label in labels_pred_by_sec[s]]
    true_markers_by_damage_class = [
        "x" if dam_label == 1 else "o" for dam_label in damage_labels
    ]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    true_colors_by_char_class = [
        all_colors[alphabet.find(char_label)] for char_label in char_labels
    ]
    # true_colors = [all_colors[int(assignment[label])] for label in labels[s]]
    for i in range(tsne_embeds.shape[0]):
        # axs[0, s].scatter(
        #     tsne_embeds[i, 0],
        #     tsne_embeds[i, 1],
        #     marker=pred_markers[i],
        #     c=pred_colors[i]['color'],
        # )
        ax.scatter(
            tsne_embeds[i, 0],
            tsne_embeds[i, 1],
            marker=true_markers_by_damage_class[i],
            c=true_colors_by_char_class[i]["color"],
        )
        # axs[0, s].set_title(f'PRED: {s}: {secnames[s]} ({len(np.unique(labels_pred_by_sec[s]))} Clusters)')
        # axs[1, s].set_title(f'TRUE: {s}: {secnames[s]} ({len(np.unique(labels_pred_by_sec[s]))} Clusters)')
    plt.savefig(output_file)

    # plot per character results too!
    char_labels = np.array(char_labels)
    for c in sorted(set(char_labels)):
        print(c)
        output_file = Path(args.output_dir) / f"tsne_{args.plot_tsne}_{c}.png"
        fig, ax = plt.subplots(figsize=(20, 20))
        pca = PCA(50)
        tsne = TSNE(2)
        cmask = char_labels == c
        tsne_embeds = tsne.fit_transform(pca.fit_transform(embeddings[cmask]))
        true_markers_by_damage_class = [
            "x" if dam_label == 1 else "o" for dam_label in damage_labels[cmask]
        ]
        true_colors_by_damage_class = [
            "red" if dam_label == 1 else "blue" for dam_label in damage_labels[cmask]
        ]
        # true_colors = [all_colors[int(assignment[label])] for label in labels[s]]
        for i in range(tsne_embeds.shape[0]):
            ax.scatter(
                tsne_embeds[i, 0],
                tsne_embeds[i, 1],
                marker=true_markers_by_damage_class[i],
                c=true_colors_by_damage_class[i],
                label="Damage" if damage_labels[cmask][i] == 1 else "Normal",
            )
        ax.legend()
        legend_without_duplicate_labels(ax)
        plt.savefig(output_file)


def prep_model(args, num_char_classes, num_channels):
    num_classes = len(args.classes)
    if args.use_cbam:
        model = models.CBAMResNet(args)
    elif 'resnet' in args.encoder_type:
        model = torch.hub.load(
            "pytorch/vision:v0.13.0",
            args.encoder_type,
            pretrained=args.use_pretrained,
        )
        model.fc = nn.Linear(model.fc.in_features, out_features=num_classes, bias=True)
    elif args.encoder_type == 'vit_b_16':
        if num_classes != 1:
            raise NotImplementedError
        model = torch.hub.load(
            "pytorch/vision:v0.13.0",
            args.encoder_type,
            pretrained=args.use_pretrained,
        )
        model.heads.head = nn.Linear(
            model.heads.head.in_features, out_features=1, bias=True
        )
    elif 'hf_hub' in args.encoder_type:
        model = timm.create_model(args.encoder_type, num_classes=num_classes, pretrained=args.use_pretrained)
        # Modify the final layer if necessary
        if hasattr(model, "fc"):  # For models like ResNet
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier"):  # For models like EfficientNet
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model = models.DamageNet(
            args, num_channels=num_channels, num_char_classes=num_char_classes
        )
    # NOTE: replaced by model's compute_loss function
    # if self.use_rpn:
    #    assert False
    # elif self.use_attn:
    #    loss_fn = BCEWithForcedAttentionLoss(reduction="mean")
    # else:
    #    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    if args.load_model:
        print("Loading model from", args.load_model)
        if args.sup_con_stage2:
            ckpt = torch.load(args.load_model, map_location=device)["model_state"]
            ckpt.pop("projection_net.net.output.weight")
            ckpt.pop("projection_net.net.output.bias")
            ckpt.pop("fc_out.weight") if "fc_out.weight" in ckpt else None
            ckpt.pop("fc_out.bias") if "fc_out.bias" in ckpt else None
            model.load_state_dict(ckpt, strict=False)
        else:
            model.load_state_dict(
                torch.load(args.load_model, map_location=device)["model_state"]
            )
    print(model)
    if "cnn" in model.__dict__:
        print(
            f"CNN trainable params: {sum(p.numel() for p in model.cnn.parameters() if p.requires_grad)}"
        )
        print(
            f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
    model = model.to(device)
    optimizer = (
        optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if args.loss_type == "sup_con"
        else optim.Adam(model.parameters(), lr=args.lr)
    )
    if args.load_optimizer:
        optimizer.load_state_dict(
            torch.load(args.load_optimizer, map_location=device)["optimizer_state"]
        )
    return model, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('char', type=str)
    parser.add_argument("train_csv", default=None, type=str)
    parser.add_argument("valid_csv", default=None, type=str)
    parser.add_argument("test_csv", default=None, type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--other_test_csvs", default="{}", type=str)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable pytorch native Automatic Mixed Precision training",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs='+',
        choices=["normal", "damaged", "bad_extraction"],
        help="Use augmentations from Augmentor library",
    )
    parser.add_argument("--jitter", action="store_true")
    parser.add_argument(
        "--best_metric",
        type=str,
        default="acc",
        choices=["acc", "p", "r", "f1", "roc_auc", "ap", "p@r=0.75"],
        help="Metric to choose model on.",
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument("--load_optimizer", default=None, type=str)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs without improvement on val set to wait before stopping.",
    )
    parser.add_argument(
        "--evaluate", default=None, type=str, help="Evaluate trained model on filelist"
    )
    # model opts
    parser.add_argument("--cnn_dim", default=128, type=int)
    parser.add_argument("--cnn_output_dim", default=512, type=int)
    parser.add_argument(
        "--use_char_embedding",
        action="store_true",
        help="Use character embeddings for multi-char class training",
    )
    parser.add_argument("--char_embedding_dim", default=32, type=int)
    parser.add_argument("--use_attn", action="store_true")
    parser.add_argument("--encoder_type", choices=[
        'resnet18',
        'resnet34',
        'resnet50',
        'vit_b_16', 
        'hf_hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k', 
        'hf_hub:timm/vit_base_patch8_224.dino', 
        'hf_hub:timm/vit_base_patch14_dinov2.lvd142m',
        'hf_hub:timm/vit_base_patch8_224.augreg_in21k_ft_in1k', 
        'hf_hub:timm/convnext_base.fb_in22k_ft_in1k'
    ], default='resnet34')
    # parser.add_argument(
    #     "--use_resnet", type=int, default=None, choices=[18, 34, 50, 101, 152]
    # )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Use pretrained resnet/vit weights (except for final random linear layer)",
    )
    # parser.add_argument("--use_vit", type=int, default=None, choices=[16, 32])
    parser.add_argument(
        "--size_transform",
        type=str,
        default="pad",
        choices=["pad", "resize", "fixed_height_resize_and_pad", "matcher"],
        help="Whether to pad or resize images to same size. Matcher uses same current transform as matcher model",
    )
    parser.add_argument("--use_cbam", action="store_true")
    parser.add_argument("--use_rpn", action="store_true")  # TODO
    parser.add_argument(
        "--loss_type",
        default="ce",
        choices=["ce", "attn_ce", "forced_cam", "focal", "object_ce", "sup_con"],
    )
    parser.add_argument(
        "--output_pooling",
        default="none",
        type=str,
        choices=["global_avg_pool_hw", "max_pool_f", "none"],
    )
    parser.add_argument("--eval_interval", default=1, type=int)
    parser.add_argument(
        "--use_template_channel",
        action="store_true",
        help="Stack template on separate channel as input to CNN",
    )
    parser.add_argument("--use_template_residual", action="store_true")
    parser.add_argument("--use_pos_weight_on_ce", action="store_true")
    parser.add_argument("--pos_weight_ce_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--limit_train_data",
        type=int,
        default=sys.maxsize,
        help="Limit train data to total amount of examples.",
    )
    parser.add_argument(
        "--choose_best_checkpoint_on",
        type=str,
        default="test",
        choices=["valid", "test", "valid_jitter", "test_jitter"],
    )
    parser.add_argument(
        "--plot_tsne",
        type=str,
        default=None,
        choices=["train", "valid", "test"],
        help="Plot tsne on data set split",
    )
    parser.add_argument(
        "--sup_con_stage2",
        action="store_true",
        help="Do stage 2 cross-entropy training of sup_con and learn linear classifier.",
    )
    parser.add_argument(
        "--sup_con_stage2_warmup_epochs",
        type=int,
        default=0,
        help="Warmup linear classifier before updating rest of network.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Optionally freeze encoder (during stage 2 sup_con training).",
    )
    # projection net options
    # parser.add_argument('--projection_net_in_dim', type=int, default=128)
    parser.add_argument("--projection_net_out_dim", type=int, default=256)
    parser.add_argument("--projection_net_hidden_dim", type=int, default=256)
    parser.add_argument(
        "--projection_net_type", type=str, choices=["mlp", "linear"], default="linear"
    )
    # sup_con hyperparams
    parser.add_argument("--sup_con_temp", type=float, default=0.1)
    parser.add_argument(
        "--sup_con_alpha",
        type=float,
        default=0.3,
        help="Weight on positive term for interpolation between pos/neg loss terms",
    )
    # CNN
    parser.add_argument("--num_conv_per_block", type=int, default=3, help="")
    parser.add_argument("--num_cnn_blocks", type=int, default=3, help="")
    # for synthetic pair data (damaged, original pairs)
    parser.add_argument('--is_pair_data', action='store_true', help='Whether the data is synthetic pair data or not')
    parser.add_argument('--apply_random_inking', action='store_true', help='Whether to apply inking augmentation during jitter or not')
    parser.add_argument('--apply_position_jitter', action='store_true', help='Whether to apply position jitter during jitter or not')
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    datasets, dataloaders = prep_datasets(args)
    model, optimizer = prep_model(
        args,
        num_char_classes=len(datasets["train"].get_char_classes_labels()),
        num_channels=datasets["train"].get_num_channels(),
    )
    # import ipdb; ipdb.set_trace()
    if args.evaluate:
        evaluate_filelist(args, datasets, model, args.evaluate)
        exit(0)
    if args.plot_tsne:
        plot_tsne(args, model, dataloaders)
        exit(0)
    if args.wandb:
        run_name = Path(args.output_dir).name 
        run_name += f'jitter{args.jitter}'
        if args.jitter:
            run_name += f'_ink{args.apply_random_inking}_pos{args.apply_position_jitter}'
        run_name += f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        wandb.init(
            project="damaged-type", entity="nvog", name=run_name
        )
        wandb.watch(model)
        wandb.config = vars(args)
    # (train_ds, valid_ds, test_ds), (train_dl, valid_dl, test_dl) = prep_datasets(args)
    print("Starting training...")
    start_time = time.time()
    # train_log = train(args, model, loss_fn, optimizer, train_dl, valid_dl, test_dl)
    augmenter = None
    if args.loss_type == "sup_con":
        augmenter = TypeAugmenter(args)
    train_log = train(args, model, optimizer, dataloaders, augmenter=augmenter)
    print("================================================")
    print(f"Finished training in {(time.time() - start_time) / 60:.1f} mins.")
    print(
        f"Best {args.choose_best_checkpoint_on} {args.best_metric}={train_log['best_' + args.choose_best_checkpoint_on + '_score']:.4f} @ epoch {train_log['best_' + args.choose_best_checkpoint_on + '_epoch']}."
    )
    # if train_log[f'best_valid_all']:
    #     print('\nbest_valid_all')
    #     for k, v in train_log[f'best_valid_all'].items():
    #         if k in {'acc', 'p', 'r', 'f1', 'roc_auc', 'ap', 'p@r=0.75'}:
    #             print(f'{k}: {v:0.3f}')
    # print('\nbest_valid_scores_info')
    # print(train_log['best_valid_scores_info'])
    for split in list(datasets.keys()):
        # if split != 'train':
        if True:
            try:
                print(
                    "Best {} {}={:.4f} @ epoch {}.".format(
                        split,
                        args.best_metric,
                        train_log[f"best_{split}_score"],
                        train_log[f"best_{split}_epoch"],
                    )
                )
                if train_log[f"best_{split}_all"]:
                    for k, v in train_log[f"best_{split}_all"].items():
                        if k in {"acc", "p", "r", "f1", "roc_auc", "ap", "p@r=0.75"}:
                            print(f"{k}: {v:0.3f}")
            except Exception:
                continue
            print(train_log[f"best_{split}_scores_info"])
