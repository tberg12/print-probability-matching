""" Contains various file handling utils and things """
import sys
from glob import glob
import random
from pathlib import Path
from torchvision import transforms
import numpy as np
import torch
from collections import defaultdict, OrderedDict
import copy
from skimage.util import random_noise
from torch.optim.lr_scheduler import LambdaLR
import math


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TeeFile:  # write to multiple files at once
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


def list_images_in_dirs(dirs, randomize=False, first_k=sys.maxsize, file_pattern="*.tif"):
    img_paths = []
    for dirpath in dirs:
        fpaths = list(glob(f"{dirpath}/{file_pattern}"))
        if len(fpaths) == 0:
            print(f"WARNING: {dirpath} does not have any files!")
        img_paths += fpaths
        print(f"Dir {dirpath} has {len(fpaths)} images")
    if randomize:
        random.shuffle(img_paths)
    return img_paths[:first_k]


def update_best_metrics(metrics, epoch):
    metrics_items = list(metrics.items())
    for metric, value in metrics_items:
        if metric.startswith('best'):
            continue
        best_metric_str = f'best_{metric}'
        best_epoch_metric_str = f'best_epoch_{metric}'
        
        if isinstance(value, float) or isinstance(value, int):
            if ('loss' in metric and value < metrics[best_metric_str]) \
                or value > metrics[best_metric_str]:
                metrics[best_metric_str] = value
                metrics[best_epoch_metric_str] = epoch
    return metrics


def get_printer_name(path_str):
    return Path(path_str).name.split('_')[0]


def get_year(path_str):
    try:
        year = int(Path(path_str).name.split('-')[0][-4:])
    except (IndexError, ValueError) as e:
        year = None
    return year


def get_book_name(path_str):
    p = Path(path_str).name.split('-')[0].split('_')[-1]
    if p[-1].isupper():
        p = p[:-1]
    return p


def get_line_num(path_str):
    try:
        return int(Path(path_str).name.split('_')[5].replace('page1rline', ''))
    except (IndexError, ValueError) as e:
        # handle cases like F_uc - (96077) Two treatises of government in... p. 84-s l. 35 c. 2.jpg
        return -1



def get_char(path_str):
    name = Path(path_str).name
    if '_uc' in name:
        return name[name.find('_uc') - 1] 
    elif 'letters' in str(path_str):
        return Path(path_str).parent.name
    elif 'uc' not in name:
        return name[name.find('.')-1]

def get_page_num(path_str):
    try:
        return int(Path(path_str).name.split('-')[1].split('_')[0])
    except (IndexError, ValueError) as e:
        # handle cases like F_uc - (96077) Two treatises of government in... p. 84-s l. 35 c. 2.jpg
        return -1


def get_char_loc(path_str):
    try:
        return int(Path(path_str).name.split('_')[6].replace('char', ''))
    except (IndexError, ValueError) as e:
        # handle cases like F_uc - (96077) Two treatises of government in... p. 84-s l. 35 c. 2.jpg
        return -1


def get_shakespeare4thfolio_part(path_str):
    page_num = int(Path(path_str).name.split('-')[1].split('_')[0][:4])
    if page_num <= 292:
        return 1
    elif 293 <= page_num <= 620:
        return 2
    elif 621 <= page_num:
        return 3


# def make_jitter_transform(change_scale=False, from_tensor=True):
def make_jitter_transform(change_scale=True, from_tensor=False, add_noise=True):
    x_translate = random.uniform(-0.6, 0.6)
    y_translate = random.uniform(-0.6, 0.6)
    rotation_angle = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.3) if change_scale else 1.0
    transform_list = []
    if from_tensor:  # if the input is a tensor, convert it to PIL image
        transform_list.append(transforms.ToPILImage(mode='L'))
    transform_list.extend(
        [
            transforms.Lambda(
                lambda img: transforms.functional.affine(
                    img,
                    angle=rotation_angle,
                    translate=(
                        int((x_translate * 10 * 224) / 224),
                        int((y_translate * 10 * 224) / 224)
                    ),
                    scale=scale,
                    shear=0.0,
                    fill=1
                    # deprecated:
                    # resample=False,
                    # fillcolor=1
                )
            ),
            # transforms.ToTensor(),
            # # add noise
            # transforms.Lambda(
            #     lambda img: random_noise(img, mode='s&p', amount=0.05, salt_vs_pepper=0.25, clip=True)
            #     # lambda img: random_noise(img, mode='gaussian', mean=0, var=0.05, clip=True)
            #     # lambda img: random_noise(img, mode='speckle', mean=0, var=0.05, clip=True)
            # ) if add_noise else lambda x: x,
            transforms.ToTensor(),
            # and finally, take the borders and threshold them to 1 
            # (fillcolor doesnt do this correctly for some reason)
            transforms.Lambda(
                lambda img: torch.where(img > 0, torch.tensor(1.0), img)
            )
        ]
    )
    return transforms.Compose(
        transform_list
    )


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
            # fnames = [fname.strip() for fname in line.split(',') if (img_root_path/fname.strip()).exists()]
            fnames = []
            for fname in line.strip().split(','):
                if Path(fname).exists():
                    fnames.append(fname.strip())
                elif (img_root_path/fname.strip()).exists():
                    fnames.append(fname.strip())
                elif 'anomaly-detection' in fname:
                    # NOTE: replace savernake path with pando path
                    fnames.append(fname.strip().replace('/home/nvog/projects/git/anomaly-detection/matching', '/graft2/code/nvog/git/matching/data'))
            if len(set(fnames)) >= 2:  # otherwise there's no match on the line and the fnames cant be considered "gold"
                match_set = set(fnames)
                # assert len(fnames) == len(match_set), str(match_set)
                for fname in match_set:
                    its_matches = copy.deepcopy(match_set)
                    its_matches.remove(fname)
                    assert len(its_matches) > 0
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
            found = total = 0
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


def create_batch(data, bsz):
    l = len(data)
    for ndx in range(0, l, bsz):
        yield data[ndx:min(ndx + bsz, l)]

    
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


def get_model_params_and_size(model):
    # Get total trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Calculate model size
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    model_size_mb = param_size / 1024**2
    model_size_gb = model_size_mb / 1024
    print(f"Trainable parameters: {trainable_params}")
    print(f"Estimated model size (GB): {model_size_gb:.2f}")
    return trainable_params, model_size_gb


class LinearWarmupCosineDecayScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = (current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return max(self.min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def get_scheduler(self):
        return LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
