import sys
from collections import defaultdict
from evaluation import load_recall_match_eval
from pathlib import Path

matches_dirs = [
    '/home/nvog/projects/git/anomaly-detection/matching/leviathan_matching_test_set/{}/test',
    '/home/nvog/projects/git/anomaly-detection/matching/areopagitica_matching_test_set/{}/test'
]
for i, matches_dir_unf in enumerate(matches_dirs):
    for c in 'ABCDEFGHKLMNPRTW':
        matches_dir = matches_dir_unf.format(c)
        if i == 1 and c not in 'DFGM':
            continue
        match_groups = set()
        with open(Path(matches_dir)/'matches.csv') as f:
            for line in f:
                if line.strip() and not line.strip().startswith('#'):
                    match_groups.add(tuple(sorted(line.strip().split())))
        print(c, 'match groups:', len(match_groups))
        img_paths, gt_matches = load_recall_match_eval(Path(matches_dir))
        print(c, len(set(gt_matches.keys())), len(set([f for s in gt_matches.values() for f in s])))
    
