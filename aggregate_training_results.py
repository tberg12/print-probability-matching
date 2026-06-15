import sys
import re
from collections import defaultdict
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt

pat = re.compile(r"^[>|\*]+ \['([A-Z])'\] (.*): (.*)", flags=re.MULTILINE)

k = '5'
limit_to_first_n_epochs = 60 // 10

log_pat = sys.argv[1]
logs = sorted(glob.glob(log_pat)) if len(sys.argv) == 2 else sorted(sys.argv[1:])
print('# logs:', len(logs))
all_results = []
for log in logs:
    print(log)
    with open(log) as l:
        all_results.extend(pat.findall(l.read()))
print(Path(logs[0]).parents[0].name)
logname = '_'.join(Path(logs[0]).parents[0].name.split('_')[1:-3])

char2res = defaultdict(lambda: defaultdict(list))
metrics = set()
for i, (char, metric, score) in enumerate(all_results):
    # print(i, char, metric, score)
    if 'Score' in metric or not metric.endswith(k):
        continue
    # import ipdb; ipdb.set_trace()
    # print(char, metric, score)
    if len(char2res[char]['_'.join(metric.split('_')[:-1])]) < limit_to_first_n_epochs:
        char2res[char]['_'.join(metric.split('_')[:-1])].append(float(score))
        metrics.add('_'.join(metric.split('_')[:-1]))

# import ipdb; ipdb.set_trace()
agg = defaultdict(dict)
plt.figure(figsize=(10,10))
colors = ['lightgray', 'darkgray', 'mediumorchid', 'purple', 'lightblue', 'darkblue', 'lightgreen', 'darkgreen', 'sandybrown', 'saddlebrown', 'lightcoral', 'crimson'] * 4
m = 0
for metric in sorted(metrics):
    if 'denom' in metric:
        continue
    print(f"{metric}:")
    if 'pct' not in metric:
        # gt_test_pos_recall_A.5
        # denom_metric = '_'.join(metric.split('_')[:-1]) + '_denom_' + metric.split('_')[-1]
        denom_metric = '_'.join(metric.split('_')) + '_denom'
        print(metric, denom_metric)
        # agg_micro['_'.join(metric.split('_')[:2])] = agg_metrics[metric] / sum(agg_metrics[denom_metric])
        cs = 'DFGM' if 'areo' in metric else sorted(char2res.keys())
        nums = np.array([char2res[c][metric] for c in cs])
        denoms = np.array([char2res[c][denom_metric][-1] for c in cs])
        # print(metric, (np.sum(nums, axis=0) / np.sum(denoms)).tolist())
        # if 'syn_valid' in metric:
        #     import ipdb; ipdb.set_trace()
        agg['micro'][metric] = (np.sum(nums, axis=0) / np.sum(denoms)).tolist()
        print(f"micro:", agg['micro'][metric])
        plt.plot(agg['micro'][metric], label=f"{metric}-micro", color=colors[m])
        m += 1
    if 'pct' in metric:
        cs = 'DFGM' if 'areo' in metric else sorted(char2res.keys())
        pcts = np.array([char2res[c][metric] for c in cs])
        # if 'syn_valid' in metric:
        #     import ipdb; ipdb.set_trace()
        agg['macro'][metric] = (np.sum(pcts, axis=0) / pcts.shape[0]).tolist()
        print(f"macro:", agg['macro'][metric])
        plt.plot(agg['macro'][metric], label=f"{metric}-macro", color=colors[m])
        m += 1

plt.legend()
plt.title(logname)
plt.ylim(0.0, 1.0)
plt.grid()
plt.savefig(f'agg_training_results_{logname}.png')


print('\n---------\nmicro gt_test_areo_recall:', agg['micro']['gt_test_areo_recall'])
i = np.argmax(agg['micro']['gt_test_areo_recall'])
print(i)
# (chosen via areo valid micro-avg r@{k})
print(f'R@{k}')
latex_list = []
for bg in ['syn_valid', 'gt_test_areo', 'gt_test_strong_neg', 'gt_test_mix_neg', 'gt_test_pos', 'gt_test_weak_neg']:
    # print('bg')
    v = agg['micro'][f'{bg}_recall'][i]
    latex_list.append(f'{v*100:0.2f}')
    print(bg, v)
print()
print(' & '.join(latex_list[:-2]))