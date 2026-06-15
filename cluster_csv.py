# takes a matching output input_csv file containing triu distances and clusters
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import SpectralClustering
import joblib


parser = argparse.ArgumentParser()
parser.add_argument('input_csv', type=str)
parser.add_argument('--output_csv', type=str, default=None)
parser.add_argument('--n_clusters', type=int, default=200)
parser.add_argument('--n_neighbors', type=int, default=10)
parser.add_argument('--affinity', type=str, choices=['precomputed_nearest_neighbors', 'precomputed'], default='precomputed')
parser.add_argument('--minmaxscale', action='store_true')
parser.add_argument('--dist_to_aff_fn', type=str, choices=['invert', 'rbf'], default='rbf')
parser.add_argument('--conversion_power', type=int, default=2)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--sigma', type=float, default=1.0)
args = parser.parse_args()


output_csv = args.output_csv

if output_csv is None:
    output_csv = str(Path(args.input_csv).with_suffix('')) + f'_clustering{args.n_clusters}.csv'

def load_adj_mat(csv_file):
    df = pd.read_csv(csv_file, names=['q', 'c', 'distance'])
    qfilelist_set = set()
    qfilelist = []
    for q in df.q.tolist():
        if q not in qfilelist_set:
            qfilelist.append(q)
            qfilelist_set.add(q)

    qfname2i = {qfilelist[i]: i for i in range(len(qfilelist))}
    qi2fname = {v: k for k, v in qfname2i.items()}

    cfilelist_set = set()
    cfilelist = []
    for c in df.c.tolist():
        if c not in cfilelist_set:
            cfilelist.append(c)
            cfilelist_set.add(c)

    cfname2i = {cfilelist[i]: i for i in range(len(cfilelist))}
    ci2fname = {v: k for k, v in cfname2i.items()}

    mat = np.zeros((len(qfname2i)+1, len(cfname2i)+1), dtype=float)

    #print(len(qfname2i))
    #print(len(cfname2i))
    
    # make symmetric
    for i, row in df.iterrows():
        mat[qfname2i[row.q], cfname2i[row.c]+1] = row.distance
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat, qfilelist, cfilelist


print('Loading csv...')
mat, qfilelist, cfilelist = load_adj_mat(args.input_csv)

plt.figure()
sns.heatmap(mat, square=True)
plt.savefig(output_csv.replace('.csv', '_heatmap.png'))

def convert_distance_mat_to_affinity_mat(distance_mat, sigma=1.0, power=2, eps=1.0):
    #return np.exp(-distance_mat ** power / (2. * sigma ** 2))
    return np.exp(-(eps * distance_mat) ** power / (2 * sigma ** 2))

def print_stats(mat):
    print(f"min = {mat.min()}, max = {mat.max()}, mean = {mat.mean()}, std = {mat.std()}")

def min_max_scaler(mat, scale_range=(0.0, 1.0)):
    scaled_min, scaled_max = scale_range
    mat_std = (mat - mat.min()) / (mat.max() - mat.min())
    mat = mat_std * (scaled_max - scaled_min) + scaled_min
    return mat

if args.affinity == 'precomputed':
    # scale distance matrix?
    if args.minmaxscale:
        print('Scaling data...')
        print(f'Old range:')
        print_stats(mat)
        mat = min_max_scaler(mat)
        print(f'New range:')
        print_stats(mat)
    print("Converting distance matrix to affinity...")
    print("Before:")
    print_stats(mat)
    if args.dist_to_aff_fn == 'invert':
        print('Inverting distance matrix to obtain affinity matrix...')
        mat = mat.max() - mat
    elif args.dist_to_aff_fn == 'rbf':
        mat_min = mat.min()
        mat_max = mat.max()
        fig, axes = plt.subplots(1, 2)
        axes[0].hist(mat, bins=30)
        mat = convert_distance_mat_to_affinity_mat(mat, power=args.conversion_power, eps=args.eps, sigma=args.sigma)
        print(f'min of {mat_min} will get converted to {convert_distance_mat_to_affinity_mat(mat_min, power=args.conversion_power, eps=args.eps)}')
        print(f'max of {mat_max} will get converted to {convert_distance_mat_to_affinity_mat(mat_max, power=args.conversion_power, eps=args.eps)}')
        axes[1].hist(mat, bins=30)
        plt.savefig('rbf.png')
    print("After:")
    print_stats(mat)
    print()
    print(f'Clustering {mat.shape[0]}x{mat.shape[1]} aff mat...')
    clustering = SpectralClustering(
            n_clusters=args.n_clusters,
            random_state=42,
            affinity='precomputed',
            n_jobs=20,
            assign_labels='kmeans',  # 'discretize'
        ).fit(mat)
    print(f"SpectralClustering's aff mat stats:")
    print_stats(clustering.affinity_matrix_)
else:
    print()
    print(f'Clustering {mat.shape[0]}x{mat.shape[1]} aff mat...')
    clustering = SpectralClustering(
            n_clusters=args.n_clusters,
            random_state=42,
            affinity='precomputed_nearest_neighbors',
            n_neighbors=args.n_neighbors,
            n_jobs=20,
            assign_labels='kmeans',  # 'discretize'
        ).fit(mat)
#print(clustering.labels_)
print('Eigenvalues:')
print(', '.join([f'{e:0.5f}' for e in clustering.eigvals_]))

spectral_clustering_path = output_csv.replace('.csv', f'_spectralclustering-{args.affinity}-{args.dist_to_aff_fn}-{args.conversion_power}-{args.sigma}.joblib')
print('Writing SpectralClustering object to', spectral_clustering_path)
joblib.dump(clustering, spectral_clustering_path)


print("TODO: change back cluster csv name")
#clustering_output = output_csv.replace('.csv', f'-{args.affinity}-{args.dist_to_aff_fn}-{args.conversion_power}-{args.sigma}.csv')
clustering_output = output_csv
print('Writing clustering to', clustering_output)
with open(clustering_output, 'w') as f:
    for i, l in enumerate(clustering.labels_):
        print(','.join([
                str(qfilelist[i]) if i == 0 else str(cfilelist[i-1]),
                str(l)
            ]), file=f)


#import ipdb; ipdb.set_trace()
print('done')


