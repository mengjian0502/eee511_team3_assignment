"""
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser(description='GEMM clustering')
# parameters
parser.add_argument('--clusters', type=int, default=4, help='number of clusters')
args = parser.parse_args()


def plotting(predict_labels, data, num_clusters):
    color = ['lightgreen', 'orange', 'lightblue', 'steelblue', 'red', 'blueviolet', 'aqua', 'g', 'tan', 'darkcyan', 'darkblue']
    markers = ['s', 'o', 'v', '^', 'x', 'D', 'P', 'X', 'h', '+']

    plt.figure(figsize=(8,8), dpi=300)

    for ii in range(num_clusters):
        plt.scatter(
            data[predict_labels == ii, 0], data[predict_labels == ii, 1],
            s=50, c=color[ii],
            marker=markers[ii], edgecolor='black',
            label=f'cluster {ii+1}'
        )

    plt.title(f'GMM: after clustering | Number of clusters=4')
    plt.legend(loc='best')
    plt.savefig(f'./figs/gemm_sklearn_cluster_{num_clusters}_genderFalse.png', bbox_inches = 'tight', pad_inches = 0)

def main():
    K = args.clusters

    if K not in [4, 6, 8, 10]:
        raise ValueError("Number of clusters must be 4, 6, 8, or 10!")

    X = np.load("./data/customer_data_original_genderFalse.npy", allow_pickle=True)

    # print(f"Empty data: {X.isnull().any().any()}")

    pca = PCA(n_components=2)
    principleComp = pca.fit_transform(X)

    gmm = GaussianMixture(n_components=K)

    gmm_x = gmm.fit(X)

    labels = gmm_x.predict(X)

    plotting(labels, principleComp, K)

    print(f'GMM: Converged = {gmm_x.converged_} | num_iter = {gmm_x.n_iter_}')



if __name__ == '__main__':
    main()