import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from kmeans import k_means

parser = argparse.ArgumentParser(description='Kmeans clustering')
# parameters
parser.add_argument('--clusters', type=int, default=4, help='number of clusters')
args = parser.parse_args()

def plotting(predict_labels, data, num_clusters, centroids, f1, f2):
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

        plt.scatter(
            centroids[ii, 0],
            centroids[ii, 1],
            marker='X',
            s=100,
            c='r'
        )

    plt.title(f'Kmeans: after clustering | Number of clusters={args.clusters}')
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.legend(loc='best')
    plt.savefig(f'./figs/kmeans_cluster_{num_clusters}_{f1}_{f2}.png', bbox_inches = 'tight', pad_inches = 0)

def main():
    clusters = args.clusters

    if clusters not in [4, 6, 8, 10]:
        raise ValueError("Number of clusters must be 4, 6, 8, or 10!")


    data_path = './data/Mall_Customers.csv'    
    attr = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    f1, f2 = 'Age', 'Spending Score (1-100)'
    
    df = pd.read_csv(data_path)
    data = df[[f1, f2]].iloc[: , :].to_numpy()


    if f1 == 'Gender':
        gender = data[:, 0]

        gender[gender=='Male'] = 5.
        gender[gender=='Female'] = 10.
        print(gender)
        data[:,0] = gender

    print(f'Shape of the data: {data.shape}')

    km = k_means(num_clusters=clusters, tol=1e-4)
    est_centroid, history_centroids, predict_labels, loss, num_iter = km.get_cluster(data)


    plotting(predict_labels, data, clusters, est_centroid, f1, f2)


if __name__ == '__main__':
    main()