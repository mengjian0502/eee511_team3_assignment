"""
k-means clustering
"""

import argparse

import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description='Kmeans clustering')
# parameters
parser.add_argument('--clusters', type=int, default=3, help='number of clusters')
args = parser.parse_args()

class k_means():
    def __init__(self, num_clusters, tol):
        self.num_clusters = num_clusters
        self.tol = tol
    
    def distance(self, x, y):
        """
        Return the euclidian distance between two points
        """
        return np.linalg.norm(x-y)

    def get_cluster(self, data):
        old_centroids = []
        
        sample_size, feature_size = data.shape 
        init_centorid_idx = np.random.randint(0, sample_size-1, size=self.num_clusters)
        est_centorid = data[init_centorid_idx]                                             # randomly fetch the centroids

        old_centroids.append(est_centorid)                                                 # record the initial estimation
        prev_centroids = np.zeros(est_centorid.shape)                                      # initialize centroid history

        classify = np.zeros((sample_size, ))
        diff = self.distance(est_centorid, prev_centroids)
        num_iter = 0
        loss = []

        while diff > self.tol:
            diff = self.distance(est_centorid, prev_centroids)
            loss.append(diff*2)                                                             # record the losses
            num_iter += 1
            print(f'Iter: {num_iter} | diff: {diff}')

            for ii, instance in enumerate(data):
                dist = np.zeros((self.num_clusters,1))
                # for each centroid:
                for jj, centroids in enumerate(est_centorid):
                    dist[jj] = self.distance(centroids, instance)
                # find the minimum distance
                classify[ii] = np.argmin(dist)
            tmp_est = np.zeros((self.num_clusters, feature_size))

            # for each cluster
            for idx in range(len(est_centorid)):
                # all points that classified to a cluster
                instance_close = [k for k in range(len(classify)) if classify[k] == idx]
                centorid = np.mean(data[instance_close], axis=0)
                tmp_est[idx, :] = centorid
            
            prev_centroids = est_centorid
            est_centorid = tmp_est
            old_centroids.append(tmp_est)

        return est_centorid, old_centroids, classify, loss, num_iter


def plotting(predict_labels, data, num_clusters):
    color = ['lightgreen', 'orange', 'lightblue', 'steelblue', 'red', 'blueviolet', 'aqua']
    markers = ['s', 'o', 'v', '^', 'x', 'D', 'P']

    plt.figure(figsize=(8,8), dpi=300)

    for ii in range(num_clusters):
        plt.scatter(
            data[predict_labels == ii, 0], data[predict_labels == ii, 1],
            s=50, c=color[ii],
            marker=markers[ii], edgecolor='black',
            label=f'cluster {ii+1}'
        )

    plt.title('Kmeans: after clustering')
    plt.legend(loc='best')
    plt.savefig(f'./figs/kmeans_cluster_{num_clusters}.png', bbox_inches = 'tight', pad_inches = 0)
            
def main():
    clusters = args.clusters
    if clusters not in [3, 5, 7]:
        raise ValueError("Number of clusters must be 3, 5, or 7!")

    # Load the data samples and the corresponding labels
    labels = np.load('./data/labels.npy')
    samples = np.load('./data/samples.npy')

    print(f'Size of the data samples: {samples.shape}')

    # data preprocessing: PCA
    pca = PCA(n_components=2)
    principleComp = pca.fit_transform(samples)
    
    # visualize the transformed data:
    plt.figure(figsize=(8,8), dpi=300)
    sns.scatterplot(principleComp[:, 0], principleComp[:, 1])
    plt.title('Before clustering')
    plt.savefig('./figs/pca_process.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    # Performing kmeans (with sklearn)
    # km = KMeans(n_clusters=3, max_iter=200)
    # predict_labels = km.fit_predict(samples)

    km = k_means(num_clusters=clusters, tol=1e-4)
    est_centroid, history_centroids, predict_labels, loss, num_iter = km.get_cluster(samples)
    
    # plotting:
    plotting(predict_labels, principleComp, clusters)

    # losses
    plt.figure(figsize=(8,8), dpi=300)
    plt.plot(np.arange(0, num_iter, 1), loss)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Cost')
    plt.xticks(np.arange(0, num_iter, 1))
    plt.title(f'Kmeans: Cost in learning | Number of clusters={clusters}')
    plt.savefig('./figs/Kmeans_loss.png', bbox_inches = 'tight', pad_inches = 0)


    

if __name__ == '__main__':
    main()