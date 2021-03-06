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
parser.add_argument('--clusters', type=int, default=4, help='number of clusters')
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



def compute_inertia(X, est_centorid, predict_labels, num_clusters):
    total_inertia = 0
    for ii in range(num_clusters):
        data = X[predict_labels == ii]
        inertia = 0
        centroid = est_centorid[ii]
        
        for jj in range(data.shape[0]):
            data_point = data[jj, :]
            inertia += np.linalg.norm(data_point - centroid)
        
        total_inertia += inertia
    
    print(f'Clusters: {num_clusters} | interia = {total_inertia}')
    return total_inertia


def plotting(predict_labels, data, num_clusters, centroids):
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
            c=color[ii],
            label=f'centroid {ii+1}'
        )

    plt.title(f'Kmeans: after clustering | Number of clusters={args.clusters}')
    plt.legend(loc='best')
    plt.savefig(f'./figs/kmeans_cluster_{num_clusters}_genderFalse.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()
            
def main():
    clusters = args.clusters
    if clusters not in [3, 4, 5, 6, 7, 8, 9, 10]:
        raise ValueError("Number of clusters must be 4, 6, 8, or 10!")

    # Load the data samples and the corresponding labels
    samples = np.load('./data/customer_data_original_genderFalse.npy', allow_pickle=True)

    # print(f"Empty data: {samples.isnull().any().any()}")
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
    
    centroid_pca = pca.fit_transform(est_centroid)

    
    # plotting:
    plotting(predict_labels, principleComp, clusters, centroid_pca)

    inertia = compute_inertia(samples, est_centroid, predict_labels, clusters)
    # losses
    plt.figure(figsize=(8,8), dpi=300)
    plt.plot(np.arange(0, num_iter, 1), loss)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Cost')
    plt.xticks(np.arange(0, num_iter, 1))
    plt.title(f'Kmeans: Cost in learning | Number of clusters={clusters}')
    plt.savefig('./figs/Kmeans_loss.png', bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    return inertia
    

if __name__ == '__main__':
    main()
