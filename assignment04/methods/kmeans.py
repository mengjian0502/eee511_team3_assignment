"""
k-means clustering
"""

import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans

class Kmeans():
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
        init_centorid = data[init_centorid_idx]                                             # randomly fetch the centroids

        old_centroids.append(init_centorid)                                                 # record the initial estimation



def main():
    # Load the data samples and the corresponding labels
    labels = np.load('../data/labels.npy')
    samples = np.load('../data/samples.npy')

    print(f'Size of the data samples: {samples.shape}')

    # data preprocessing: PCA
    pca = PCA(n_components=2)
    principleComp = pca.fit_transform(samples)
    
    # visualize the transformed data:
    plt.figure(figsize=(8,8), dpi=300)
    sns.scatterplot(principleComp[:, 0], principleComp[:, 1])
    plt.title('Before clustering')
    plt.savefig('../figs/pca_process.png')

    

if __name__ == '__main__':
    main()