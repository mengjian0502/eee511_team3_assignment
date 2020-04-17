
import argparse
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from gmm import *

# Set debug mode
DEBUG = True

parser = argparse.ArgumentParser(description='GEMM clustering')
# parameters
parser.add_argument('--clusters', type=int, default=3, help='number of clusters')
args = parser.parse_args()

def main():
    # Load data
    Y = np.load("./data/samples.npy")
    matY = np.matrix(Y, copy=True)

    # The number of models
    K = args.clusters

    if K not in [3, 5, 7]:
        raise ValueError("Number of clusters must be 3, 5, or 7!")
    
    # Calculate the Gmm models parameters 
    num_iter = 200
    mu, cov, alpha, log_likelihood = GMM_EM(matY, K, num_iter)
    
    print(log_likelihood.shape)

    # According to the GMM model, cluster the sample data, one model corresponds to one category
    N = Y.shape[0]
    # Find the responsiveness matrix of each model to the sample under the current model parameters
    gamma = getExpectation(matY, mu, cov, alpha)
    # For each sample, find the model index with the most responsiveness as its category identifier
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    # Put each sample in the corresponding category list

    marker = ['rs', 'bo', 'gv', 'kp', 'yh', 'c<', 'm>']
    plt.figure(figsize=(8,8), dpi=300)
    for kk in range(K):
        cluster = np.array([Y[i] for i in range(N) if category[i] == kk])
        plt.plot(cluster[:, 0], cluster[:, 1], marker[kk], label=f"class{kk+1}")

    plt.legend(loc="best")
    plt.title("GMM Clustering By EM Algorithm")
    plt.savefig(f'./figs/gemm_cluster_{K}.png')

    plt.figure(figsize=(5,5), dpi=300)
    plt.plot(np.arange(1, num_iter+1, 1), log_likelihood.mean(axis=1))
    
    plt.title(f'GEMM: Log likelihood | Num clusters = {K}')
    plt.xlabel('Iterations')
    plt.savefig(f'./figs/gemm_loglike_{K}.png', bbox_inches = 'tight', pad_inches = 0)

if __name__ == '__main__':
    main()