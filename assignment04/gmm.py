
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

DEBUG = True

######################################################
#  Debug output function
#  Control output by global variable DEBUG
######################################################
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)


######################################################
# Gaussian distribution density function of the kth model
# Each i line represents the occurrence probability of the i sample in each model
# Return one-dimensional list
######################################################
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)


######################################################
#E Step: Calculate the responsiveness of each model to the sample
#Y is a sample matrix, one row per sample, and a column vector when there is only one feature
#mu is a mean multi-dimensional array, each row represents the mean of each feature of a sample
#cov is an array of covariance matrices, alpha is an array of model responsivity
######################################################
def getExpectation(Y, mu, cov, alpha):
    #  number of samples
    N = Y.shape[0]
    #  number of models
    K = alpha.shape[0]

    
    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"

    # Responsiveness matrix, rows correspond to samples, columns correspond to responsivity
    gamma = np.mat(np.zeros((N, K)))

    # Calculate the probability of occurrence of all samples in each model, rows correspond to samples, columns correspond to models
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    # Calculate the responsivity of each model to each sample
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma


######################################################
# M Step: Iterate model parameters
# Y is the sample matrix, gamma is the responsivity matrix
######################################################
def maximize(Y, gamma):
    # Number of samples and features
    N, D = Y.shape
    # Number of models
    K = gamma.shape[1]

    #Initialization parameter value
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    # Update the parameters of each model
    for k in range(K):
        # The sum of the responsivity of the kth model to all samples
        Nk = np.sum(gamma[:, k])
        # Update mu
        # Average each feature
        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
        # update cov
        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)
        # upudate alpha
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha


######################################################
# Data preprocessing
# Scale all data between 0 and 1
######################################################
def scale_data(Y):
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y


######################################################
# Initialize model parameters
# shape is a binary representation of the sample size, (number of samples, number of features)
# K represents the number of models
######################################################
def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    debug("Parameters initialized.")
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


######################################################
# Gaussian mixture model EM algorithm
# Given the sample matrix Y, calculate the model parameters
# K is the number of models
# times is the number of iterations
######################################################
def GMM_EM(Y, K, times):
    Y = scale_data(Y)
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    debug("{sep} Result {sep}".format(sep="-" * 20))
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha
