"""
convert .mat file to numpy array
"""

import numpy as np
from scipy.io import loadmat

def main():
    filename = './Assignment4_D1_2.mat'
    data = loadmat(filename)

    labels = np.asarray(data['labels'][0])
    samples = np.asarray(data['samples'])

    np.save('./labels.npy', labels)
    np.save('./samples.npy', samples)

if __name__ == '__main__':
    main()


