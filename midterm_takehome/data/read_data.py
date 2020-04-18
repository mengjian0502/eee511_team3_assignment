"""
Data Extraction: 

Fetch the data from .csv file to numpy array
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def main():
    data_path = './Mall_Customers.csv'
    attr = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    df = pd.read_csv(data_path, usecols=attr)


    data = df.to_numpy()
    # gender = data[:, 0]

    # gender[gender=='Male'] = 5.
    # gender[gender=='Female'] = 10.
    # data[:,0] = gender

    # data = MinMaxScaler().fit_transform(data)

    np.save('./customer_data_original_genderFalse.npy', data)

if __name__ == '__main__':
    main()


