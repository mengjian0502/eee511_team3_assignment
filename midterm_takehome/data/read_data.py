"""
Data Extraction: 

Fetch the data from .csv file to numpy array
"""

import numpy as np
import pandas as pd


def main():
    data_path = './Mall_Customers.csv'
    attr = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    df = pd.read_csv(data_path, usecols=attr)

    data = df.to_numpy()
    gender = data[:, 0]

    gender[gender=='Male'] = 1.
    gender[gender=='Female'] = 2.
    data[:,0] = gender

    for ii in range(data.shape[1]):
        data[:, ii] = data[:, ii] / np.max(data[:, ii])

    
    np.save('./customer_data.npy', data)

if __name__ == '__main__':
    main()


