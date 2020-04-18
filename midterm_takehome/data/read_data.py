"""
Data Extraction: 

Fetch the data from .csv file to numpy array
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def main():
    data_path = './Mall_Customers.csv'
    attr = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    df = pd.read_csv(data_path, usecols=attr)


    data = df.to_numpy()
    gender = data[:, 0]

    gender[gender=='Male'] = 5.
    gender[gender=='Female'] = 10.
    data[:,0] = gender
    
    data = MinMaxScaler().fit_transform(data)

    np.save('./customer_data_minmax_scale.npy', data)

if __name__ == '__main__':
    main()


