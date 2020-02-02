"""
dataset testing:

visualize the time series & shape 
"""

import numpy as np


def main():
    mg_series = np.load('./data.npy')
    
    print(f"the shape of the time series: {mg_series.shape}")

    mg_series = mg_series.reshape(mg_series.shape[0],1)

    np.save('./mg_series_train.npy', mg_series)

if __name__ == '__main__':
    main()