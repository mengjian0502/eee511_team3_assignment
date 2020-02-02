"""
EEE511 team assignment 01 - team 03

widrow_hoff LMS algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error 


parser = argparse.ArgumentParser(description='Widrow Hoff LMS algorithm')

parser.add_argument('--epochs', type=int, help='number of iterations')
parser.add_argument('--eta', type=float)
parser.add_argument('--theta', type=float)
parser.add_argument('--time_step', type=int, default=2, help='the number of samples extracted from the time series in each epoch')

args = parser.parse_args()
def main():
    train_target_path = ''                          # training set (time series)

    weights = np.zeros([args.time_step, 1])         # initialize the weight vector as a t_step by 1 vector

    train_target = dataloader(train_target_path)

    for epoch in range(args.epochs):
        weights, mse = train(args.eta, args.time_step, train_target, weights)
        print(mse)


def dataloader(target_path):
    """
    Load the pre-generated dataset
    """
    target = np.load(target_path)                   # load the target
    # print(f'size of the dataset: {dataset.shape}')
    
    return target


def train(eta, time_step, target, weight):
    """
    LMS algorithm training
    """
    num_window = target.shape[0] // time_step      # number of windows
    loss_sum = 0

    for ii in range(num_window):
        t_start, t_end = ii, ii + time_step
        x = np.arange(t_start, t_end, 1)            # time steps

        y_true = target[t_start, t_end]

        y_pred = weight @ x                         # compute the prediction

        loss = (y_true - y_pred)**2

        loss_sum += loss                             # sum up the loss

        weight = weight - eta * (y_pred - y_true) * x

    mse = loss_sum / target.shape[0]                # compute the mse error

    return weight, mse
    
    

if __name__ == '__main__':
    main()