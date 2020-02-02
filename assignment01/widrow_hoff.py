"""
EEE511 team assignment 01 - team 03

widrow_hoff LMS algorithm

Widrow, Bernard, and Marcian E. Hoff. Adaptive switching circuits. 
No. TR-1553-1. Stanford Univ Ca Stanford Electronics Labs, 1960.
"""

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns; sns.set()
import os


parser = argparse.ArgumentParser(description='Widrow Hoff LMS algorithm')

# parameters
parser.add_argument('--epochs', type=int, default=1, help='number of iterations')
parser.add_argument('--eta', default=5e-3, type=float)
parser.add_argument('--time_step', type=int, default=2, help='the number of samples extracted from the time series in each epoch')

# results saving
parser.add_argument('--save_path', type=str, default='./save/', help='folder to save the results/plots')

args = parser.parse_args()

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
        print(f'save path: {args.save_path} created')
    else:
        print(f'save path already exist')

    train_target_path = './mackey-glass/data/mg_series_5000.npy'                                # dataset (time series)

    weights = np.zeros([args.time_step, 1])                                                     # initialize the weight vector as a t_step by 1 vector

    dataset = dataloader(train_target_path)                                                     # load the full size mackey glass series

    train_target = dataset[:4000]                                                               # split the dataset into training set & test set
    test_target = dataset[4000:5000]

    print(f"number of samples for training: {train_target.shape}")
    print(f"number of samples for test: {test_target.shape}")


    for epoch in range(args.epochs):
        train_pred, weights, train_loss, train_mse = train(args.eta, train_target, weights)     # training phase
        test_pred, test_loss, test_mse = test(test_target, weights)                             # test phase

        print(f'[Epoch: {epoch+1}], training error: {train_mse}, test loss: {test_mse}')

    # plot the prediction
    plt.figure(figsize=(8,6))
    
    plt.plot(range(dataset.shape[0]), dataset, linewidth=2, label='original time series')
    plt.plot(range(train_target.shape[0]-args.time_step), 
            train_pred, linestyle='-.', label='training prediction')
    plt.plot(range(train_target.shape[0], train_target.shape[0] + test_target.shape[0]-args.time_step), 
            test_pred, linestyle=':', label='test prediction')

    plt.title(f'LMS algorithm with MG time series: \n training average loss {train_mse[0]}, test average loss: {test_mse[0]}')
    plt.xlabel('time')
    plt.ylabel('$x(t)$')

    plt.legend(loc='best')

    plt.savefig(args.save_path+'prediction.png', dpi=300)

    plt.close()


    # plot the loss
    plt.figure(figsize=(8,8))
    plt.plot(range(train_target.shape[0]-args.time_step), abs(train_loss), linewidth=2)
    plt.xlabel('time')
    plt.ylabel('Training loss $|d(t)-y(t)|$')
    plt.grid(True)

    plt.savefig(args.save_path+'training_loss.png', dpi=300)

    plt.close()

def dataloader(target_path):
    """
    Load the pre-generated dataset from makey_glass time series
    ------
    Parameters
    target_path: the directory of the dataset

    out:
    target: numpy array of the dataset
    """
    target = np.load(target_path)                                                              # load the dataset
    # print(f'size of the dataset: {dataset.shape}')
    
    return target


def train(eta, target, weight):
    """
    LMS algorithm training function
    ------
    Parameters:
    eta: learning rate of the algorithm
    target: training set
    weight: weight value of the connection

    out:
    Numpy array of the time series prediction
    Updated weights
    Training loss (numpy array)
    mse: average training loss
    """
    loss_sum = 0                                                                            # initialize the sum of the loss

    series_pred = []
    series_loss = []

    for ii in range(target.shape[0]-args.time_step):  
        
        t_start, t_end = ii, ii + args.time_step                                            # begining of the window & end of the window

        x = target[t_start: t_end].reshape(args.time_step, 1)                               # pick the current + past data
        y_true = target[t_end]                                                              # target output

        y_pred = weight.T @ x                                                               # linear combination 
        loss = (y_true - y_pred)                                                            # compute the loss

        if ii % 50 == 0:
            print(f'training loss: {loss[0]}')                                              # print the loss every 50 samples

        weight = weight + eta * loss * x                                                    # update the weights

        loss_sum += loss

        series_pred.append(y_pred[0])
        series_loss.append(loss[0])
    
    mse = loss_sum / target.shape[0]                                                        # compute the average training loss


    return np.array(series_pred), weight, np.array(series_loss), mse


def test(target, weight):
    """
    LMS algorithm for Mackey Glass Time Series prediction
    -----
    Parameters:
    Target: test set of the algorithm
    Weight: weights of the connections

    out:
    Numpy array of the time series prediction
    Test loss (numpy array)
    Average test loss
    """
    loss_sum = 0

    series_pred = []
    series_loss = []

    for ii in range(target.shape[0]-args.time_step): 
        
        t_start, t_end = ii, ii + args.time_step                                        # begining of the window & end of the window

        x = target[t_start: t_end].reshape(args.time_step, 1)                           # pick the current + past data
        y_true = target[t_end]                                                          # target output

        y_pred = weight.T @ x                                                           # linear combination 
        loss = (y_true - y_pred)                                                        # compute the loss

        if ii % 50 == 0:
            print(f'test loss: {loss[0]}')

        loss_sum += loss

        series_pred.append(y_pred[0])
        series_loss.append(loss[0])

    mse = loss_sum / target.shape[0]                                                    # compute the average training loss

    return np.array(series_pred), np.array(series_loss), mse



if __name__ == '__main__':
    main()