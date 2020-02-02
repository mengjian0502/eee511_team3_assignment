"""
EEE511 team assignment 01 - team 03

widrow_hoff LMS algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


parser = argparse.ArgumentParser(description='Widrow Hoff LMS algorithm')

# parameters
parser.add_argument('--epochs', type=int, default=1, help='number of iterations')
parser.add_argument('--eta', default=5e-3, type=float)
parser.add_argument('--theta', type=float)
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

    train_target_path = './mackey-glass/data/mg_series_5000.npy'           # dataset (time series)

    weights = np.zeros([args.time_step, 1])                                # initialize the weight vector as a t_step by 1 vector

    dataset = dataloader(train_target_path)

    train_target = dataset[:4000]
    test_target = dataset[4000:5000]

    print(f"number of samples for training: {train_target.shape}")
    print(f"number of samples for test: {test_target.shape}")


    for epoch in range(args.epochs):
        train_pred, weights, train_loss, train_mse = train(args.eta, train_target, weights)
        test_pred, test_loss, test_mse = test(test_target, weights, epoch, args.save_path)

        print(f'[Epoch: {epoch+1}], training error: {train_mse}, test loss: {test_mse}')

    plt.figure(figsize=(8,6))
    
    plt.plot(range(dataset.shape[0]), dataset, linewidth=2, label='original time series')
    plt.plot(range(train_target.shape[0]-args.time_step), train_pred, linestyle='-.', label='training prediction')
    plt.plot(range(train_target.shape[0], train_target.shape[0] + test_target.shape[0]-args.time_step), test_pred, linestyle=':', label='test prediction')

    plt.title(f'LMS algorithm with MG time series: \n training average loss {train_mse[0]}, test average loss: {test_mse[0]}')
    plt.xlabel('time')
    plt.ylabel('$x(t)$')

    plt.legend(loc='best')

    plt.savefig(args.save_path+'prediction.png', dpi=300)

    plt.close()

        

def dataloader(target_path):
    """
    Load the pre-generated dataset from makey_glass time series
    """
    target = np.load(target_path)                                          # load the target
    # print(f'size of the dataset: {dataset.shape}')
    
    return target


def train(eta, target, weight):
    """
    LMS algorithm training
    """
    loss_sum = 0

    series_pred = []
    series_loss = []

    for ii in range(target.shape[0]-args.time_step):  
        
        t_start, t_end = ii, ii + args.time_step

        x = target[t_start: t_end].reshape(args.time_step, 1)
        y_true = target[t_end]

        y_pred = weight.T @ x
        loss = (y_true - y_pred)

        if ii % 50 == 0:
            print(f'training loss: {loss}')

        weight = weight + eta * loss * x

        loss_sum += loss

        series_pred.append(y_pred[0])
        series_loss.append(loss[0])
    
    mse = loss_sum / target.shape[0]


    return np.array(series_pred), weight, np.array(series_loss), mse


def test(target, weight, epoch, save_path):
    """
    LMS algorithm for Mackey Glass Time Series prediction
    """
    loss_sum = 0

    series_pred = []
    series_loss = []

    for ii in range(target.shape[0]-args.time_step): 
        
        t_start, t_end = ii, ii + args.time_step

        x = target[t_start: t_end].reshape(args.time_step, 1)
        y_true = target[t_end]

        y_pred = weight.T @ x
        loss = (y_true - y_pred)

        if ii % 50 == 0:
            print(f'test loss: {loss}')

        loss_sum += loss

        series_pred.append(y_pred[0])
        series_loss.append(loss[0])

    mse = loss_sum / target.shape[0]

    return np.array(series_pred), np.array(series_loss), mse



if __name__ == '__main__':
    main()