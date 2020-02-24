import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import os
import torch.utils.data as data
import matplotlib.pyplot as plt
import random
import seaborn as sns;sns.set()
from torch.autograd import Variable
from utils import *


parser = argparse.ArgumentParser(description='Kaggle Happiness Score Regression')
# parameters
parser.add_argument('--lr', type=float, default=0.1, help='learning rate of training')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum of training')
parser.add_argument('--epochs', type=int, default=200, help='training epochs')
parser.add_argument('--val_perc', type=float, default=0.2, help='ratio of validation set')
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--cuda', default=True, help='use cuda')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.5],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')


# save results
parser.add_argument('--save_path', type=str, default='./save_lr/', help='folder to save the results/plots')
parser.add_argument('--prt_freq', type=int, default=1, help='print frequency')

args = parser.parse_args()

class LR_Model(nn.Module):
    def __init__(self):
        """
        Linear regression with multiple features
        """
        super(LR_Model, self).__init__()
        self.linear = torch.nn.Linear(6, 1)  # six features in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


def main():
    # devices
    use_cuda = args.cuda and torch.cuda.is_available()                                                                                              # check the if the cuda is available
    device = torch.device("cuda" if use_cuda else "cpu")                                                                                            # define the device                                                                                                       # dimensionality of hidden layer
    if not os.path.isdir(args.save_path):                                                                                                           # create the save path
        os.makedirs(args.save_path)
        print(f'save path: {args.save_path} created')
    else:
        print(f'save path already exist')

    # log 
    log = open(os.path.join(args.save_path,                                                                                                         # create the log file
                            'log_mlp_log.txt'), 'w')
    # combine dataset
    data_dir = f'./dataset/world-happiness/'                                                                                                        # data dir

    print_log(f"Use device: {device}, cuda availablity: {torch.cuda.is_available()}", log)      

    if not os.path.exists(data_dir+'input_features.pt'):
        data_read(data_dir)
    else:
        print(f'dataset already exists')
        features = torch.load(data_dir+'input_features.pt', map_location=device)
        score = torch.load(data_dir+'happy_score.pt', map_location=device)
        print_log(f"size of the dataset: {features.size(0)}", log)

        train_dataset, train_target = features[:int((1-args.val_perc) * features.size(0)), :], score[:int((1-args.val_perc) * features.size(0))]    # dataset splitting: training set
        valid_dataset, valid_target = features[int(args.val_perc * features.size(0)):-1, :], score[int(args.val_perc * features.size(0)):-1]        # dataset splitting: validation set

    print_log(f"training set size: {train_dataset.size()}", log)                                                                                    # size of the training set
    print_log(f"validation set size: {valid_target.size()}", log)                                                                                   # size of the validation set

    training_loader = train_loader(train_dataset.float(), train_target.float(), args.batch)                                                         # define the dataloader for training & validaiton
    val_loader = valid_loader(valid_dataset.float(), valid_target.float(), args.batch)

    model = LR_Model().to(device)                                                                                                                     # map the model to the selected device
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)                                                                       # optimizer of training: SGD

    val_loss = []
    train_loss = []

    for e in range(1, args.epochs+1):
        current_learning_rate, current_momentum = adjust_learning_rate(                                                                             # adjust the learning rate based on the pre-defined schedule and ratio
            optimizer, e, args.gammas, args.schedule)
        
        t_loss = train(model, training_loader, optimizer, e, args.epochs, device)                                                                   # training
        v_loss = valid(model, e, val_loader, device)                                                                                                # validation

        train_loss.append(t_loss)                                                                                                                   # fetch the loss
        val_loss.append(v_loss)

        if e == 1:                                                                                                                                  # initialize the checkpoint
            best = v_loss
            is_best = False
            save_checkpoint({
                'epoch': e,
                'state_dict': model.state_dict(),
                'best_loss': v_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_path)
            best_loss = v_loss
        else:       
            is_best = v_loss < best                                                                                                                 # save the best model (early stop/saving)
            save_checkpoint({
                'epoch': e,
                'state_dict': model.state_dict(),
                'best_loss': v_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_path)

        if e % args.prt_freq == 0:
            if is_best: 
                print_log(f"Best model obtained: epoch = {e}, saving...", log)
                best_loss = v_loss
            print_log(f"Epoch: {e}/{args.epochs}, training loss: {t_loss}, validation_loss: {v_loss}, LR={current_learning_rate}, momentum={current_momentum}, best accuracy {best_loss}", log)
    
    # plotting
    plt.figure(figsize=(8,6))
    plt.plot([ii for ii in range(args.epochs)], train_loss, label='train')
    plt.plot([ii for ii in range(args.epochs)], val_loss, label='valid')
    plt.xlabel(f"Epoch", fontsize=14, fontweight='bold')
    plt.ylabel(f"MSE loss", fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.savefig(args.save_path+f'curve.png', dpi=300, bbox_inches = 'tight', pad_inches = 0)

# training
def train(model, train_loader, optimizer, epoch, total_epoch, device):
    model.train()
    loss_fn = nn.MSELoss()

    loss_sum = 0
    for b, (bx, by) in enumerate(train_loader):
        x = Variable(bx.float(), requires_grad=False).to(device)                                                                                    # wrap up the input batch
        y = Variable(by.float(), requires_grad=False).to(device)    

        y_hat = model(x)                                                                                                                            # forward pass                                                                
        loss = loss_fn(y_hat, y)                                                                                                                    # compute the loss

        loss_sum += loss.item() * len(bx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 1 == 0:
            print(f"Train Epoch {epoch}, [{b+1} / {len(train_loader)}] \t Loss:{loss.item()} \t")                                                   # print log

    avg_loss = loss_sum / len(train_loader.dataset)                                                                                                 # compute the mean error of the batch

    return avg_loss
    
# validation
def valid(model, epoch, valid_loader, device):
    model.eval()                                                                                                                                    # switch the model to evaluation 
    loss_fn = nn.MSELoss()

    valid_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)                                                                                       # load the data to the selected device

            output = model(data)                                                                                                                    # forward pass

            valid_loss += loss_fn(output, target).item() * len(data)

    valid_loss /= len(valid_loader.dataset)
    return valid_loss

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed decayed based on the pre-defined schedule"""
    lr = args.lr
    mu = args.momentum

    if optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


if __name__ == '__main__':
    main()