"""
EEE511 team assignment 02 - team 03 

Kaggle Happiness Score Regression
https://www.kaggle.com/unsdsn/world-happiness
"""

# Dependencies
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import os
from sklearn.linear_model import LinearRegression
from utils import *

parser = argparse.ArgumentParser(description='Kaggle Happiness Score Regression')

# parameters
parser.add_argument('--year', type=int, default=2015, help='which year')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate of training')
parser.add_argument('--epochs', type=int, default=100, help='training epochs')
parser.add_argument('--val_perc', type=float, default='ratio of validation set')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--cuda', default=True, help='use cuda')


# save results
parser.add_argument('--save_path', type=str, default='./save/', help='folder to save the results/plots')

args = parser.parse_args()

class Net(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, D_h)
        self.linear2 = nn.Linear(D_h, D_out)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.linear1(x))
        out = self.relu(self.linear2(out))
        return out

def main():
    # devices
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
        print(f'save path: {args.save_path} created')
    else:
        print(f'save path already exist')

    # combine dataset
    data_dir = f'./dataset/world-happiness/'        # data path

    if not os.path.exists(data_dir+'input_features.pt'):
        data_read(data_dir)
    else:
        print(f'dataset already exists')
        features = torch.load(data_dir+'input_features.pt', map_location=device)
        score = torch.load(data_dir+'happy_score.pt', map_location=device)
        print(f"size of the dataset: {features.size(0)}")

    # dataset splitting needed.
    
    
    
    
def train(model, train_loader, optimizer, epoch, total_epoch, device):
    model.train()
    loss_fn = nn.MSELoss()

    loss_sum = 0
    for b, (bx, by) in enumerate(train_loader):
        x = Variable(bx.float(), requires_grad=False).to(device)
        y = Variable(by.float(), requires_grad=False).to(device)

        optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        loss_sum += loss.item() * len(bx)

        loss.backward()
        optimizer.step()

        if b % 50 == 0:
            print(f"Train Epoch {epoch}, [{b * len(bx)} / {len(train_loader.dataset)}] \t Loss:{loss.item()} \t")
    
    avg_loss = loss_sum / len(train_loader.dataset)


    if epoch % 1 == 0:
        print(f"Epoch: {epoch}; Training loss: {avg_loss}\n")

    return avg_loss
    
    

def valid(model, epoch, valid_loader, device):
    model.eval()
    loss_fn = nn.MSELoss()

    valid_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            valid_loss += loss_fn(output, target).item() * len(data)

    valid_loss /= len(valid_loader.dataset)

    print(f"Epoch: {epoch}; Validation loss: {valid_loss}")
    return valid_loss



if __name__ == '__main__':
    main()



