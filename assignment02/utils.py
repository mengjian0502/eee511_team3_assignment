"""
EEE511 team assignment 02 - team 03 

Kaggle Happiness Score Regression - utils.py
https://www.kaggle.com/unsdsn/world-happiness
"""

# Dependencies
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import shutil

def to_tensor(sat_val):
    """
    convert the input data into pytorch tensor
    """
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return out

def data_read(data_path, tensor=False):
    """
    extract input features & output from the .csv file

    The following features will be extracted as the input: 
    - GDP
    - Family
    - Health
    - Freedom
    - Generosity
    - Corruption
    """
    attr = ['Score', 'GDP', 'Family', 'Health', 'Freedom', 'Generosity', 'Corruption']
    year = np.arange(2015, 2020, 1)                                                     # csv file: from 2015 to 2019
    
    score = []
    features = []

    dataset_size = 0

    for yy in year:
        file_path = data_path + f'{yy}.csv'                                             # define the relative path of the .csv file
        df = pd.read_csv(file_path, usecols=attr).to_numpy()                            # read the csv file
        
        sc = df[:, 0]
        f = df[:, 1:]
        
        score.append(sc)
        features.append(f)

        dataset_size += sc.shape[0]

    score = np.concatenate(score, axis=0)
    features = np.concatenate(features, axis=0)

    # np.save(data_path+'input_features.npy', features)                                 # save the extracted data into .npy file
    # np.save(data_path+'happy_score.npy', score.reshape(dataset_size, 1))              # save the happiness score into .npy file

    torch.save(to_tensor(score.reshape(dataset_size, 1)), data_path+'happy_score.pt')   # save the extracted data into .pt file
    torch.save(to_tensor(features), data_path+'input_features.pt')                      # save the happiness score into .pt file

def train_loader(t_dataset, t_target, batchsize):
    """
    Input: tensor dataset
    Output: data loader of training set
    """

    dataset_Tensor = data.TensorDataset(t_dataset, t_target)

    return data.DataLoader(
        dataset = dataset_Tensor,
        batch_size=batchsize,
        shuffle=True
    )

def valid_loader(v_dataset, v_target, batchsize):
    """
    Input: tensor dataset
    Output: data loader of validation set
    """

    validset_Tensor = data.TensorDataset(v_dataset, v_target)

    return data.DataLoader(
        dataset=validset_Tensor,
        batch_size=batchsize, 
        shuffle=True
    )

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth'):
    if is_best is True:
        torch.save(state, save_path+'model_best.pth')
    else:
        torch.save(state, save_path+filename)

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()