"""
EEE511 team assignment 02 - team 03 

Kaggle Happiness Score Regression - utils.py
https://www.kaggle.com/unsdsn/world-happiness
"""

# Dependencies
import numpy as np
import pandas as pd
import torch

def to_tensor(sat_val):
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
    attr = ['Score','GDP', 'Family', 'Health', 'Freedom', 'Generosity', 'Corruption']
    year = np.arange(2015, 2020, 1)
    
    score = []
    features = []

    dataset_size = 0

    for yy in year:
        file_path = data_path + f'{yy}.csv'
        df = pd.read_csv(file_path, usecols=attr).to_numpy()
        
        sc = df[:, 0]
        f = df[:, 1:-1]
        
        score.append(sc)
        features.append(f)

        dataset_size += sc.shape[0]

    score = np.concatenate(score, axis=0)
    features = np.concatenate(features, axis=0)

    # np.save(data_path+'input_features.npy', features)
    # np.save(data_path+'happy_score.npy', score.reshape(dataset_size, 1))

    torch.save(to_tensor(score.reshape(dataset_size, 1)), data_path+'happy_score.pt')
    torch.save(to_tensor(features), data_path+'input_features.pt')

        