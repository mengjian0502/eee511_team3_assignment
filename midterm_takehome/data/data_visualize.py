import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def main():
    data_path = './Mall_Customers.csv'
    attr = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

    df = pd.read_csv(data_path, usecols=attr)


    data = df.to_numpy()
    gender = data[:, 0]

    gender[gender=='Male'] = 5.
    gender[gender=='Female'] = 10.
    data[:,0] = gender
    
    X = df.drop(['CustomerID', 'Gender'], axis=1)
    sns.pairplot(df.drop('CustomerID', axis=1), hue='Gender', aspect=1.5)
    plt.savefig('../figs/data_visualize.png', dpi=300)

if __name__ == '__main__':
    main()