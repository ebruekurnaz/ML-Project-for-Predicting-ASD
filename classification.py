import pandas as pd
import numpy as np


def load_data():
    # Read data
    train_data = pd.read_csv("data/train.csv")

    # Remove zeros from data
    train_data = train_data.loc[:, (train_data != 0).any(axis=0)]

    # Change data to numpy array
    train_data = train_data.values
    col_size = len(train_data[0]) - 1

    # Take last column
    train_y = train_data[:,col_size]

    # Remove last colum
    train_x = train_data[:,0:col_size]
    print(train_x)
    print(train_y)


load_data()