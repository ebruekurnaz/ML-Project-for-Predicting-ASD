import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def load_data(file_name):
    # Read data
    train_data = pd.read_csv("data/{}.csv".format(file_name))

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
    return train_x, train_y

train_x, train_y = load_data("train")
test_x, test_y = load_data("test")


clf = SVC(kernel='linear')
clf.fit(train_x, train_y)
pred_y = clf.predict(train_x)
print(accuracy_score(train_y,pred_y))