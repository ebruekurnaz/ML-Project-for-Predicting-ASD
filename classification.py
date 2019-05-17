import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import csv


def load_train_data():
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

    return train_x, train_y


def load_test_data():
    # Read data
    test_data = pd.read_csv("data/test.csv")

    # Remove zeros from data
    test_data = test_data.loc[:, (test_data != 0).any(axis=0)]

    return test_data.values

def train_random_forest(X_train, y_train):

	# Instantiate rf
    rf = RandomForestRegressor(n_estimators=25,
            random_state=2)
            
	# Fit rf to the training set    
    rf.fit(X_train, y_train) 

    return rf

def train_using_gini(X_train, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 

# Load Datas
train_x, train_y = load_train_data()
test_x = load_test_data()

# Apply SVC
#clf = SVC(kernel='rbf', gamma = 100, C = 100).fit(train_x,train_y)
#clf.fit(train_x, train_y)
#pred_y = clf.predict(test_x)


#Apply Decision Tree
#clf_gini = train_using_gini(train_x, train_y)
#pred_y = clf_gini.predict(test_x)

#Apply K means
kmeans = KMeans(n_clusters=3)
kmeans.fit(train_x,train_y)

pred_y = kmeans.predict(test_x)


# Write to submission file
submission_file = [["ID", "Predicted"]]
for i, prediction in enumerate(pred_y):
    submission_file.append([i + 1, int(prediction)])

with open('submission.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(submission_file)

# print(accuracy_score(train_y,pred_y))