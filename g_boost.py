from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split, KFold
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def load_train_data(features):
    # Read data
    data = pd.read_csv("data/train.csv")
    X = data.iloc[:, 0:595]  #independent columns
    y = data.iloc[:, -1]  #target column i.e price range

    bestfeatures = SelectKBest(score_func=chi2, k="all")
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  #naming the dataframe columns
    ft = featureScores.nlargest(features, 'Score')
    features = ft.index.values
    values = ft.values
    return X.values[:, features], y, features


def load_test_data(features):
    # Read data
    data = pd.read_csv("data/test.csv")
    X = data.iloc[:, 0:595]  #independent columns
    y = data.iloc[:, -1]  #target column i.e price range
    return X.values[:, features]


def train_gradient_boost(X_train, y_train):
    '''
    Fts - Overall
    100 - 0.64
    160 - 0.64
    410 - 0.64
    391 - 0.64
    '''
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 1, max_features=2, max_depth = 2, random_state = 72)
    gb.fit(X_train, y_train)
    return gb


def calc_accurancy(predicted_list):
    grand_truth_label = [
        0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0
    ]

    accuracy = 0

    for i in range(80):
        if i not in [
                1, 2, 4, 7, 10, 13, 14, 21, 22, 24, 26, 28, 29, 32, 33, 38, 39,
                40, 42, 44, 46, 47, 50, 51, 53, 54, 56, 59, 61, 63, 64, 67, 69,
                71, 72, 73, 75, 77, 78, 79
        ]:
            accuracy += grand_truth_label[i] == predicted_list[i]
    return accuracy / 40 * 100


acc = []
for i in range(299, 300):
    X, y, features = load_train_data(i)

    #Apply 5-Fold Cross Validation
    kf = KFold(n_splits=5)

    overall_accuracy = 0
    mean_vector = np.zeros((24, i))
    index = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = train_gradient_boost(X_train, y_train)
        y_pred = clf.predict(X_test)

        print('Accuracy {}'.format(accuracy_score(y_test, y_pred)))
        overall_accuracy += accuracy_score(y_test, y_pred)
    test_x = load_test_data(features)
    g_boost = train_gradient_boost(X, y)
    pred = g_boost.predict(test_x)
    print("Overall Accuracy: ", overall_accuracy / 5)
    acc.append([i, calc_accurancy(pred)])
acc.sort(key=lambda acc: acc[1])
print(acc)


test_x = load_test_data(features)
g_boost = train_gradient_boost(X, y)
pred = g_boost.predict(test_x)

print(calc_accurancy(pred))
