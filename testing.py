from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from scipy.sparse import *
from construct_W import construct_W
import matplotlib.pyplot as plt
from matplotlib import colors as cl
from sklearn.ensemble import AdaBoostClassifier
from SIMLR import SIMLR_LARGE


def lap_score(X, **kwargs):
    X = X.values
    # if 'W' is not specified, use the default W
    if 'W' not in kwargs.keys():
        W = construct_W(X)

    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp) / D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp) / D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000

    # compute laplacian score for all features
    score = 1 - np.array(np.multiply(L_prime, 1 / D_prime))[0, :]
    return np.transpose(score)


def feature_ranking(score):
    idx = np.argsort(score, 0)
    return idx


def select_features_filter(X, y, ft_num):
    bestfeatures = SelectKBest(score_func=chi2, k="all")
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    ft = featureScores.nlargest(ft_num, 'Score')
    features = ft.index.values
    return features


def select_features_laplace(X, y, ft_num):
    score = lap_score(X)
    features = feature_ranking(score)
    return features[:ft_num]


def select_features_wrapper(X, y, ft_num):
    model = sfs(
        KNeighborsClassifier(n_neighbors=25, p=5),
        k_features=ft_num,
        forward=True,
        verbose=2,
        cv=4,
        n_jobs=-1,
        scoring='accuracy')
    model.fit(X, y)
    print(model.k_feature_idx_)
    return model.k_feature_idx_


def load_train_data(ft_num):
    # Read data
    data = pd.read_csv("data/train.csv")
    X = data.iloc[:, 0:595]
    y = data.iloc[:, -1]
    X = X.drop(["X3", "X31", "X32", "X127", "X128", "X590"], axis=1)
    features = select_features_filter(X, y, ft_num)
    return X.values[:, features], y, features


def load_test_data(features):
    # Read data
    data = pd.read_csv("data/test.csv")
    X = data.iloc[:, 0:595]  #independent columns
    X = X.drop(["X3", "X31", "X32", "X127", "X128", "X590"], axis=1)
    return X.values[:, features]


def train_simlr(X):
    c = 2
    simlr = SIMLR_LARGE(c, 11)
    S, F, val, ind = simlr.fit(X)
    y_pred = simlr.fast_minibatch_kmeans(F, c)
    return y_pred


def train_gradient_boost(X_train, y_train):
    '''
    Fts - Overall
    100 - 0.64
    160 - 0.64
    410 - 0.64
    391 - 0.64
    '''
    gb = GradientBoostingClassifier(
        n_estimators=6,  # 6
        learning_rate=1,
        max_features=2,
        max_depth=2,
        random_state=289)
    gb.fit(X_train, y_train)
    return gb


def train_decision_tree(X_train, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(
        criterion="gini",
        random_state=1,
        max_depth=3,
        min_samples_leaf=4,
        min_samples_split=0.5,
        max_features=240)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# Change range and function parameters to calculate best parameters
# It prints the all result after sorting their accuracy
acc = []
for i in range(10, 11):
    X, y, features = load_train_data(341)

    #Apply 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=95)

    overall_accuracy = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = train_gradient_boost(X_train, y_train)
        y_pred = clf.predict(X_test)

        print('Accuracy {}'.format(accuracy_score(y_test, y_pred)))
        overall_accuracy += accuracy_score(y_pred, y_test)

    test_x = load_test_data(features)
    g_b = train_gradient_boost(X, y)
    pred = g_b.predict(test_x)

    print("Overall Accuracy: ", overall_accuracy / 5)
    acc.append([i, overall_accuracy / 5])
acc.sort(key=lambda acc: acc[1])
print(acc)

submission_file = [["ID", "Predicted"]]
for i, prediction in enumerate(pred):
    submission_file.append([i + 1, int(prediction)])

with open('submission.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(submission_file)
