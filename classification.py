import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
import csv
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


def load_train_data():
    # Read data
    data = pd.read_csv("data/train.csv")
    X = data.iloc[:,0:595]  #independent columns
    y = data.iloc[:,-1]    #target column i.e price range


def select_features_filter(X, y, ft_num):
    bestfeatures = SelectKBest(score_func=chi2, k="all")
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  #naming the dataframe columns
    print(featureScores.nlargest(ft_num, 'Score'))
    ft = featureScores.nlargest(ft_num, 'Score')
    features = ft.index.values
    return features


def select_features_wrapper(X, y, ft_num):
    # model = sfs(
    #     GradientBoostingRegressor(random_state=7, learning_rate=0.5),
    #     k_features=ft_num,
    #     forward=True,
    #     verbose=2,
    #     cv=5,
    #     n_jobs=-1,
    #     scoring='r2')
    # model.fit(X, y)
    # print(model.k_feature_idx_)
    # return model.k_feature_idx_
    return [2, 30, 31, 126, 127, 137, 347, 406, 589]


def load_train_data(ft_num):
    # Read data
    data = pd.read_csv("data/train.csv")
    X = data.iloc[:, 0:595]  #independent columns
    y = data.iloc[:, -1]  #target column i.e price range

    features = select_features_wrapper(X, y, ft_num)
    return X.values[:, features], y, features


def load_test_data(features):
    # Read data
    data = pd.read_csv("data/test.csv")
    X = data.iloc[:, 0:595]  #independent columns
    y = data.iloc[:, -1]  #target column i.e price range

    return X.values[:, features]


def train_random_forest(X_train, y_train):

    # Instantiate rf
    rf = RandomForestClassifier(n_estimators=100, random_state=3)

    # Fit rf to the training set
    rf.fit(X_train, y_train)

    return rf


def train_decision_tree(X_train, y_train):

    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(
        criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


def train_linear_regression(X_train, y_train):
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    return lm


def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    return knn


def train_logistic_regression(X_train, y_train):
    logmodel = LogisticRegression(solver='lbfgs')
    logmodel.fit(X_train, y_train)
    return logmodel


def train_adaboost(X_train, y_train):
    '''
    Fts - Overall
    149 - 0.64
    107 - 0.625
    150 - 0.625
    '''
    bdt = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        algorithm="SAMME",
        n_estimators=100,
        random_state=0)
    bdt.fit(X_train, y_train)
    return bdt


def train_gradient_boost(X_train, y_train):
    '''
    Fts - Overall
    100 - 0.64
    160 - 0.64
    410 - 0.64
    391 - 0.64
    '''
    '''
    Wrapper - Overall
    10 - 0.6583333333333333 (rand_state = 0)
    10 - 0.6166666666666667 (rand_state = 1)
    10 - 0.6 (rand_state = 2)
    10 - 0.575 (rand_state = 4)
    10 - 0.65 (rand_state = 5)
    10 - 0.675 (rand_state = 6)
    10 - 0.691 (rand_state = 7)
    10 - 0.658 (rand_state = 8)
    10 - 0.691 (rand_state = 10)
    9 - 0.7 (rand_state = 7) ****
    '''
    #62
    gb = GradientBoostingClassifier(
        n_estimators=20,
        learning_rate=0.5,
        max_features=2,
        max_depth=2,
        random_state=0)
    gb.fit(X_train, y_train)
    return gb


def calculate_error(mean_val, X):
    return np.sum((mean_val - X)**2) / len(X)


def make_prediction(d_t, knn, rand_forest, x_t, mean_val):
    errors = []
    for i in range(5):
        errors.append([i + 1, calculate_error(mean_val[i], x_t)])
    errors.sort(key=lambda errors: errors[1])

    pred = []
    if (errors[0][0] == 1 or errors[0][0] == 2):
        pred = knn.predict([x_t])
    elif (errors[0][0] == 3):
        pred = rand_forest.predict([x_t])
    else:
        pred = d_t.predict([x_t])

    return pred


# data = pd.read_csv("data/train.csv")
# X = data.iloc[:,0:595]  #independent columns
# y = data.iloc[:,-1]    #target column i.e price range
# model=sfs(RandomForestRegressor(n_estimators=10),k_features=10,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')
# model.fit(X,y)
# print(model.k_feature_idx_)
# # (18, 30, 126, 127, 199, 347, 442, 445, 579, 589)
# # (2, 30, 31, 126, 127, 137, 219, 347, 406, 589)

# Load Datas
acc = []
for i in range(9, 10):
    X, y, features = load_train_data(i)
    test_x = load_test_data(features)
    avg = 0
    for j in range(10):
        #Apply 5-Fold Cross Validation
        kf = KFold(n_splits=10, shuffle=True)

        overall_accuracy = 0
        mean_vector = np.zeros((24, i))
        index = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = train_gradient_boost(X_train, y_train)
            y_pred = clf.predict(X_test)

            # print(np.mean(X_train[:,0]))
            # print(np.mean(X, axis = 1))

            mean_vector[index] = np.mean(X_test, axis=0)

            # print('Accuracy {}'.format(accuracy_score(y_test, y_pred)))
            overall_accuracy += accuracy_score(y_test, y_pred)
            index += 1
        print("Overall Accuracy: ", overall_accuracy / 10)
        avg += overall_accuracy / 10
        # acc.append([i, overall_accuracy / 5])
    print(avg / 10)
    X, y, features = load_train_data(i)
    test_x = load_test_data(features)
    clf = train_gradient_boost(X, y)
    test_pred = clf.predict(test_x)

    submission_file = [["ID", "Predicted"]]
    for i, prediction in enumerate(test_pred):
        submission_file.append([i + 1, int(prediction)])
    
    with open('submission.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(submission_file)
exit()

''' Decision Tree
Accuracy 0.3333333333333333
Accuracy 0.5416666666666666
Accuracy 0.4166666666666667
Accuracy 0.5
Accuracy 0.7916666666666666
Overall Accuracy:  0.5166666666666667
'''
''' AdaBoost
Accuracy 0.7083333333333334
Accuracy 0.5
Accuracy 0.5416666666666666
Accuracy 0.4583333333333333
Accuracy 0.5833333333333334
Overall Accuracy:  0.5583333333333333
'''
''' KNN
Accuracy 0.625
Accuracy 0.625
Accuracy 0.5
Accuracy 0.4166666666666667
Accuracy 0.5416666666666666
Overall Accuracy:  0.5583333333333333
'''
''' RandomForest
Accuracy 0.5416666666666666
Accuracy 0.5416666666666666
Accuracy 0.7083333333333334
Accuracy 0.4583333333333333
Accuracy 0.5833333333333334
Overall Accuracy:  0.5666666666666667
'''
''' Logistic
Accuracy 0.4166666666666667
Accuracy 0.5833333333333334
Accuracy 0.5833333333333334
Accuracy 0.5
Accuracy 0.6666666666666666
Overall Accuracy:  0.5583333333333333
'''
''' SVC
Accuracy 0.5416666666666666
Accuracy 0.5833333333333334
Accuracy 0.5416666666666666
Accuracy 0.4583333333333333
Accuracy 0.625
Overall Accuracy:  0.55
'''

d_t = train_decision_tree(X, y)
knn = train_knn(X, y)
rand_forest = train_random_forest(X, y)


#Apply 5-Fold Cross Validation
kf = KFold(n_splits = 5)


#To generate test set randomly and split it from the data
#X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2) 

# Apply SVC
#clf = SVC(kernel='rbf', gamma = 100, C = 100).fit(train_x,train_y)
#clf.fit(train_x, train_y)
#pred_y = clf.predict(test_x)


#Apply Decision Tree
#clf_gini = train_using_gini(train_x, train_y)
#pred_y = clf_gini.predict(test_x)

#Apply K means
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(train_x,train_y)
# pred_y = kmeans.predict(test_x)


overall_accuracy = 0;
mean_vector = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = train_decision_tree(X_train, y_train)    
    y_pred = clf.predict(X_test)
    
    # print(np.mean(X_train[:,0]))
    # print(np.mean(X, axis = 1))
    
    

    print("TEST: ",  test_index)
    print('Accuracy {}'.format(accuracy_score(y_test, y_pred)))  
    overall_accuracy += accuracy_score(y_test, y_pred)


clf = train_decision_tree(X, y)
test_pred = clf.predict(test_x)

# Write to submission file
submission_file = [["ID", "Predicted"]]
for i, prediction in enumerate(test_pred):
    submission_file.append([i + 1, int(prediction)])

with open('submission.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(submission_file)

print("Overall Accuracy: ", overall_accuracy / 5)
# print(accuracy_score(train_y,pred_y))