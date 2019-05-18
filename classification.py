import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import csv


def load_train_data():
    # Read data
    data = pd.read_csv("data/train.csv")
    X = data.iloc[:,0:595]  #independent columns
    y = data.iloc[:,-1]    #target column i.e price range

    bestfeatures = SelectKBest(score_func=chi2, k="all")
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features
    ft = featureScores.nlargest(200,'Score')
    features = ft.index.values
    values = ft.values   

    print(np.sum(values[:,-1]))
    return X.values[:, features], y, features


def load_test_data(features):
    # Read data
    data = pd.read_csv("data/test.csv")
    X = data.iloc[:,0:595]  #independent columns
    y = data.iloc[:,-1]    #target column i.e price range

    return X.values[:, features]

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


def get_pca_results(X, Y, x_test, y_test):
    results = []
    for i in range(1,90):
        # pca = PCA(n_components=i)  
        pca = PCA(n_components=i)
        x_train = pca.fit_transform(X)

        clf = train_using_gini(X,Y) 
        clf.fit(x_train, Y)

        # clf = SVC(kernel='sigmoid').fit(train_x,train_y)
        # clf.fit(x_train, train_y)

        # clf = train_using_gini(x_train, train_y)

        # Predicting the Test set results
        x_t = pca.transform(x_test)
        y_pred = clf.predict(x_t)

        results.append([i, accuracy_score(y_test, y_pred)])
    results.sort(key = lambda results: results[1], reverse=True) 
    return results


# Load Datas
X, y, features = load_train_data()
test_x = load_test_data(features)

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

for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	# results = get_pca_results(X_train, y_train, X_test, y_test)
	#print(results)

	# pca = PCA(n_components=results[0][0])
	# pca = PCA(n_components=results[0][0])
	# x_train = pca.fit_transform(X_train)

	clf = train_using_gini(X_train, y_train)
	# x_test = pca.transform(X_test)
    
	
	y_pred = clf.predict(X_test)
	print("TEST: ",  test_index)
	print('Accuracy {}'.format(accuracy_score(y_test, y_pred)))  
	overall_accuracy += accuracy_score(y_test, y_pred)


# pca = PCA(n_components=results[0][0])
# x_train = pca.fit_transform(X)
clf = train_using_gini(X, y)
# test_x = pca.transform(test_x)
test_pred = clf.predict(test_x)

# Write to submission file
submission_file = [["ID", "Predicted"]]
for i, prediction in enumerate(test_pred):
    submission_file.append([i + 1, int(prediction)])

with open('submission.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(submission_file)


print("Overall Accuracy: " , overall_accuracy/5)
# print(accuracy_score(train_y,pred_y))