## Necessary libraries for matrix operations
import pandas as pd
import numpy as np

# Gradient Boost Classifier
from sklearn.ensemble import GradientBoostingClassifier

# Feature Selection Algorithms and Functions
from sklearn.feature_selection import chi2, SelectKBest

# Import csv to write to File
import csv


# Load and return train and test data
def loadData(Xpaths):
    '''
    Input:
        Xpaths (List that contains paths to files)
    Output:
        trainData, testData (pandas object)
    '''
    # Read test and train data
    trainData = pd.read_csv("data/{}.csv".format(Xpaths[0]))
    testData = pd.read_csv("data/{}.csv".format(Xpaths[1]))
    return trainData, testData


# Feature Selection method (Filter)
def filterFeatureSelection(X, y, numOfFeatures):
    ## Calculate feature scores (Score function chi2)
    bestfeatures = SelectKBest(score_func=chi2, k="all")
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    # Concatenate Features and releated Score to Visualize
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']

    # Select Top K Features
    ft = featureScores.nlargest(numOfFeatures, 'Score')
    '''
    Return Feature List (E.g)
    output : [512, 89, 450, 10]
    '''
    features = ft.index.values
    return features


# Find optimal features and train data
def preprocessing(trainData):
    # Set Features to X
    X = trainData.iloc[:, 0:595]

    # Set Target values to Y (labels)
    y = trainData.iloc[:, -1]

    # Drop zero columns and Select top K features with Filter Method
    X = X.drop(["X3", "X31", "X32", "X127", "X128", "X590"], axis=1)
    featureIndexes = filterFeatureSelection(X, y, numOfFeatures=341)

    # Return Samples and Targer with Selected Features
    return X.values[:, featureIndexes], y, featureIndexes


# Train the model
def trainModel(XtrainNew, Ytrain):
    # Initialize Classifier Object with Neccessary Parameters
    gradBoost = GradientBoostingClassifier(
        n_estimators=6,
        learning_rate=1,
        max_features=2,
        max_depth=2,
        random_state=289)

    # Train the Model
    gradBoost = gradBoost.fit(XtrainNew, Ytrain)
    return gradBoost


# Second submission on Kaggle with different params
# Change Selected Best K filters to 281 from 341, Before running this function
def trainModelSecondSubmission(XtrainNew, Ytrain):
    gradBoost = GradientBoostingClassifier(
        n_estimators=3,
        learning_rate=1,
        max_features=2,
        max_depth=2,
        random_state=288)
    # Train the Model
    gradBoost = gradBoost.fit(XtrainNew, Ytrain)
    return gradBoost


# Apply preprocessing to testData
def transformTestSamples(testData, selectedFeatures):
    X = testData.iloc[:, 0:595]
    X = X.drop(["X3", "X31", "X32", "X127", "X128", "X590"], axis=1)
    return X.values[:, selectedFeatures]


# Predict test samples
def predict(model, Xtest):
    '''
    Input:
        Trained model (Classifier Object)
        Test Samples  (DxN Matrix)
    Output:
        Predictions (Dx1 Matrix)
    '''
    return model.predict(Xtest)


def writeOutput(predictedValues):
    '''
    Write predicted values to submission.csv
    1, 0
    2, 1
    3, 0
    '''
    # Create nested lists
    submissionFile = [["ID", "Predicted"]]
    for i, prediction in enumerate(predictedValues):
        submissionFile.append([i + 1, int(prediction)])

    # Write to file
    with open('submission.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(submissionFile)


## Run the program and produce the output
# Load datas
Xpaths = ["train", "test"]
trainData, testData = loadData(Xpaths)

# Feature Selection and Filtering
Xtrain, Ytrain, selectedFeatures = preprocessing(trainData)

# Train the model and assign classifier object
gradientBoost = trainModel(Xtrain, Ytrain)

# Transform and Predict Test Samples
XtestNew = transformTestSamples(testData, selectedFeatures)

# Predict the output label
predictedValues = predict(gradientBoost, XtestNew)

# Write to submission.csv
writeOutput(predictedValues)
