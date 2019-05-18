import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

data = pd.read_csv("data/train.csv")
X = data.iloc[:,0:595]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

# model = ExtraTreesClassifier()
# model.fit(X,y)
# print(model.feature_importances_) 
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(20).plot(kind='barh')
# plt.savefig('ft_selection.png')


#apply SelectKBest class to extract top 10 best features
# bestfeatures = SelectKBest(score_func=chi2, k="all")
# fit = bestfeatures.fit(X,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))  #print 10 best features
# ft = featureScores.nlargest(300,'Score')
# indexes = ft.index.values
# values = ft.values   
# print(indexes)
# print(np.sum(values[:,-1]))


X = X.values

svm = LinearSVC()
# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(svm, 10)
rfe = rfe.fit(X, y)
# print summaries for the selection of attributes
print(rfe.ranking_[:10])