#!/usr/bin/env python                                                                                                                                                                                                                       
# coding=utf-8

"""
多种机器学习在titanic数据中的结果比较
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/train.csv"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

## Load library
#sklearn
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, auc, roc_curve
from sklearn import ensemble, linear_model, neighbors, svm ,tree, neural_network
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#load package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Load data
train_original = pd.read_csv('../input/train.csv')
test_original = pd.read_csv('../input/test.csv')
#print(train_original.sample(10))
#print(train_original.index)
#print(train_original.columns)
print(train_original.head())
import sys;sys.exit()
total = [train_original, test_original]
#print(train_original['Salutation'])


## Data Cleaning
for dataset in total:
    dataset['Salutation'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    #print(dataset['Salutation'])
    #print(pd.crosstab(dataset['Salutation'], dataset['Sex']))

#print(pd.crosstab(train_original['Salutation'], train_original['Sex']))

#print(pd.crosstab(test_original['Salutation'], test_original['Sex']))

for dataset in total:
    dataset['Salutation'] = dataset['Salutation'].replace(['Lady', 'Countess', 'Don', 'Dr', 
                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Col', 'Capt'], 'Rare')
    dataset['Salutation'] = dataset['Salutation'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Salutation'] = dataset['Salutation'].replace('Mme', 'Mrs')
    dataset['Salutation'] = pd.factorize(dataset['Salutation'])[0]
    
print(pd.crosstab(train_original['Salutation'], train_original['Sex']))

print(pd.crosstab(test_original['Salutation'], test_original['Sex']))

## Select column
#clean unused variable
train = train_original.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test  = test_original.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#print(train)
total = [train, test]
print(train.shape, test.shape)

## Detect and fill the missing data
# detect the missing data in train dataset
print(train.isnull().sum())

# create function to replace missing data with the median value
def fill_missing_age(dataset):
    for i in range(0, 5):
        median_age = dataset[dataset['Salutation']==i]['Age'].median()
        #print(median_age)
        #dataset['Age'] = dataset['Age'].fillna(median_age)
        dataset['Age'] = dataset.apply(lambda row:median_age if ((row['Salutation']==i) and np.isnan(row['Age'])) else row['Age'], axis=1)
    return dataset

print(pd.unique(train[train['Age'].isnull()]['Salutation']))
#print('########')
train = fill_missing_age(train)
#print(train.isnull().sum())
#import sys;sys.exit()

# Embarked missing cases
train[train['Embarked'].isnull()]
train['Embarked'] = train['Embarked'].fillna('C')

##Detecting the missing data in 'test' dataset is done to get the insight which column consist missing data.
##as it is shown below, there are 2 column which have missing value. 
##they are 'Age' and 'Fare' column. The same function is used in order to filled the missing 'Age' value. 
##missing 'Fare' value is filled by finding the median of 'Fare' value in the 'Pclass' = 3 and 'Embarked' = S.
print(test.isnull().sum())
print(test[test['Age'].isnull()].head())
test = fill_missing_age(test)
print(test[test['Fare'].isnull()])

# creat function to replace missing Fare data with the median
def fill_missing_fare(dataset):
    median_fare = dataset[(dataset['Pclass']==3) & (dataset['Embarked']=='S')]['Fare'].median()
    dataset['Fare'] = dataset['Fare'].fillna(median_fare)
    return dataset
test = fill_missing_fare(test)

## Recheck for misssing data
print(train.isnull().any())
print(test.isnull().any())

## Date Preprocessing
# discretize `Age` feature
for dataset in total:
    dataset.loc[dataset['Age'] <= 9, 'Age'] = 0
    dataset.loc[(dataset["Age"] > 9) & (dataset["Age"] <= 19), "Age"] = 1
    dataset.loc[(dataset["Age"] > 19) & (dataset["Age"] <= 29), "Age"] = 2
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[(dataset["Age"] > 29) & (dataset["Age"] <= 39), "Age"] = 3
    dataset.loc[dataset["Age"] > 39, "Age"] = 4
    
# discretize 'Fare' feature
print(pd.qcut(train['Fare'], 8).value_counts())
for dataset in total:
    dataset.loc[dataset["Fare"] <= 7.75, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.75) & (dataset["Fare"] <= 7.91), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 9.841), "Fare"] = 2
    dataset.loc[(dataset["Fare"] > 9.841) & (dataset["Fare"] <= 14.454), "Fare"] = 3   
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 24.479), "Fare"] = 4
    dataset.loc[(dataset["Fare"] >24.479) & (dataset["Fare"] <= 31), "Fare"] = 5   
    dataset.loc[(dataset["Fare"] > 31) & (dataset["Fare"] <= 69.487), "Fare"] = 6
    dataset.loc[dataset["Fare"] > 69.487, "Fare"] = 7
    
# factorized 2 of the column whic are 'Sex' and 'Embarked'
for dataset in total:
    dataset['Sex'] = pd.factorize(dataset['Sex'])[0]
    dataset['Embarked']= pd.factorize(dataset['Embarked'])[0]
    
print(train.head())


## Checking the correlation between features
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

## Spliting the data
x = train.drop('Survived', axis = 1)
y = train['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=1)


## Performance Comparision
# List of Machine Learning Algorithm(MLA) used.

MLA = [
    #ensemble methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    
    #gaussian processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    ]

## Train the data into the model and calculate the performance
MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)
row_index = 0
for alg in MLA:
    predicted = alg.fit(x_train, y_train).predict(x_test)
    fp, tp, th = roc_curve(y_test, predicted)
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(x_train, y_train), 4)
    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(x_test, y_test), 4)
    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(y_test, predicted)
    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(y_test, predicted)
    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)
    row_index+=1
    
MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    
print(MLA_compare)

## plot ROC curve 
#index = 1
#for alg in MLA:
#    predicted = alg.fit(x_train, y_train).predict(x_test)
#    fp, tp, th = roc_curve(y_test, predicted)
#    roc_auc_mla = auc(fp, tp)
#    MLA.__name = alg.__class__.__name__
#    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC=%.2f)' %(MLA.__name, roc_auc_mla)
#    index += 1
    
#plt.title('ROC Curve comparison')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([0,1])
#plt.ylim([0,1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')    
#plt.show()