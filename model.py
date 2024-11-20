# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 19:40:37 2024

@author: LENOVO
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# loading the dataset to pandas DataFrame
loan_dataset = pd.read_csv("E:\college\sem 6\mini project\project\loanstatus.csv")

type(loan_dataset)

# printing the first 5 rows of the dataframe
loan_dataset.head()

# number of rows and columns
loan_dataset.shape

# statistical measures
loan_dataset.describe()

# number of missing values in each column
loan_dataset.isnull().sum()

# dropping the missing values
loan_dataset = loan_dataset.dropna()

# number of missing values in each column
loan_dataset.isnull().sum()

# number of missing values in each column
loan_dataset.isnull().sum()

# label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

# printing the first 5 rows of the dataframe
loan_dataset.head()

# Dependent column values
loan_dataset['Dependents'].value_counts()

# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# dependent values
loan_dataset['Dependents'].value_counts()

# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

# separating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']

print(X)
print(Y)

X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

print(X.shape, X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')

#training the support Vector Macine model
classifier.fit(X_train,Y_train)

import joblib

# Assuming 'model' is your trained machine learning model
joblib.dump(classifier, 'trained_model.pkl')

print(type(classifier))




































