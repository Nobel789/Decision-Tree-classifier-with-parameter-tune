#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:22:43 2023

@author: myyntiimac
"""

#dicission tree classifier
#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("/Users/myyntiimac/Desktop/Social_Network_Ads.csv")
df.head()

#define variable
X = df.iloc[:,[2,3]].values
X
Y = df.iloc[:,-1]

#spliting the variable
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size = 0.20,random_state = 0 )
# Training the Naive Bayes model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="gini",splitter="best",max_depth=2,min_samples_split=3,min_samples_leaf=2) 
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
ac
#check the bias
bias = classifier.score(X_train, y_train)
bias
#check the variance
bias = classifier.score(X_train, y_train)
bias

#we find the bias is .99 in default, lets change some parameter in clasifier model and check its decrease or not
#(criterion="gini",splitter="random",max_depth=2,min_sample_split=2,min_samples_leaf=2)		
#(criterion="gini",splitter="random",max_depth=2,min_sample_split=2,min_samples_leaf=2)		
