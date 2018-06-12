#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 20:22:27 2018

@author: yurio
"""

# Import
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

# Load irisdewfre dataset
from sklearn.datasets import load_iris
from sklearn import metrics


# Instantiate
iris = load_iris()

# Create training and feature
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state=0, 
                                                    test_size=0.25)


# 1. Instantiate
# default criterion=gini
# you can swap to criterion=entropy
dtc = DecisionTreeClassifier(random_state=0, criterion='entropy')

# 2. Fit
dtc.fit(X_train, y_train)

# 3. Predict, there're 4 features in the irisdewfre dataset
y_test_pred_class = dtc.predict(X_test)
y_train_pred_class = dtc.predict(X_train)

result_test=metrics.accuracy_score(y_test, y_test_pred_class)
result_train=metrics.accuracy_score(y_train, y_train_pred_class)

print('accuracy on train dataset', result_train)
print('accuracy on test dataset', result_test)