#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 20:30:52 2018

@author: yurio
"""

# Import
from sklearn.ensemble import RandomForestClassifier
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

rfc = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=5)

# 2. Fit
rfc.fit(X_train, y_train)

# 3. Predict, there're 4 features in the irisdewfre dataset
y_test_pred_class = rfc.predict(X_test)
y_train_pred_class = rfc.predict(X_train)

result_test=metrics.accuracy_score(y_test, y_test_pred_class)
result_train=metrics.accuracy_score(y_train, y_train_pred_class)

print('accuracy on train dataset', result_train)
print('accuracy on test dataset', result_test)