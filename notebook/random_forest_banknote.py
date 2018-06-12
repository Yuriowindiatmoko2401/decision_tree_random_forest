#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:00:05 2018

@author: yurio
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
from sklearn import metrics
import pickle

def load_data(filename, percentfortrain):
    data=np.genfromtxt(filename, delimiter=',')
    size=len(data)
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    for i in range(size):
        x_data = data[i][0:4]
        y_data = data[i][4]
        rn=random.random()
        if rn<percentfortrain:
            #train
             x_train.append(x_data)
             y_train.append(y_data)

            #for j in range(0, 3):
            #    x_train.append(data[i])

        else:
            #test
            x_test.append(x_data)
            y_test.append(y_data)

    return x_train, y_train, x_test, y_test

# Set random seed
np.random.seed(0)

x_train, y_train, x_test, y_test=load_data('data_banknote_authentication.csv', 0.75)

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf.fit(x_train, y_train)

# Create actual english names for the plants for each predicted plant class
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

print ('========features importances=====================')
# Create confusion matrix
# View a list of the features and their importance scores
print (clf.feature_importances_)

accuracy_train = metrics.accuracy_score(y_train, y_train_pred)
print ('accuracy_train',accuracy_train)
accuracy_test = metrics.accuracy_score(y_test, y_test_pred)
print ('accuracy_test',accuracy_test)

# save to file
with open('randomforest_model.mdl', 'wb') as output:
    pickle.dump(clf, output)