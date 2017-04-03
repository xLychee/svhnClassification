#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:04:35 2017

@author: xlychee
"""

import pandas as pd
import time
import random
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import sys


def curtime():
    return time.asctime(time.localtime(time.time()))
    

print curtime()+" Program begin "

df = pd.read_csv('../train.csv')

train_size = 70000

X = np.array(df.iloc[:train_size,1:])

mu = np.mean(X,axis=0)
sigma = np.std(X,axis=0)
X_train = (X-mu)/sigma
y_train = np.array(df.iloc[:train_size,0]) 

print curtime()+" Training data ready "
print 'Training data size: ', sys.getsizeof(X_train)

df2 = pd.read_csv('../test.csv')

X_test = np.array(df2)

#X_test = random.sample(X_test,100)

X_test = (X_test-mu)/sigma

print curtime() + ' Test data ready '

#test_X = np.array(df.iloc[-100:,1:])

#test_y = np.array(df.iloc[-100:,0])
#test_X = np.array(df.iloc[:100,1:])
#test_y = np.array(df.iloc[:100,0])
#gnb = GaussianNB()

#model= OneVsOneClassifier(SVC())
model = linear_model.LogisticRegression(solver='lbfgs',tol=0.1,multi_class='ovr')

model.fit(X_train,y_train)

print curtime()+" Model fitting done "

y_test=model.predict(X_test)

print curtime()+" prediction done "



#scores = cross_val_score(model, X_norm, y, cv=5)

#model.fit(X_norm,y).predict()
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#print("Number of mislabeled points out of a total %d points : %d" % (test_X.shape[0],(test_y != y_pred).sum()))
