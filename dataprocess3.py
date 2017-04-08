#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:30:30 2017

@author: xlychee
"""


import numpy as np
import pandas as pd

df = pd.read_csv('../extra.csv')

X = np.array(df.iloc[:,1:])
X = X.reshape(32, 32, 3, -1).transpose(3,0,1,2)
y = np.array(df.iloc[:,0])
print "extra data:", X.shape

m = X.shape[0]
assert m==y.shape[0]
mask = np.random.choice(m, 120000)
X2 = X[mask]
y2 = y[mask]


X1 = np.load('../train_X.npy')
y1 = np.load('../train_y.npy')

#X2 = np.load('../extra_X.npy')
#y2 = np.load('../extra_y.npy')

X3 = np.vstack((X1,X2))
y3 = np.hstack((y1,y2))
assert X3.shape[0] == y3.shape[0]
print y3.shape

np.save('../new_X.npy',X3)
np.save('../new_y.npy',y3)
