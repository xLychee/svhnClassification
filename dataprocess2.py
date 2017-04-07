#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:49:03 2017

@author: xlychee
"""


import pandas as pd
import numpy as np

df = pd.read_csv('../extra.csv')

X = np.array(df.iloc[:,1:])
X = X.reshape(32, 32, 3, -1).transpose(3,0,1,2)
y = np.array(df.iloc[:,0])

m = X.shape[0]
assert m==y.shape[0]
mask = np.random.choice(m, 60000)
X = X[mask]
y = y[mask]

np.save('../extra_X',X)
np.save('../extra_y',y)