#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:12:25 2017

@author: xlychee
"""

import pandas as pd
import numpy as np

df = pd.read_csv('../train.csv')

X = np.array(df.iloc[:,1:])
X = X.reshape(32, 32, 3, -1).transpose(3,0,1,2)
y = np.array(df.iloc[:,0])

#import matplotlib.pyplot as plt
#plt.imshow(X[2])
np.save('../train_X',X)
np.save('../train_y',y)

#b = np.load('../train_y.npy')


df2 = pd.read_csv('../test.csv')
X = np.array(df2)
X = X.reshape(32, 32, 3, -1).transpose(3,0,1,2)
np.save('../test_X',X)

