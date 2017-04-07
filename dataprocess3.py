#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:30:30 2017

@author: xlychee
"""


import numpy as np


X1 = np.load('../train_X.npy')
y1 = np.load('../train_y.npy')

X2 = np.load('../extra_X.npy')
y2 = np.load('../extra_y.npy')

X3 = np.vstack((X1,X2))
y3 = np.hstack((y1,y2))
assert X3.shape[0] == y3.shape[0]
print y3.shape

np.save('../new_X.npy',X3)
np.save('../new_y.npy',y3)
