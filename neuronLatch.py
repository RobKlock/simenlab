#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:49:32 2021

@author: Rob Klock
Initialization of a two-unit closed-loop system that propogates a 1 forever
"""

import numpy as np
import math
import sys
sys.path.append('/Users/robertklock/Documents/Class/Fall20/SimenLab/simenlab')
import perceptron as p
import matplotlib.pyplot as plt

def sigmoid(l, total_input, b):
    f = 1/ (1 + np.exp(-l * (total_input - b)))
    return f

# Setup our time series data, which is a series of zeros with two batches of 1's from
# 20-30 and 50-60
data = np.zeros((1,100))
data[0][20:30] = 1
data[0][50:60] = 1

weights = np.array([[2,  1],     # 1->1, 1->2
                    [-1,  2]])   # 2->1, 2->2
    
beta = 1.2
lmbd = 4
v_hist = np.array([[0, 0]]).T    

steps = 0 
tau = 1
dt = 0.01
l = np.array([[lmbd, lmbd]]).T     
    
bias = np.array([[1, 1]]).T 

v_hist = np.array([[0, 0]]).T    
v = np.array([[0, 0]]).T              

leak = -1
sig_idx= [0,1]

for i in data[0]:
    
    activations = weights @ v # sum of activations, inputs to each unit
    activations[0] = i + activations[0]    
    activations[1] = sigmoid(1, activations[1], bias[1])
    
    dv = (1/tau) * (((-v) + activations) * dt + (np.sqrt(dt) * np.random.normal(0,1, (2,1))) / tau) # add noise using np.random
    v = v + dv
    
    v_hist = np.concatenate((v_hist,v), axis=1)

plt.figure(1)
plt.plot(v_hist[0,:]) 
plt.plot(v_hist[1,:])
plt.legend(["v1","v2"], loc=0)
plt.ylabel("activation")
plt.xlabel("time")
plt.grid('on')
plt.title("Units 1 and 2 Activations")