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
import matplotlib.pyplot as plt

def sigmoid(l, total_input, b):
    f = 1/ (1 + np.exp(-l * (total_input - b)))
    return f

def graph_sigmoid(l, I, b):
    f=1/(1+np.exp(-l * (I-b)))
    return f
# Setup our time series data, which is a series of zeros with two batches of 1's from
# 20-30 and 50-60
data = np.zeros((1,300))
data[0][20:30] = 1
data[0][50:60] = 1

weights = np.array([[1.5,  -2],     # 1->1, 2->1
                    [1.0,  1.75]])   # 1->2, 2->2
    
beta = 1.2
lmbd = 3.5
v_hist = np.array([[0, 0]]).T    

steps = 0 
tau = 1
dt = .1
l = np.array([[lmbd, lmbd]]).T     
    
bias = np.array([[beta, beta]]).T 

v_hist = np.array([[0, 0]]).T    
v = np.array([[0, 0]]).T              
activations = [0.0,0.0]
sig_idx= [0,1]

for i in data[0]:
    
    activations = weights @ v # sum of activations, inputs to each unit
    activations[0] = i + activations[0]    
    activations[1] = sigmoid(lmbd, activations[1], bias[1])
    dv = (1/tau) * ((-v + activations) * dt * np.sqrt(dt))+ np.random.normal(.5,.1, (2,1))  #add noise using np.random
    #dv = (1/tau) * (((-v) + activations) * dt) + (np.sqrt(dt)) / tau) add noise using np.random
    
    v = v + dv
    
    v_hist = np.concatenate((v_hist,v), axis=1)

fig, axs = plt.subplots(3)
fig.suptitle("Unit Activations and Stimulus")
axs[0].plot(v_hist[1,:]) 
axs[0].set_ylabel("OUT Unit Activation")
axs[0].grid('on')
axs[1].plot(v_hist[0,:])
axs[1].set_ylabel("IN Unit Activation")
axs[1].grid('on')
axs[2].plot(data[0])
axs[2].set_ylabel("Input Stimulus")
plt.xlabel("Time")
plt.grid('on')

x_axis_vals = np.arange(-2, 3, .1)

plt.figure()
plt.plot(x_axis_vals, graph_sigmoid(l[1], x_axis_vals, bias[1] - v[1]))
x1 = [0, 3]
y2 = [0, 1/weights[1,1] * 3]
plt.plot(x1,y2, label = "strength of unit 2")
plt.ylim([0,1])
plt.legend([ "sigmoid internal", "strength of unit 2"], loc = 0)
plt.title("activation of 2 against sigmoid")
plt.grid('on')
plt.show()
