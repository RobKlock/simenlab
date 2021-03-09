#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:02:18 2021

@author: Rob Klock
Winner-take all modules for AGI
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

def sq_error(t, o):
    
    # T is target (desired) output
    # O is actual output
    return (t - o) ** 2

def state(unit_index, weights, v):
    activations = weights[unit_index,:] @ v
    return activations[0]

def energy(weights, v):
    return (-1/2) * v.T * weights * v
    
# Setup our time series data, which is a series of zeros with two batches of 1's from
# 20-30 and 50-60
dt = .01
data1 = np.zeros((1,round(300/dt)))
data1[0][round(20/dt):round(30/dt)] = 1


data2 = np.zeros((1,round(300/dt)))
data2[0][round(50/dt):round(60/dt)] = 3

weights = np.array([[0,     0,      0,      0,      0,      0],       # 1->1, 2->1, 3->1, 4->1, 5->1, 6->1
                    [0,     0,      0,      0,      0,      0],       # 1->2, 2->2, 3->2, 4->2, 5->2, 6->2
                    [2.0,   0,      2.0,    -1.0,   -1.0,     0],       # 1->3, 2->3, 3->3, 4->3, 5->3, 6->3
                    [-2.0,  2.0,    -1.0,   2.0,    -1.0,      0],       # 1->4, 2->4, 3->4, 4->4, 5->4, 6->4
                    [0,     -2.0,   -1.0,      -1,     2.0,    0],       # 1->5, 2->5, 3->5, 4->5, 5->5, 6->5
                    [0,     0,      1.0,    1.0,    1.0,    0]])      # 1->6, 2->6, 3->6, 4->6, 5->6, 6->6
    
beta = 1.2 
lmbd = 4
v_hist = np.array([[0, 0,0,0,0,0]]).T    
noise = 0.03
steps = 0 
tau = 1
l = np.array([[lmbd, lmbd,lmbd,lmbd,lmbd, lmbd]]).T     
    
bias = np.array([[1.2, 1.2,beta,beta,beta, beta]]).T 

v_hist = np.array([[0, 0,0,0,0,0]]).T    
v = np.array([[0, 0,0,0,0,0]]).T              
net_in = [0.0,0.0,0.0,0.0,0.0,0.0]
sig_idx= [0,5]

for i in range (0, data1.size):
    
    net_in = weights @ v # sum of activations, inputs to each unit
    net_in[0] = sigmoid(lmbd, data1[0][i], bias[0])    
    net_in[1] = sigmoid(lmbd, data2[0][i], bias[1])
    net_in[2:5] = sigmoid(l[2:5], net_in[2:5], bias[2:5])
    # dv = (1/tau) * ((-v + activations) * dt) # No noise
    dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (6,1)))  # Add noise using np.random
    #dv = (1/tau) * (((-v) + activations) * dt) + (np.sqrt(dt)) / tau) add noise using np.random
    
    v = v + dv
    
    v_hist = np.concatenate((v_hist,v), axis=1)


activation_plot_xvals = np.arange(0, 300, dt)
plt.figure()
activation_plot_xvals = np.arange(0, 300, dt)
plt.plot(activation_plot_xvals, v_hist[0,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[1,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[2,0:-1], dashes = [2,2])
plt.plot(activation_plot_xvals, v_hist[3,0:-1], dashes = [3,3])
plt.plot(activation_plot_xvals, v_hist[4,0:-1], dashes = [2,2])
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["v1","v2","v3","v4", "v5"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.grid('on')
plt.show()

plt.figure()
plt.plot(activation_plot_xvals, v_hist[4,0:-1], dashes = [2,2])
    
fig, axs = plt.subplots(3)
activation_plot_xvals = np.arange(0, 300, dt)
fig.suptitle("Unit Activations and Stimulus")
axs[0].plot(activation_plot_xvals, v_hist[4,0:-1]) 
axs[0].set_ylabel("OUT Unit Activation")
axs[0].set_ylim([0,1])
axs[0].grid('on')
axs[1].plot(activation_plot_xvals, v_hist[3,0:-1])
axs[1].set_ylabel("IN Unit Activation")
axs[1].set_ylim([0,1])
axs[1].grid('on')
axs[2].plot(activation_plot_xvals, data1[0])
axs[2].set_ylabel("Input Stimulus")
axs[2].set_ylim([0,1])
plt.xlabel("Time Steps")
plt.grid('on')