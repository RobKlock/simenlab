#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 19:24:22 2021

@author: Robert Klock
A simulation of a neural network learning a time sequence using adaption rules
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

def piecewise_linear(v, cutoff, bias):
    if ((v - (bias/2)) < 0):
        return 0
    elif ((v - (bias/2) >= 0) and (v - (bias/2) < cutoff)):
        return (v - (bias/2)) / cutoff
    else:
        return 1
    
def graph_pl(v, cutoff, bias):
    f = np.zeros(v.size)
    for i in range (0, v.size):
        if (v[i] - (bias/2) < 0):
            f[i] = 0
        elif (v[i] - (bias/2) >= 0) and (v[i] - (bias/2) < cutoff):
            f[i] = ((v[i] - (bias/2)) / cutoff)
        else:
            f[i] = 1
    return f

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

   
# Setup our time series data, which is a series of zeros with two batches of 1's from
# 20-30 and 50-60
dt = .01
data1 = np.zeros((1,round(300/dt)))
data1[0][round(20/dt):round(40/dt)] = 1
data1[0][round(20/dt):] = 1
#data1[0][round(70/dt):round(90/dt)] = 
interval2 = np.zeros((1,round(300/dt)))
interval2[0][round(40/dt):round(60/dt)] = 1
interval3 = np.zeros((1,round(300/dt)))
interval3[0][round(60/dt):round(120/dt)] = 1
interval4 = np.zeros((1,round(300/dt)))
interval4[0][round(110/dt):round(120/dt)] = 1

stretch = .5
ramp_bias = 0.1
other_term = 1
weights = np.array([[2,     0,  0,  -20,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],     # 1->1, 2->1, 3->1 4->1
                    [.157,  2,  0,  -1,       0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],      # 1->2, 2->2, 3->2
                    [0,     1,  2,    0,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],     # 1->3, 2->3, 3->3
                    [0,     0,  1,    0,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],
                    # 1->4, 2->4, 3->4 ... 
                    [0,     0,  1,    0,      2,  0,  0,-10,    0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,     .157,2,  0,-1,     0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  1,  2, 0,     0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  .9, 0,     0, 0,  0,  0,   0,  0,  0,  0],
                    
                    [0,     0,  0,    0,      0,  0,  1, 0,     2, 0,  0,-10,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0, .0811, 2,  0, -1,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, .9,  2,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  .9,  0,   0,  0,  0,  0],
                    
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  1,  0,   2,  0,  0,  -10],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,  .275,2,  0, -1],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,   0,  1,  2,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,   0,  0,  1,  0]])          
                         
weights2 = np.array([[2,     0,  0,  -20,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],     # 1->1, 2->1, 3->1 4->1
                    [other_term * (ramp_bias / 2) + stretch * (weights[1][0] - (ramp_bias / 2)),  2,  0,  -1,       0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],      # 1->2, 2->2, 3->2
                    [0,     1,  2,    0,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],     # 1->3, 2->3, 3->3
                    [0,     0,  1,    0,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],
                    # 1->4, 2->4, 3->4 ... 
                    [0,     0,  1,    0,      2,  0,  0,-10,    0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      other_term * (ramp_bias / 2) + stretch * (weights[5][4] - (ramp_bias / 2)),2,  0,-1,     0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  1,  2, 0,     0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  .9, 0,     0, 0,  0,  0,   0,  0,  0,  0],
                    
                    [0,     0,  0,    0,      0,  0,  1, 0,     2, 0,  0,-10,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0, other_term * (ramp_bias / 2) + stretch * (weights[9][8] - (ramp_bias / 2)), 2,  0, -1,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, .9,  2,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  .9,  0,   0,  0,  0,  0],
                    
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  1,  0,   2,  0,  0,  -10],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,  other_term * (ramp_bias / 2) + stretch * (weights[13][12] - (ramp_bias / 2)),2,  0, -1],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,   0,  1,  2,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,   0,  0,  1,  0]])  
beta = 1.2
third_unit_beta = 1.1
lmbd = 4
v_hist = np.array([np.zeros(weights.shape[0])]).T 
v_hist_test = np.array([np.zeros(weights.shape[0])]).T 
noise = 0.0
steps = 0 
tau = 1
delta_A = 0
l = np.array([[lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd]]).T   
#l = np.full([ weights.shape[0] ], lmbd).T  
interval_1_slope = weights[1][1]
bias = np.array([[beta, ramp_bias, beta, beta, beta, ramp_bias, beta, beta, beta, ramp_bias, beta, beta, beta, ramp_bias, beta, beta]]).T 
 
v = np.array([np.zeros(weights.shape[0])]).T 
net_in = np.zeros(weights.shape[0])
net_in_test = np.zeros(weights.shape[0])
timer_learn = False  
early = False
for i in range (0, data1.size):
    # No learning in this model, all weights are hard-coded for interval timing
                     
    net_in = weights2 @ v      
    
    # Transfer functions
    net_in[0] = sigmoid(l[0], data1[0][i] + net_in[0], bias[0])    
    net_in[1] = piecewise_linear(net_in[1], interval_1_slope, bias[1])
    net_in[2:5] = sigmoid(l[2:5], net_in[2:5], bias[2:5])
    
    net_in[5] = piecewise_linear(net_in[5], interval_1_slope, bias[5])
    net_in[6:9] = sigmoid(l[6:9], net_in[6:9], bias[6:9])
    
    net_in[9] = piecewise_linear(net_in[9], interval_1_slope, bias[9])
    net_in[10:13] = sigmoid(l[10:13], net_in[10:13], bias[10:13])
    #net_in[10:11] = sigmoid(l[10:11], net_in[10:11], bias[10:11])
    #net_in[12] = sigmoid(l[12], net_in[12], bias[12])
    net_in[13] = piecewise_linear(net_in[13], interval_1_slope, bias[13])
    net_in[14:] = sigmoid(l[14:], net_in[14:], bias[14:])
#    net_in[10:13] = sigmoid(l[10:13], net_in[10:13], bias[10:13])
    #net_in[13] = piecewise_linear(net_in[13], interval_1_slope, bias[13])
    #net_in[13:] = sigmoid(l[13:], net_in[13:], bias[13:])
            
    dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
    v = v + dv            
    v_hist = np.concatenate((v_hist,v), axis=1)
    
activation_plot_xvals = np.arange(0, 300, dt)
plt.figure()
activation_plot_xvals = np.arange(0, 300, dt)
plt.plot(activation_plot_xvals, v_hist[0,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[1,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[2,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[3,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, data1[0])
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["v1","v2", "v3", "v4"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.title("First module of timers")
plt.grid('on')
plt.show()

activation_plot_xvals = np.arange(0, 300, dt)
plt.figure()
activation_plot_xvals = np.arange(0, 300, dt)
plt.plot(activation_plot_xvals, v_hist[4,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[5,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[6,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[7,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, interval2[0])
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["v5", "v6", "v7", "v8", "interval 2"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.title("Second module of timers")
plt.grid('on')
plt.show()

activation_plot_xvals = np.arange(0, 300, dt)
plt.figure()
activation_plot_xvals = np.arange(0, 300, dt)
plt.plot(activation_plot_xvals, v_hist[8,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[9,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[10,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[11,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, interval3[0])
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["v9", "v10", "v11", "v12", "interval 3"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.title("Third module of timers")
plt.grid('on')
plt.show()

activation_plot_xvals = np.arange(0, 300, dt)
plt.figure()
activation_plot_xvals = np.arange(0, 300, dt)
plt.plot(activation_plot_xvals, v_hist[12,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[13,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[14,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[15,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, interval4[0])
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["v13", "v14", "v15", "v16", "interval 4"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.title("Fourth module of timers")
plt.grid('on')
plt.show()

activation_plot_xvals = np.arange(0, 300, dt)
plt.figure()
activation_plot_xvals = np.arange(0, 300, dt)
plt.plot(activation_plot_xvals, v_hist[1,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[2,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[5,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[6,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[9,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[10,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[13,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[14,0:-1], dashes = [2,2]) 
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["timer 1","timer 2", "timer 3", "timer 4"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.title("All timers")
plt.grid('on')
plt.show()

#x_axis_vals = np.arange(-2, 3, dt)
#plt.figure()
#plt.plot(x_axis_vals, graph_sigmoid(l[2], x_axis_vals, bias[2] - v[2]))
#x1 = [0, 3]
#y2 = [0, 1/weights[2,2] * 3]
#plt.plot(x1,y2, label = "strength of unit 2")
#plt.ylim([0,1])
#plt.legend([ "sigmoid internal", "strength of unit 2"], loc = 0)
#plt.title("activation of OUT Unit against sigmoid")
#plt.grid('on')
#plt.show()
#
#x_axis_vals = np.arange(-2, 3, dt)
#plt.figure()
#plt.plot(x_axis_vals, graph_pl(x_axis_vals, pl_slope, bias[1]))
#x1 = [0, 3]
#y2 = [0, 1/weights[1,1] * 3]
#plt.plot(x1,y2, label = "strength of unit 2")
#plt.ylim([-.1,1.1])
#plt.legend([ "piecewise linear", "strength of unit 2"], loc = 0)
#plt.title("activation of RAMP Unit against piecewise linear")
#plt.grid('on')
#plt.show()

#x_axis_vals = np.arange(-2, 3, dt)
#plt.figure()
#plt.plot(x_axis_vals, graph_sigmoid(l[0], x_axis_vals, bias[0] - v[0]))
#x1 = [0, 3]
#y2 = [0, 1/weights[0,0] * 3]
#plt.plot(x1,y2, label = "strength of unit 1")
#plt.ylim([0,1])
#plt.legend([ "sigmoid internal", "strength of unit 2"], loc = 0)
#plt.title("activation of FIRST Unit against sigmoid")
#plt.grid('on')
#plt.show()
#
#fig, axs = plt.subplots(4)
#activation_plot_xvals = np.arange(0, 300, dt)
#fig.suptitle("Unit Activations and Stimulus")
#axs[0].plot(activation_plot_xvals, v_hist[2,0:-1]) 
#axs[0].set_ylabel("OUT Unit Activation")
#axs[0].set_ylim([0,1])
#axs[0].grid('on')
#axs[1].plot(activation_plot_xvals, v_hist[1,0:-1])
#axs[1].set_ylabel("MID Unit Activation")
#axs[1].set_ylim([0,1])
#axs[1].grid('on')
#axs[2].plot(activation_plot_xvals, v_hist[0,0:-1])
#axs[2].set_ylabel("IN Unit Activation")
#axs[2].set_ylim([0,1])
#axs[2].grid('on')
#axs[3].plot(activation_plot_xvals, data1[0])
#axs[3].set_ylabel("Input Stimulus")
#axs[3].set_ylim([0,1])
#plt.xlabel("Time Steps")
#plt.grid('on')
#plt.show()


#activation_plot_xvals = np.arange(0, 300, dt)
#plt.figure()
#activation_plot_xvals = np.arange(0, 300, dt)
#plt.plot(activation_plot_xvals, v_hist_test[0,0:-1], dashes = [2,2]) 
#plt.plot(activation_plot_xvals, v_hist_test[1,0:-1], dashes = [1,1])
#plt.plot(activation_plot_xvals, v_hist_test[2,0:-1], dashes = [1,1])
#plt.plot(activation_plot_xvals, v_hist_test[3,0:-1], dashes = [1,1])
#plt.ylim([0,1])
##plt.plot(v2_v1_diff, dashes = [5,5])
#plt.legend(["v1_learned","v2_learned", "v3_learned", "v4_learned"], loc=0)
#plt.ylabel("activation")
#plt.xlabel("steps")
#plt.grid('on')
#plt.title("Timer after learning")
#plt.show()
