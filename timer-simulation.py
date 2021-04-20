#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:36:22 2021

@author: Rob Klock

Model of situations where the world isnt timeable. Driving question is: when do you allocate new timers?

"""
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import math
import sys

sys.path.append('/Users/robertklock/Documents/Class/Fall20/SimenLab/simenlab')


fig, ax = plt.subplots(1, 1)
mean, var, skew, kurt = stats.norm.stats(moments='mvsk')
x = np.linspace(stats.norm.ppf(0.01),
                stats.norm.ppf(0.99), 100)
ax.plot(x, stats.norm.pdf(x),
       'r-', lw=3, alpha=.3, label='norm pdf')
rv = stats.norm()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
vals = stats.norm.ppf([0.001, 0.5, 0.999])
np.allclose([0.001, 0.5, 0.999], stats.norm.cdf(vals))
r = stats.norm.rvs(size=1000)
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()
## generate the data and plot it for an ideal normal curve## x-axis for the plot
x_data = np.arange(-5, 20, 0.001)## y-axis as the gaussian
y_data1 = stats.norm.pdf(x_data, 3, 3)## plot data
#y_data2 = stats.norm.pdf(x_data, 15, 6)
y_data = y_data1
plt.plot(x_data, y_data)
plt.show()

def plot_curves(bias, lambd = 4, slf_excitation = 2, v = 0):
    x_axis_vals = np.arange(-2, 3, dt)
    plt.plot(x_axis_vals, graph_sigmoid(lambd, x_axis_vals, bias - v))
    x1 = [0, 3]
    y2 = [0, 1/slf_excitation * 3]
    plt.plot(x1,y2, label = "strength of module 1 OFF")
    plt.ylim([0,1])
    plt.legend([ "sigmoid internal", "strength of unit 4"], loc = 0)
    plt.title("activation of 4 Unit against sigmoid BOT")
    plt.grid('on')
    plt.show() 

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

distributions = {
        "pBA": np.random.normal(100, .1),
        "pCA": np.random.normal(100, .1),
        "pCB": np.random.normal(),
        "pBC": np.random.normal()}
   
# Setup our time series data, which is a series of zeros with two batches of 1's from
# 20-30 and 50-60
dt = .1

''' Establish Intervals '''
data1 = np.zeros((1,round(800/dt)))
data1[0][round(distributions["pBA"]/dt):round((distributions["pBA"] + 20)/dt)] = 1
data1[0][round(distributions["pBA"] + 20/dt):] = 0
interval1 = np.zeros((1,round(300/dt)))
interval1[0][round(20/dt):round(40/dt)] = 1
interval2 = np.zeros((1,round(300/dt)))
interval2[0][round(40/dt):round(60/dt)] = 1
interval3 = np.zeros((1,round(300/dt)))
interval3[0][round(60/dt):round(120/dt)] = 1
interval4 = np.zeros((1,round(300/dt)))
interval4[0][round(117/dt):round(127/dt)] = 1

stretch = .4
ramp_bias = 0.1

weights = np.array([[2,     0,  0,   -1,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],     # 1->1, 2->1, 3->1 4->1
                    [.157,  2,  0,   -1,       0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],      # 1->2, 2->2, 3->2
                    [0,     1,  2,   -1,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],     # 1->3, 2->3, 3->3
                    [0,     0,  1,    2,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],
                     
                    [0,     0,  1,    0,      2,  0,  0,-1,    0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,     .157,2,  0,-1,     0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  1,  2, -1,     0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  1, 2,     0, 0,  0,  0,   0,  0,  0,  0],
                    
                    [0,     0,  0,    0,      0,  0,  1, 0,     2, 0,  0,-1,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0, .0811, 2,  0, -1,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 1,  2, -1,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  1,  2,   0,  0,  0,  0],
                    
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  1,  0,   2,  0,  0,  -1],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,  .275,2,  0, -1],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,   0,  1,  2,  -1],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,   0,  0,  1,  2]])          
                         

stretched_weights = np.array([[2,     0,  0,  -1,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],     # 1->1, 2->1, 3->1 4->1
                    [(ramp_bias / 2) + stretch * (weights[1][0] - (ramp_bias / 2)),  2,  0,  -1,       0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],      # 1->2, 2->2, 3->2
                    [0,     1,  2,    -1,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],     # 1->3, 2->3, 3->3
                    [0,     0,  1,    2,      0,  0,  0,  0,    0, 0,  0,  0,   0,  0,  0,  0],
                    
                    [0,     0,  1,    0,      2,  0,  0,-1,    0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      (ramp_bias / 2) + stretch * (weights[5][4] - (ramp_bias / 2)),2,  0,-1,     0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  1,  2, -1,     0, 0,  0,  0,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  1, 2,     0, 0,  0,  0,   0,  0,  0,  0],
                    
                    [0,     0,  0,    0,      0,  0,  1, 0,     2, 0,  0,-1,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     (ramp_bias / 2) + stretch * (weights[9][8] - (ramp_bias / 2)), 2,  0, -1,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 1,  2,  -1,   0,  0,  0,  0],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  1,  2,   0,  0,  0,  0],
                    
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  1,  0,   2,  0,  0,  -1],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,  (ramp_bias / 2) + stretch * (weights[13][12] - (ramp_bias / 2)),2,  0, -1],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,   0,  1,  2,  -1],
                    [0,     0,  0,    0,      0,  0,  0, 0,     0, 0,  0,  0,   0,  0,  1,  2]])  
beta = 1.2
inhibition_unit_bias = 1.4
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
bias = np.array([[beta, ramp_bias, beta, inhibition_unit_bias, beta, ramp_bias, beta, inhibition_unit_bias, beta, ramp_bias, beta, inhibition_unit_bias, beta, ramp_bias, beta, inhibition_unit_bias]]).T 
 
v = np.array([np.zeros(weights.shape[0])]).T 
net_in = np.zeros(weights.shape[0])
net_in_test = np.zeros(weights.shape[0])
timer_learn = False  
early = False
for i in range (0, data1.size):
    # No learning in this model, all weights are hard-coded for interval timing
    if i == 1:
        x_axis_vals = np.arange(-2, 3, dt)
        plt.figure()
        plt.plot(x_axis_vals, graph_sigmoid(l[3], x_axis_vals, bias[3] - v[3]))
        x1 = [0, 3]
        y2 = [0, 1/weights[3,3] * 3]
        plt.plot(x1,y2, label = "strength of module 1 OFF")
        plt.ylim([0,1])
        plt.legend([ "sigmoid internal", "strength of unit 4"], loc = 0)
        plt.title("activation of 4 Unit against sigmoid BOT")
        plt.grid('on')
        plt.show() 
        
             
    # net_in = weights @ v      
    net_in = stretched_weights @ v
    
    # Transfer functions
    net_in[0] = sigmoid(l[0], data1[0][i] + net_in[0], bias[0])    
    net_in[1] = piecewise_linear(net_in[1], interval_1_slope, bias[1])
    net_in[2:5] = sigmoid(l[2:5], net_in[2:5], bias[2:5])
    
    net_in[5] = piecewise_linear(net_in[5], interval_1_slope, bias[5])
    net_in[6:9] = sigmoid(l[6:9], net_in[6:9], bias[6:9])
    
    net_in[9] = piecewise_linear(net_in[9], interval_1_slope, bias[9])
    net_in[10:13] = sigmoid(l[10:13], net_in[10:13], bias[10:13])
   
    net_in[13] = piecewise_linear(net_in[13], interval_1_slope, bias[13])
    net_in[14:] = sigmoid(l[14:], net_in[14:], bias[14:])

    dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
    v = v + dv            
    v_hist = np.concatenate((v_hist,v), axis=1)
    
plt.figure()
activation_plot_xvals = np.arange(0, 800, dt)
plt.plot(activation_plot_xvals, v_hist[1,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, data1[0])
plt.plot(activation_plot_xvals, v_hist[2,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[5,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[6,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[9,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[10,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[13,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[14,0:-1], dashes = [2,2]) 
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["timer 1", "module 1 output switch","timer 2", "module 2 output switch", "timer 3", "module 3 output switch", "timer 4", "module 4 output switch"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.title("All timers")
plt.grid('on')
plt.show()
    