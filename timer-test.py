#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:12:57 2021

@author: Robert Klock
Timer update rule verification for single module
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys

sys.path.append('/Users/robertklock/Documents/Class/Fall20/SimenLab/simenlab')

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

def piecewise_linear(v, bias):
    if ((v - (bias) + .5) <= 0):
        return 0
    elif (((v - (bias + .5))> 0) and ((v - (bias + .5)) < 1)):
        return ((v - (bias)))
    else:
        return 1
    
def graph_pl(v, bias, c):
    f = np.zeros(v.size)
    for i in range (0, v.size):
        if ((v[i] - (bias))/c < 0):
            f[i] = 0
        elif ((v[i] - (bias))/c) >= 0 and ((v[i] - (bias))/c < 1):
            f[i] = ((v[i] - (bias))/c)
        else:
            f[i] = 1
    return f

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

pBA = np.random.normal(100, 10, 1)
pCA = np.random.normal(300, 10, 1)
pCB = np.random.normal(100, 10, 1)
pBC = np.random.normal(100, 10, 1)

events = {
        "pBA": np.random.normal(50, 5, 1)[0],
        "pCA": np.random.normal(100, 5, 1)[0],
        "pCB": np.random.normal(),
        "pBC": np.random.normal()}
   
# Setup our time series data, which is a series of zeros with two batches of 1's from
# 20-30 and 50-60
dt = .1
total_duration = 300
''' Establish Events '''
data1 = np.zeros((1,round(total_duration/dt)))
data1[0][0:round(events["pBA"]/dt)] = 1
event1 = np.zeros((1,round(total_duration/dt)))
event1[0][round(events["pBA"]/dt)] = 1
event2 = np.zeros((1,round(total_duration/dt)))
event2[0][round(events["pCA"]/dt)] = 1


stretch = 1
ramp_bias = 1
lmbd = 4
weights = np.array([[2,     0,  0,   0],         # 1->1, 2->1, 3->1 4->1
                    [.8,    2,  0,   0],     # 1->2, 2->2, 3->2
                    [0,     1,  2,   0],        # 1->3, 2->3, 3->3
                    [0,     0,  1,    2]])               
                         
 
beta = 1.2
inhibition_unit_bias = .5
third_unit_beta = 1.1
v = np.array([np.zeros(weights.shape[0])]).T 
v_test = np.array([np.zeros(weights.shape[0])]).T 
v_hist = np.array([np.zeros(weights.shape[0])]).T 
v_hist_test = np.array([np.zeros(weights.shape[0])]).T 
net_in = np.zeros(weights.shape[0])
noise = 0.00
steps = 0 
tau = 1
delta_A = 0
l = np.array([[lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd]]).T   
#l = np.full([ weights.shape[0] ], lmbd).T  
interval_1_slope = 1
bias = np.array([[beta, ramp_bias, beta, inhibition_unit_bias, beta, ramp_bias, beta, inhibition_unit_bias]]).T 
 
net_in_test = np.zeros(weights.shape[0])
timer_learn_1 = True  
timer_learn_2 = True  
early_1 = False
early_2 = False
SN1 = 0
for i in range (0, data1.size):
    z = 1.2
#    if (i == round(events["pBA"]/dt)) and (timer_learn_1 == True) and (not early_1):
##         If we hit our target late
##         Do the late update
#        print("old weights", weights[1][0])
#        Sn = weights[1][0]
#        B = ramp_bias
#        Vt = v[1][-1]
#        print("VT: ", Vt)
#        A = z - Vt
#        drift = (Sn - B + .5)
#        dS = drift * (A/Vt)
##        drift = Sn - B + .5
##        d_A = drift * ((z-Vt)/Vt)
#        weights[1][0] = weights[1][0]+ dS
#        timer_learn_1 = False
#        print("new weights", weights[1][0])
    net_in = weights @ v      
    # Transfer functions
    net_in[0] = sigmoid(l[0], data1[0][i] + net_in[0], bias[0])    
    net_in[1] = sigmoid(l[1], net_in[1], bias[1])
    net_in[2:4] = sigmoid(l[2:4], net_in[2:4], bias[2:4])

    dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
    v = v + dv            
    v_hist = np.concatenate((v_hist,v), axis=1)
    
    if (v[1] > .8) and timer_learn_1 == True:
        early_1 = True
        print('early')
        if i < round(events["pBA"]/dt):
            # We're still in the interval, so we keep updating
            drift = (weights[1][0] - ramp_bias) + .5
            d_A = (- ((drift ** 2)/z)) * dt
            weights[1][0] = weights[1][0] + d_A
        else:
            timer_learn_1 = False
            print(weights[1][0])

x_axis_vals = np.arange(-2, 3, dt)
plt.figure()
plt.plot(x_axis_vals, graph_sigmoid(l[1], x_axis_vals, bias[1] - v_hist[1][0]))
plt.plot(x_axis_vals, graph_sigmoid(l[1], x_axis_vals, bias[1] - v_hist[1][round(events["pBA"]/dt) - 100]))
plt.plot(x_axis_vals, graph_sigmoid(l[1], x_axis_vals, bias[1] - v_hist[1][round(events["pBA"]/dt)]), dashes=[2,2])
plt.plot(x_axis_vals, graph_sigmoid(l[1], x_axis_vals, bias[1] - v_hist[1][round(events["pBA"]/dt) + 100]))
x1 = [0, 3]
y2 = [0, 1/weights[1,1] * 3]
plt.plot(x1,y2, label = "strength of inh unit")
plt.ylim([0,1])
plt.legend(["strength of inh start", "strength of inh unit -100 event", "strength of inh unit event","strength of inh unit +100 event", "inh unit reference line"], loc = 0)
plt.title("activation of inhibition Unit against sigmoid")
plt.grid('on')
plt.show() 

plt.figure()
activation_plot_xvals = np.arange(0, total_duration, dt)
plt.plot(activation_plot_xvals, v_hist[0,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[1,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, event1[0])
plt.plot(activation_plot_xvals, event2[0])
plt.plot(activation_plot_xvals, v_hist[3,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[2,0:-1], dashes = [2,2]) 
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["first unit", "timer 1", "event 1", "event 2", "module 1 inhibition unit", "module 1 output switch"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.title("All timers before learning")
plt.grid('on')
plt.show()

x_axis_vals = np.arange(-2, 3, dt)
plt.figure()
plt.plot(x_axis_vals, graph_sigmoid(l[3], x_axis_vals, bias[3] - v_hist[3][1]))
plt.plot(x_axis_vals, graph_sigmoid(l[3], x_axis_vals, bias[3] - v_hist[3][round(events["pBA"]/dt) - 100]))
plt.plot(x_axis_vals, graph_sigmoid(l[3], x_axis_vals, bias[3] - v_hist[3][round(events["pBA"]/dt)]), dashes=[2,2])
plt.plot(x_axis_vals, graph_sigmoid(l[3], x_axis_vals, bias[3] - v_hist[3][round(events["pBA"]/dt) + 100]))
x1 = [0, 3]
y2 = [0, 1/weights[3,3] * 3]
plt.plot(x1,y2, label = "strength of inh unit")
plt.ylim([0,1])
plt.legend(["strength of inh start", "strength of inh unit -10 event", "strength of inh unit event","strength of inh unit +10 event", "inh unit reference line"], loc = 0)
plt.title("activation of inhibition Unit against sigmoid")
plt.grid('on')
plt.show() 
        
v = np.array([np.zeros(weights.shape[0])]).T 
v_test = np.array([np.zeros(weights.shape[0])]).T 
v_hist_test = np.array([np.zeros(weights.shape[0])]).T 
net_in_test = np.zeros(weights.shape[0])
data1 = np.zeros((1,round(total_duration/dt)))
data1[0][0:round(4/dt)] = .95
print(SN1)
for i in range (0, data1.size):
    net_in_test = weights @ v_test      
    # Transfer functions
    net_in_test[0] = sigmoid(l[0], data1[0][i] + net_in_test[0], bias[0])    
    net_in_test[1] = sigmoid(l[1], net_in_test[1], bias[1])
    net_in_test[2:5] = sigmoid(l[2:4], net_in_test[2:4], bias[2:4])

    dv = (1/tau) * ((-v_test + net_in_test) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
    v_test = v_test + dv            
    v_hist_test = np.concatenate((v_hist_test,v_test), axis=1)
    
plt.figure()
activation_plot_xvals = np.arange(0, total_duration, dt)
plt.plot(activation_plot_xvals, v_hist_test[0,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist_test[1,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, event1[0])
plt.plot(activation_plot_xvals, event2[0])
plt.plot(activation_plot_xvals, v_hist_test[2,0:-1], dashes = [2,2]) 
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["unit 1", "timer 1", "event 1", "event 2","output unit", "inhibition unit"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.title("All timers after learning")
plt.grid('on')
plt.show()