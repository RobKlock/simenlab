#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 22:23:39 2021

@author: Robert Klock
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
    elif (((v - (bias) + .5)> 0) and ((v - (bias) + .5) < 1)):
        return ((v - (bias) + .5))
    else:
        return 1
    
def graph_pl(v, bias, c):
    f = np.zeros(v.size)
    for i in range (0, v.size):
        if ((v[i] - (bias)) <= 0):
            f[i] = 0
        elif ((v[i] - (bias))) > 0 and ((v[i] - (bias)) < 1):
            f[i] = ((v[i] - (bias)))
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
        "pBA": np.random.normal(100, 5, 1)[0],
        "pCA": np.random.normal(100, 5, 1)[0],
        "pCB": np.random.normal(),
        "pBC": np.random.normal()}
   
# Setup our time series data, which is a series of zeros with two batches of 1's from
# 20-30 and 50-60
dt = .1
total_duration = 300
''' Establish Events '''
data1 = np.zeros((1,round(total_duration/dt)))
data1[0][0:round(4/dt)] = 1
event1 = np.zeros((1,round(total_duration/dt)))
event1[0][round(events["pBA"]/dt)] = 1
event2 = np.zeros((1,round(total_duration/dt)))
event2[0][round(events["pCA"]/dt)] = 1


stretch = 1
ramp_bias = 1
lmbd = 4
weights = np.array([[2,     0,  0,   -.4,      0,  0,  0,  0],     # 1->1, 2->1, 3->1 4->1
                    [.5485272872630207,    1,  0,   -.4,       0,  0,  0,  0],      # 1->2, 2->2, 3->2
                    [0,     .5, 2,    0,      0,  0,  0,  0],     # 1->3, 2->3, 3->3
                    [0,     0,  1,    2,      0,  0,  0,  0],
                     
                    [0,     0,  0,    0,      0,  0,  0, -.5],
                    [0,     0,  0,    0,     .4, 1,  0, -.5],
                    [0,     0,  0,    0,      0,  .5,  2, -.5],
                    [0,     0,  0,    0,      0,  0,  1, 2]])         
                         
 
beta = 1.2
inhibition_unit_bias = 1.25
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
l = np.full((weights.shape[0],1), lmbd) 
interval_1_slope = 1
bias = np.array([[beta, ramp_bias, beta, inhibition_unit_bias, beta, ramp_bias, beta, inhibition_unit_bias]]).T 
 
net_in_test = np.zeros(weights.shape[0])
timer_learn_1 = True  
timer_learn_2 = True  
early_1 = False
early_2 = False
for i in range (0, data1.size):
    net_in = weights @ v      
    # Transfer functions
    net_in[0] = sigmoid(l[0], data1[0][i] + net_in[0], bias[0])    
    net_in[1] = piecewise_linear(net_in[1], bias[1])
    net_in[2] = sigmoid(l[2], net_in[2], bias[2])
    #net_in[2:4] = sigmoid(l[2:4], net_in[2:4], bias[2:4])    
    net_in[3] = sigmoid(l[3], net_in[3], bias[3])
    net_in[4] = sigmoid(l[4], net_in[4], bias[4])
    net_in[5] = piecewise_linear(net_in[5], bias[5])
    net_in[6:8] = sigmoid(l[6:8], net_in[6:8], bias[6:8])
    dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
    v = v + dv                
    v_hist = np.concatenate((v_hist,v), axis=1)
    z = .99
    
    """=== Early Timer Update Rules ==="""
    early_threshold = .99
    
    # Force v3 to be off during learning
    if timer_learn_1 and i < round(events["pBA"]/dt):
        v[2] = 0

    
    if (v[1] >= z) and timer_learn_1 == True:
        early_1 = True
        if i < round(events["pBA"]/dt):
            # We're still in the interval, so we keep updating
            # Drift for PL assuming a slope of 1
            A = (weights[1][0] * v[0] + (weights[1][3] * v[3]) - bias[1] + .5)
            dA = (-((A**2)/z) * dt)
    
            weights[1][0] = weights[1][0] + dA  
        else:
            #print("early")
            timer_learn_1 = False
            #print("else: ",weights[1][0])               
#    if i < round(events["pBA"]/dt):
#        timer_learn_1 = False
    #print("i: ", i, "weight: ",weights[1][0])         
    """=== Late Timer Update Rules ==="""                
    if (v[1] > 0) and (i > round(events["pBA"]/dt)) and (timer_learn_1 == True) and (not early_1):
#         If we hit our target late
#         Do the late update
        timer_learn_1 = False
        z = .99
        Vt = net_in[1][-1]
        print("Vt in late update: ", Vt)
        print("V[0] in late update: ", v[0])
        v[2] = 1
        # drift = (weights[1][0] - bias[1] + .5)
        """=== LOOK HERE! Timer Update Rules ==="""
        drift = ((weights[1][0] * v[0]) - bias[1] + .5)
        d_A = drift * ((z-Vt)/Vt)
        weights[1][0] = weights[1][0] + d_A
        #print(weights[1][0])
        #print("late")
        #print("new weights", weights[1][0])
###

x_axis_vals = np.arange(-2, 3, dt)
plt.figure()
plt.plot(x_axis_vals, graph_pl(x_axis_vals, bias[1], 1))
x1 = [0, 3]
y2 = [0, 1/weights[1,1] * 3]
plt.plot(x1,y2, label = "strength of unit 2")
plt.ylim([-.1,1.1])
plt.legend([ "piecewise linear", "strength of unit 2"], loc = 0)
plt.title("activation of RAMP Unit against piecewise linear")
plt.grid('on')
plt.show()

plt.figure()
activation_plot_xvals = np.arange(0, total_duration, dt)
plt.plot(activation_plot_xvals, v_hist[0,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[1,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, event1[0])
#plt.plot(activation_plot_xvals, event2[0])
plt.plot(activation_plot_xvals, v_hist[2,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[3,0:-1], dashes = [2,2]) 
#plt.plot(activation_plot_xvals, v_hist[5,0:-1], dashes = [1,1])
#plt.plot(activation_plot_xvals, v_hist[6,0:-1], dashes = [2,2]) 
#plt.plot(activation_plot_xvals, v_hist[7,0:-1], dashes = [2,2])  
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["first unit", "timer 1", "event 1", "module 1 output switch",  "module 1 inhibition unit"], loc=0)
plt.ylabel("activation")
plt.xlabel("Time units")
plt.title("Timer before learning")
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

# Test the results of the learning rule 
for i in range (0, data1.size):
    net_in_test = weights @ v_test      
    # Transfer functions
    net_in_test[0] = sigmoid(l[0], data1[0][i] + net_in_test[0], bias[0])    
    net_in_test[1] = piecewise_linear(net_in_test[1], bias[1])
    net_in_test[2:5] = sigmoid(l[2:5], net_in_test[2:5], bias[2:5])
    
    net_in_test[5] = piecewise_linear(net_in_test[5], bias[5])
    net_in_test[6:9] = sigmoid(l[6:9], net_in_test[6:9], bias[6:9])

    dv = (1/tau) * ((-v_test + net_in_test) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
    v_test = v_test + dv            
    v_hist_test = np.concatenate((v_hist_test,v_test), axis=1)
    
plt.figure()
activation_plot_xvals = np.arange(0, total_duration, dt)
plt.plot(activation_plot_xvals, v_hist_test[0,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist_test[1,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, event1[0])
# plt.plot(activation_plot_xvals, event2[0])
plt.plot(activation_plot_xvals, v_hist_test[2,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist_test[3,0:-1], dashes = [2,2]) 
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["unit 1", "timer 1", "event 1", "output unit", "inhibition unit"], loc=0)
plt.ylabel("activation")
plt.xlabel("Time Units")
plt.title("Timer after learning")
plt.grid('on')
plt.show()

def multiple_trials(n=5, s = 1, noise = 0):
    ramp_bias = 1
    dt = .1
    stretch_factor = s
    total_duration = round(300 / stretch_factor)
    weights = np.array([[2,     0,  0,   -.4,      0,  0,  0,  0],     # 1->1, 2->1, 3->1 4->1
                        [0.7,  1,  0,   -.4,       0,  0,  0,  0],      # 1->2, 2->2, 3->2
                        [0,     .5,  2,   -.4,      0,  0,  0,  0],     # 1->3, 2->3, 3->3
                        [0,     0,  1,    2,      0,  0,  0,  0],
                         
                        [0,     0,  1,    0,      2,  0,  0,-.5],
                        [0,     0,  0,    0,     .7, 1,  0,-.5],
                        [0,     0,  0,    0,      0,  .5,  2, -.5],
                        [0,     0,  0,    0,      0,  0,  1, 2]]) 
    
    stretch_weights =  np.array([[2,     0,  0,   -.5,      0,  0,  0,  0],     # 1->1, 2->1, 3->1 4->1
                        [.55,  1,  0,   -.5,       0,  0,  0,  0],      # 1->2, 2->2, 3->2
                        [0,     .5,  2,   -.5,      0,  0,  0,  0],     # 1->3, 2->3, 3->3
                        [0,     0,  1,    2,      0,  0,  0,  0],
                         
                        [0,     0,  1,    0,      2,  0,  0,-.5],
                        [0,     0,  0,    0,     .55, 1,  0,-.5],
                        [0,     0,  0,    0,      0,  .5,  2, -.5],
                        [0,     0,  0,    0,      0,  0,  1, 2]]) 
    #np.array([[2,     0,  0,   -1,      0,  0,  0,  0],     # 1->1, 2->1, 3->1 4->1
#                        [0.3,  1,  0,   -1,       0,  0,  0,  0],      # 1->2, 2->2, 3->2
#                        [0,     .55,  2,   -1,      0,  0,  0,  0],     # 1->3, 2->3, 3->3
#                        [0,     0,  1,    2,      0,  0,  0,  0],
#                         
#                        [0,     0,  1,    0,      2,  0,  0,-1],
#                        [0,     0,  0,    0,     .03, 1,  0,-1],
#                        [0,     0,  0,    0,      0,  .55,  2, -1],
#                        [0,     0,  0,    0,      0,  0,  1, 2]]) 
    plt.figure()
    activation_plot_xvals = np.arange(0, total_duration, dt)
    for i in range(n):
        events = {
        "pBA": np.random.normal(70, 5, 1)[0] / s,
        "pCA": np.random.normal(140, 5, 1)[0] / s,
        "pCB": np.random.normal(),
        "pBC": np.random.normal()}
        ''' Establish Events '''
        data1 = np.zeros((1,round(total_duration/dt)))
        data1[0][0:round(4/dt)] = 1
        event1 = np.zeros((1,round(total_duration/dt)))
        event1[0][round(events["pBA"] / stretch_factor /dt)] = 1
        event2 = np.zeros((1,round(total_duration/dt)))
        event2[0][round(events["pCA"] / stretch_factor /dt)] = 1      
                                 
         
        beta = 1.2
        inhibition_unit_bias = 1.5
        lmbd = 4
        v_hist = np.array([np.zeros(weights.shape[0])]).T 
        noise = noise
        tau = 1
        l = np.array([[lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd]]).T   
        #l = np.full([ weights.shape[0] ], lmbd).T  
        bias = np.array([[beta, ramp_bias, beta, inhibition_unit_bias, beta, ramp_bias, beta, inhibition_unit_bias]]).T 
         
        v = np.array([np.zeros(weights.shape[0])]).T 
        net_in = np.zeros(weights.shape[0])
        timer_learn_1 = True  
        timer_learn_2 = True  
        early_1 = False
        early_2 = False
        stretched = False if s == 1 else True
        for j in range (0, data1.size):   

            net_in = weights @ v if not stretched else stretch_weights @ v  

            # Transfer functions
            net_in[0] = sigmoid(l[0], data1[0][j] + net_in[0], bias[0])    
            net_in[1] = piecewise_linear(net_in[1], bias[1])
            net_in[2:5] = sigmoid(l[2:5], net_in[2:5], bias[2:5])
            net_in[5] = piecewise_linear(net_in[5], bias[5])
            net_in[6:8] = sigmoid(l[6:8], net_in[6:8], bias[6:8])
        
            dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
            v = v + dv            
            v_hist = np.concatenate((v_hist,v), axis=1)
            v[4:] = 0
            z = .99
            ''' Module 1 
                Early Update Rule '''
            if (v[1] >= z) and timer_learn_1 == True:
                early_1 = True
                if j < round(events["pBA"]/dt):
                    # We're early
                    if not stretched:
                        print("early update multiple trials no stretch")
                        A = (weights[1][0] * v[0] + (weights[1][3] * v[3]) - bias[1] + .5)
                        dA = (-((A**2)/z) * dt)
                        weights[1][0] = weights[1][0] + dA  
                        
                    else:
                        print("early update multiple trials")
                        drift = ((ramp_bias + stretch * (stretch_weights[1][0] - ramp_bias)) - bias[1] + .5)
                        d_A = (- (drift ** 2)/z) * dt
                        stretch_weights[1][0] = stretch_weights[1][0] + d_A  
                else:
                    timer_learn_1 = False
            
            ''' Module 1 
                Late Update Rule '''
            if (net_in[1][-1] > 0) and (j > round(events["pBA"]/dt/s)) and (timer_learn_1 == True) and (not early_1):
                # If we hit our target late
                # Do the late update
                
                timer_learn_1 = False
                z = net_in[0][-1]
                z = .99
                Vt = net_in[1][-1]
                if not stretched:
                    drift = ((weights[1][0] * v[0]) - bias[1] + .5) 
                    d_A = drift * ((z-Vt)/Vt)
                    weights[1][0] = weights[1][0] + d_A
                else:
                    drift = (((ramp_bias + stretch * (stretch_weights[1][0] - ramp_bias)) * v[0]) - bias[1] + .5) 
                    d_A = drift * ((z-Vt)/Vt)
                    stretch_weights[1][0] = stretch_weights[1][0] + d_A
            
            ''' Module 2
                Early Update Rule 
            if (v[5] >= z) and timer_learn_2 == True:
                early_2 = True
                if i < round(events["pCA"]/dt / s):
                    if not stretched:
                        # We're still in the interval, so we keep updating
                        A = (weights[5][4] * v[4] + (weights[4][5] * v[5]) - bias[5] + .5)
                        dA = (-((A**2)/z) * dt)
                        weights[5][4] = weights[5][4] + dA    
                        
                    else: 
                        # We're still in the interval, so we keep updating
                        drift = (weights[5][4] - bias[5] + .5) 
                        d_A = (- (drift ** 2)/z) * dt
                        weights[5][4] = weights[5][4] + d_A  
                else:
                    timer_learn_2 = False
                    
             Module 2 
                Late Update Rule 
            if (v[5] > 0) and (i > round(events["pCA"]/dt)) and (timer_learn_2 == True) and (not early_2):
                # If we hit our target late
                # Do the late update
                timer_learn_2 = False
                z = net_in[5][-1]
                z = .99
                Vt = net_in[5][-1]
                if not stretched:
                    drift = (weights[5][4] * v[4]) - ramp_bias + .5
                    d_A = drift * ((z-Vt)/Vt)
                    weights[5][4] = weights[5][4] + d_A
                else:
                    drift = (weights[5][4] * v[4]) - ramp_bias + .5
                    d_A = drift * ((z-Vt)/Vt)
                    weights[5][4] = weights[5][4] + d_A
                    '''
       
        plt.plot(activation_plot_xvals, event1[0], 'k', alpha = .6)
        plt.plot(activation_plot_xvals, event2[0], 'k', alpha = .6)      
        plt.plot(activation_plot_xvals, v_hist[1,0:-1], 'b', dashes = [2,2]) 
        plt.plot(activation_plot_xvals, v_hist[5,0:-1], 'r', dashes = [2,2]) 
        plt.ylim([0,1])
     
        #plt.plot(v2_v1_diff, dashes = [5,5])
    plt.legend(["event 1", "event 2", "timer 1", "timer 2"], loc=0)
    plt.ylabel("activation")
    plt.xlabel("steps")
    plt.title("Timer behavior multiple trials")
    plt.grid('on')
    plt.show()


