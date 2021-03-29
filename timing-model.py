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
#data1[0][round(70/dt):round(90/dt)] = 1
#data1[0][round(100/dt):round(120/dt)] = 1


data2 = np.zeros((1,round(300/dt)))
data2[0][round(50/dt):round(60/dt)] = 1

weights = np.array([[2,   0,      0, 0],      # 1->1, 2->1, 3->1
                    [.1,     2,    0, 0],      # 1->2, 2->2, 3->2
                    [0,     0,      0.1, 0],      # 1->3, 2->3, 3->3
                    [0,     0,      0, 0]])      # 1->4, 2->4, 3->4
                         
    
beta = 1.2
ramp_bias = .1
third_unit_beta = 1.1
lmbd = 4
v_hist = np.array([[0, 0, 0, 0]]).T 
v_hist_test = np.array([[0, 0, 0, 0]]).T      
noise = 0.0
steps = 0 
tau = 1
delta_A = 0
l = np.array([[lmbd, lmbd, lmbd, lmbd]]).T     
pl_slope = weights[1][1]
bias = np.array([[beta, ramp_bias, beta, beta]]).T 
 
v = np.array([[0.0, 0.0, 0.0, 0.0]]).T            
net_in = [0.0,0.0, 0.0, 0.0]
net_in_test = [0.0,0.0, 0.0, 0.0]
timer_learn = False  
early = False
for i in range (0, data1.size):
    
    A = 1 / weights[1][0]
    
    # If the network gets a signal, start learning its duration
    if data1[0][i] == 1:
        timer_learn = True 
    
   
    
    if (v[1] >= net_in[0]) and timer_learn == True:
        early = True
        print("early")
        if (data1[0][i] == 1):
            # We're still in the interval, so we keep updating
            drift = (weights[1][0] - bias[1] / 2)
            z = 1.2
            d_A = (- (drift ** 2)/z) * dt
            print(d_A)
            weights[1][0] = weights[1][0] + d_A
            print(weights[1][0])
        else:
            timer_learn = False
            print(weights)
            
            
    net_in = weights @ v # sum of activations, inputs to each unit        
    net_in[0] = sigmoid(l[0], data1[0][i] + net_in[0], bias[0])    
    #net_in[1] = sigmoid(l[1], net_in[0], bias[1])
    net_in[1] = piecewise_linear(net_in[1], pl_slope, bias[1])
    net_in[2] = sigmoid(l[2], net_in[2], bias[2])
    
            
    dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (4,1)))  # Add noise using np.random
    v = v + dv            
    v_hist = np.concatenate((v_hist,v), axis=1)
    
    if (data1[0][i] == 0) and (timer_learn == True) and (not early):
        # If we hit our target late
        # Do the late update
        timer_learn = False
        Sn = weights[1][0]
        B = bias[1]
        z = net_in[0][-1]
        # z = 1 <- Rivest has this for notation sake in their paper
        z = 1.2
        Vt = net_in[1][-1]
        drift = (weights[1][0] - bias[1] / 2)
        d_A = drift * ((z-Vt)/Vt)
        
        weights[1][0] = weights[1][0] + d_A
        # The threshold on ramp-up values is C
        # The input weight to the ramp is w
        # The bias of the ramp unit is \beta
        # Thus the drift imposed by the weight is \epsilon = w - \beta
        # learning_rate * (W_end - beta) * (C - Cn) / Cn) <- Prof Simen's
    
activation_plot_xvals = np.arange(0, 300, dt)
plt.figure()
activation_plot_xvals = np.arange(0, 300, dt)
plt.plot(activation_plot_xvals, v_hist[0,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist[1,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[2,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist[3,0:-1], dashes = [1,1])
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["v1","v2", "v3", "v4"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.title("Timer before learning")
plt.grid('on')
plt.show()

x_axis_vals = np.arange(-2, 3, dt)
plt.figure()
plt.plot(x_axis_vals, graph_sigmoid(l[2], x_axis_vals, bias[2] - v[2]))
x1 = [0, 3]
y2 = [0, 1/weights[2,2] * 3]
plt.plot(x1,y2, label = "strength of unit 2")
plt.ylim([0,1])
plt.legend([ "sigmoid internal", "strength of unit 2"], loc = 0)
plt.title("activation of OUT Unit against sigmoid")
plt.grid('on')
plt.show()

x_axis_vals = np.arange(-2, 3, dt)
plt.figure()
plt.plot(x_axis_vals, graph_pl(x_axis_vals, pl_slope, bias[1]))
x1 = [0, 3]
y2 = [0, 1/weights[1,1] * 3]
plt.plot(x1,y2, label = "strength of unit 2")
plt.ylim([-.1,1.1])
plt.legend([ "piecewise linear", "strength of unit 2"], loc = 0)
plt.title("activation of RAMP Unit against piecewise linear")
plt.grid('on')
plt.show()

x_axis_vals = np.arange(-2, 3, dt)
plt.figure()
plt.plot(x_axis_vals, graph_sigmoid(l[0], x_axis_vals, bias[0] - v[0]))
x1 = [0, 3]
y2 = [0, 1/weights[0,0] * 3]
plt.plot(x1,y2, label = "strength of unit 1")
plt.ylim([0,1])
plt.legend([ "sigmoid internal", "strength of unit 2"], loc = 0)
plt.title("activation of FIRST Unit against sigmoid")
plt.grid('on')
plt.show()

fig, axs = plt.subplots(4)
activation_plot_xvals = np.arange(0, 300, dt)
fig.suptitle("Unit Activations and Stimulus")
axs[0].plot(activation_plot_xvals, v_hist[2,0:-1]) 
axs[0].set_ylabel("OUT Unit Activation")
axs[0].set_ylim([0,1])
axs[0].grid('on')
axs[1].plot(activation_plot_xvals, v_hist[1,0:-1])
axs[1].set_ylabel("MID Unit Activation")
axs[1].set_ylim([0,1])
axs[1].grid('on')
axs[2].plot(activation_plot_xvals, v_hist[0,0:-1])
axs[2].set_ylabel("IN Unit Activation")
axs[2].set_ylim([0,1])
axs[2].grid('on')
axs[3].plot(activation_plot_xvals, data1[0])
axs[3].set_ylabel("Input Stimulus")
axs[3].set_ylim([0,1])
plt.xlabel("Time Steps")
plt.grid('on')
plt.show()

v_test = np.array([[0.0, 0.0, 0.0, 0.0]]).T     
for i in range (0, data1.size):
    
    net_in_test = weights @ v_test # sum of activations, inputs to each unit
    A = 1 / weights[1][0]
    z = .999
    # If the network gets a signal, start learning its duration
    net_in_test[0] = sigmoid(l[0], data1[0][i] + net_in_test[0], bias[0])    
    net_in_test[1] = piecewise_linear(net_in_test[1], pl_slope, bias[1])
    net_in_test[2] = sigmoid(l[2], net_in_test[2], bias[2])
    
    # dv = (1/tau) * ((-v + activations) * dt) # No noise
    dv = (1/tau) * ((-v_test + net_in_test) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (4,1)))  # Add noise using np.random
    #dv = (1/tau) * (((-v) + activations) * dt) + (np.sqrt(dt)) / tau) add noise using np.random
    v_test = v_test + dv
    v_hist_test = np.concatenate((v_hist_test,v_test), axis=1)


activation_plot_xvals = np.arange(0, 300, dt)
plt.figure()
activation_plot_xvals = np.arange(0, 300, dt)
plt.plot(activation_plot_xvals, v_hist_test[0,0:-1], dashes = [2,2]) 
plt.plot(activation_plot_xvals, v_hist_test[1,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist_test[2,0:-1], dashes = [1,1])
plt.plot(activation_plot_xvals, v_hist_test[3,0:-1], dashes = [1,1])
plt.ylim([0,1])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(["v1_learned","v2_learned", "v3_learned", "v4_learned"], loc=0)
plt.ylabel("activation")
plt.xlabel("steps")
plt.grid('on')
plt.title("Timer after learning")
plt.show()


fig, axs = plt.subplots(2)
activation_plot_xvals = np.arange(0, 300, dt)
fig.suptitle("Timer behavior before and after learning")
axs[0].plot(activation_plot_xvals, v_hist[1,0:-1]) 
axs[0].plot(activation_plot_xvals, v_hist[0,0:-1]) 
axs[0].plot(activation_plot_xvals, data1[0][0:]) 
axs[0].set_ylabel("Activation")
axs[0].set_ylim([0,1])
axs[0].grid('on')
axs[1].plot(activation_plot_xvals, v_hist_test[1,0:-1], dashes = [1,1])
axs[1].plot(activation_plot_xvals, v_hist_test[0,0:-1], dashes = [1,1])
axs[1].plot(activation_plot_xvals, data1[0][0:]) 
axs[1].set_ylabel("Activation")
axs[1].set_ylim([0,1])
axs[1].grid('on')
plt.xlabel("Time Steps")
plt.grid('on')
plt.show()