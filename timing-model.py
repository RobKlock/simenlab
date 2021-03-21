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
    if ((v - bias) < 0):
        return 0
    else:
        return (v - bias) / cutoff
    
def graph_pl(v, cutoff, bias):
    f = np.zeros(v.size)
    for i in range (0, v.size):
        if (v[i] - bias < 0):
            f[i] = 0
        elif (v[i] - bias >= 0) and (v[i] - bias < cutoff):
            f[i] = ((v[i] - bias) / cutoff)
        else:
            f[i] = 1
    return f
   
# Setup our time series data, which is a series of zeros with two batches of 1's from
# 20-30 and 50-60
dt = .01
data1 = np.zeros((1,round(300/dt)))
data1[0][round(20/dt):round(40/dt)] = 1
data1[0][round(70/dt):round(90/dt)] = 1
data1[0][round(130/dt):round(135/dt)] = 1


data2 = np.zeros((1,round(300/dt)))
data2[0][round(50/dt):round(60/dt)] = 1

weights = np.array([[2,   0,      0, 0],      # 1->1, 2->1, 3->1
                    [.1,     2,    0, 0],      # 1->2, 2->2, 3->2
                    [0,     0,      0.1, 0],      # 1->3, 2->3, 3->3
                    [0,     0,      0, 0]])      # 1->4, 2->4, 3->4
                         
    
beta = 1.2
third_unit_beta = 1.1
lmbd = 4
v_hist = np.array([[0, 0, 0, 0]]).T    
noise = 0.0
steps = 0 
tau = 1
delta_A = 0
l = np.array([[lmbd, lmbd, lmbd, lmbd]]).T     
pl_slope = weights[1][1]
bias = np.array([[beta, beta - 1.15, beta, beta]]).T 

v_hist = np.array([[0, 0, 0, 0]]).T    
v = np.array([[0.0, 0.0, 0.0, 0.0]]).T              
net_in = [0.0,0.0, 0.0, 0.0]
timer_learn = False  
for i in range (0, data1.size):
    
    net_in = weights @ v # sum of activations, inputs to each unit
    A = 1 / weights[1][0]
    z = 1
    # If the network gets a signal, start learning its duration
    if data1[0][i] == 1:
        timer_learn = True 

    if (data1[0][i] == 0) and (timer_learn == True):
        x_end = net_in[1]
        delta_A = A * ((z - x_end)/x_end)
        timer_learn = False
        weights[1][0] = weights[1][0] + delta_A
        print(delta_A)
    
    
    net_in[0] = sigmoid(l[0], data1[0][i] + net_in[0], bias[0])    
    net_in[1] = piecewise_linear(net_in[1], pl_slope, bias[1])
    net_in[2] = sigmoid(l[2], net_in[2], bias[2])
    
    # dv = (1/tau) * ((-v + activations) * dt) # No noise
    dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (4,1)))  # Add noise using np.random
    #dv = (1/tau) * (((-v) + activations) * dt) + (np.sqrt(dt)) / tau) add noise using np.random
    v = v + dv
    v_hist = np.concatenate((v_hist,v), axis=1)


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
plt.legend([ "piecewise linar", "strength of unit 2"], loc = 0)
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