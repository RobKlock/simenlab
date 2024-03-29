#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:42:01 2021

@author: robklock
"""

# Early Timer Update Rule with new Drift calculation

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import time
from scipy.integrate import odeint
sys.path.append('/Users/robertklock/Documents/Class/Fall20/SimenLab/simenlab')
SMALL_FLOAT = .000000001
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
        if ((v[i] - (bias) + .5) <= 0):
            f[i] = 0
        elif ((v[i] - (bias) + .5)) > 0 and ((v[i] - (bias) + .5) < 1):
            f[i] = ((v[i] - (bias) + .5))
        else:
            f[i] = 1
    return f

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def etr(z,t):
    A = (.7 - .5 + .5)
    #z = .99999
    # A = (timer_weight - timer_bias + .5)
    dAdt = -(A**2)/z
    return dAdt

pBA = np.random.normal(300, 10, 1)
pCA = np.random.normal(300, 10, 1)
pCB = np.random.normal(100, 10, 1)
pBC = np.random.normal(100, 10, 1)

events = {
        "pBA": np.random.normal(250, 0, 1)[0],
        "pCA": np.random.normal(100, 5, 1)[0],
        "pCB": np.random.normal(),
        "pBC": np.random.normal()}

# Setup our time series data, which is a series of zeros with two batches of 1's from
# 20-30 and 50-60
dt = 0.1
total_duration = 300
''' Establish Events '''
data1 = np.zeros((1,round(total_duration/dt)))
ramp_slope = np.zeros((1,round(total_duration/dt)))
event1 = np.zeros((1,round(total_duration/dt)))
event1[0][round(events["pBA"]/dt)] = 1
data1[0][0:round(4/dt)] = 1
event2 = np.zeros((1,round(total_duration/dt)))
event2[0][round(events["pCA"]/dt)] = 1
learning_phase = np.zeros((1,round(total_duration/dt)))
learning_start = 0
learning_end = 0
#learning_range = [0,round(events["pBA"])/dt]

stretch = 1
ramp_bias = 1
lmbd = 4
weights = np.array([[2,     0,  0,   -.4],         # 1->1, 2->1, 3->1 4->1
                    [.6,    1,  0,   -.4],      # 1->2, 2->2, 3->2
                    [0,     .5, 2,   -.4],     #2->3, 3->3
                    [0,     0,  1,    2]])

beta = 1.2
inhibition_unit_bias = 1.2
third_unit_beta = 1.1
v = np.array([np.zeros(weights.shape[0])]).T
v_test = np.array([np.zeros(weights.shape[0])]).T
v_hist = np.array([np.zeros(weights.shape[0])]).T
v_hist_test = np.array([np.zeros(weights.shape[0])]).T

net_in = np.array([np.zeros(weights.shape[0])]).T
net_in_squashed = np.array([np.zeros(weights.shape[0])]).T
net_in_hist = np.array(np.array([np.zeros(weights.shape[0])])).T
noise = 0.00
steps = 0
tau = 1
delta_A = 0
l = np.array([[lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd, lmbd]]).T
#l = np.full([ weights.shape[0] ], lmbd).T
interval_1_slope = 1
bias = np.array([[beta, ramp_bias, beta, inhibition_unit_bias, beta, ramp_bias, beta, inhibition_unit_bias]]).T

net_in_test = np.zeros(weights.shape[0])
timer_learn_1 = False
timer_learn_2 = True
early_1 = False
early_2 = False
SN1 = 0
numpy_z = np.ones([1])
timer_input = np.zeros(weights.shape[0])
for i in range (0, data1.size):
    net_in = weights @ v
    if i % 1000 == 0:
        print("printing timer input at step: ", i, net_in[1])
    # Transfer functions
    net_in_squashed[0] = 1 #sigmoid(l[0], data1[0][i] + net_in[0], bias[0])
    net_in_squashed[1] = piecewise_linear(net_in[1], bias[1])
    #timer_input[i] = net_in[1] - bias[1] + .5
    #net_in[1] = sigmoid(l[1], net_in[1], bias[1])
    net_in_squashed[2:4] = sigmoid(l[2:4], net_in[2:4], bias[2:4])
    dv = (1/tau) * ((-v + net_in_squashed) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
    v = v + dv
    v_hist = np.concatenate((v_hist,v), axis=1)
    net_in_hist = np.concatenate((net_in_hist, net_in), axis=1)
    z = .99 #1 - SMALL_FLOAT
    """=== Early Timer Update Rules ==="""
    #early_threshold = 1
    # Record ramp slope for plotting
    ramp_slope[0][i] = weights[1][0]
    if i < round(events["pBA"])/dt:
        v[2] = 0
        v[0] = 1
    '''
    if i == round(events["pBA"])/dt:
        print("should stop here")
        print("early")
        #timer_learn_1 = False
        print("else: ",weights[1][0])
        print("event: ", events["pBA"])
    '''
    # .50988 is about the target weight
    if i < round(events["pBA"])/dt:

        if (v[1][0] >= z):
            timer_learn_1 = True
            learning_phase[0][i] = 1
        # 5/25 notes
        # net in and weights * activation ought to be the same
        # figure out how to debug effectively using pdb
        # master pdb
        # subtract input from itself from net_in
        if timer_learn_1:
            if learning_start == 0:
                learning_start = i
        #if i < round(events["pBA"]/dt):
            # We're still in the interval, so we keep updating
            # Drift for PL assuming a slope of 1
            # A = Drift, z = threshold
            #A = (net_in[1] - bias[1] + .5)
            #A = (weights[1][0] * v[0] - bias[1] + .5)
            A = (weights[1][0] * v[0] + (weights[1][3] * v[3]) - bias[1] + .5)
            dA = (-((A**2)/z) * dt)

            #print("updating...")
            weights[1][0] = weights[1][0] + dA
            if weights[1][0] < 0:
                weights[1][0] = 0
                print(weights[1][0])
                #input()
            '''
            drift = ((weights[1][0]) - bias[1] + .5)
            d_A = (- (drift ** 2)/z) * dt
            weights[1][0] = weights[1][0] + d_A
            '''
            # print("i: ", i, "weight: ",weights[1][0])
        else:
            '''
            print("not in timer 1")
            print("v1: ", v[1])
            print("timer 1 learn: ", timer_learn_1)
            '''
#    if i < round(events["pBA"]/dt):
#        timer_learn_1 = False
    #print("i: ", i, "weight: ",weights[1][0])
    """=== Late Timer Update Rules ==="""
#    if (v[1] > 0) and (i > round(events["pBA"]/dt)) and (timer_learn_1 == True) and (not early_1):
##         If we hit our target late
##         Do the late update
#        timer_learn_1 = False
#        z = .99
#        Vt = net_in[1][-1]
#        v[2] = 1
#        # drift = (weights[1][0] - bias[1] + .5)
#        """=== LOOK HERE! Timer Update Rules ==="""
#        drift = ((weights[1][0] * v[0]) - bias[1] + .5)
#        d_A = drift * ((z-Vt)/Vt)
#        weights[1][0] = weights[1][0] + d_A
#        print(weights[1][0])
#        print("late")
#        print("new weights", weights[1][0])
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
plt.axvspan(learning_start*dt, round(events["pBA"]), alpha=.2, color='red')
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
net_in = np.zeros(weights.shape[0])
v_hist_test = np.array([np.zeros(weights.shape[0])]).T
#data1 = np.zeros((1,round(total_duration/dt)))
#data1[0][0:round(4/dt)] = .95
print("learned weights: ", weights)
for i in range (0, data1.size):
    v[2] = 0
    v[0] = 1
    net_in = weights @ v
    # Transfer functions
    net_in[0] = sigmoid(l[0], data1[0][i] + net_in[0], bias[0])
    #net_in[1] = piecewise_linear(net_in[1], bias[1])
    net_in[1] = sigmoid(l[1], net_in[1], bias[1])
    net_in[2:4] = sigmoid(l[2:4], net_in[2:4], bias[2:4])
    #net_in[2:4] = piecewise_linear(net_in[2:4], bias[2:4])
    dv = (1/tau) * ((-v + net_in) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
    v = v + dv
    v_hist_test = np.concatenate((v_hist_test,v), axis=1)

'''
for i in range (0, data1.size):
    net_in_test = weights @ v_test
    # Transfer functions
    net_in_test[0] = sigmoid(l[0], data1[0][i] + net_in_test[0], bias[0])
    #net_in_test[1] = sigmoid(l[1], net_in_test[1], bias[1])
    net_in_test[1] = piecewise_linear(net_in_test[1], bias[1])
    v_test[0] = 1
    net_in_test[2:4] = sigmoid(l[2:4], net_in_test[2:4], bias[2:4])

    dv = (1/tau) * ((-v_test + net_in_test) * dt) + (noise * np.sqrt(dt) * np.random.normal(0, 1, (weights.shape[0],1)))  # Add noise using np.random
    v_test = v_test + dv
    v_hist_test = np.concatenate((v_hist_test,v_test), axis=1)
'''
plt.figure()
xvals = np.arange(0, total_duration, dt)
plt.plot(xvals, ramp_slope[0])
# plot net_in 1 - weights[1][0] over time
# plot weights[1][0]*v[0]
#plt.plot(xvals, net_in_hist[1][0:-1], ramp_slope[0])
plt.plot(xvals, event1[0])
#plt.plot(xvals, timer_input[0])
# plt.plot(xvals, learning_start[0])
plt.axvspan(learning_start*dt, round(events["pBA"]), alpha=.2, color='red')
plt.ylim([0,1])
plt.legend(["ramp slope", "net input", "event"])
plt.ylabel("ramp slope")
plt.xlabel("Time Units")
plt.title("Timer weight during trial")
plt.grid('on')
plt.show()

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