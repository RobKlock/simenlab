#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 21:58:21 2021

@author: Robert Klock

Let S_m be the set of state s the model has observed so far
Let F_m be the set of time distribution F(tau|s,s') in the model so far
Let lambda be the learning rate 
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from timer_module import TimerModule as TM

#print(TM.getSamples())

def getSampleFromParameters(params):
    """
    Parameters
    ----------
    params : TYPE array
        Loc, Scale, Dist TYPE (0 normal, 1 exp) in that order.

    Returns 
    -------
    a sample from provided distribution

    """
    if params[2] == 0:
        return np.random.normal(params[0], params[1], 1)
    else:
        return np.exponential(params[1], 1)

# def earlyUpdateRule()

def lateUpdateRule(vt, timer_weight, v0=1, z = .99, bias = 1):
    """
    Parameters
    ----------
    v0: activation of IN unit
    z: desired activation, threshold
    Vt: timer unit activation
    bias: bias of timer unit
    timer_weight: the weight of the timer

    Returns 
    -------
    The corrected timer weight for the associated event

    """
    
    drift = ((timer_weight * v0) - bias + .5)
    d_A = drift * ((z-vt)/vt)
    ret_weight = timer_weight + d_A
    return ret_weight
    
def earlyUpdateRule(vt, timer_weight, v0=1, z = .99, bias = 1):
    """
    Parameters
    ----------
    v0: activation of IN unit
    z: desired activation, threshold
    Vt: timer unit activation
    bias: bias of timer unit
    timer_weight: the weight of the timer

    Returns 
    -------
    The corrected timer weight for the associated event

    """
    
    """=== LOOK HERE! Timer Update Rules ==="""
    drift = ((timer_weight * v0) - bias + .5)
    d_A = drift * ((vt-z)/vt)
    ret_weight = timer_weight - d_A
    return ret_weight

def activationAtIntervalEnd(weight, interval_length):
    drift = ((weight * 1) - 1 + .5)
    height = drift * interval_length
    return float(height)
    

# def main(s0, T=1000, dt = 1, num_events=2):
    
# Establish two probability distributions for P(A|B) and P(B|A)
P_AB = [np.random.randint(20,30), np.random.randint(10,15), 0]
P_BA = [np.random.randint(20,30), np.random.randint(10,15), 0]

s0 = 0
T = 50
dt = .1 
num_events = 2
event_at = np.random.normal(P_AB[0], P_AB[1], 1)


# print(getSampleFromParameters(P_AB))
timer_module_1 = TM(timer_weight=.534)
timer_module_2 = TM(timer_weight=.534)

timer_weight_1 = timer_module_1.block
timer_weight_2 = timer_module_2.block

# Loop through A, B, and C 
# Each is relative to the other 
# Once the event is pulled, calculate the early or late update rule 
# based on what the timer would have activated at (its current weight) and update
# accordingly 


# Learn to predict the sequences 
# Focus on getting te timers to record the correct event 
# Then the fun stuff comes with what do with the collected data
# print(t)
# Time is zeros with 1's at events 
# time = np.zeros(round(T / dt))
# time[round(event_time) : round(event_time + (1 / dt))] = 1
# Begin time 
# for i in range (0, T):
    
    
S_m = s0
F_m = 0
# for i in range (1, T):
    # Observe tau_t, s_t
    # if (s_t-1, s_t) is in Fm then
    #   Update F(tau_t | s_t-1, s_t)
    # else
    #    F(tau_i | s_t-1, st) = IG(tau_t, ??)
    #    F_m = F_m union F(tau_t | s_t-1)

# def Update(s, tau, s_prime):
   # let mu, lambda, be the parameters of F(tau_t | s_t-1, s_t)
   # mu = (1-alpha) * mu + alpha(tau)
   # lambda = 
# main(0, 1000)
# Idea: have three timers that keep track of the upper and lower deviation of the 
# event

timer_value = activationAtIntervalEnd(timer_module_1.timerWeight(), event_at)
t2v = activationAtIntervalEnd(timer_module_2.timerWeight(), event_at+1)                                  
plt.figure()
plt.subplot(121)
activation_plot_xvals = np.arange(0, T, dt)
plt.plot([event_at], [timer_value], 'ro')
plt.plot([0,event_at], [0, timer_value])
plt.plot([event_at], [t2v], 'go')
plt.plot([0,event_at], [0, t2v], '--g')
#plt.plot(activation_plot_xvals, v_hist[0,0:-1], dashes = [2,2]) 
plt.vlines(event_at, 0,1)
plt.ylim([0,1])
plt.xlim([0,T])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(['timer activation = {:.2f}'.format(timer_value), "timer activation",
           "early timer act.", "early timer act"], loc=0)
plt.ylabel("activation")
plt.xlabel("Time units")
plt.title("Timer")
plt.grid('on')
#plt.show()

if timer_value > 1:
    timer_weight = earlyUpdateRule(timer_value, timer_module_1.timerWeight())
    timer_module_1.setTimerWeight(timer_weight)
    
    t2 = earlyUpdateRule(t2v, timer_module_2.timerWeight())
    timer_module_2.setTimerWeight(t2)
else:
    timer_weight = lateUpdateRule(timer_value, timer_module_1.timerWeight())
    timer_module_1.setTimerWeight(timer_weight)
    
    t2 = lateUpdateRule(t2v, timer_module_2.timerWeight())
    timer_module_2.setTimerWeight(t2)

timer_value = activationAtIntervalEnd(timer_module_1.timerWeight(), event_at)
t2v = activationAtIntervalEnd(timer_module_2.timerWeight(), event_at-1)                                  

plt.subplot(122)
activation_plot_xvals = np.arange(0, T, dt)
plt.plot([event_at], [timer_value], 'ro')
plt.plot([0,event_at], [0, timer_value])
plt.plot([event_at], [t2v], 'go')
plt.plot([0,event_at], [0, t2v], '--g')
#plt.plot(activation_plot_xvals, v_hist[0,0:-1], dashes = [2,2]) 
plt.vlines(event_at, 0,1)
plt.ylim([0,1])
plt.xlim([0,T])
#plt.plot(v2_v1_diff, dashes = [5,5])
plt.legend(['timer activation = {:.2f}'.format(timer_value), "timer activation",
           "early timer act.", "early timer act"], loc=0)
plt.ylabel("activation")
plt.xlabel("Time units")
plt.title("Timer")
plt.grid('on')
plt.show()

""" Unfortunately I dont think this code is relevant anymore, but I'm including
it down here in case it becomes relevant again 
# Establish Events
event_probs = np.random.rand(num_events - 1) 
# Add 0 and 1 to that array
event_probs = np.append(event_probs, 0)
event_probs = np.append(event_probs, 1)
event_probs = np.sort(event_probs)
event_weights = np.zeros(num_events)

for i in range (0, event_weights.size):
    event_weights[i]=(event_probs[i + 1] - event_probs[i])
    
# Re calculate the events so instead of having [.25, .25, .25, .25]
# you have [.25, .50, .75, 1]
event_weights = np.sort(event_weights)
print(event_weights)
event_weights_for_sampling = event_weights

for i in range (0, event_weights_for_sampling.size - 1):
    event_weights_for_sampling[i+1] = event_weights_for_sampling[i+1] + event_weights_for_sampling[i]
print("sampling")
print(event_weights_for_sampling)
# Roll a dice to determine which event happens 
dice_roll = np.random.rand()
event_distributon = 0

for i in range(0, event_weights_for_sampling.size):
   if dice_roll <= event_weights_for_sampling[i]:
       event_distribution = i + 1 
       break
print(dice_roll)
print(event_distribution)
        
event_time = TM.getSamples()[0]
print(event_time)
"""