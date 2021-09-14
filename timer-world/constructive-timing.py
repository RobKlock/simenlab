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
import random
from timer_module import TimerModule as TM
from labellines import labelLine, labelLines

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

def lateUpdateRule(vt, timer_weight, learning_rate, v0=1, z = .99, bias = 1):
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
    ret_weight = timer_weight + (learning_rate * d_A)
    return ret_weight
    
def earlyUpdateRule(vt, timer_weight, learning_rate, v0=1, z = .99, bias = 1):
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
    ret_weight = timer_weight - (learning_rate * d_A)
    return ret_weight

def activationAtIntervalEnd(weight, interval_length):
    drift = ((weight * 1) - 1 + .5)
    height = drift * interval_length
    return float(height)
    

# def main(s0, T=1000, dt = 1, num_events=2):
    
# Establish two probability distributions for P(A|B) and P(B|A)
P_A = [np.random.randint(20,50), np.random.randint(5,20), 0]
P_AB = [np.random.randint(20,50), np.random.randint(5,20), 0]
P_BA = [np.random.randint(30,50), np.random.randint(5,20), 0]

s0 = 0
dt = .1 
y_lim=2
num_events = 200
noise = 0.0
learning_rate = 1

# print(getSampleFromParameters(P_AB))
timer_module_1 = TM(timer_weight=.534)
timer_module_2 = TM(timer_weight=.534)

timer_weight_1 = timer_module_1.block
timer_weight_2 = timer_module_2.block

timer_1_running_act = 0

events = np.zeros(num_events)
# Establish first event
events[0] = np.random.normal(P_A[0],P_A[1], 1)
for i in range (1,num_events):
    if i % 2 == 0:
        events[i] = events[i-1] + np.random.normal(P_AB[0],P_AB[1], 1)
    else:
        events[i] = events[i-1] + np.random.normal(P_BA[0],P_BA[1], 1)
        
# (events)
T = events[-1] + 100
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
    
    
""" 
To add noise, take the event, calculate the 
timer activation at the event, and add noise based on a Gaussian that has a variance 
based on the amount of time thats gone by
Va  = kT
k = the noise parameter, squared

Once the noise is applied, you apply the learning update rule to that 

Try out a two timer sequence
"""

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
plt.figure()
plt.suptitle(f'Centers = {P_AB[0]}, {P_BA[0]} Spreads = {P_AB[1]}, {P_BA[1]}')
A = np.zeros(num_events)
colors = ['blue', 'magenta', 'lawngreen', 'orange']
for i in range (0,num_events): 
    """ Plotting """ 
    event = events[i]
    learning_rate = learning_rate
   
    # plt.subplot(121)
    if i % 2 == 0:
        # Event is type A
        timer_1_value = activationAtIntervalEnd(timer_module_1.timerWeight(), event) + np.random.normal(0.01, .01 * (noise*noise)*event, 1)
        plt.plot([event], [timer_1_value], marker='o',c=colors[0])
        timer_1_running_act = timer_1_running_act + timer_1_value
        if i >= 1:
            plt.plot([events[i-1],event], [0, timer_1_value], linestyle = "dashed",  c=colors[0], alpha=0.8)
        else:
            plt.plot([0,event], [0, timer_1_value], linestyle = "dashed", c=colors[0], alpha=0.8)
        
        
        # Update Rules
        if timer_1_value > 1:
            timer_weight = earlyUpdateRule(timer_1_value, timer_module_1.timerWeight(), learning_rate)
            timer_module_1.setTimerWeight(timer_weight)
        else:
            timer_weight = lateUpdateRule(timer_1_value, timer_module_1.timerWeight(), learning_rate)
            timer_module_1.setTimerWeight(timer_weight)
        
        plt.vlines(event, 0,y_lim, label="v", color=colors[0])
    
    else:
        # Event is type B
        timer_2_value = activationAtIntervalEnd(timer_module_2.timerWeight(), event) + np.random.normal(0.01, .01 * (noise*noise)*event, 1)         
        plt.plot([event], [timer_2_value], c=colors[1], marker='o')
        
        if i >= 1:
            plt.plot([events[i-1],event], [0, timer_2_value], linestyle="dashed",  c=colors[1], alpha=0.8)
        else:
            plt.plot([0,event], [0, timer_2_value],  c=colors[1], alpha=0.8)
        
        # Update Rules
        if timer_2_value > 1:
            timer_weight = earlyUpdateRule(timer_2_value, timer_module_2.timerWeight(), learning_rate)
            timer_module_2.setTimerWeight(timer_weight)
        
        else:
            timer_weight = lateUpdateRule(timer_2_value, timer_module_2.timerWeight(), learning_rate)
            timer_module_2.setTimerWeight(timer_weight)
        plt.vlines(event, 0,y_lim, label="v", color=colors[1])
            
    if y_lim>1:
            plt.hlines(1, 0, T, alpha=0.2, color='black')
            
    if i == num_events-1:
        print(f"timer 1 average activation: {timer_1_running_act/(num_events/2)} ")
    
    plt.ylim([0,y_lim])
    plt.xlim([0,T])
    plt.ylabel("activation")
    plt.xlabel("Time units")
    plt.title("Timer")
    plt.grid('on')
"""        
    # Wire up two timers for two distinct events 
    event_a = np.random.normal(P_AB[0], P_AB[1], 1)
    event_b = event_a + np.random.normal(P_BA[0], P_BA[1], 1)
    A[i] = event_a
    
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    marker = random.randint(0,11)
    
    timer_1_value = activationAtIntervalEnd(timer_module_1.timerWeight(), event_a) + np.random.normal(0.01, .01 * (noise*noise)*event_a, 1)
    timer_2_value = activationAtIntervalEnd(timer_module_2.timerWeight(), event_b) + np.random.normal(0.01, .01 * (noise*noise)*event_b, 1)                                
    
    plt.subplot(121)
    plt.plot([event_a], [timer_1_value], marker='o',c=color)
    plt.plot([0,event_a], [0, timer_1_value],  c=color, alpha=0.8)
    plt.plot([event_b], [timer_2_value], c=color, marker='o')
    plt.plot([event_a, event_b], [0, timer_2_value], c=color, linestyle='dashed')
    
    if y_lim>1:
        plt.hlines(1, 0, T, alpha=0.5, linestyle="dashed", color='black')
    
    plt.vlines(event_a, 0,y_lim, label="v")
    plt.vlines(event_b, 0,y_lim, label="v", linestyle="dashed", color='r')
    plt.ylim([0,y_lim])
    plt.xlim([0,T])
    #plt.plot(v2_v1_diff, dashes = [5,5])
    #plt.legend(['timer activation = {:.2f}'.format(timer_1_value), "timer activation",
               #"early timer act.", "early timer act"], loc=0)
    # plt.legend(['timer activation = {:.2f}'.format(timer_1_value), "timer activation"],loc=0)
    plt.ylabel("activation")
    plt.xlabel("Time units")
    plt.title("Timer")
    plt.grid('on')
    #labelLines(plt.gca().get_lines(), align=False, fontsize=14)
    #plt.show()
 
    if timer_1_value > 1:
        timer_weight = earlyUpdateRule(timer_1_value, timer_module_1.timerWeight())
        timer_module_1.setTimerWeight(timer_weight)
        
    else:
        timer_weight = lateUpdateRule(timer_1_value, timer_module_1.timerWeight())
        timer_module_1.setTimerWeight(timer_weight)
        
        
    if timer_2_value > 1:
        timer_weight = earlyUpdateRule(timer_2_value, timer_module_2.timerWeight())
        timer_module_2.setTimerWeight(timer_weight)
        
    else:
        timer_weight = lateUpdateRule(timer_2_value, timer_module_2.timerWeight())
        timer_module_2.setTimerWeight(timer_weight)
    
    
    timer_1_value = activationAtIntervalEnd(timer_module_1.timerWeight(), event_a) + np.random.normal(0.01, .01 * (noise*noise)*event_a, 1)
    timer_2_value = activationAtIntervalEnd(timer_module_2.timerWeight(), event_b) + np.random.normal(0.01, .01 * (noise*noise)*event_b, 1)                               
    
    plt.subplot(122)
    plt.plot([event_a], [timer_1_value], marker='o', c=color)
    plt.plot([0,event_a], [0, timer_1_value], c=color, alpha=0.8)
    
    plt.plot([event_b], [timer_2_value], c=color, marker='o')
    plt.plot([event_a, event_b], [0, timer_2_value], c=color, linestyle='dashed')
    
    if y_lim>1:
        plt.hlines(1, 0, T, alpha=0.5, linestyle="dashed", color='black')
    "Early Timer Plots Below"
    plt.vlines(event_a, 0,y_lim)
    plt.vlines(event_b, 0,y_lim, label="v", linestyle="dashed", color='r')
    plt.ylim([0,y_lim])
    plt.xlim([0,T])
    #plt.plot(v2_v1_diff, dashes = [5,5])
    #plt.legend(['timer activation = {:.2f}'.format(timer_1_value), "timer activation",
               #"early timer act.", "early timer act"], loc=0)
    # plt.legend(['timer activation = {:.2f}'.format(timer_1_value), "timer activation"],loc=0)
    #plt.legend()
    plt.ylabel("activation")
    plt.xlabel("Time units")
    plt.title("Timer")
    plt.grid('on')
    """

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