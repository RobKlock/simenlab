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
from scipy.stats import invgauss   

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

def lateUpdateRule(vt, timer_weight, learning_rate, v0=1.0, z = 1, bias = 1):
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
    
    #drift = ((timer_weight * v0) - bias + .5)
    drift = (timer_weight * v0)
    d_A = drift * ((1-vt)/vt)
    ret_weight = timer_weight + d_A
    return ret_weight
    
def earlyUpdateRule(vt, timer_weight, learning_rate, v0=1.0, z = 1, bias = 1):
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
    #drift = ((timer_weight * v0) - bias + .5)
    drift = (timer_weight * v0)
    d_A = drift * ((vt-z)/vt)
    ret_weight = timer_weight - (learning_rate * d_A)
    return ret_weight

def activationAtIntervalEnd(weight, interval_length, c):
    # drift = ((weight * 1) - 1 + .5)
    act = weight * interval_length + (c * math.sqrt(weight) * np.random.normal(0, 1) * math.sqrt(interval_length))
    # height = drift * interval_length
    return float(act)
    

# def main(s0, T=1000, dt = 1, num_events=2):
    
# Establish two probability distributions for P(A|B) and P(B|A)
P_A = [30, 1] #[np.random.randint(20,50), np.random.randint(5,20), 0]
P_AB = [30, 1] #[np.random.randint(20,50), np.random.randint(5,20), 0]

P_BA = [30, 1] #[np.random.randint(30,50), np.random.randint(5,20), 0]

s0 = 0
dt = .1 
y_lim=2
num_events = 500
# Internal noise - timer activation
noise = 0.05
learning_rate = 1
STANDARD_INTERVAL = 200

# Alternatively, use >1 distributions for a compound dist func. Pull all samples at once 
P_AB2 = TM.getSamples(num_events, num_normal = 2, num_exp = 1, num_dists = 3, ret_params = False)
#plt.hist(P_AB2, bins=40, color='black')

# print(getSampleFromParameters(P_AB))
timer_module_1 = TM(timer_weight=.5)
timer_module_2 = TM(timer_weight=.175)

timer_weight_1 = timer_module_1.block
timer_weight_2 = timer_module_2.block

timer_1_running_act = 0
timer_1_ramping_weights = []
timer_2_ramping_weights = []
timer_2_events = []
events = np.zeros(num_events)

timer_1_mu = 10
timer_1_loc = 1
alpha = 0.8

# Establish first event
#events[0] = np.random.normal(P_A[0],P_A[1], 1)
events[0] = np.random.normal(STANDARD_INTERVAL,1, 1)
for i in range (1,num_events):
    if i % 2 == 0:
        # events[i] = events[i-1] + P_AB2[i] 
        #events[i] = events[i-1] + np.random.normal(P_AB[0],P_AB[1], 1)
        #events[i] = events[i-1] + np.random.normal(np.random.randint(20,50),10, 1)
        #events[i] = events[i-1] + np.random.exponential(STANDARD_INTERVAL,1)
        #events[i] = events[i-1] +  np.random.normal(STANDARD_INTERVAL,1, 1)
        events[i] = events[i-1] + invgauss.rvs(1, 200, size=1)
    else:
        #events[i] = events[i-1] + np.random.normal(P_BA[0],P_BA[1], 1)
        events[i] = events[i-1] + np.random.normal(STANDARD_INTERVAL,1, 1)
        timer_2_events.append(np.random.normal(STANDARD_INTERVAL,1, 1)[0])
        
T = events[-1] + 100

S_m = s0
F_m = 0
""" 
Record the ramp activations of every event that occurs
Store the correct ramp slopes in an array
This just helps the framing of getting everything stored


Start to calculate reward based on a reward(gain) function
A function that defines reward, determined by the event duration (a rectangle -- all or nothing, gaussian, etc. Can also
                                                                  favor different outcomes)

Then explore different approaches--what if you acted on the weighted outcome of the previous 5 interactions? 
"""

plt.figure()
plt.suptitle(f'Internal Noise = {noise}')
A = np.zeros(num_events)
colors = ['blue', 'magenta', 'lawngreen', 'orange']
for i in range (0,num_events): 
    """ Plotting """ 
    event = events[i]
   
    if i % 2 == 0:
        # Event is type A
        #print("event A")
        
        #timer_1_running_act = timer_1_running_act + timer_1_value
        if i >= 1:
            #timer_1_value = activationAtIntervalEnd(timer_module_1.timerWeight(), np.random.normal(timer_1_mu, 1, 1), noise)
            timer_1_value = activationAtIntervalEnd(timer_module_1.timerWeight(), event-events[i-1], noise)
            plt.plot([events[i-1],event], [0, timer_1_value], linestyle = "dashed",  c=colors[0], alpha=0.8)
            plt.plot([event], [timer_1_value], marker='o',c=colors[0])
        else:
            timer_1_value = activationAtIntervalEnd(timer_module_1.timerWeight(), event, noise)
            plt.plot([0,event], [0, timer_1_value], linestyle = "dashed", c=colors[0], alpha=0.8)
            plt.plot([event], [timer_1_value], marker='o',c=colors[0])
        
        
        # Update Rules
        if timer_1_value > 1:
            #print("early update rule")
            timer_1_mu = ((1-learning_rate) * 1) + learning_rate*(event-events[i-1])
            timer_weight = earlyUpdateRule(timer_1_value, timer_module_1.timerWeight(), learning_rate)
            timer_1_ramping_weights.append(timer_weight)
            timer_module_1.setTimerWeight(timer_weight)
        else:
            #print("late update rule")
            timer_1_mu = ((1-learning_rate) * 1) + learning_rate*(event-events[i-1])
            timer_weight = lateUpdateRule(timer_1_value, timer_module_1.timerWeight(), learning_rate)
            timer_1_ramping_weights.append(timer_weight)
            timer_module_1.setTimerWeight(timer_weight)
        
        plt.vlines(event, 0,y_lim, label="v", color=colors[0])
    
    else:
        #print("event B")
        # Event is type B
        
        if i >= 1:
            timer_2_value = activationAtIntervalEnd(timer_module_2.timerWeight(), event - events[i-1], noise)
            plt.plot([events[i-1],event], [0, timer_2_value], linestyle="dashed",  c=colors[1], alpha=0.8)
        else:
            timer_2_value = activationAtIntervalEnd(timer_module_2.timerWeight(), event, noise)
            plt.plot([0,event], [0, timer_2_value],  c=colors[1], alpha=0.8)
        
        # Update Rules
        if timer_2_value > 1:
            timer_weight = earlyUpdateRule(timer_2_value, timer_module_2.timerWeight(), learning_rate)
            timer_2_ramping_weights.append(1/timer_weight)
            timer_module_2.setTimerWeight(timer_weight)
        
        else:
            timer_weight = lateUpdateRule(timer_2_value, timer_module_2.timerWeight(), learning_rate)
            timer_2_ramping_weights.append(1/timer_weight)
            timer_module_2.setTimerWeight(timer_weight)
        plt.plot([event], [timer_2_value], c=colors[1], marker='o')
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

plt.figure()
f, (ax1, ax2) = plt.subplots(2, 1)
plt.subplot(211)
ax1 = plt.hist(timer_2_ramping_weights, bins=20, range=(160,240))
plt.subplot(212)
ax2 = plt.hist(timer_2_events, bins=80, range=(160,240))
    
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