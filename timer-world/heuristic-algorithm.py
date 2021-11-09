#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 21:02:41 2021

@author: Rob Klock

Heuristic Algorithm Exploration
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
    #drift = ((timer_weight * v0) - bias + .5)
    drift = (timer_weight * v0)
    d_A = drift * ((vt-z)/vt)
    ret_weight = timer_weight - (learning_rate * d_A)
    return ret_weight

def activationAtIntervalEnd(timer, ramp_index, interval_length, c):
    act = timer.timers[ramp_index] * interval_length
    for i in range (0, len(act)):
        act[i] = act[i] + c * np.sqrt(act[i]) * np.random.normal(0, 1) * math.sqrt(interval_length)
    return act
    

def responseTime(weight, threshold):
    return threshold/weight

def reward(activation, margin=.025):
    # Squared duration of error from event
    if 1 - activation <= margin:
        return 1
    else:
        return 0
    
def score_decay(response_time, event_time):
    #print(f'response_time: {response_time}, event_time: {event_time}')
    diff = event_time - response_time
    #print(f'diff: {diff}')
    if diff <= 0:
        return 0  
    else:
        #return 0.02**(1.0-diff)
        return 2**(-diff/2)

def update_rule(timer_values, timer, timer_indices, v0=1.0, z = 1, bias = 1):
    for idx, value in zip(timer_indices, timer_values):
        if value > 1:
            ''' Early Update Rule '''
            timer_weight = earlyUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
            timer.setTimerWeight(timer_weight, idx)
        else:
            ''' Late Update Rule '''
            timer_weight = lateUpdateRule(value, timer.timerWeight(idx), timer.learningRate(idx))
            timer.setTimerWeight(timer_weight, idx)
        

N_EVENT_TYPES=2 # Number of event types (think, stimulus A, stimulus B, ...)
NUM_EVENTS=40 # Total amount of events across all types
Y_LIM=2 # Plotting limit
NOISE=0.001 # Internal noise - timer activation
LEARNING_RATE=.8
STANDARD_INTERVAL=200
RESPONSE_THRESHOLD=0.95
ALPHA = 0.8
colors = ['b', 'g', 'r', 'c', 'm', 'y']

events_with_type = TM.getSamples(NUM_EVENTS, num_normal = 2, num_exp = 0)
events_with_type = TM.getSamples(NUM_EVENTS, num_normal = 4, num_exp = 0, standard_interval = 200)
event_occurances = (list(zip(*events_with_type))[0])
plt.hist(event_occurances, bins=80, color='black')

events = np.zeros(NUM_EVENTS)
events_with_type[0][0] = event_occurances[0]

for i in range (1,NUM_EVENTS):
     events_with_type[i][0] = events_with_type[i-1][0] + events_with_type[i][0]

# Time axis for plotting        
T = events_with_type[-1][0] + 100

# Timer with 20 ramps, all initialized to be very highly weighted
timer=TM(1,20)

# TODO: Make the initial weight update continuous so we can see the process better

plt.figure()

timer.eventDict()[0] = [0,1,2]
timer.eventDict()[1] = [3,4,5]
free_indices = np.arange(5,20)
# Timers are allocated to an event type based on the timer's eventDict object
# eventDict takes in an event type as a key and gives an array of timer indices 
# for that object as the value

first_event = True

for idx, event in enumerate(events_with_type):
    event_time = event[0]
    event_type = int(event[1])
    
    if event_type not in timer.eventDict():
        # Allocate a new timer for this event type 
        # need protection if we run out of timers 
        timer.eventDict()[event_type] = free_indices[:3]
        free_indices = free_indices[3:]
        
    event_timer_index = timer.eventDict()[event_type]
   
    if first_event:
        first_event= False                   
        timer_value = activationAtIntervalEnd(timer, event_timer_index, event_time, NOISE)
        # TODO: set up response times correctly
        response_time = responseTime(timer.timerWeight(), RESPONSE_THRESHOLD)
        #timer.setScore(ramp_index, timer.getScore(ramp_index) + score_decay(response_time, event_time))
        
        for i in timer_value:
            plt.plot([0,event_time], [0, i], linestyle = "dashed", c=colors[event_type], alpha=0.5)
            plt.plot([event_time], [i], marker='o',c=colors[event_type],  alpha=0.2) 

    else:
        prev_event = events_with_type[idx-1][0]
        timer_value = activationAtIntervalEnd(timer, event_timer_index, event_time - events_with_type[idx-1][0], NOISE)
        response_time = responseTime(timer.timerWeight(), RESPONSE_THRESHOLD)
        #timer.setScore(ramp_index, timer.getScore(ramp_index) + score_decay(response_time, event_time))       
        
        for i in timer_value:
            plt.plot([prev_event,event_time], [0, i], linestyle = "dashed",  c=colors[event_type], alpha=0.5)
            plt.plot([event_time], [i], marker='o',c=colors[event_type], alpha=0.2) 
    
    update_rule(timer_value, timer, event_timer_index)    
    
    # TODO: Rest of the heuristic (scores, reallocation, etc)

    plt.vlines(event, 0,Y_LIM, label="v", color=colors[event_type], alpha=0.5)
    if Y_LIM>1:
        plt.hlines(1, 0, T, alpha=0.2, color='black')
  
    plt.ylim([0,Y_LIM])
    plt.xlim([0,T])
    plt.ylabel("activation")
    plt.xlabel("Time")
    plt.grid('on')
   