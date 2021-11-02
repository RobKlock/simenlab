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
   
        
"""
Do next:
Generalize this out to have more complex worlds
Mixture Distributions
Examine a few cases where distributions are spread out, or unpredictable
vs very predictable
Look at the rewards the models earn and get a feel for it 

Reward function: Decay exponentially backwards from the actual event time
You get rewarded based on how close you are to that 

Goal: Hit the timing threshold at the time of the event
Either correct completely or some proportion less than 1

Fixed/Peak Interval: You just get rewarded for the first press
Every time you press before the event, you lose a little bit of points. But the
first time after you get a lot of rewards

2nd: You get rewarded based on how correctly you respond
Two thresholds for beat the clock and DRL (differential reinforcement of low rates of responding)

Look at how accuracy drops as the standard deviation drops or increases


Notes 10/23
Make each event a tuple of type (event time, Event Type)
Start the world, and have timers automatically allocated for each event type

Notes 10/26
Upon each event, fire off many infinitiely valued timers. If we encounter a new event type, one of those
timers must 
Keep an array of records associated with each ramp
A timer is a combination of switches and ramps 

Start with 20 ramps per event
Code has to be event-centric

Say A occurs at 5 seconds, it has a bunch of ramps (some are frozen)
Ramps with infinite weights fire off and immediately start correcting
A subsequent event occurs and you assign one of the ramps to that event
Ramps have associated scores that freeze them or enable them to be reallocated

As score goes to zero, learning rate goes to 1

Score comes from the reward function
Learning rate is e^-(k * score)

A subset of them are adaptable 
Each ramp has a weight associated with its confidence in measuring 
How well did it do
What is the theoretical maximum performance 
Keep an array of scores 
Flip between the two versions of the early update rule
How do you get the max reward? By being on time always

Need to simulate peak interval task as well (start and stop threshold)

Flip a coin on each ramp about if itll be updated on that trial
All different approaches about which ones should be flexible and which arent

"""
""" === Global Variables === """

n_event_types=3 # Number of event types (think, stimulus A, stimulus B, ...)
num_events = 100 # Total amount of events across all types
y_lim=2 # Plotting limit
noise = 0.0 # Internal noise - timer activation
learning_rate = .8
STANDARD_INTERVAL = 200
response_threshold = 0.95
alpha = 0.8


# TODO: Establish probability distributions for all combinations of event types

events_with_type = TM.getSamples(num_events, num_normal = 2, num_exp = 1)

event_occurances = (list(zip(*events_with_type))[0])
plt.hist(event_occurances, bins=80, color='black')

events = np.zeros(num_events)
events_with_type[0][0] = event_occurances[0]

for i in range (1,num_events):
     events_with_type[i][0] = events_with_type[i-1][0] + events_with_type[i][0]

# Time axis for plotting        
T = events_with_type[-1][0] + 100

# Array of timers for the simulation, new timers get allocated when new events appear
timers=[TM(.5, 3), TM(timer_weight=.175)]
timers[0].setTimerWeight(.55,1)
timers[0].setTimerWeight(.45,2)
# In this declaration we make a timer that has 3 distinct weights. We then set those weights
timer_1 = TM(.5, 3)
timer_1.setTimerWeight(.55,1)
timer_1.setTimerWeight(.45,2)

timer_2 = TM(timer_weight=.175)

plt.figure()
# plt.suptitle(f'Internal Noise = {noise}')
colors = ['b', 'g', 'r', 'c', 'm', 'y']
timers_dict = {}
timers_dict[0] = 0
timers_dict[1] = 1
first_event = True
for idx, event in enumerate(events_with_type):
    event_time = event[0]
    event_type = int(event[1])
    
    if event_type not in timers_dict:
        # Allocate a new timer for this event type 
        # TODO: Run this by prof. simen to see if its fine
        index = len(timers)
        timers.append(TM(.5))
        timers_dict[event_type] = index
        
    timer_idx = timers_dict[event_type]
    timer = timers[timer_idx]
   
    if first_event:
        first_event= False                   
        timer_value = activationAtIntervalEnd(timer.timerWeight(), event_time, noise)
        response_time= responseTime(timer.timerWeight(), response_threshold)
        timer_1.setScore(timer.getScore() + score_decay(response_time, event_time))
        
        plt.plot([0,event_time], [0, timer_value], linestyle = "dashed", c=colors[event_type], alpha=0.8)
        plt.plot([event_time], [timer_value], marker='o',c=colors[event_type]) 

    else:
        prev_event = events_with_type[idx-1][0]
        timer_value = activationAtIntervalEnd(timer.timerWeight(), event_time - events_with_type[idx-1][0], noise)
        response_time = responseTime(timer.timerWeight(), response_threshold)
        timer_1.setScore(timer.getScore() + score_decay(response_time, event_time))       
        
        plt.plot([prev_event,event_time], [0, timer_value], linestyle = "dashed",  c=colors[event_type], alpha=0.8)
        # plt.plot([events[idx-1][0],event_time], [0, timer_value], linestyle = "dashed", c=colors[0], alpha=0.8)
        plt.plot([event_time], [timer_value], marker='o',c=colors[event_type]) 
        
        # if len(timer.timers) > 1:
        #     for weight in range(1,len(timer.timers)):
        #         timer_value = activationAtIntervalEnd(timer.timerWeight(weight), event_time, noise)
        #         plt.plot([event_time], [timer_value], marker='o',c=colors[event_type]) 
    
    # Update Rules
    if timer_value > 1:
        ''' Early Update Rule '''
        # timer_1_mu = ((1-learning_rate) * 1) + learning_rate*(event-events[i-1])
        
        timer_weight = earlyUpdateRule(timer_value, timer.timerWeight(), timer.learningRate())
        timer.setTimerWeight(timer_weight)
        
        # if len(timer.timers) > 1:
        #     for weight in range(1,len(timer.timers)):
        #          timer_weight = earlyUpdateRule(timer_value + .05, timer.timerWeight(weight), timer.learningRate())
        #          timer.setTimerWeight(timer_weight)
        # timer_weight2 = earlyUpdateRule(timer_1_value_2 + .05, timers[0].timerWeight(1), learning_rate)
        # timers[0].setTimerWeight(timer_weight2, 1)
        
        # timer_weight3 = earlyUpdateRule(timer_1_value_3 - .05, timers[0].timerWeight(2), learning_rate)
        # timers[0].setTimerWeight(timer_weight3, 2)
        
        
    else:
        ''' Late Update Rule '''
        #timer_1_mu = ((1-learning_rate) * 1) + learning_rate*(event-events[i-1])
        
        timer_weight = lateUpdateRule(timer_value, timer.timerWeight(), learning_rate)
        timer.setTimerWeight(timer_weight)
        
        # timer_weight2 = lateUpdateRule(timer_1_value_2  + .05, timer_1.timerWeight(1), learning_rate)
        # timer_1.setTimerWeight(timer_weight2, 1)
        
        # timer_weight3 = lateUpdateRule(timer_1_value_3 - .05 , timer_1.timerWeight(2), learning_rate)
        # timer_1.setTimerWeight(timer_weight3, 2)
                    
    
    plt.vlines(event, 0,y_lim, label="v", color=colors[event_type], alpha=0.5)
    if y_lim>1:
        plt.hlines(1, 0, T, alpha=0.2, color='black')
            
    # if i == num_events-1:
    #     print(f"timer 1 average activation: {timer_1_running_act/(num_events/2)} ")
    
    plt.ylim([0,y_lim])
    plt.xlim([0,T])
    plt.ylabel("activation")
    plt.xlabel("Time")
    
    # ax = plt.subplot(211)
    # ax.set_title("Timer Ramping Activity")
    # # ax = plt.subplot(212)
    # ax.set_title("Timer 1 Responses")
    plt.grid('on')


# plt.figure()
# f, (ax1, ax2) = plt.subplots(2, 1)
# plt.subplot(211)
# plt.subplot(212)
# ax2 = plt.hist(timer_2_events, bins=80, range=(160,240))
print (f'noise: {noise}')
print(f'learning rate: {learning_rate}')
print (f'Num Events: {num_events}')
print (f'Timer 1 - {timer_1.getScore()}')
print (f'Timer 2 - {timer_2.getScore()}')
    