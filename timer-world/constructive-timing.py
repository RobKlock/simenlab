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

print(TM.getSamples())

def main(s0, T, num_events=3):
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
    event = 0
    
    for i in range(0, event_weights_for_sampling.size):
       if dice_roll <= event_weights_for_sampling[i]:
           event = i + 1 
           break
    print(dice_roll)
    print(event)
            
    
    
    
    timer1 = TM
    print(timer1.getSamples(num_samples=40))

    events = TM.getSamples(num_samples=3)
    print(events)
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
main(0, 1000)