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
    # Need to make a new TimerModule object and THEN call the method
    timer1 = TM.new()
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
   