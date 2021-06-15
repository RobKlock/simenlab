#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:52:53 2021

@author: Robert Klock
Now that the update rules are working we can move to building dynamic circuits that time real-world events

This program consists of:
        A world that have stimuli that make up time intervals (a light goes off and an explosion follows 10 seconds later)
        A circuit that keeps track of the time by learning the time in between stimuli and grows dynamically
        
Input:
    A series of events A, B, C... that trigger the circuit to start timing 
    
Output:
    Predictions of events via a learned weight matrix which times the events 
    
Qs:
    What kinds of events?
    
    What kinds of time between them?
        The duration of the first timer is the time between event A and event B
            B follows A with a probability density function 
            On any given trial, B may occur exactly a second after A
            but it could also occur subject to a probability function
            
    How best to predict the distribution of an event?
        Timer A could have a population of ramps that have different weights
        Different ramps would hit threshold at different times 
        
        To predict when B occurs, you could take the average of these units.
        The switch takes in a weighted average of all ramps from A, with higher probability
        timers having a higher weight
        
        Use all ramps to make a prediction, but generate new ramps on new trials 
        How do you know if a ramp already times the trial? 
            You can record everything or just use one ramp, but those are both extreme
            Instead, ramps that are close to the trial length get tuned and other ramps
            aren't adjusted. This can be modulated by a learning rate 
            If a ramp is adjusted a certain # of times, allocate new ramps to increase
            precision in that probability
            
        Check to see which timers hit close to the next timeable event B 
        If you dont start enough ramps and they all miss, you just miss timing that event
        Theres two extra parameters: number of ramps and spread of the ramps 
        
    What allows you to add units?
        Not sure yet, but theres probably an optimal approach for different worlds
        Goal: A heuristic approach that works in most worlds but may not be the best
            If B relative to A is exponential theres no way to predict that and no 
            ramps should be allocated. Temporally unpredictable events shouldn't be attempted 
            its memoryless
            
Piecewise linear Hysterisis function? Make all units piecewise linear for easier multiplications
    
For Friday, show several examples of the learning rules working and the a two-unit module 
learning

The infinitude of the impossible
"""

"""
Problems from Gabriele and Francois

XOR in time
music timers 
what can you learn without having new questions to answer

short-long interval
need a time sequence or period detector that only goes off if its pattern
is matched

build a few patterns by hand and then decide whats needed to actually
detect their structure
hierarchy or perception
inverse hierarchy of action
"""

"""
Generate a sequence of events, an alphabet of symbols
Probabilities of each event following each other. Matrix of events:
        A        B              C
    A        PDE for B|A   PDE for C_t|A_0
    B PDE A|B    
    C

Deal with a world where things are very predictable (B always occurs after A)
or not predictable (exponential)
See iPad notes for probability stuff
Can have a perceptual module that uses backprop to decide which events are best
to keep track of/pay attention to/learn
"""

import sys
sys.path.append(".")
from timer_module import TimerModule as TM

# Timer Event:

newTimer = TM()

timer_weight = newTimer.getTimerWeight()

print(timer_weight)

