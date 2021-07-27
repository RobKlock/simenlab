#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:14:22 2021

@author: Robert Klock

Class defining a timer module and related useful methods
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy import signal
from scipy.optimize import minimize 
from scipy.stats import norm
"""
Scikit-Learn has useful Gaussian Mixture for probabilities 
Be careful because they probably have two models: one for fitting
and another for generating samples.

Alternatively:
    3 components with different weights
        Flip a coin proportional to those weights 
        and draw a random sample from those individual 
        distributions. This works for just Gaussian mixtures
        
        Weighted sum of their CDFs. Then you can just pull from
        the CDF. If its less than the weight you draw from that sample
        
        N components of exponentials and Gaussians. 
        Normal variable, find out which weight range it falls into
        Then just draw a sample from the distribution that falls in line with
    
    https://pytorch.org/docs/stable/distributions.html?highlight=mixture#torch.distributions.mixture_same_family.MixtureSameFamily
    
    Package would roll the dice for you, give a number, and 
    draw the sample. Since we want to combine exponentials and normals,
    its better to build out from scratch but is still relatively simple
    
Initializing Timers:
    
    Semi-Markov Model
    Get a Stimulus Event 
        Assume that event is the mean of the distribution
        Update the mean with new samples 
    Then get another Stimulus and another Event
        Repeat from previous stimulus
    
    For new stimulus, calculate the likelihood of it belonging
    to each distribution and update the running average of the mean
    
    If its really off, create a new distribution
    Need some bias to avoid overfitting. Bias has 
    two components: one to keep track of distribution and 
    another to keep track of speed
    
    Default timer to capture new events and update to 
    record new events and probabilities 
    
    Keep track of variability to fit standard deviation
    
    First, choose a family
    
    Need to define and assign distribution families
    
    
    Dont want to record everything or else youll just overfit
    
    Neural networks currently tend to just learn every frame of video or all data
    instead of learning a timer interval to expect an event
    
    DDMs have good properties: they are speed adaptable through bias,
    can represent means and standard deviations with them.
    
    Model is:
        As soon as a stimulus occurs, I start multiple potential timers. When the event occurs,
        I store information about which ramp was closest. Or you could allocate 5 timers for each event
        and adjust their fan to represent the event's standard deviation
    
    Rivest will write down a basic rule to use
    Events A and B and their independent to get started
    
"""

class TimerModule:
    
    ZEROS_BLOCK = np.zeros((4,4))
    BIAS_BLOCK = np.array([1.2, 1, 1.2, 1.25])
    def __init__(self,timer_weight = 1):
        self.timer_weight=timer_weight
        block = np.array([[2, 0, 0, -.4],
                          [self.timer_weight, 1, 0, -.4],
                          [0, .55, 2, 0],
                          [0, 0, .9, 2]])
        self.block = block
    
    def timerWeight(self):
        return self.timer_weight
    
    def setTimerWeight(self, weight):
        self.timer_weight = weight
    
    def timerBlock(self):
        return self.block
    
    def buildWeightMatrixFromWeights(timerModules):
        if not isinstance(timerModules, list):
            raise TypeError('Timer modules should be in a list')
        else:
            module_count = len(timerModules)
            weights = np.kron(np.eye(module_count), np.ones((4,4)))
            idx = (0,1)
            for i in range (0, module_count): 
                # t = np.kron(np.eye(module_count), np.ones((4,4)))
               
                weights[0+(4*i):0+(4*(i+1)), 0+(4*i):0+(4*(i+1))] = timerModules[i] #np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])
                print(weights)
                # for j in range (0, len(timerModules)):
                 
    def buildWeightMatrixFromModules(timerModules):
        if not isinstance(timerModules, list):
            raise TypeError('Timer modules should be in a list')
        else:
            module_count = len(timerModules)
            weights = np.kron(np.eye(module_count), np.ones((4,4)))
            idx = (0,1)
            for i in range (0, module_count): 
                # t = np.kron(np.eye(module_count), np.ones((4,4)))
               
                weights[0+(4*i):0+(4*(i+1)), 0+(4*i):0+(4*(i+1))] = timerModules[i].timerBlock() #np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])
                # we only add connecting weight between modules if there are more than 1
                # and we only do it n-1 times
                if (module_count - i > 1):
                    weights[4+(4*i), 2+(4*i)] = 1
        return weights
                # for j in range (0, len(timerModules)):
    
    def updateV(v):
        # Expands v for a single additional module
        new_v = np.vstack((v,[[0], [0], [0], [0]]))
        return new_v
    
    def updateVHist(v_hist):
        # Expands v_hist for a single additional module
        history = v_hist[0].size
        prior_activity = np.array([np.zeros(history)])
        
        return np.concatenate((v_hist, prior_activity), axis=0)
    
    def updateL(l, lmbd):
        # Expands l for a single additional module
        add_l = np.full((4,1), lmbd)
        return np.concatenate((l, np.array(add_l)))
    
    def updateBias(b):
        add_b = b[:4]
        return np.concatenate((b, add_b))
 
    def generate_sample(x,mu,sigma,y):
        x_start = .5
        
        CDF = lambda x: (1/2)*(1 + erf((x - mu)/(sigma*(sqrt(2)))))
        func = lambda x: (CDF(x) - y)**2

        result = minimize(func, x_start)
        print(result.x)
        print(result.fun)

    
    def getSamples(num_samples = 1, num_normal = 2, num_exp = 0, num_dists = 2):
        """
        A function that generates random times from a probability 
        distribution that is the weighted sum of exponentials and Gaussians.
        """
        # Gotta add better error handling
        if num_normal + num_exp != num_dists:
            suggested_method = f'getSamples({num_samples}, {num_normal}, {num_exp}, {num_normal + num_exp})'
            raise ValueError(f"num_normal and num_exp must sum to num_dists! Did you mean {suggested_method}?")
        
        # To get N random weights that sum to 1, add N-1 random numbers to an array
        weights_probs = np.random.rand(num_dists - 1) 
        # Add 0 and 1 to that array
        weights_probs = np.append(weights_probs, 0)
        weights_probs = np.append(weights_probs, 1)
        # Sort them
        weights_probs = np.sort(weights_probs)
        weights = np.zeros(num_dists)
        # After establishing the weight array, iterate through the probabilities 
        # and declare the Nth weight to be the difference between the entries at the N+1 and N-1
        # indices of the probability array
        for i in range (0, (weights_probs.size - 1)):
            weights[i]=(weights_probs[i + 1] - weights_probs[i])
        # Declare distribution types (1 is exp, 0 is normal)
        dist_types = np.random.rand(num_dists)
        # Declare num_dists centers and std deviations 
        locs = []
        scales = []
        loc1 = np.random.randint(10,30)
        loc2 = np.random.randint(10,30) 
        scale1 = math.sqrt(np.random.randint(5, 10))
        scale2 = math.sqrt(np.random.randint(5, 10))
            
        # Establish our distributions and weights in the same loop
        for i in range (0, num_dists):
            # Round our dist_type entries
            dist_types[i] = round(dist_types[i])
            if (i < weights_probs.size - 1):
                weights[i]=(weights_probs[i + 1] - weights_probs[i])    
            locs.append(np.random.randint(10,30))
            scales.append(math.sqrt(np.random.randint(5, 10)))
        weights = np.sort(weights)
        
        # Roll a dice N times
        samples = []
        # Roll our dice N times
        # I hate that this is O(N * D)
        for i in range(0, num_samples):
            dice_roll = np.random.rand(1)
        
            # Find which range it belongs in
            for dist in range (0, num_dists):
                if (dice_roll < weights[dist]):
                    # The roll falls into this weight, draw our sample
                    if dist_types[dist] == 1:
                        sample = np.random.exponential(scales[dist], 1)
                    else:
                        sample = np.random.normal(locs[dist], scales[dist], 1)
                    samples.append(sample[0])
                else:
                    continue
        return np.asarray(samples)

# Useful visualization, just comment it out 
# plt.hist(TimerModule.getSamples(num_samples=1000, num_normal=1, num_exp = 1, num_dists = 2), bins=40, color='black')

            
'''
if something unusual happens, i release some timers 
if you repeat the stimulus, you can look it up in memory to see when it last happened
once thats implemented, start buildig out some heuristics or biases
sum of squares of error of timers. If they exceed some level you make 
some decisions about garbage collection

each ramp has a weight that gets updated/garbage collected depending on its 
error  
'''
                