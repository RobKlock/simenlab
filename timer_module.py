#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:14:22 2021

@author: Robert Klock

Class defining a timer module
"""
import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy.stats as stats
from scipy import signal
from scipy.optimize import minimize 

class TimerModule:
    
    ZEROS_BLOCK = np.zeros((4,4))
    BIAS_BLOCK = np.array([1.2, 1, 1.2, 1.25])
    def __init__(self,timer_weight = 1):
        self.timer_weight=timer_weight
        block = np.array([[2, 0, 0, -.4],
                          [self.timer_weight, 1, 0, -.4],
                          [0, .5, 2, 0],
                          [0, 0, 1, 2]])
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
    
    def getProbabilities(event_count):
        delta = 1e-4
        mu = np.random.randint(10,20)
        mu2 = np.random.randint(10,20)
        print(mu)
        variance = np.random.randint(5, 10)
        print(variance)
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        big_grid = np.arange(-5,60,delta)
        large_x = np.linspace(-1, 50,100)
        
        norm1 = stats.norm(mu, 8)
        cdf1 = stats.norm.cdf(large_x, mu, sigma)
        plt.plot(x, norm1.pdf(large_x))
        pdf1 = norm1.pdf(large_x)
        # Define a discretized probability mass function to convolude later on 
        pmf1 = norm1.pdf(big_grid)*delta
        print("Integral over norm1 pdf: "+str(np.trapz(pdf1, large_x)))
        print("Sum of norm1 pmf2: "+str(sum(pmf1)))
        
        norm2 = stats.norm(mu2, sigma)
        cdf2 = stats.norm.cdf(large_x, mu2, sigma)
        plt.plot(x, norm2.pdf(large_x))
        pdf2 = norm2.pdf(large_x) 
        pmf2 = norm2.pdf(big_grid) * delta
        print("Integral over norm2 pdf: "+str(np.trapz(pdf2, large_x)))
        print("Sum of norm2 pmf2: "+str(sum(pmf2))+"\n")
        
        print("Integral over sum pdf: "+str(np.trapz(pdf1+pdf2, large_x)))
        print("Integral over normed sum pdf: "+str(np.trapz((pdf1+pdf2)/2, large_x)))
        summed_pdf = (pdf1+pdf2)/2
        summed_cdf = stats.norm.cdf(pdf1) # calculate the cdf - also discrete
        
        plt.figure("cdfs")
        # plot the cdf
        plt.plot(large_x, cdf1)
        plt.show()
        
        plt.figure("cdf2")
        # plot the cdf
        plt.plot(large_x, cdf2)
        plt.show()
        
        plt.figure("cdf sum")
        # plot the cdf
        plt.plot(large_x, (cdf1 + cdf2)/2)
        plt.show()
        
        #plt.plot(x, stats.norm.pdf(x, mu + 10, sigma))
        #plt.plot(x, stats.expon.pdf(x, 8, sigma))
        plt.figure()
        plt.plot(large_x, (stats.norm.pdf(large_x, mu, sigma) + stats.norm.pdf(large_x, mu + 10, sigma)) / 2)
        plt.show()
        
        conv_pdf = signal.fftconvolve(pdf1,pdf2,'same')
        print("Integral over convoluted pdf: "+str(np.trapz(conv_pdf, large_x)))
        
        
        conv_pmf = signal.fftconvolve(pmf1,pmf2,'same')
        print("Sum of convoluted pmf: "+str(sum(conv_pmf)))
        
        pdf1 = pmf1/delta
        pdf2 = pmf2/delta
        conv_pdf = conv_pmf/delta
        print("Integration of convoluted pdf: " + str(np.trapz(conv_pdf, big_grid)))
        
        plt.figure()
        plt.plot(big_grid,pdf1, label='Uniform')
        plt.plot(big_grid,pdf2, label='Gaussian')
        plt.plot(big_grid,conv_pdf, label='Sum')
        plt.legend(loc='best'), plt.suptitle('PDFs')
        plt.show()
        
                