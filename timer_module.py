#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:14:22 2021

@author: Robert Klock

Class defining a timer module
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy import signal
from scipy.optimize import minimize 
from scipy.stats import norm

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
        #print("Integration of convoluted pdf: "   str(np.trapz(conv_pdf, big_grid)))
        
        plt.figure()
        plt.plot(big_grid,pdf1, label='Uniform')
        plt.plot(big_grid,pdf2, label='Gaussian')
        plt.plot(big_grid,conv_pdf, label='Sum')
        plt.legend(loc='best'), plt.suptitle('PDFs')
        plt.show()
        
    def getSamples(num_samples = 1):
        """
        A function that generates random times from a probability 
        distribution that is the weighted sum of exponentials and Gaussians.

        To get a random sample from just one normal distribution, one can 
        sample a uniform random variable, then take the result, and take the
        inverse of the Gaussian cdf of that value. 
 
        Algorithm:
            1) Generate a uniform rand, y
            2) Use PPF to draw inverse sample from CDF
        """
        
       # data = np.zeros((num_samples,1))
       # y = np.random.rand(num_samples,1)
       #weights for disribution
        w1 = 0.5
        w2 = 0.5
        
    
        # Mixture distribution weights
        loc1 = np.random.randint(10,25)
        loc2 = np.random.randint(10,25)
        #print(loc1)
        scale1 = math.sqrt(np.random.randint(5, 10))
        scale2 = math.sqrt(np.random.randint(5, 10))
        x1_rand = np.random.normal(loc1, scale1, 1000)
        x2_rand = np.random.normal(loc2, scale2, 1000)
        plt.hist(x1_rand, bins=20)
        plt.xlabel('X')
        plt.ylabel('Norm1')
        plt.title('Normal Distribution 1')
                  
        plt.figure()
        plt.hist(x2_rand, bins=20)
        plt.xlabel('X')
        plt.ylabel('Norm2')
        plt.title('Normal Distribution 2')
        
        # Refit loc and scales for distributions
        loc, scale = stats.norm.fit(x1_rand)
        loc2, scale2 = stats.norm.fit(x2_rand)
        
        # Linearly spaced values of x
        x = np.linspace(start = 0, stop = 30, num = 800)
        
        # PDF for dist1
        pdf1 = stats.norm.pdf(x, loc=loc, scale=scale)
        plt.figure()
        # Plot
        plt.plot(x, pdf1, color='black')
        plt.xlabel('X')
        plt.ylabel('PDF1')
        plt.title('PDF for Distribution 1')
        
        # PDF for dist2
        pdf2 = stats.norm.pdf(x, loc=loc2, scale=scale2)
        plt.figure()
        # Plot
        plt.plot(x, pdf2, color='black')
        plt.xlabel('X')
        plt.ylabel('PDF2')
        plt.title('PDF for Distribution 2')
        
        # Wsum PDF
        sum_pdf = (w1 * pdf1) + (w2 * pdf2)
        plt.figure()
        # Plot
        plt.plot(x, sum_pdf, color='black')
        plt.xlabel('X')
        plt.ylabel('sum_pdf')
        plt.title('PDF for sum_pdf')

    #10000 samples should look more like the PPF
    # Solvers vs minimizers
        
        # CDFs
        cdf1 = stats.norm.cdf(x, loc=loc, scale=scale)
        cdf2 = stats.norm.cdf(x, loc=loc2, scale=scale2)
        # Use the CDF inverse to calculate the likelihood of that value or less
        cdf_at_10 = stats.norm.cdf(10, loc=loc, scale=scale)
        cdf2_at_10 = stats.norm.cdf(10, loc=loc2, scale=scale2)
        
        
        plt.figure()
        # Plot
        plt.plot(x, cdf1, color='black')
        plt.vlines(10, 0, cdf_at_10, linestyle=':')
        plt.hlines(cdf_at_10, -5, 10, linestyle=':')
        plt.xlabel('X')
        plt.ylabel('CDF1 = P(x<=X)')
        plt.title('CDF for Distribution 1')
        
        plt.figure()
        # Plot
        plt.plot(x, cdf2, color='black')
        plt.vlines(10, 0, cdf2_at_10, linestyle=':')
        plt.hlines(cdf2_at_10, -5, 10, linestyle=':')
        plt.xlabel('X')
        plt.ylabel('CDF2 = P(x<=X)')
        plt.title('CDF for Distribution 2')
        
        #fun = lambda y: y - ((w1 * stats.norm.cdf(x, loc=loc, scale=scale)) + (w2 * stats.norm.cdf(x, loc=loc2, scale=scale2)))
        #res = minimize(fun, [.5], method='Nelder-Mead', tol=1e-6)
        #print(res)
        
        plt.figure()
        # Plot
        sum_cdf = (w1 * cdf1) + (w2 * cdf2)
        plt.plot(x, sum_cdf, color='black')
        #plt.vlines(10, 0, cdf2_at_10, linestyle=':')
        #plt.hlines(cdf2_at_10, -5, 10, linestyle=':')
        plt.xlabel('X')
        plt.ylabel('Sum CDF')
        plt.title('CDF SUM')
        
        # Percent point function
        # instead of minimizing, we could just use ppf?
        cdf1_ = np.linspace(start=0, stop=1, num=10000)
        cdf2_ = np.linspace(start=0, stop=1, num=10000)
        x_ = stats.norm.ppf(cdf1_, loc=loc, scale=scale)
        x2_ = stats.norm.ppf(cdf2_, loc=loc2, scale=scale2)
        sum_ppf_ = (w1 * x_) + (w2 * x2_)
    
        plt.figure()
        # Plot
        plt.plot(cdf1_, x_, color='black')
        random_sample = np.random.rand()
        cdf1_inv_sample = stats.norm.ppf(random_sample, loc=loc, scale=scale)
        cdf_min = stats.norm.ppf(0.001, loc=loc, scale=scale)
        #print(random_sample)
        #print(cdf1_inv_sample)
        plt.vlines(random_sample, cdf_min, cdf1_inv_sample, linestyle=':')
        plt.hlines(cdf1_inv_sample, 0, random_sample, linestyle=':')
        plt.xlabel('X')
        plt.ylabel('PPF1')
        plt.title('PPF for Distribution 1')
        
        plt.figure()
        # Plot
        plt.plot(cdf1_, sum_ppf_, color='black')
        random_sample2 = np.random.rand()
        cdf_sum_inv_sample = (w1 * stats.norm.ppf(random_sample2, loc=loc, scale=scale)) + (w2 * stats.norm.ppf(random_sample2, loc=loc2, scale=scale2))
        print(random_sample2)
        print(cdf_sum_inv_sample)
        plt.vlines(random_sample2, 0, cdf_sum_inv_sample, linestyle=':')
        plt.hlines(cdf_sum_inv_sample, 0, random_sample2, linestyle=':')
        plt.xlabel('X')
        plt.ylabel('PPFs')
        plt.title('Weighted Sum PPF')
        
        samples = np.zeros(800)
        ten_thousand_samples = np.random.rand(800)
        plt.figure()
        s_index = 0
        
        for sample in ten_thousand_samples:
            val = (w1 * stats.norm.ppf(sample, loc=loc, scale=scale)) + (w2 * stats.norm.ppf(sample, loc=loc2, scale=scale2))
            samples[s_index] = val
            plt.plot(val, sample, 'b.')
            s_index = s_index + 1
            
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Inverse of samples')    
        
        textstr = '\n'.join((
            r'$\mu_1=%.2f$' % (loc1,),
            r'$\sigma_1=%.2f$' % (scale1, ),
            r'$\mu_2=%.2f$' % (loc2,),
            r'$\sigma_2=%.2f$' % (scale2, ),
            r'$w_1=%.2f$' % (w1,),
            r'$w_2=%.2f$' % (w2, ),
            'n=800'))
        ax1 = plt.subplot(311)
        ax1.set_title('Random PPF Samples Histogram')
        ax1.hist(samples, bins = 40, range=[0,30])
        ax2 = plt.subplot(312)
        ax2.set_title('Weighted Distribution Sum PDF')
        ax2.plot(x, sum_pdf, color='black')
        ax3 = plt.subplot(313)
        ax3.set_title('Weighted CDF')
        ax3.plot(x, sum_cdf, color='black')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
                          
   
                
        #cdf1 = stats.norm.cdf(num_samples, mu, sigma)
        #pdf1 = norm1.pdf(x)
       # plt.plot(x, pdf1)
        #plt.plot(x, norm1.pdf(x))
        #plt.plot(x, norm.cdf(x))
        #plt.plot(x, norm1)
    
        

                        
        #cdf2 = stats.norm.cdf(x, mu2, sigma)
        #plt.plot(x, cdf2)
       
        #dist = (w1 * norm1) + (w2 * norm2)
        #plt.plot((w1 * norm1(num_samples)) + (w2 * norm2(num_samples)), title="dist")
        # difference = y - (w1*norm.cdf(x,0,1) + w2*norm.cdf(x,3,0.1))^2 
      
       #for i in range(1,num_samples):
        #    difference = y[i] - (w1*norm.cdf(x,0,1) + w2*norm.cdf(x,3,0.1))^2 
            

                