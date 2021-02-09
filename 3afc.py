#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 22:41:24 2021
Three-alternative forced choice circuit
@author: Rob Klock
"""
import numpy as np
import matplotlib.pyplot as plt

#Coodinates for random data
mu, sigma = 0, .5
x1 = np.random.normal(mu, sigma, size = (1200, 1))
y1 = np.random.normal(mu, sigma, size = (1200, 1))

#500 instances of data classified as a 1 and data classified as 0 at a time
zeros = np.zeros((1200,1))
ones = np.ones((1200,1))

#p_weights = p.gradient_descent(train_context_1, 0.15, 1000)
context_1_weights = [ 1       , -1,  -1] #fixed weights so we dont call GD every time we run
context_2_goal_weights = [0,    2.5,    -2.5]

#Combine pairs of X and Y
data = np.append(x1, y1, axis = 1)
labels = np.empty((data.shape[0], 2))
    #Labels random data according to passed in line
    #Takes in data and weights
    #Appends two values, a context 1 label and a context 2 label (opposites)
    #Context one classifies data above the line as 1, below as 0, and
    #visa versa for context 2
for i in range (0, data.shape[0]):
    if (data[i][1] >= 1 * data[i][0] + weights1[0]) and (data[i][1] >= (-1/weights2[2]) * data[i][0] * weights2[1] + weights2[0]):
        labels[i] = [1,1]
    
    elif (data[i][1] >= weights1[1] * data[i][0] + weights1[0]) and (data[i][1] <= (-1/weights2[2]) * data[i][0] * weights2[1] + weights2[0]):
        labels[i] = [1,0]
    
    elif (data[i][1] <= weights1[1] * data[i][0] + weights1[0]) and (data[i][1] >= (-1/weights2[2]) * data[i][0] * weights2[1] + weights2[0]):
        labels[i] = [0,1]
    
    else:
        labels[i] = [0,0]
return labels


xx = np.linspace(0, 1, 10)
above = data[data[:, 2] == 0]
below = data[data[:,2] == 1]
plt.scatter(above[0:,0:1], above[0:,1:2], alpha=0.80, marker='^', label = "Context: 1, Class: 1")

def sigmoid(l, total_input, b, sig_idx):
    f = 1/ (1 + np.exp(-l[sig_idx] * (total_input - b[sig_idx])))
    #sig_index is the index of the activcation function we want   
    return f

#Plt.pause for debugging 
'''Weights'''
circuit_weights = np.array([[0,    -1,      0,     0],    #1->1  2->1, 3->1, 4->1
                    [-1,    0,      0,     0],            #1->2, 2->2, 3->2, 4->2
                    [1.23,  0,      2,  0],             #1->3, 2->3, 3->3, 4->3
                    [0,     1.23,   0,      2]])        #1->4, 2->4, 3->4  4->4
    
    
    #Weighted sum prediction
def p_predict(row, weights, ap=False):
    activation = weights[0]
    for i in range(0,2):
        activation += weights[i + 1] * row[i]
    #if (1/(1 + math.exp(-activation))) >= .5:
    if not ap:
        if activation > 0:
            return  1
        else: 
            return -1
    else:
        if activation > 0:
            return -1
        else:
            return 1
    
def diffusion_predict(p, ap, datum, label, circuit_weights = circuit_weights, plot = False):                        
    steps = 0 
    tau = 1
    dt = 0.01
    context_timer = 0
    p_weights = p
    ap_weights = ap
    internal_noise = 0.15
    sensor_noise = 0.2
    decision_threshold = .8
    v_hist = np.array([[0, 0, 0, 0]]).T    
    v = np.array([[0, 0, 0, 0]]).T              
    p_hist = np.array([0])
    ap_hist = np.array([0])
    perceptron_activation_scalar = .8
    l = np.array([[4, 4, 4, 4]]).T                  
    bias = np.array([[1, 1, 1, 1]]).T 
  
    bias = bias * 1.1
    sig_idx= [2,3]
    
    #Repeatedly have perceptron and AP classify row, feed into circuit
    #until the circuit makes a classification (v3 or v4 is > decision threshold)
    while (v[3] < decision_threshold) and (v[2] < decision_threshold):
        row = datum
        nn = np.random.normal(0, .1, 2) * sensor_noise
        #noise = np.append(nn, data[row_idx][-1])
        noisyRow = np.add(nn, row[:2])
        
        p_classification = p_predict(noisyRow, p_weights) * perceptron_activation_scalar
        ap_classification = p_predict(noisyRow, ap_weights, ap=True) * perceptron_activation_scalar
        
        steps += 1 
        
        activations = circuit_weights @ v                              #weighted sum of activations, inputs to each unit
        activations[0] = p_classification + activations[0]     
        activations[1] = ap_classification + activations[1]    
        activations[2:] = sigmoid(l, activations[2:], bias, sig_idx)
        dv = tau * ((-v + activations) * dt + internal_noise * np.sqrt(dt) * np.random.normal(0,1, (4,1))) # add noise using np.random
        v = v + dv
        
        v_hist = np.concatenate((v_hist,v), axis=1)
        p_hist = np.append(p_hist, p_classification)
        ap_hist = np.append(ap_hist, ap_classification)
    
    context_timer += 1
    classification = 1 if ( (v[2] >= decision_threshold) and (v[3] <= decision_threshold)) else 0
    
    #Check to see if our classification is right and add to running accuracy 
        
                
        
    if plot:
        plt.figure(1)
        plt.plot(v_hist[0,:]) 
        plt.plot(v_hist[1,:])
        plt.plot(v_hist[2,:])
        plt.plot(v_hist[3,:])
        plt.legend(["v1","v2","v3","v4"], loc=0)
        plt.ylabel("activation")
        plt.xlabel("time")
        plt.grid('on')
        plt.title("Units 1-4 Activations")
        
        
        #Historical classifcation of the noisy data by p and ap
        plt.figure(2)
        plt.plot(p_hist)
        plt.plot(ap_hist)
        plt.legend(["p classification", "ap classification"], loc=0)
        plt.title("Perceptron and Antiperceptron Activations")
        
                
        plt.figure(3) 
        plt.plot(v_hist[1,:] - v_hist[0,:])
        plt.ylabel("v1 v2 difference")
        plt.xlabel("time")
        plt.grid('on')
        plt.title("V1 V2 Difference (Drift Diffusion)")
    return [classification, steps]