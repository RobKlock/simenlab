#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:46:01 2020
Contextual classification in a 2afc model
@author: Robert Klock
Executive function, adaptive behavior, machine learning
Questions 12/15:
        How should our data actually change to know we're in a new context?
            Generate random data with new decision line and reclassify it
            
        Do we want ap and p to be able to operate in different contexts?
            nope. They operate under the same context
            
        Going to rewrite this to use numpy arrays. Perceptron weights will be
        stored in an array with size (7contexts, |data) (so, 1 context for sonar data
        would yield a numpy array of shape (1, 60) 2 contexts yields (2, 60))
            Good.
            
        When do we want to switch to a new contetxt?
            Get 5 incorrect classifications in a row?
            Some percentage incorrect that we've seen so far?
            
"""
import numpy as np
from csv import reader
import math
import sys
sys.path.append('/Users/robertklock/Documents/Class/Fall20/SimenLab/simenlab')
import perceptron as p
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

#Version two just has a normal distribution of data with a line passing through 

#Set up dummy data. 4 normal distributions of points classified as a 1 or 0 with values between 0 and 1
mu, sigma, d = .5, 0.2, .3 # mean and standard deviation

def label_data(data, weights):
    labels = np.empty((data.shape[0], 2))
    #Labels random data according to passed in line
    #Takes in data and weights
    #Appends two values, a context 1 label and a context 2 label (opposites)
    #Context one classifies data above the line as 1, below as 0, and
    #visa versa for context 2
    for i in range (0, data.shape[0]):
        if data[i][1] >= weights[1] * data[i][0] + weights[0]:
            labels[i] = [1,0]
        else:
            labels[i] = [0,1]
    return labels

#Weighted sum prediction
def p_predict(row, weights, ap=False):
    activation = weights[0]
    for i in range(0,2):
        activation += weights[i + 1] * row[i]
    #if (1/(1 + math.exp(-activation))) >= .5:
    if not ap:
        if activation > 0:
            return  0
        else: 
            return 1
    else:
        if activation > 0:
            return 1
        else:
            return 0

#Coodinates for random data
x1 = np.random.normal(mu, sigma, size = (1200, 1))
y1 = np.random.normal(mu, sigma, size = (1200, 1))

#500 instances of data classified as a 1 and data classified as 0 at a time
zeros = np.zeros((1200,1))
ones = np.ones((1200,1))

#Combine pairs of X and Y
data = np.append(x1, y1, axis = 1)
labels = label_data(data, [1, -1, -1])
data = np.append(data, labels, axis = 1)


#Training sets for each quadrant of data (quadrants 1 and 2 = 1, 3 and 4 = 0)
train =  data[:400]
test = data[400:]

#p_weights = p.gradient_descent(train_context_1, 0.15, 1000)
p_weights = [ 1       , -1,  -1] #fixed weights so we dont call GD every time we run

#Plot data and decision line
plt.figure("Context 1")
xx = np.linspace(0, 1, 10)
above = data[data[:, 2] == 0]
below = data[data[:,2] == 1]
plt.scatter(above[0:,0:1], above[0:,1:2], alpha=0.80, marker='^', label = "Context: 1, Class: 1")
plt.scatter(below[0:,0:1], below[0:,1:2], alpha=0.80, marker='o',  label = "Context: 1, Class: 0")

a = -p_weights[1]/p_weights[2]
#yy = a * xx - p_weights[0] / p_weights[2]
yy = (-1 / p_weights[2]) * p_weights[1] * xx + p_weights[0]
plt.plot(xx, yy, '-g', label = "Context 1 Weights")  # solid green
#plt.plot(x, (sgd_clf.intercept_[0] - (sgd_clf.coef_[0][0] * x)) / sgd_clf.coef_[0][1])
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
accuracy = 0
correct_context_1 = 0
context_2_weights = [0, 0, 0]
context_1_timer = 0
context_2_timer = 0
context_1 = True
context_2 = False
for i in range (test.shape[0]):
    context_1_timer += 1
    datum = test[i]
    
    if (p_predict(datum, p_weights) == datum[2]):
        correct_context_1 += 1
    
    if (i > 0):
        running_accuracy = correct_context_1 / (i + 1)
        print(running_accuracy)

#Context 2
correct_context_2 = 0
for i in range (test.shape[0]):
    datum = test[i]
    #Now we see accuracy in context 2 with context 1 weights
    if (p_predict(datum, p_weights) == datum[3]):
        correct_context_2 += 1
    
    running_accuracy = (correct_context_1 + correct_context_2) / (i + 801)
    print(running_accuracy)
    
    if (running_accuracy < .98):
        break
    
context_2_timer += 1
    #We start getting things wrong, so we need to switch to a new state with 
    #its own weights. Switch on how well youve been doing
        
    #Decide when to swtich
    #Update context 2 weights 
    #analyze
    #print(correct / 800)
        
#Plot context 2
"""
plt.figure("Context 2")
plt.scatter(test_q1_context_2[0:,0:1], test_q1_context_2[0:,1:2], alpha=0.80, marker='^')
plt.scatter(test_q2_context_2[0:,0:1], test_q2_context_2[0:,1:2], alpha=0.80, marker='^')
plt.scatter(test_q3_context_2[0:,0:1], test_q3_context_2[0:,1:2], alpha=0.80, marker='^')
plt.scatter(test_q4_context_2[0:,0:1], test_q4_context_2[0:,1:2], alpha=0.80)
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""

'''Helper Functions'''

    
def sigmoid(l, total_input, b, sig_idx):
    f = 1/ (1 + np.exp(-l[sig_idx] * (total_input - b[sig_idx])))
    #sig_index is the index of the activcation function we want   
    return f

#Plt.pause for debugging 
'''Weights'''
circuit_weights = np.array([[0,    -.3,      0,     -1],    #1->1  2->1, 3->1, 4->1
                    [-.3,    0,      -1,     0],            #1->2, 2->2, 3->2, 4->2
                    [1.23,  0,      2.087,  0],             #1->3, 2->3, 3->3, 4->3
                    [0,     1.23,   0,      2.087]])        #1->4, 2->4, 3->4  4->4
    

"""
def trial(w = circuit_weights, p = p_weights, ap = ap_weights, row_idx = 9, plot = False, plot_num = 1, contexts = 2):
    zeros = np.zeros((contexts - 1,61))
    p_context_weights = np.concatenate((p, zeros), axis = 0)
    ap_context_weights = np.concatenate((ap, zeros), axis = 0)                              
    correct = 0
    
    context = 0
   
    
    for i in range (0, plot_num): 
        steps = 0 
        tau = 1
        dt = 0.01
        weights = w
        context_timer = 0
        p_weights = p[context]
        ap_weights = ap[context]
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
            row = data[row_idx + plot_num]
            nn = np.random.normal(0, .1, (1, 60)) * sensor_noise
            
            nn = np.append(nn, data[row_idx][-1])
            noisyRow = np.add(nn, row)
            
            p_classification = p_predict(noisyRow, p_weights) * perceptron_activation_scalar + 1
            ap_classification = p_predict(noisyRow, ap_weights, ap=True) * perceptron_activation_scalar + 1
            
            steps += 1 
            
            activations = weights @ v                              #weighted sum of activations, inputs to each unit
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
    
        if (plot_num == 1 and v[2] >= decision_threshold):
            return [1, steps]
        if(plot_num == 1 and v[3] >= decision_threshold):
            return [-1, steps]
        if(i == plot_num-1):
            plt.show()
"""