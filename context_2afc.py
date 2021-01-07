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

#Set up dummy data. 4 normal distributions of points classified as a 1 or 0 with values between 0 and 1
mu, sigma, d = .5, 0.06, .3 # mean and standard deviation

#Coodinates for random data
x1 = np.random.normal(mu + d, sigma, size = (700, 1))
y1 = np.random.normal(mu + d, sigma, size = (700, 1))
x2 = np.random.normal(mu - d, sigma, size = (700, 1))
y2 = np.random.normal(mu + d, sigma, size = (700, 1))
x3 = np.random.normal(mu - d, sigma, size = (700, 1))
y3 = np.random.normal(mu - d, sigma, size = (700, 1))
x4 = np.random.normal(mu + d, sigma, size = (700, 1))
y4 = np.random.normal(mu - d, sigma, size = (700, 1))

#500 instances of data classified as a 1 and data classified as 0 at a time
zeros = np.zeros((500,1))
ones = np.ones((500,1))

#Combine pairs of X and Y data for each quadrant
quad1 = np.append(x1, y1, axis = 1)
quad2 = np.append(x2, y2, axis = 1)
quad3 = np.append(x3, y3, axis = 1)
quad4 = np.append(x4, y4, axis = 1)


#Training sets for each quadrant of data (quadrants 1 and 2 = 1, 3 and 4 = 0)
train_q1 =  np.append(quad1[:300], ones[:300], axis = 1)
train_q2 =  np.append(quad2[:300], ones[:300], axis = 1)
train_q3 =  np.append(quad3[:300], zeros[:300], axis = 1)
train_q4 =  np.append(quad4[:300], zeros[:300], axis = 1)
test_q1 = np.append(quad1[300:], ones[:400], axis = 1)
test_q2 = np.append(quad2[300:], ones[:400], axis = 1)
test_q3 = np.append(quad3[300:], zeros[:400], axis = 1)
test_q4 = np.append(quad4[300:], zeros[:400], axis = 1)

#Training sets for context 1
train_context_1 =  np.append(train_q1, train_q2, axis = 0)
train_context_1 =  np.append(train_context_1, train_q3, axis = 0)
train_context_1 =  np.append(train_context_1, train_q4, axis = 0)
np.random.shuffle(train_context_1)

#Testing set for context 1
test_context_1 =  np.append(test_q1, test_q2, axis = 0)
test_context_1 =  np.append(test_context_1, test_q3, axis = 0)
test_context_1 =  np.append(test_context_1, test_q4, axis = 0)

#Testing set for context 2 (quadrants 1, 2, and 4 are now 1, quadrant 3 is 0)
test_q1_context_2 = np.append(quad1[300:], ones[:400], axis = 1)
test_q2_context_2 = np.append(quad2[300:], ones[:400], axis = 1)
test_q3_context_2 = np.append(quad3[300:], ones[:400], axis = 1)
test_q4_context_2 = np.append(quad4[300:], zeros[:400], axis = 1)

test_context_2 =  np.append(test_q1_context_2, test_q2_context_2, axis = 0)
test_context_2 =  np.append(test_context_2, test_q3_context_2, axis = 0)
test_context_2 =  np.append(test_context_2, test_q4_context_2, axis = 0)

#Context 2 data (quadrants 1,2, 4 are 1, quadrant 3 is 0)


#test_c1 = np.append(train_c1[, ones[150:], axis = 1)
#sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
#sgd_clf.fit(train_c1[:,:2], train_c1[:,2])

#p_weights = p.gradient_descent(train_context_1, 0.15, 1000)
p_weights = [ 0.3       , -0.02454649,  0.34936848]
xx = np.linspace(0, 1, 10)
plt.scatter(quad1[0:,0:1], quad1[0:,1:2], alpha=0.80)
plt.scatter(quad2[0:,0:1], quad2[0:,1:2], alpha=0.80)
plt.scatter(quad3[0:,0:1], quad3[0:,1:2], alpha=0.80)
plt.scatter(quad4[0:,0:1], quad4[0:,1:2], alpha=0.80)
a = -p_weights[1]/p_weights[2]
#yy = a * xx - p_weights[0] / p_weights[2]
yy = (-1 / p_weights[2]) * p_weights[1] * xx + p_weights[0]
plt.plot(xx, yy, '-g')  # solid green
#plt.plot(x, (sgd_clf.intercept_[0] - (sgd_clf.coef_[0][0] * x)) / sgd_clf.coef_[0][1])
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

'''Helper Functions'''
#Loads in sonar data. 
#Rocks are -1, metal is 1
def load_data(filename):
    #Load sonar data
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    # Convert string column to float
    for row in dataset:
        for column in range(0,  len(row)):
            if(row[column] == 'R'):
                row[column] = -1
            if(row[column] == 'M'):
                row[column] = 1
            else:
                row[column] = float(row[column])
    data = np.array(dataset)
    return data

#Weighted sum prediction
def p_predict(row, weights, ap=False):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    #if (1/(1 + math.exp(-activation))) >= .5:
    if not ap:
        if activation > 0.5:
            return  1
        else: 
            return 0
    else:
        if activation > 0.5:
            return 1
        else:
            return 0
    
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
    

data = load_data('sonar1.all-data')
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