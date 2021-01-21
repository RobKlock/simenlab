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
from matplotlib import colors
from sklearn.linear_model import SGDClassifier

#Version two just has a normal distribution of data with a line passing through 

#Set up dummy data. 4 normal distributions of points classified as a 1 or 0 with values between 0 and 1
mu, sigma, d = .5, 0.2, .3 # mean and standard deviation
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
    
#How can we use the circuit to change our weights?
    
def predict(c1_weights, c2_weights, datum, label, circuit_weights = circuit_weights, plot = False):                        
    steps = 0 
    tau = 1
    dt = 0.01
    context_timer = 0
    p_weights = c1_weights
    ap_weights = c2_weights
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

def unthresholded_predict(row, weights, ap=False):
    activation = weights[0]
    for i in range(0,2):
        activation += weights[i + 1] * row[i]
    return activation
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
context_1_weights = [ 1       , -1,  -1] #fixed weights so we dont call GD every time we run

#Plot data and decision line
plt.figure("Context 1")
xx = np.linspace(0, 1, 10)
above = data[data[:, 2] == 0]
below = data[data[:,2] == 1]
plt.scatter(above[0:,0:1], above[0:,1:2], alpha=0.80, marker='^', label = "Context: 1, Class: 1")
plt.scatter(below[0:,0:1], below[0:,1:2], alpha=0.80, marker='o',  label = "Context: 1, Class: 0")

a = -context_1_weights[1]/context_1_weights[2]
#yy = a * xx - p_weights[0] / p_weights[2]
yy = (-1 / context_1_weights[2]) * context_1_weights[1] * xx + context_1_weights[0]
plt.plot(xx, yy, '-g', label = "Context 1 Weights")  # solid green
#plt.plot(x, (sgd_clf.intercept_[0] - (sgd_clf.coef_[0][0] * x)) / sgd_clf.coef_[0][1])
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
plt.figure("Context 2")
above = data[data[:, 3] == 1]
below = data[data[:, 3] == 0]
plt.scatter(above[0:,0:1], above[0:,1:2], alpha=0.80, marker='o', label = "Context: 2, Class: 1")
plt.scatter(below[0:,0:1], below[0:,1:2], alpha=0.80, marker='^',  label = "Context: 2, Class: 0")
plt.show()
accuracy = 0
correct_context_1 = 0
context_2_weights = np.random.rand(3)
context_1_timer = 0
context_2_timer = 0
context_1 = True
context_2 = False
for i in range (test.shape[0]):
    context_1_timer += 1
    datum = test[i]
    
    if (p_predict(datum, context_1_weights) == datum[2]):
        correct_context_1 += 1
    
    if (i > 0):
        running_accuracy = correct_context_1 / (i + 1)
        #print(running_accuracy)

#Context 2
correct_context_2 = 0
left_off_index = 0
accuracy_threshold = 0.98
context_retrain = False
for i in range (test.shape[0]):
    datum = test[i]
    #Now we see accuracy in context 2 with context 1 weights
    if (p_predict(datum, context_1_weights) == datum[3]):
        correct_context_2 += 1
    
    running_accuracy = (correct_context_1 + correct_context_2) / (i + 801)
    #print(running_accuracy)
    
    if (running_accuracy < accuracy_threshold):
        left_off_index = i
        break

context_retrain = True 
retrain_weights = np.empty((50,3))
    
#cmap = colors.LinearSegmentedColormap('custom', cdict)

for i in range (left_off_index, test.shape[0]):
    context_2_timer += 1
    
    if context_retrain:
        for j in range (0, 50):
            #retrain weights
             # retrain a perceptron and antiperceptron on context 2 
             # have those compete
            circuit_values = predict(context_1_weights, context_2_weights, data[i+j,:2], data[i+j, 3]) #change to better name like "diffusion_predict"
            # >> Do we want to have two perceptrons competing here? OR is context1 and c2 weights sufficient
            # completely separate perceptron/antiperceptron pair 
           
            prediction = circuit_values[0]
            steps = circuit_values[1]
            #prediction = unthresholded_predict(datum, context_2_weights
            #make learning rate tied to prediction time
            error = datum[3] - prediction
            #Affine function of steps
            learning_rate = (.5 - ((steps * .0001)) / 2) #Any better ideas for this?
            
            #learning_rate = (1 / (steps + 0))
            #Explore making this logistic
            # >> Here 
            bias_delta = learning_rate * error
            context_2_weights[0] = context_2_weights[0] + (learning_rate * error)
            for k in range(1, 3):
                context_2_weights[k] = context_2_weights[k] + learning_rate * error * data[i+j, k-1]
                retrain_weights[j] = context_2_weights
            #print(context_2_weights)
            
        context_retrain = False 
    else: 
        prediction = p_predict(data[i], context_2_weights)
        if (prediction == datum[3]):
            correct_context_2 += 1
    
    if (i == test.shape[0] - 1):
        print(correct_context_2)
        plt.figure()
        xx = np.linspace(0,1,10)
        for j in range (0, 50):
            yy2 = (-1 / retrain_weights[j][2]) * retrain_weights[j][1] * xx + retrain_weights[j][0]
            plt.plot(xx, yy2, c = (0.1, 0.1, .02 * j, .02 * j), label = "Context 2 Weights") 
        plt.plot(xx, yy, '-r', label = "Context 1 Weights")
        #plt.axis([0, 10.0, -10.0, 10.0])
        plt.show()
        #print(p_predict(datum, context_2_weights))
    #print(predict(context_1_weights, context_2_weights, datum, datum[3]))
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



