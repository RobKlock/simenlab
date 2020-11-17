#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 00:06:22 2020

@author: Rob Klock
Restarting! AGGGGGHHH!!!!!!!

"""

import numpy as np
from csv import reader
import math
import sys
sys.path.append('/Users/robertklock/Documents/Class/Fall20/SimenLab/simenlab')
import perceptron as p
import matplotlib.pyplot as plt

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
                row[column] = 0
            if(row[column] == 'M'):
                row[column] = 1
            else:
                row[column] = float(row[column])
    data = np.array(dataset)
    return data

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    
    return  (1 / (1 + math.exp(-activation)))

def sigmoid(l, total_input, b, sig_idx):
    #squashes all input values to a range (0,1)
    f = 1/ (1 + np.exp(-l[sig_idx] * (total_input - b[sig_idx])))
    #sig_index is the index of the activcation function we want 
    
    return f

#M = 1, R = 0
data = load_data('sonar1.all-data')

#Get training samples 
train_data = data[np.random.choice(data.shape[0], 104, replace = False), :]

#Train perceptron, store weights
p_weights = p.gradient_descent(train_data, 0.1, 100)

#Train antiperceptron, store weights 
ap_weights = p.gradient_descent(train_data, 0.1, 100, antiperceptron = True)

weights = np.array([[0,    -2,      0,     -2],  #1->1  2->1, 3->1, 4->1
                    [-2,    0,      -2,     0],  #1->2, 2->2, 3->2, 4->2
                    [1.23,  0,      2.087,  0],  #1->3, 2->3, 3->3, 4->3
                    [0,     1.23,   0,      2.087]])   #1->4, 2->4, 3->4, 4->4

#For a row n in the dataset
def trial(w, p, ap, row_idx, plot = False):
    steps = 0 
    tau = 1
    dt = 0.1
    weights = w
    p_weights = p
    ap_weights = ap
    noise = 0.15
    v_hist = np.array([[0, 0, 0, 0]]).T    
    v = np.array([[0, 0, 0, 0]]).T              # values for each neuron
    
    l = np.array([[4, 4, 4, 4]]).T                  #steepness of activation fxn
    bias = np.array([[1, 1, 1, 1]]).T 
    #bias is responsible for increasing the amount of input a neuron needs/doesnt need to activate 
    bias = bias * 1.23
    sig_idx= [0,1,2,3]
    
    #Repeatedly have perceptron and AP classify row, feed into circuit
    #until the circuit makes a classification (v3 or v4 is > 1)
    while (v[3] < 1) and (v[2] < 1):
        row = data[row_idx]
        #Make sure the data isnt negative
        nn = np.random.rand(1, 60)
        nn = np.append(nn, data[row_idx][-1])
        noisyRow = np.add(nn, row)
        #noisyData = np.vstack((noisyData, noisyRow))
        #P is HIGH for 1s
        stimulus = predict(noisyRow, p_weights)
        #AP is HIGH for 0s
        stimulus_delta = predict(noisyRow, ap_weights)
        steps += 1 
        
        activations = weights @ v    #weighted sum of activations, inputs to each unit
        activations[0] = stimulus + activations[0] #previously self_input[0] + stiulus
        activations[1] = stimulus_delta + activations[1] #previously self_input[1] + stimulus
        activations = sigmoid(l, activations, bias, sig_idx)
        dv = tau * ((-v + activations) * dt + noise * np.sqrt(dt) * np.random.normal(0,1, (4,1))) # add noise using np.random
        v = v + dv
        
        v_hist = np.concatenate((v_hist,v), axis=1)
    
    #The issue was there were two variables named noise, one for adding noise to the row and one for noise in dv
    if plot:
        plt.figure()
        plt.plot(v_hist[0,:]) 
        plt.plot(v_hist[1,:])
        plt.plot(v_hist[2,:])
        plt.plot(v_hist[3,:])
        plt.legend(["v1","v2","v3","v4"], loc=0)
        plt.ylabel("activation")
        plt.xlabel("time")
        plt.grid('on')
        plt.show()
    #plot v2 - v1 
    #will probably always reach the same level when the network makes a decision
    #Unit 3 predicts 1s (metal), unit 4 predicts 0s (rocks)
    if(v[3] >= 1):
        return [0, steps]
    if(v[2] >= 1):
        return [1, steps]
  
def evaluate_circuit(n = 208, eval_perceptron = True):
        
    test_idxs = np.random.choice(data.shape[0], n, replace = False)
    correct_circuit = 0
    accuracy_circuit = 0.0
    
    correct_perceptron = 0
    accuracy_perceptron = 0.0
    
    for i in test_idxs:
        #print(trial(weights, p_weights, ap_weights, i, plot = False))
        #print(data[i][-1])
        #print()
        trial_result = trial(weights, p_weights, ap_weights, i, plot = False)
        if trial_result[0] == data[i][-1]:
            correct_circuit += 1
        
        if(eval_perceptron):
            #This uses multiple noisy training rows over n epochs, should we instead train the perceptron on a single noisy row n times?
            perceptron_training_data = data[test_idxs, :-1]
            perceptron_noisy_data = perceptron_training_data + np.random.rand(n, 60) 
            classes = data[[test_idxs], [-1]]
            classes = classes.reshape(n, 1)           
            perceptron_noisy_data = np.append(perceptron_noisy_data, classes, axis = 1)
            
            perceptron = p.gradient_descent(perceptron_noisy_data, 0.1, trial_result[1])
            
            if p.predict(data[i], perceptron):
                correct_perceptron += 1
    #Train perceptron on noisy data for as many epochs as it took the circuit to decide
    #determine perceptrons accuracy 
    
    #print(perceptron)
    accuracy_circuit = (correct_circuit / n) * 100
    accuracy_perceptron = (correct_perceptron / n) * 100
    print("circuit accuracy: ", accuracy_circuit)
    print("perceptron accuracy:", accuracy_perceptron)
    
    

#to do: make sure perceptron/v1/return val are all lined up for a given classification
#compare against a perceptron trained w the same amount of trials on noisy data


