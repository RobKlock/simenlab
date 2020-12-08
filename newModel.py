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
                row[column] = -1
            if(row[column] == 'M'):
                row[column] = 1
            else:
                row[column] = float(row[column])
    data = np.array(dataset)
    return data

def predict(row, weights, piecewise = False, unbounded = False, sigmoid = False, ap = False):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    if piecewise:
        if ap:
            #return activation
            if activation < -0.5:
                return 1
            else:
                return 0
        else:
            #return activation
            if activation > 0.5:
                return 1
            else:   
                return 0
    if unbounded:
        return activation
    if sigmoid:
        return  (1 / (1 + math.exp(-activation)))

def ap_predict(row,weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return (1 / (1 + math.exp(activation)))

def sigmoid(l, total_input, b, sig_idx):
    #squashes all input values to a range (0,1)
    f = 1/ (1 + np.exp(-l[sig_idx] * (total_input - b[sig_idx])))
    #sig_index is the index of the activcation function we want   
    return f

#M = 1, R = 0
data = load_data('sonar1.all-data')


#Get training samples 
train_data = data[np.random.choice(data.shape[0], 150, replace = True), :]
#rng = np.random.default_rng()

#train_data = rng.shuffle(data)

#train_data2 = data[np.random.choice(data.shape[0], 154, replace = False), :]

#Train perceptron, store weights
#p_weights = p.gradient_descent(train_data, 0.01, 600)
#ap_weights = p.gradient_descent(train_data, 0.01, 600, antiperceptron = True)
#ap_weights = [-0.55629858,  0.89203385,  0.34666907, -0.44980223,  1.04757964,  0.53757167,
# -0.40469565, -0.8770542,  -1.3698451,   0.21673824,  0.16285401,  0.63985225,
#  1.15393013, -0.80287275, -0.02125667,  0.63025255, -0.36644349,  0.11719943,
#  0.10822826, -0.146744,    0.15896858, -0.10971499, -0.04627238,  0.08041704,
#  0.27218873,  0.00867797, -0.13303348,  0.40030014, -0.3146386,  -0.05086975,
#  0.65502934, -1.0231504,   0.43352633, -0.11903075,  0.01172933,  0.57977145,
# -0.79967504, -0.04192846, -0.05788221,  0.49054684, -0.57015977, -0.05814346,
#  0.11253256, -0.26984774,  0.54954347,  0.32377145,  0.74961652,  0.39953977,
#  0.5553721,  0.79020755,  0.30655165,  0.95602438,  0.10446539,  0.99808969,
#  0.62234621,  0.21749132,  0.345626,    0.37328169,  0.48229927,  0.34150254,
#  1.04644338]
ap_weights = [ -0.33,       -0.019774,   -.027099,   -0.044141,   -0.08618,    -0.033431,
 0.0061105,  -0.0346125,  -0.0544955,  -0.0236925,  -0.0355805,  -0.1077745,
  -0.109684,   -0.0421365, 0.002039,  0.0857735, 0.048519,  0.00991,
  -0.027473,   -0.002009,  0.012095,   -0.053062,   -0.024014,   -0.08682,
  -0.036538,  0.0733745, 0.0986845,  -0.0095075,  -0.078641,  0.0389175,
  -0.0130335,  -0.0022795,  -0.117009,   -0.0269505, 0.0625085, 0.036384,
 0.0407765, 0.062081,   -0.118647,   -0.075233,  0.017397,   -0.0302475,
  -0.0456765,  -0.042965,   -0.072804,   -0.043611, 0.0191205,  -0.03065,
  -0.053188,   -0.0262975,  -0.0054895,  -0.01852,    -0.014349,   -0.015545,
  -0.00592,    -0.002709,   -0.006073,   -0.009757,   -0.007356,   -0.011061,
  -0.007584 ]

p_weights = [ 0.33,       0.019774,   0.027099,   0.044141,   0.08618,    0.033431,
 -0.0061105,  0.0346125,  0.0544955,  0.0236925,  0.0355805,  0.1077745,
  0.109684,   0.0421365, -0.002039,  -0.0857735, -0.048519,  -0.00991,
  0.027473,   0.002009,  -0.012095,   0.053062,   0.024014,   0.08682,
  0.036538,  -0.0733745, -0.0986845,  0.0095075,  0.078641,  -0.0389175,
  0.0130335,  0.0022795,  0.117009,   0.0269505, -0.0625085, -0.036384,
 -0.0407765, -0.062081,   0.118647,   0.075233,  -0.017397,   0.0302475,
  0.0456765,  0.042965,   0.072804,   0.043611,  -0.0191205,  0.03065,
  0.053188,   0.0262975,  0.0054895,  0.01852,    0.014349,   0.015545,
  0.00592,    0.002709,   0.006073,   0.009757,   0.007356,   0.011061,
  0.007584 ]

#ap_weights = p_weights @ -1 

def perceptron_accuracy(weights, ap=False):
    accuracy = 0
    right = 0
    for i in range (0,208):
        v = predict(data[i], weights, piecewise = True, )
        """
        if not ap:
            if v >= .5:
                v = 1
            if v < .5:
                v = 0
        else:
            if v < .5:
                v = 1
            if v >= .5:
                v = 0
        """   
        if v == data[i][-1]:
            right += 1
    
    accuracy = (right / 208) * 100
    return accuracy 
        

#Train antiperceptron, store weights 


weights = np.array([[0,    -2,      0,     -2],        #1->1  2->1, 3->1, 4->1
                    [-2,    0,      -2,     0],        #1->2, 2->2, 3->2, 4->2
                    [1.23,  0,      2.087,  0],            #1->3, 2->3, 3->3, 4->3
                    [0,     1.23,   0,      2.087]])       #1->4, 2->4, 3->4, 4->4
#To help tune up, run a version with just linear v1 and v2 
#Then, plot the difference, gather some info to guide the weights for the nonlinear model
    
#For a row n in the dataset
def perceptronval():
    for i in range(0,10):
    
        pp = p.gradient_descent(train_data, (0.2 - (0.15 * i)), 700)
        print(perceptron_accuracy(pp))
        print(pp)
    
    
def trial(w = weights, p = p_weights, ap = ap_weights, row_idx = 8, plot = False):
    steps = 0 
    tau = 1
    dt = 0.01
    weights = w
    p_weights = p
    ap_weights = ap
    internal_noise = 0.2
    sensor_noise = 0.2
    decision_threshold = 1
    v_hist = np.array([[0, 0, 0, 0]]).T    
    v = np.array([[0, 0, 0, 0]]).T              # values for each neuron
    p_hist = np.array([0])
    ap_hist = np.array([0])
    perceptron_activation_scalar = .8
    l = np.array([[4, 4, 4, 4]]).T                  #steepness of activation fxn
    bias = np.array([[1, 1, 1, 1]]).T 
    #bias is responsible for increasing the amount of input a neuron needs/doesnt need to activate 
    bias = bias * 1.1
    sig_idx= [0,1,2,3]
    #Repeatedly have perceptron and AP classify row, feed into circuit
    #until the circuit makes a classification (v3 or v4 is > 1)
    while (v[3] < decision_threshold) and (v[2] < decision_threshold):
        row = data[row_idx]
        #Make sure the data isnt negative
        nn = np.random.normal(0, .1, (1, 60)) * sensor_noise
        
        nn = np.append(nn, data[row_idx][-1])
        noisyRow = np.add(nn, row)
        #noisyData = np.vstack((noisyData, noisyRow))
        #P is HIGH for 1s
        p_classification = predict(noisyRow, p_weights, piecewise = True) * perceptron_activation_scalar
        #Histogram of response times 
        
        #AP is HIGH for 0s
        #ap_classification = 1 - ap_predict(noisyRow, p_weights) * perceptron_activation_scalar
        ap_classification = predict(noisyRow, ap_weights, piecewise = True, ap = True) * perceptron_activation_scalar
        steps += 1 
        
        
        activations = weights @ v    #weighted sum of activations, inputs to each unit
        activations[0] = p_classification + activations[0] #previously self_input[0] + stiulus
        activations[1] = ap_classification + activations[1] #previously self_input[1] + stimulus
        #activations = sigmoid(l, activations, bias, sig_idx)
        dv = tau * ((-v + activations) * dt + internal_noise * np.sqrt(dt) * np.random.normal(0,1, (4,1))) # add noise using np.random
        v = v + dv
        
        v_hist = np.concatenate((v_hist,v), axis=1)
        p_hist = np.append(p_hist, p_classification)
        ap_hist = np.append(ap_hist, ap_classification)
        #if (steps > 2000):
        #    break
            
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
        plt.title("Units 1-4 Activations")
        plt.show()
        
        #Historical classifcation of the noisy data by p and ap
        plt.figure()
    
        plt.plot(p_hist)
        plt.plot(ap_hist)
        plt.legend(["p classification", "ap classification"], loc=0)
        plt.title("Perceptron and Antiperceptron Activations")
        plt.show()
                
        plt.figure()
        plt.plot(v_hist[1,:] - v_hist[0,:])
        plt.ylabel("v1 v2 difference")
        plt.xlabel("time")
        plt.grid('on')
        plt.title("V1 V2 Difference (Drift Diffusion)")
        plt.show()
    
    #plot v2 - v1 
    #will probably always reach the same level when the network makes a decision
    #Unit 3 predicts 1s (metal), unit 4 predicts 0s (rocks)
    
    if(v[3] >= decision_threshold):
        return [1, steps]
    if(v[2] >= decision_threshold):
        return [-1, steps]
    #else:
    #    return [-1 , 1000]
    #Plot out a bunch of examples and superimpose them
    #The circuit should be classifying things perfectly 
def evaluate_circuit(n = 208, eval_perceptron = True):
        
    test_idxs = np.random.choice(data.shape[0], n, replace = False)
    correct_circuit = 0
    accuracy_circuit = 0.0
    
    correct_perceptron = 0
    accuracy_perceptron = 0.0
    #test_idxs = 208
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
            
            if p.predict(data[i], perceptron) == data[i][-1]:
                correct_perceptron += 1
    #Train perceptron on noisy data for as many epochs as it took the circuit to decide
    #determine perceptrons accuracy 
    
    #print(perceptron)
    accuracy_circuit = (correct_circuit / n) * 100
    accuracy_perceptron = (correct_perceptron / n) * 100
    print("circuit accuracy: ", accuracy_circuit)
    print("perceptron accuracy:", accuracy_perceptron)
    
#9-17 to dos: fix the circuit to increase accuracy 

#to do: make sure perceptron/v1/return val are all lined up for a given classification
#compare against a perceptron trained w the same amount of trials on noisy data


