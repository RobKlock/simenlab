f#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 17:25:28 2020

@author: Robert Klock
A model consisting of a perceptron, antiperceptron, and neural switch circuit
Goal: Accurate classification of noisy mine data 
"""

from random import seed
from random import randrange
from csv import reader
import copy
import math
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import preprocessing 
import sys
#If you have the perceptron.py file, add the path where it is here so we can import some of its functions
sys.path.append('/Users/robertklock/Documents/Class/Fall20/SimenLab/simenlab')

import perceptron as p

#Loads and processes our csv into floats, encodes M as 0 and R as 1
def load_data(filename):
    dataset = list() 
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            for i in range(0,60):
                row[i] = float(row[i])
            if(row[60] == 'M'):
                row[60] = float(0)
            if(row[60] == 'R'):
                row[60] = float(1)
            dataset.append(row)
    return dataset

def sigmoid(l, total_input, b):
    f = 1/ (1 + np.exp(-l * (total_input - b)))
    return f
     
dataset = np.asarray(load_data("sonar1.all-data"))
ROW_LENGTH = len(dataset[0])
COLUMNS = len(dataset)
#noisy_data = p.noisyData(dataset)
#perceptron = p.perceptron
#antiperceptron = p.antiperceptron


p_random_indices = np.random.choice(110, size = 110, replace = False)
p_train_data = dataset[p_random_indices, :110]

ap_random_indices = np.random.choice(110, size = 110, replace = False)
ap_train_data = dataset[ap_random_indices, :110]

#test_data = np.delete(noisy_data, random_indices, 0)
p_weights = p.gradient_descent(p_train_data, 0.1, 200)
ap_weights = p.gradient_descent(p_train_data, 0.1, 200, antiperceptron = True)

#scores = p.evaluate_algorithm(dataset, p.perceptron, 3, .01, 500)

c_weights = np.array([[0, -.5, 0, -.5],   #1->1 2->1 3->1 4->1
                      [-.5, 0, -.5, 0],   #1->2 2->2 3->2 4->2
                      [2.23, 0, 6, -6],  #1->3 2->3 3->3 4->3
                      [0, 2.23, 0, 2.5]]) #1->4 2->4 3->4 4->4

weights = np.array([[0,    -2,      0,     -2],  #1->1  2->1, 3->1, 4->1
                    [-2,    0,      -2,     0],  #1->2, 2->2, 3->2, 4->2
                    [1.23,  0,      2.087,  0],  #1->3, 2->3, 3->3, 4->3
                    [0,     1.23,   0,      2.087]])   #1->4, 2->4, 3->4, 4->4
                
  
l = np.array([[4,4,4,4]]).T
bias = np.array([[1,1,1,1]]).T

bias = bias * 1.3

v4 = list()

row = dataset[162,:]

#Make sure the data isnt negative
noise = np.random.rand(1, 60)
noise = np.append(noise, row[60])

def main():
    #Prove accuracy by comparing to perceptron and running over all examples 
    
    #For a row n in the dataset
    #for row in dataset:
    #row = dataset[162]
    v = np.array([[0,0,0,0]]).T
    #v_hist = np.array([[0, 0, 0, 0,]]).T  
    tau = 1
    dt = 0.05
    steps = 0 
    v_hist = np.array([[0, 0, 0, 0]]).T    
    #Repeat until the decision network makes a classification - output switch units, 
            # *need to decide when a decision is made 
            #Now, a decision is reached when the activation of unit 3 or 4 is >= 1.1
    while (v[3][0] <= 1.1) and (v[2][0] <= 1.1) and (steps < 10) :
        steps += 1 
        row = dataset[162]
        #Make sure the data isnt negative
        noise = np.random.rand(1, 60)
        noise = np.append(noise, 0)
        #print(noise)
        
        #Add noise to the row
        noisyRow = np.add(noise, row)
        
        #Let the perceptron and antiperceptron predict it - unthresholded
        p_prediction = p.predict(noisyRow, p_weights)
        ap_prediction = p.predict(noisyRow, ap_weights)
        
        #feed those values into the decision network
        activations = weights @ v 
        print(activations)
        activations[0][0] = p_prediction + activations[0][0]
        activations[1][0] = ap_prediction + activations[1][0]
        
        activations = sigmoid(l, activations, bias)    
        v_hist = np.concatenate((v_hist,v), axis=1)
        #if (steps > 5):
        #    print(signal.savgol_filter(v_hist[3,:], 5, 4))
        #if (steps == 5):
        #    print(p_prediction)
        #    print(ap_prediction)
        #    print(v_hist)
        dv = tau * ((-v + activations) * dt + noise * np.sqrt(dt) * np.random.normal(0,1, (4,1))) # add noise using np.random
        v = v + dv
        
        if (v[3][0] > 1) or (v[2][0] > 1):
            break
        
    
    plt.figure()
    plt.plot(v_hist[0,:], dashes = [2,2]) 
    plt.plot(v_hist[1,:], dashes = [1,1])
    plt.plot(v_hist[2,:], dashes = [2,2])
    plt.plot(v_hist[3,:], dashes = [3,3])
    #plt.plot(v2_v1_diff, dashes = [5,5])
    plt.legend(["v1","v2","v3","v4"], loc=0)
    plt.ylabel("activation")
    plt.xlabel("steps")
    plt.grid('on')
    plt.show()
    
    #plt.figure()
    #smoothed_v4 = signal.savgol_filter(v_hist[3,:], 901, 4)
    #smoothed_v3 = signal.savgol_filter(v_hist[2,:], 901, 4)
    #plt.plot(smoothed_v4)
    #plt.plot(smoothed_v3)
    #plt.legend(["smooth v4", "smooth v3"], loc = 0)
    #plt.grid('on')
    #plt.show()
    
    print(steps)
    print(steps * 60)


                
        #Repeat until the decision network makes a classification - output switch units, 
            # *need to decide when a decision is made 
    #Go to row n+1
        
    #alternative 1:
    #train perceptron and anti perceptron - gradient descent
    #train P and AP as normal,
        #Go back to data/validation set
        #Now run noisy rows through perceptron and antiperceptron and have decision network classify
    
    #Now we can change how we train
        #Add noise to rows before we train / during each iteration of training 
        #Decision network decides, then we adjust P and AP's weights (over the course of k steps = amt of steps it took to get a decision)
    
    #Compare that performance to a perceptron getting trained with noisy data over K steps
    #Adjust weights on every single sample
    
        
    #Train the perceptron based on those decision netowrks' classifications
    #compare to perceptron using equal number of steps 
"""
    for i in range(0, len(dataset)):
        print(p.predict(dataset[i], p_weights))
        print(p.predict(dataset[i], ap_weights))
        print(p.predict(dataset[i], p_weights) + p.predict(dataset[i], ap_weights))
        print("")
    
    noise = 0.00 #noise inherent in the brain
   
    dt = 0.5 #dt should roughly be proportional to the amount of data we process
    #in 9-22, it was 0.005 for 5000 steps, so 25/x
    tau = 1
    p_activations = np.array([])
    ap_activations = np.array([])
    v = np.array([[0,0,0,0]]).T
    v_hist = np.array([[0, 0, 0, 0,]]).T   
    
  
    #rand_index = np.random.randint(0,208,1)[0]
    rand_index = 166
    print(rand_index)
    #for each row processed...
    for j in range (0, ROW_LENGTH - 1):
        
        #p_activations = np.insert(p_activations, p.process(dataset[0][i], i, p_weights))
        #ap_activations = np.insert(ap_activations, p.process(dataset[0][i], i, ap_weights))
        p_activation = p.process(dataset[rand_index][j], j, p_weights)
        #p_activation = (1 / (1 + math.exp(-(p.process(dataset[rand_index][j], j, p_weights)))))
        #print("P activation", p_activation)
        ap_activation = p.process(dataset[rand_index][j], j, ap_weights)
        #ap_activation = (1 / (1 + math.exp(-(p.process(dataset[rand_index][j], j, ap_weights)))))
        #print("P, AP:", p_activation, ap_activation)
        p_activations = np.append(p_activations, p_activation)
        ap_activations = np.append(ap_activations, ap_activation)
                                  
        activations = c_weights @ v
        #print(v)
        activations[0][0] = p_activation + activations[0][0]
        activations[1][0] = ap_activation + activations[1][0]
        
        activations = sigmoid(l, activations, bias)    
        v_hist = np.concatenate((v_hist,v), axis=1)
        
        dv = tau * ((-v + activations) * dt + noise * np.sqrt(dt) * np.random.normal(0,1, (4,1))) # add noise using np.random
        v = v + dv
        
#    print("Trial %1d : %1d" % (j, dataset[rand_index][60]))
    print(p.predict(dataset[rand_index], p_weights))
    
    vio_plot_data = [p_activations, ap_activations]
    violinplot = plt.figure()
    ax = violinplot.add_axes([0,0,1,1])
    bp = ax.violinplot(vio_plot_data)
    plt.grid('on')
    plt.show()
    
    plt.figure()
    plt.plot(p_activations)
    plt.plot(ap_activations)
    plt.show()
    
    plt.figure()
    plt.plot(v_hist[0,:], dashes = [2,2]) 
    plt.plot(v_hist[1,:], dashes = [1,1])
    plt.plot(v_hist[2,:], dashes = [2,2])
    plt.plot(v_hist[3,:], dashes = [3,3])
    #plt.plot(v2_v1_diff, dashes = [5,5])
    plt.legend(["v1","v2","v3","v4"], loc=0)
    plt.ylabel("activation")
    plt.xlabel("columns of data")
    plt.grid('on')
    plt.show()
    """
    

        
        