#!/usr/bin/env python3
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
import scipy as signal
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
noisy_data = p.noisyData(dataset)
perceptron = p.perceptron
antiperceptron = p.antiperceptron

p_random_indices = np.random.choice(101, size = 101, replace = False)
p_train_data = noisy_data[p_random_indices, :61]

ap_random_indices = np.random.choice(101, size = 101, replace = False)
ap_train_data = noisy_data[ap_random_indices, :61]

#test_data = np.delete(noisy_data, random_indices, 0)
p_weights = p.gradient_descent(p_train_data, 0.1, 50)
ap_weights = p.gradient_descent(ap_train_data, 0.1, 50, antiperceptron = True)

c_weights = np.array([[0, .5, 0, -3],   #1->1 2->1 3->1 4->1
                      [.5, 0, -3, 0],   #1->2 2->2 3->2 4->2
                      [2.23, 0, 2.5, 0],  #1->3 2->3 3->3 4->3
                      [0, 2.23, 0, 2.5]]) #1->4 2->4 3->4 4->4

  
l = np.array([[2,2,2,2]]).T
bias = np.array([[1,1,1,1]]).T

bias = bias * 1
    

def main():
    noise = 0.05 #noise inherent in the brain
   
    dt = 0.4 #dt should roughly be proportional to the amount of data we process
    #in 9-22, it was 0.005 for 5000 steps, so 
    tau = 1
    p_activations = np.array([])
    ap_activations = np.array([])
    v = np.array([[0,0,0,0]]).T
    v_hist = np.array([[0, 0, 0, 0,]]).T   
    
  
    #rand_index = np.random.randint(0,208,1)[0]
    rand_index = 162
    print(rand_index)
    #for each row processed...
    for j in range (0, ROW_LENGTH - 1):
        
        #p_activations = np.insert(p_activations, p.process(dataset[0][i], i, p_weights))
        #ap_activations = np.insert(ap_activations, p.process(dataset[0][i], i, ap_weights))
        p_activation = p.process(dataset[rand_index][j], j, p_weights)
        #print("P activation", p_activation)
        ap_activation = p.process(dataset[rand_index][j], j, ap_weights)
        #print("P, AP:", p_activation, ap_activation)
        
        p_activations = np.append(p_activations, p_activation)
        ap_activations = np.append(ap_activations, ap_activation)
                                  
        activations = c_weights @ v
        activations[0][0] = p_activation + activations[0][0]
        activations[1][0] = ap_activation + activations[1][0]
        
        activations = sigmoid(l, activations, bias)    
        v_hist = np.concatenate((v_hist,v), axis=1)
        
        dv = tau * ((-v + activations) * dt + noise * np.sqrt(dt) * np.random.normal(0,1, (4,1))) # add noise using np.random
        v = v + dv
        
#    print("Trial %1d : %1d" % (j, dataset[rand_index][60]))

    plt.figure()
    plt.plot(v_hist[0,:], dashes = [2,2]) 
    plt.plot(v_hist[1,:], dashes = [1,1])
    plt.plot(v_hist[2,:], dashes = [2,2])
    plt.plot(v_hist[3,:], dashes = [3,3])
    #plt.plot(v2_v1_diff, dashes = [5,5])
    plt.legend(["v1","v2","v3","v4"], loc=0)
    plt.ylabel("activation")
    plt.xlabel("time (arbitrary units)")
    plt.grid('on')
    plt.show()
    
main()
        
        