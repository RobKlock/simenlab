#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:51:51 2020

@author: RobertKlock
Returns weights for p and ap so we dont retrain at every run
"""
import sys
sys.path.append('/Users/robertklock/Documents/Class/Fall20/SimenLab/simenlab')
from csv import reader
import numpy as np
import perceptron as p
import newModel as n

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

def perceptron_accuracy(weights, ap=False):
    accuracy = 0
    right = 0
    for i in range (0,208):
        if not ap:
            v = p.predict(data[i], weights = weights)
        else:
            v = p.predict(data[i], weights = weights, ap = True)
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
        if not ap:
            if v == data[i][-1]:
                right += 1
        else:
            if v != data[i][-1]:
                right += 1
    
    accuracy = (right / 208) * 100
    return accuracy 

#M = 1, R = 0
data = load_data('sonar1.all-data')
    
    
    #Get training samples 
#train_data = data[ np.random.choice(208, 75, replace = False),:]
train_data_0 = data[np.random.choice(97, 90, replace = False), :]
train_data_1 = data[np.random.choice(np.arange(97, 208), 90, replace = False), :]

train_data = np.concatenate((train_data_0, train_data_1), axis = 0)
np.random.shuffle(train_data)

#def main():
    #rng = np.random.default_rng()
    
    #train_data = rng.shuffle(data)
    
    #train_data2 = data[np.random.choice(data.shape[0], 154, replace = False), :]
    
    #Train perceptron, store weights
#p_weights = p.gradient_descent(train_data, 0.005, 900)
ap_weights = p.gradient_descent(train_data, 0.005, 900, antiperceptron = True)
#p.gradient_descent(train_data, 0.005, 900, antiperceptron = True)
print(ap_weights)
print(perceptron_accuracy(ap_weights, ap = True))
#print(p_weights)
#print(perceptron_accuracy(p_weights))
#print(ap_weights)
   
#return(ap_weights)
    
#main()

###89.423% accuracy for perceptron
"""
[ 0.415      0.3567415  0.2803695 -0.3303615  0.1614965  0.280231
 -0.0106485 -0.1240805 -0.346292   0.288558  -0.0213655  0.196917
  0.0558015  0.096608  -0.107925   0.0672705 -0.0314365 -0.1765775
  0.067532   0.0159805  0.1567985 -0.2244875  0.1458415 -0.0670525
  0.173409  -0.054886  -0.112063   0.1712425 -0.146532   0.008575
  0.2224625 -0.416645   0.2457135 -0.133196  -0.1065415  0.292066
 -0.1786365 -0.1106975 -0.0352985  0.146277  -0.20811    0.017258
  0.1354105 -0.054501   0.1112255  0.065701  -0.0407725 -0.205622
  0.653725   0.41649   -0.1497025  0.180644   0.112318   0.088075
  0.1160365 -0.0359335  0.0138575 -0.137104  -0.030442   0.062433
  0.0565735]
"""