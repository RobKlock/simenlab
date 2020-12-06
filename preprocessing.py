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

def main():
    #M = 1, R = 0
    data = load_data('sonar1.all-data')
    
    
    #Get training samples 
    train_data = data[np.random.choice(data.shape[0], 150, replace = True), :]
    #rng = np.random.default_rng()
    
    #train_data = rng.shuffle(data)
    
    #train_data2 = data[np.random.choice(data.shape[0], 154, replace = False), :]
    
    #Train perceptron, store weights
    p_weights = p.gradient_descent(train_data, 0.01, 600)
    ap_weights = p.gradient_descent(train_data, 0.01, 600, antiperceptron = True)
    print(p_weights)
    print(ap_weights)
    return([p_weights, ap_weights])
    
main()