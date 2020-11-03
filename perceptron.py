#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:06:26 2020

@author: Rob Klock
A three unit perceptron for classification. Prep for more complicated models 
INPUT : strength of sonar chirps bouncing off different services
OUTPUT : M or R, for "mine" or "rock", expressed as 1 or 0
"""
from random import seed
from random import randrange
from csv import reader
import math
import matplotlib.pyplot as plt

#Load sonar data
def load_data(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        if(row[column] == 'R'):
            row[column] = 0
        if(row[column] == 'M'):
            row[column] = 1
        else:
            row[column] = float(row[column])

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
        #for each fold, add random data from the dataset
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

def activation(weight_i, x_i, bias):
    
    return (weight_i * x_i) + bias

 #step transfer function
def transfer(act):
    #return 1/(1+math.exp(-act))
    if (act > 0):
        return 1
    else:
        return -1

def weight_iteration(w, learning_rate, expected, predicted, input_val):
    
    return w + learning_rate * (expected - predicted) * input_val

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    
    return  1.0 if activation >= 0.0 else 0.0

#weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

def gradient_descent(training_data, learning_rate, num_epoch, antiperceptron = False):
    #outputs a set of trained weights
    #initialize each weight to zero, bias is first index
    weights = [0.0 for i in range(len(training_data[0]))]
    error_arr = []
    for epoch in range(num_epoch):
        sum_error = 0.0
        for row in training_data:
            #make a prediction with the given weights
            prediction = predict(row, weights)
            #calculate error
            #adjust here to make anti-perceptron (1 - row[-1])
            #error = (1 - row[-1]) - prediction
            error = row[-1] - prediction if not antiperceptron else (1 - row[-1]) - prediction
            #sum and square error
            sum_error += error**2
            #update bias
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row)-1):
                #update weights according to weight_i = weight_i + (learning_rate*error*input)
                weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]
        error_arr.append(sum_error)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
    #print("weights:", weights)
    #plt.figure()
    #plt.plot(weights, 'bo')
    #plt.scatter(x = range(61), y = weights)
    #plt.show()
    #plt.figure()
    #plt.plot(error_arr)
    #plt.show()
    return weights

def perceptron(train, test, learning_rate, num_epoch):
    predictions = list()
    weights = gradient_descent(train, learning_rate, num_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return(predictions)
    
#seed(1)
filename='sonar(1).all-data'
dataset=load_data(filename)

#This preps the data (makes R M numeric, for example)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0])-1)

dataset2 = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
weights = gradient_descent(dataset2, 0.01, 20)

n_folds = 2
l_rate = 0.01
n_epoch = 200

#plt.figure()
#scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
#print('Scores: %s' % scores)
#print('mean accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

#for row in dataset:
#	prediction = predict(row, weights)
#	print("Expected=%d, Predicted=%d" % (row[-1], prediction))