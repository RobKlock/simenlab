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
		row[column] = float(row[column].strip())

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
    
    if (act > 0):
        return 1
    else:
        return 0

def weight_iteration(w, learning_rate, expected, predicted, input_val):
    
    return w + learning_rate * (expected - predicted) * input_val

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    
    return 1.0 if activation >= 0.0 else 0.0

#weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

def stochastic_gradient_descent(data, learning_rate, num_epoch):
    weights = [0.0 for i in range(len(data[0]))]
    for epoch in range(num_epoch):
        sum_error = 0.0
        for row in data:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
    return weights

def perceptron(train, test, learning_rate, num_epoch):
    predictions = list()
    weights = stochastic_gradient_descent(train, learning_rate, num_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return(predictions)
    
seed(1)
filename='sonar.all-data'
dataset=load_data(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0])-1)

n_folds = 6
l_rate = 0.005
n_epoch = 1000
scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('mean accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

#for row in dataset:
#	prediction = predict(row, weights)
#	print("Expected=%d, Predicted=%d" % (row[-1], prediction))