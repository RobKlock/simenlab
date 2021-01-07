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
import copy
import math
import numpy as np
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
            row[column] = 1
        if(row[column] == 'M'):
            row[column] = -1
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
    #construct and evaluate k models, then estimate the overall performance as the mean of all models' errors
    #each model is evaluated on its classification accuracy
#Returns an array with split up data
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    #print("datasetcopy:", dataset_copy)
    #fold size is how many elements are in each fold
    fold_size = int(len(dataset) / n_folds)
    #print("foldsize:", fold_size)
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
    #*args is learning rate and number of epochs
    #Get the subsections of trained data
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        #print("trainset", train_set)
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        #call the perceptron to train and test itself
        predicted = algorithm(train_set, test_set, *args)
        
        #print("predicted", predicted)
        #actual values are penultimate entry in each fold row
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def predict(row, weights, ap=False):
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
    #if (activation > 0):
    #    return 1
    #if (activation < 0):
    #    return 0
    #else:
    #    return 0 
    #return  (1 / (1 + math.exp(-activation)))

def weightedSum(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
        
def ap_predict(row,weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return (1 / (1 + math.exp(activation)))

def process(datum, index, weights):
    activation = weights[0]
    activation += weights[index + 1] * datum
    
    return activation

def gradient_descent(training_data, learning_rate, num_epoch, antiperceptron = False, plot = False, plotMany = False):
    #outputs a set of trained weights
    #initialize each weight to zero, bias is first index
    #weights = np.random.rand(len(training_data[0]))
    weights = np.zeros(len(training_data[0]))
    error_arr = np.array([])
    for epoch in range(num_epoch):
        sum_error = 0.0
        #learning_rate = ((num_epoch/100) - epoch) * learning_rate
        for row in training_data:
            #make a prediction with the given weights
            prediction = predict(row, weights)
            #calculate error
            #adjust here to make anti-perceptron (1 - row[-1])
            #error = (1 - row[-1]) - prediction
            if not antiperceptron:
                error = row[-1] - prediction
            else:
                error = (-1 * (row[-1] - 1)) - prediction
                #error = (row[-1] - prediction) * -1 
           
            #error = row[-1] - prediction if not antiperceptron else (1 - row[-1]) - prediction
            #sum and square error
            sum_error += error**2
            
            
            #update bias
            bias_delta = learning_rate * error
            weights[0] = weights[0] + bias_delta
            #Plot the last ten 
            for i in range(len(row) - 1):
                #update weights according to weight_i = weight_i + (learning_rate*error*input)
                weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]
        np.append(error_arr, sum_error)
    
    if plot:
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
        print("weights:", weights)
        plt.figure()
        plt.plot(weights, 'bo')
    return weights

#Perceptron Function
    #Take in a training set, test set, learning rate, and epoch number
    #Learns weights based on a training set, then returns predictions for the test set
def perceptron(train, test, learning_rate, num_epoch):
    predictions = list()
    weights = gradient_descent(train, learning_rate, num_epoch)
    #plt.figure(2)
    #plt.plot(weights, label = "perceptron weights")
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return(predictions)

def antiperceptron(train, test, learning_rate, num_epoch):
    predictions = list()
    weights = gradient_descent(train, learning_rate, num_epoch, antiperceptron = True)
    #plt.figure(2)
    #plt.plot(weights, label = "antiperceptron weights")
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return(predictions)

"""
class antiperceptron:
    def __init__(self, weights, learning_rate):
  """     

#Add Gaussian noisy data to sonar data
#Default makes some of the mine data negative 
def noisyData(dataset, mu = 0, sigma = 0.05):
    #dataset_noisy = dataset[:]
    #print(dataset_noisy)
    
    dataset_noisy = copy.deepcopy(dataset)
    #print("inside function, dataset id: ", id(dataset))
    #print("inside function, dataset id: ", id(dataset_noisy))
    for row in range(len(dataset_noisy)):
        #print("row:", dataset_noisy[row])
        for datum in range (len(dataset_noisy[row]) - 1):
            d = dataset_noisy[row][datum]
            n = d + np.random.normal(mu, sigma, 1)
            dataset_noisy[row][datum] = n[0]
    
    return dataset_noisy

def main():
    filename='sonar1.all-data'
    dataset=load_data(filename)
    #This preps the data (makes R M numeric, for example)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
        str_column_to_int(dataset, len(dataset[0])-1)
    
    
    n_folds = 6
    l_rate = 0.01
    n_epoch = 50
    scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
    #antiscores = evaluate_algorithm(dataset, antiperceptron, n_folds, l_rate, n_epoch)
    print('Scores: %s' % scores)
    #print('Anti Scores: %s' % antiscores)
    print('mean accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

    
#dataset3 = [[1,2,3,0],[4,5,6,1]]
#print("dataset3 id:", id(dataset3))
#noisyData3 = list(noisyData(dataset))
#print("noisydata3 id", id(noisyData3))


"""
#train them at the same time
scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
antiscores = evaluate_algorithm(dataset, antiperceptron, n_folds, l_rate, n_epoch)
plt.figure(1)
plt.plot(scores, 'bo')
plt.plot(antiscores, 'ro')
plt.show()
print('Scores: %s' % scores)
print('mean accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
"""

#for row in dataset:
#	prediction = predict(row, weights)
#	print("Expected=%d, Predicted=%d" % (row[-1], prediction))

###
#def activation(weight_i, x_i, bias):
#    return (weight_i * x_i) + bias
#
# #step transfer function
#def transfer(act):
#    #return 1/(1+math.exp(-act))
#    if (act > 0):
#        return 1
#    else:
#        return -1

#def weight_iteration(w, learning_rate, expected, predicted, input_val):
#    return w + learning_rate * (expected - predicted) * input_val
