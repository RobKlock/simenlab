#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:18:41 2020
For our meeting on September 22nd
This code implements the decision making circuit comprised of four neurons

I+delta -> O -> Switch<-
          . |
          | .
      I -> 0 -> Switch <-

@author: robertklock
"""
import noise
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
#Perlin Noise function, not imporant
#adapted from Adrian's Soapbox via https://adrianb.io/2014/08/09/perlinnoise.html
def pNoise(): 
    octaves = 10        #levels of "detail
    persistence = 1.8   #amplitude
    lacunarity = 2      #how much the detail is altered at each octave
    return noise.pnoise2(0, 1, octaves = octaves, persistence = persistence, lacunarity = lacunarity, repeatx = 12, repeaty = 10, base = 10)

"""Plotting functions"""    
#please read the check_initial function's comment first, check is a modification on check initial
def check(v,l,b,weights,n, func): 
    #plots the activation function against <reference line?>, at any particular time step, depends on when you call it
    lmda = l[n-1,0] #lambda of node n (n here starts at 1, correspond to node number in matrix, so -1 to get index)
    b = b[n-1][0] - (weights[n-1,0:n-1] @ v[0:n-1,:])- (weights[n-1,n:] @ v[n:,:]) #subtract external input from bias
    # feeadback input + external input - bias is same as feedback input - (bias - external input)
    w = weights[n-1,n-1] #self feedback weight of node n
    stepsize = 0.05
    if w<0:
        stepsize=stepsize*(-1)
    feedback_input = np.arange(-(w+stepsize+2),w+stepsize+2,stepsize)
    
    if func == "s": #sigmoid
        f = 1/ (1 + np.exp(-lmda * (feedback_input - b)))
    elif func == "r": #"relu"
        x = feedback_input.reshape(-1,1)
        temp = lmda*(x-b)
       
        temp = np.concatenate((np.zeros((temp.shape[0],1)),temp),axis = 1)
        temp = np.max(temp,axis = 1).reshape(-1,1)
        temp = np.concatenate((np.ones((temp.shape[0],1)),temp),axis = 1)
        f = np.min(temp, axis =1 ).reshape(-1,1)
    #reference line
    #plt.plot(np.arange(0,w,0.1),1/w*np.arange(0,w,0.1))
    plt.plot(feedback_input,1/w*feedback_input) #output * w = feedback input, so output = 1/w* feedback input

    #activation function
    plt.plot(feedback_input,f)    
    plt.xlabel("self feedback input at t+1")
    plt.ylabel("self output at t")
    if func == "s":
        func = "sigmoid"
    else:
        func = "relu"
            
    plt.legend(["reference",func], loc=0)
    plt.grid()
    plt.title("v"+str(n)+"_current")
    plt.show()
    
def check_initial(v,l,b,weights,n, func): 
    #plots the initial activation function position, func specifies whether it's sigmoid ("s") or relu ("r")
    lmda = l[n-1,0]  #lambda of node n (n here starts at 1, correspond to node number in matrix, so -1 to get index)
    b = b[n-1][0]   #bias of node n
    w = weights[n-1,n-1]  #self feedback weight of node n
    stepsize = 0.1
    if w<0:
        stepsize=stepsize*(-1)
    feedback_input = np.arange(-(w+stepsize+2),w+stepsize+2,stepsize)
    
    if func == "s": #sigmoid
        f = 1/ (1 + np.exp(-lmda * (feedback_input - b)))
    elif func == "r": #"relu"
        x = feedback_input.reshape(-1,1)
        temp = l*(x-b)
       
        temp = np.concatenate((np.zeros((temp.shape[0],1)),temp),axis = 1)
        temp = np.max(temp,axis = 1).reshape(-1,1)
        temp = np.concatenate((np.ones((temp.shape[0],1)),temp),axis = 1)
        f = np.min(temp, axis =1 ).reshape(-1,1)
    #reference line
    #plt.plot(np.arange(0,w,0.1),1/w*np.arange(0,w,0.1))
    plt.plot(feedback_input,1/w*feedback_input) #output * w = feedback input, so output = 1/w* feedback input

    #activation function
    plt.plot(feedback_input,f)    
    plt.xlabel("self feedback input at t+1")
    plt.ylabel("self output at t")
    if func == "s":
        func = "sigmoid"
    else:
        func = "relu"
            
    plt.legend(["reference",func], loc=0)
    plt.grid()
    plt.title("v"+str(n)+"_initial")
    plt.show()
"""End Plotting Functions"""

def piecewise(step, threshold):
    #step is the current step of the function, 
    #threshold is the step in which the value will become one (0 until step 2000 (threshold) then 1)
    if (step < threshold):
        return 0
    else:
        return 1
    
#capacitor activation function
def sigmoid(l, total_input, b, sig_idx):
    #squashes all input values to a range (0,1)
    f = 1/ (1 + np.exp(-l[sig_idx] * (total_input - b[sig_idx])))
    #sig_index is the index of the activcation function we want 
    
    return f
    
def relu(l,x,b, relu_idx):
    temp = l[relu_idx]*(x-b[relu_idx])
    temp = np.concatenate((np.zeros((temp.shape[0],1)),temp),axis = 1)
    temp = np.max(temp,axis = 1).reshape(-1,1)
    temp = np.concatenate((np.ones((temp.shape[0],1)),temp),axis = 1)
    temp = np.min(temp, axis =1 ).reshape(-1,1)
    return temp
    
def graph_sigmoid(l, I, b):
    f=1/(1+np.exp(-l * (I-b)))
    return f

    
def trial(steps = 5000, nums = 1, plot = False):
    
    noise = 0.15 #noise inherent in the brain
   
    dt = 0.005
    tau = 1
    mean, scale, delta = 1, 1, .05 #was previously 1.4, 2, 0.5, changing the scale from 1 to 5 is cool too
    
    init_val = np.random.normal(mean, scale)
    input_layer = np.array([[init_val], [init_val + delta]])
    #print("input layer: ", input_layer7
    #normal random inputs Note: Im not sure what an appropriate value is for delta
    
    sig_idx= [0,1,2,3,4, 5] #which nodes activation should be sigmoidal
    
    #relu_idx= [1,1] #we want 2 and 3 to ramp up linearly, so units 2 and 3 are relu (index 1 and 2)
    
    #steps = 5000
    
    weights = np.array([[0,    -2,      0,     -2,     0,  0],  #1->1  2->1, 3->1, 4->1, 5->1, 6->1
                        [-2,    0,      -2,     0,     0,  0],  #1->2, 2->2, 3->2, 4->2, 5->2, 6->2
                        [1.23,  0,      2.087,  0,     0,  0],  #1->3, 2->3, 3->3, 4->3, 5->3, 6->3
                        [0,     1.23,   0,      2.087, 0,  0],  #1->4, 2->4, 3->4, 4->4, 5->4, 6->4
                        [0,     0,      0,      1.3,   1,  -2],  #1->5, 2->5, 3->5, 4->5, 5->5, 6->5
                        [0,     0,    1.3,      0,     -2,  1]]) #1->6, 2->6, 3->6, 4->6, 5->6, 6->6

    # To get 4 on and stay on, 2->4 > 1.22. Increasing that will adjust how quickly v4 ramps up 
    # To get 4 on and off, 2->4 should be 
    
    #Get the model to accumulate inputs to pick which one is bigger
    #with high accuracy and as fast as possible 
    
    v = np.array([[0, 0, 0, 0, 0, 0]]).T              # values for each neuron
    v_hist = np.array([[0, 0, 0, 0, 0, 0]]).T       # a list of columns of activation values at each time stamp
    piecewise_vals = [0]
    v2_v1_diff = [0]
    input_vals = np.array([]).T
    input_vals_delta = np.array([]).T
    
    l = np.array([[4, 4, 4, 4, 12, 12]]).T                  #steepness of activation fxn
    bias = np.array([[1, 1, 1, 1, 1, 1]]).T 
    #bias is responsible for increasing the amount of input a neuron needs/doesnt need to activate 
    bias = bias * 1.23
    x = np.linspace(-10, 20, 100) 
    z = 1/(1 + np.exp(-x)) 
    plot_switch = True 
    plot_switch_off_again = False
    decisions = []
    
    v4_decision = []
    v3_decision = []
    
    for t in range (nums): 
        #run through a trial
        decisionbool = False
        
        for i in range (steps):
            #establish stimulus to units 1 and 2 
            #scale is noise out in the world 
            stimulus = np.random.normal(mean, scale, 1)
            stimulus_delta = np.random.normal(mean + delta, scale, 1)
            
            #input_layer[0,0] = external_input
            #input_layer[1,0] = external_input_delta
            #note: the at operator does matrix multiplication, instead of element-by-element mult
           
            activations = weights @ v    #weighted sum of activations, inputs to each unit
            activations[0] = stimulus + activations[0] #previously self_input[0] + stiulus
            activations[1] = stimulus_delta + activations[1] #previously self_input[1] + stimulus
            
            activations = sigmoid(l, activations, bias, sig_idx)
        
            #self_input[[0,1]] = relu(l,self_input[relu_idx],bias, relu_idx)   
            #slope = -v + output + np.random.normal(0,0,v.shape) #slope is dv/dt,  then we add noise using np.random
            x_axis_vals = np.arange(-2, 3, .1)
            dv = tau * ((-v + activations) * dt + noise * np.sqrt(dt) * np.random.normal(0,1, (6,1))) # add noise using np.random
            v = v + dv
           
            """
            if ((i == 1) & (trial == 1)):
                print("Starting activations: \n", v)
                plt.plot(x_axis_vals, graph_sigmoid(l[3], x_axis_vals, bias[3] - v[1]))
                x1 = [0, 4]
                y1 = [0, 1/weights[3,3] * 4]
                plt.plot(x1,y1, label = "strength of switch")
                plt.ylim([0,1])
                plt.plot()
                plt.legend([ "sigmoid internal", "strength of unit 4"], loc = 0)
                plt.grid('on')
                plot_switch = False 
                plot_switch_off_again = True
                plt.title("activation of 4 against sigmoid - start")
                plt.show()  
            
            if ( (v[3] > 0.8) & (plot_switch == True) & (trial == 1)):
                print(v)
                plt.plot(x_axis_vals, graph_sigmoid(l[3], x_axis_vals, bias[3] - v[1]))
                x1 = [0, 4]
                y1 = [0, 1/weights[3,3] * 4]
                plt.plot(x1,y1, label = "strength of switch")
                plt.ylim([0,1])
                plt.plot()
                plt.legend([ "sigmoid internal, unit 4 on", "strength of unit 4"], loc = 0)
                plt.grid('on')
                plot_switch = False 
                plot_switch_off_again = True
                plt.title("activation of 4 against sigmoid - unit 4 activation > 0.8")
                plt.show()  
                
                
            if ( (v[3] < 0.4) & (plot_switch_off_again == True) & (i > 2000) & (trial == 1)):
                print("v: \n", v)
                print("v[3]:", v[3], "steps: ", i)
                print("Plot switch off again triggered")
                plt.plot(x_axis_vals, graph_sigmoid(l[3], x_axis_vals, bias[3] - v[1]))
                x1 = [0, 3]
                y1 = [0, 1/weights[3,3] * 3]
                plt.plot(x1,y1, label = "strength of switch")
                plt.ylim([0,1])
                plt.plot()
                plt.legend([ "sigmoid internal", "strength of unit 4"], loc = 0)
                plt.grid('on')
                plt.title("activation of 4 against sigmoid - unit 4 < 0.4 but was > 0.8")
                plot_switch_off_again = False 
                plt.show() 
            """    
            if ((i == steps-1) & (plot == True)):
                plt.figure()
                plt.plot(x_axis_vals, graph_sigmoid(l[2], x_axis_vals, bias[2] - v[0]))
                x1 = [0, 3]
                y1 = [0, 1/weights[3,3] * 3]
            
                plt.plot(x1,y1, label = "strength of switch")
                plt.ylim([0,1])
                #idx = np.argwhere(np.diff(np.sign(z - 1/self_input))).flatten()
                #plt.plot(x[idx], self_input[idx], 'ro')
                #plt.plot( (1/ self_input[3]))
                plt.legend([ "sigmoid internal", "strength of unit 4"], loc = 0)
                plt.title("activation of 4 against sigmoid at n-1 step")
                plt.grid('on')
                print(weights)
                print(activations)
                plt.show() 
            
                plt.figure()
                plt.plot(x_axis_vals, graph_sigmoid(l[3], x_axis_vals, bias[3] - v[1]))
                x1 = [0, 3]
                y2 = [0, 1/weights[2,2] * 3]
                plt.plot(x1,y2, label = "strength of unit 3")
                plt.ylim([0,1])
                plt.legend([ "sigmoid internal", "strength of unit 3"], loc = 0)
                plt.title("activation of 3 against sigmoid at n-1 step")
                plt.grid('on')
                plt.show()
                
                
            input_vals = np.append(input_vals, stimulus, axis=0)
            input_vals_delta = np.append(input_vals, stimulus_delta, axis=0)
            v_hist = np.concatenate((v_hist,v), axis=1)
            piecewise_vals = np.append(piecewise_vals, [piecewise(i, 2500)], axis = 0)
            v2_v1_diff = np.append(v2_v1_diff, [v[1] - v[0]])
            
            #Add decision data to decision array
            if (i == steps - 1):
                if(v[4] > 0.9):
                    decisions.append(1)
                else:
                    decisions.append(0)
            
            if ((not decisionbool) & (v[4] > 0.9)):
                v4_decision.append(i)
                decisionbool = True
                
            if ((not decisionbool) & (v[3] > 0.9)):
                v3_decision.append(i)
                decisionbool = True
                
                    
    
        if((plot) & (i == steps-1)):
            plt.figure()
            plt.plot(v2_v1_diff)
            smooted_diff = signal.savgol_filter(v2_v1_diff, 901, 4)
            plt.plot(smooted_diff)
            plt.legend(["input differece"], loc = 0)
            plt.ylabel("input")
            plt.xlabel("time (arbitrary units)")
            plt.grid('on')
            plt.show()
                    
            plt.figure()
            plt.plot(v_hist[0,:], dashes = [2,2]) 
            plt.plot(v_hist[1,:], dashes = [1,1])
            plt.plot(v_hist[2,:], dashes = [2,2])
            plt.plot(v_hist[3,:], dashes = [3,3])
            plt.plot(v_hist[4,:], dashes = [3,3])
            plt.plot(v_hist[5,:], dashes = [4,4])
            plt.plot(piecewise_vals, dashes = [4,4])
            #plt.plot(v2_v1_diff, dashes = [5,5])
            plt.legend(["v1","v2","v3","v4", "v5 - The \"sure\" v4", "v6", "piecewise"], loc=0)
            plt.ylabel("activation")
            plt.xlabel("time (arbitrary units)")
            plt.grid('on')
            plt.show()
            
            plt.figure()
            plt.violinplot(v4_decision)
            plt.show()
        
       
        #plt.hist(decisions)
        #plt.show
        
        #Add np.mean
        Sum = sum(decisions)
        avg = Sum / nums
    print(avg)
    
    

    """"
    plt.plot(input_vals)
    plt.legend("input values", loc=0)
    plt.show()
    
    plt.plot(input_vals_delta)
    plt.legend("input values + delta", loc=0)
    plt.show()
    """
    """
    plt.hist(input_vals, linewidth=2, color= "r")
    plt.legend(["input values. delta = %d" % (delta)], loc=0)
    plt.show()
    
    plt.hist(input_vals_delta, linewidth=2, color= "r")
    plt.legend(["input values delta. delta = %d" % (delta)], loc=0)
    plt.show()
    #print(v_hist)
    """
    
    
def main():
    trial(plot=True)
    