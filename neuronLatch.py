#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:49:32 2021

@author: Rob Klock
Initialization of a two-unit closed-loop system that propogates a 1 forever
"""

import numpy as np
import math
import sys
sys.path.append('/Users/robertklock/Documents/Class/Fall20/SimenLab/simenlab')
import perceptron as p
import matplotlib.pyplot as plt

data = np.zeros((1,100))

data[0][20:30] = 1
data[0][50:60] = 1