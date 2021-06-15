#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:14:22 2021

@author: Robert Klock

Class defining a timer module
"""
import numpy as np

class TimerModule:
    
	def __init__(self,timer_weight = 1):
		self.timer_weight=timer_weight
        block = np.array([[2,   0,  0,  -.4],
                         [timer_weight, 1, 0, -.4],
                         [0,    .5, 2,  0],
                         [0,    0,  1,  2]])
        self.block = block
        
	def getTimerWeight(self):
		return self.timer_weight
    
    def setTimerWeight(self, weight):
        self.timer_weight = weight