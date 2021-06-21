#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:14:22 2021

@author: Robert Klock

Class defining a timer module
"""
import numpy as np

class TimerModule:
    ZEROS_BLOCK = np.zeros((4,4))
    def __init__(self,timer_weight = 1):
        self.timer_weight=timer_weight
        block = np.array([[2, 0, 0, -.4],
                          [self.timer_weight, 1, 0, -.4],
                          [0, .5, 2, 0],
                          [0, 0, 1, 2]])
        self.block = block
    
    def timerWeight(self):
        return self.timer_weight
    
    def setTimerWeight(self, weight):
        self.timer_weight = weight
    
    def timerBlock(self):
        return self.block
    
    def buildWeightMarix(timerModules):
        if not isinstance(timerModules, list):
            raise TypeError('Timer modules should be in a list')
        else:
            idx = (0,1)
            for i in range (0, len(timerModules)):
                t = np.kron(np.eye(4), np.ones((4,4)))
                print(t)
                t[0+(4*i):0+(4*(i+1)), 0+(4*i):0+(4*(i+1))] = np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2]])
                print(t)
                # for j in range (0, len(timerModules)):