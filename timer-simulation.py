#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:36:22 2021

@author: Rob Klock

Model of situations where the world isnt timeable. Driving question is: when do you allocate new timers?

"""
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
mean, var, skew, kurt = stats.norm.stats(moments='mvsk')
x = np.linspace(stats.norm.ppf(0.01),
                stats.norm.ppf(0.99), 100)

ax.plot(x, stats.norm.pdf(x),
       'r-', lw=3, alpha=.3, label='norm pdf')
rv = stats.norm()

ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
vals = stats.norm.ppf([0.001, 0.5, 0.999])

np.allclose([0.001, 0.5, 0.999], stats.norm.cdf(vals))
r = stats.norm.rvs(size=1000)

ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)

ax.legend(loc='best', frameon=False)

plt.show()
## generate the data and plot it for an ideal normal curve## x-axis for the plot
x_data = np.arange(-5, 20, 0.001)## y-axis as the gaussian
y_data1 = stats.norm.pdf(x_data, 3, 3)## plot data
y_data2 = stats.norm.pdf(x_data, 15, 6)
y_data = y_data1 + y_data2
plt.plot(x_data, y_data)
plt.show()
