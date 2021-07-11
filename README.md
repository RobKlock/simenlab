# Simen Lab Projects

# Introduction
This repository includes code related to various research projects under the advisement of Professor Patrick Simen at Oberlin College. 
Most projects rely upon the implementation of the drift-diffusion model at the neural-network level. This model uses one-shot learning for interval timing.

There are quite a few different programs here. They all contain machine learning models that use Recurrent Neural Networks that use differential equations. The most recent work is in the **timer-world** folder.

| File  | Description |
| ------------- |:-------------:|
| NeuronLatch.py      | I implemented the model shown on page 726 of the paper "A symbolic/subsymbolic interface protocol for cognitive modeling" by Simen and Polk (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2919065/). The two neurons together behave as a threshold latch. Their behavior is dictated by equation (28) in the aformentioned paper.   |
| timer-world      | Files related to ongoing research with Simen (Oberlin) and Rivest (Royal Military College of Canada).     |





Current State: 60 unit input using mines sonar data from UCI fed into perceptron/anti-perceptron model, fed into decision network 

We have designed a model that takes input from two sources, a perceptron and an anti-perceptron. These make opposite classifications about  
