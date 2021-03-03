# simenlab
files related to research with Professor Patrick Simen

There are quite a few different programs here. They all contain machine learning models that use Recurrent Neural Networks that use differential equations. The most recent work is in NeuronLatch.py

## NeuronLatch.py

I implemented the model shown on page 726 of the paper "A symbolic/subsymbolic interface protocol for cognitive modeling" by Simen and Polk (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2919065/). The two neurons together behave as a threshold latch. Their behavior is dictated by equation (28) in the above paper (shown below):

```math
SE = \frac{\sigma}{\sqrt{n}}
```

Current State: 60 unit input using mines sonar data from UCI fed into perceptron/anti-perceptron model, fed into decision network 

We have designed a model that takes input from two sources, a perceptron and an anti-perceptron. These make opposite classifications about  
