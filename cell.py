import numpy as nn
import random

def sigmoid(x):
    return 1 / (1 + nn.exp(-x))
    
def sigmoidD(x):
    s = sigmoid(x)
    return s * (1 - s)
    
weight = random.random()
bias = random.random()
start = 0.1
target = 1.0
lr = 0.1

def forward(x):
    z = weight * x + bias
    return sigmoid(z)
    
for i in range(1, 10001):
    z = weight * start + bias
    output = sigmoid(z)
    error = output - target
    dout = error * sigmoidD(z)
    weight -= lr * dout * start
    bias -= lr *dout
    
print(round(forward(start), 2))
