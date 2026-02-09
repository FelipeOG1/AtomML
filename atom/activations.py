import numpy as np

def sigmoid(z):return 1/(1 + np.exp(-z))

def linear(z):return z

def relu(z):return np.maximum(0,z)

def softmax(z):return np.exp(z) / np.sum(np.exp(z),axis=1,keepdims=True)



