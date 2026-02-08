import numpy as np

def sigmoid(z):return 1/(1 + np.exp(-z))

def linear(z):return z

def relu(z):return np.max(0,z)


