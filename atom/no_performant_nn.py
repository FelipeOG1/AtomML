import random
from atom.engine import Scalar
class Neuron:
    def __init__(self, nin):
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Scalar(random.uniform(-1,1))
    def __call__(self, x):
        z = sum([xi * wi for xi, wi in zip(x,  self.w)], self.b) 
        return z.relu()

class layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(self.neurons) == 1 else outs

class MLP:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
            
        
   
