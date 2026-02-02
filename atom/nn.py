import numpy as np
from dataclasses import dataclass,field
from typing import Callable
from atom.activations import sigmoid,relu,linear


@dataclass
class BinaryCrossentropy:
    y_hat:np.ndarray
    y:np.ndarray

    def compute_loss(self):return -self.y*np.log(self.y_hat) - (1-self.y) * np.log(1-self.y_hat)
    
    def compute_cost(self):return np.mean(self.compute_loss())

class Dense:

    ACTIVATIONS = {'sigmoid':sigmoid,
                   'relu':relu,
                   'linear':linear}
    
    def __init__(self,units: int,activation: str):
        if not  activation in self.ACTIVATIONS:
            raise ValueError("Activation not allowed")
    
        self.units = units
        self.w: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.activation_function: Callable[[np.ndarray],np.ndarray] = self.ACTIVATIONS[activation]
    
 

@dataclass
class Sequential:
    layers: list[Dense]

    def predict(self,x:np.ndarray):
        a = x
        for layer in self.layers:
            z = (a @ layer.w) + layer.b
            a = layer.activation_function(z)
        return a

    def __getitem__(self,position:int):
        return self.layers[position]
    
    def set_weights(self,weights:list[np.ndarray]):
        assert len(weights) == 2 * len(self.layers)
        for index,layer in enumerate(self.layers):
            layer.w,layer.b = weights[index * 2],weights[index * 2 + 1]

    
