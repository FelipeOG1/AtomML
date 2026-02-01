import numpy as np
from dataclasses import dataclass,field
from typing import Callable
from activations import sigmoid,relu,linear


@dataclass
class BinaryCrossentropy:
    y_hat:np.ndarray
    y:np.ndarray

    def compute_loss(self):return -self.y*np.log(self.y_hat) - (1-self.y) * np.log(1-self.y_hat)
    
    def compute_cost(self):return np.mean(self.compute_loss())


@dataclass
class Dense:
    units: int
    activation_key: str
    
    w: np.ndarray = field(False)
    b: np.ndarray = field(False)
    activation_function: Callable[[np.ndarray],np.ndarray] = field(False)
    
    def __post_init__(self):
        if not self.activation_key in ('sigmoid','relu','linear'):
            raise ValueError("Not implemented activation function")
        
        activation_map = { 'sigmoid':sigmoid,
                           'relu':relu,
                           'linear':linear
                         }
        
        self.activation_function = activation_map[self.activation_key]
           
class Sequential:
    def __init__(self,layers:list[Dense]):
        self.layers = layers

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

    
 
