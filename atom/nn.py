import numpy as np
from dataclasses import dataclass


def g(z):return 1/(1+np.exp(-z))

@dataclass
class BinaryCrossentropy:
    y_hat:np.ndarray
    y:np.ndarray

    def compute_loss(self):return -self.y*np.log(self.y_hat) - (1-self.y) * np.log(1-self.y_hat)
    
    def compute_cost(self):return np.mean(self.compute_loss())


@dataclass
class Dense:
    units:int
    w:np.ndarray = field(False)
    b:np.ndarray = field(False)
    activation_key:str
    activation_function = field(False)
    
    def __post_init__(self):
        if not self.activation_key in ('sigmoid','relu','linear'):
            raise ValueError("Not implemented activation function")
        
        fucntions = {'sigmoid':g,
                     ''
                     }
        
    
    def __postinit
class Sequential:
    def __init__(self,layers:list[Layer]):
        self.layers = layers

    
    def predict(self,x:np.ndarray):
        a = x
        for layer in self.layers:
            a = g((a @ layer.w) + layer.b)
        return a

    def __getitem__(self,position:int):
        return self.layers[position]
    
    def set_weights(self,weights:list[np.ndarray]):
        assert len(weights) == 2 * len(self.layers)
        for index,layer in enumerate(self.layers):
            layer.w,layer.b = weights[index * 2],weights[index * 2 + 1]

    
 
