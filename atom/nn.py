import numpy as np
from dataclasses import dataclass,field
from typing import Callable
from atom.activations import sigmoid,relu,linear
from typing import Optional

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
    
    def __init__(self,
                 units: int,
                 activation: str,
                 input_shape: tuple[int,int] | None = None,
                 ):
        
        if not  activation in self.ACTIVATIONS:
            raise ValueError("Activation not allowed")
    
        self.units = units
        self.input_shape = input_shape
        self.activation_function: Callable[[np.ndarray],np.ndarray] = self.ACTIVATIONS[activation]
        self.w: np.ndarray | None = None
        self.b: np.ndarray | None = None
        
        if self.input_shape:
            self._init_w_b()
     
    def _init_w_b(self, w: np.ndarray | None = None, b: np.ndarray | None = None):
        if not self.input_shape:
            raise ValueError("Unknown input shape")
        self.w = w if w is not None else np.random.randn(self.input_shape[1], self.units) * 0.01
        self.b = b if b is not None else np.zeros((1, self.units))
  

    def set_weights(self,w: np.ndarray,b: np.ndarray):
        self._init_w_b(w=w,b=b)

            
    def build(self,input_shape: tuple[int,int]):
        if not isinstance(input_shape,tuple) or len(input_shape) != 2:
            raise ValueError("Invalid input shape")
        self.input_shape = input_shape
        self._init_w_b()    
        
@dataclass
class Sequential:
    layers: list[Dense]

    def _set_weights_layers(self,input_shape: tuple[int,int]):
        current_dim = input_shape[-1]
        for layer in self.layers:
            layer.build((input_shape[0],current_dim))
            current_dim = layer.units

            
    def __post_init__(self):
        if getattr(self.layers[0],"input_shape",None) is not None:
            input_shape = self.layers[0].input_shape
            self._set_weights_layers(input_shape)


    def __getitem__(self,position: int):
        return self.layers[position]

    def __iter__(self):
        return iter(self.layers)
    
    
    def build(self,input_shape: tuple)->None:
        self._set_weights_layers(input_shape)
            
    def predict(self,x:np.ndarray):
        a: np.ndarray = x
        for layer in self.layers:
            z = (a @ layer.w) + layer.b
            a = layer.activation_function(z)
        return a

   
    def set_weights(self,weights:list[np.ndarray]):
        assert len(weights) == 2 * len(self.layers)
        for index,layer in enumerate(self.layers):
            layer.w,layer.b = weights[index * 2],weights[index * 2 + 1]

   
