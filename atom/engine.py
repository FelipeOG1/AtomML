import  dataclasses
from typing import Self


class Scalar:
    def __init__(self, data: float,
                 _op: str = '',
                 _children: tuple = ()
                 ):
        
        self.grad = 0       
        self.data = data
        self._prev = set(_children)
        self._op = _op
        
    
    def __repr__(self)->str:
        return f"Value(data={self.data})"


    def __add__(self, other: 'Scalar')-> 'Scalar':
        return Scalar(data=self.data + other.data,
                      _children=(self, other),
                      _op='+'
                      )

    def __mul__(self, other: 'Scalar')-> 'Scalar':
        return Scalar(data=self.data * other.data,
                      _children=(self, other),
                      _op='*'
                      )













    
    
 
       
