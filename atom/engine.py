
from __future__ import annotations
class Scalar:
    def __init__(self, data: float,
                 _children: tuple = (),
                 _op: str = ''
                 ):

        self.grad = 0
        self.data = data
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        
    def __add__(self, other: Scalar) -> Scalar:
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(data=self.data + other.data,
                      _children=(self, other),
                      _op='+'
                      )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward


        return out
    
    def __mul__(self, other: Scalar | float) -> Scalar:
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(data=self.data * other.data, _children=(self, other), _op='*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

          
    def __pow__(self, other: int | float) -> Scalar:
        out = Scalar(data=self.data**other,
                     _children=(self, ),
                     _op=f'**{other}'
                     )

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Scalar(data=0 if self.data < 0 else self.data,
                     _children=(self,),
                     _op='ReLu'
                     )

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):

        visited_nodes: set[Scalar] = set()
        topo: list[Scalar] = []

        def build_topo(node: Scalar):
            if node not in visited_nodes:
                visited_nodes.add(node)
                for child in node._prev:
                    build_topo(child)

                topo.append(node)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __neg__(self): 
        return self * -1

    def __radd__(self, other): 
        return self + other

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other): 
        return other + (-self)

    def __rmul__(self, other): 
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
