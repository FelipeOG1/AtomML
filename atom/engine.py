import  dataclasses

class Scalar:
    def __init__(self, data, _op='', _children=()):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
    



