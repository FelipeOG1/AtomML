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
        
    
    def __repr__(self)->str:
        return f"Value(data={self.data})"


    def __add__(self, other: 'Scalar')-> 'Scalar':
        out = Scalar(data=self.data + other.data,
                      _children=(self, other),
                      _op='+'
                      )
       
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out._backward = _backward

        return out

    def __mul__(self, other: 'Scalar')-> 'Scalar':
        out = Scalar(data=self.data * other.data,
                        _children=(self, other),
                        _op='*'
                      )

        
        return out
   
    def relu(self):
        out = Scalar(data=0 if self.data < 0 else self.data,
                     _children=(self,),
                     _op='ReLu'
                     )

        return out










    
    
 
       
