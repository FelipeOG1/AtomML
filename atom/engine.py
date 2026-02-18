class Scalar:
    def __init__(self, data: float,
                 _children: tuple = (),
                 _op: str = ''
                 ):
        
        self.grad = 0       
        self.data = data
        self._prev = set(_children)
        self._op = _op
        
    
    def __repr__(self)->str:
        return f"Value(data={self.data})"


    def __add__(self, other: 'Scalar')-> 'Scalar':
        scalar = Scalar(data=self.data + other.data,
                      _children=(self, other),
                      _op='+'
                      )

        return scalar

    def __mul__(self, other: 'Scalar')-> 'Scalar':
        scalar = Scalar(data=self.data * other.data,
                        _children=(self, other),
                        _op='*'
                      )
        return scalar

   
    def relu(self):
        out = Scalar(data=0 if self.data < 0 else self.data,
                     _children=(self,),
                     _op='ReLu'
                     )

        return out










    
    
 
       
