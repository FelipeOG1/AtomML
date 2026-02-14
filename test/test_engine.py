from atom.engine import Scalar


class TestScalar:
    
    def test_add(self):
        a = Scalar(3.0)
        b = Scalar(2.0)
        
        d = a + b

        
        assert d.data == 5.0
        assert a in d._prev
        assert b in d._prev
        
    def test_mul(self):
        a = Scalar(3.0)
        b = Scalar(2.0)
        
        d = a * b

        
        assert d.data == 6.0
        assert a in d._prev
        assert b in d._prev
    
        
        
