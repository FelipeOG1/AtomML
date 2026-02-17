from atom.engine import Scalar


class TestScalar:
    
    def test_add(self):
        a = Scalar(3.0)
        b = Scalar(2.0)
        
        d = a + b
        
        assert d.data == 5.0
        assert a in d._prev
        assert b in d._prev
        assert d._op == '+'
    def test_mul(self):
        a = Scalar(3.0)
        b = Scalar(2.0)
        
        d = a * b

        
        assert d.data == 6.0
        assert a in d._prev
        assert b in d._prev
        assert d._op == '*'
        
    def test_multiple_ops(self):
        a = Scalar(30.0)
        b = Scalar(10.0)
        c = Scalar(2.0)
        d = a + b
        
        e = c*d + a

        assert d.data ==  40.0
        assert a in d._prev
        assert b in d._prev
        
        assert e.data == 110.0
        assert any(x.data == 80 for x in e._prev)
        assert a in e._prev
        
        
 
    

    
        
        
