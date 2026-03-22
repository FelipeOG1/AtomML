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
        
        
    def test_relu(self):
        s = Scalar(-30.1)
        s2 = Scalar(33.3)
        
        r1 = s.relu()
        r2 = s2.relu()
        
        assert r1.data == 0
        assert r2.data == 33.3
        assert s in r1._prev
        assert s2 in r2._prev
        assert r1._op == 'ReLu'
        assert r2._op == 'ReLu'

    def test_neg(self):
        a = Scalar(5.0)
        b = -a
        
        assert b.data == -5.0
        assert a in b._prev

    def test_pow(self):
        a = Scalar(3.0)
        b = a**2
        
        assert b.data == 9.0
        assert a in b._prev
        assert b._op == '**2'

    def test_backward(self):
        x1 = Scalar(2.0)
        x2 = Scalar(0.0)
        w1 = Scalar(-3.0)
        w2 = Scalar(1.0)
        b = Scalar(6.8813735870195432)
        n = x1*w1 + x2*w2 + b
        
