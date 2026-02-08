import pytest
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from atom.nn import Dense,Sequential


class TestDense:
    def test_init_validations(self):
        with pytest.raises(ValueError):
             Dense(units=2,activation='foo')
            
        valid_dense = Dense(units=10,activation='sigmoid')
        assert valid_dense
    
    def test_build(self):
        
        layer = Dense(units=10,activation='sigmoid')
        with pytest.raises(ValueError):
            layer.build(input_shape=(3,3,100))

        layer.build(input_shape=(1000,400))
        
        assert layer.w.shape == (400,10)
        assert layer.b.shape == (1,10)


    def test_set_weights(self):
         with pytest.raises(ValueError):
            print("NASHEE")
            layer = Dense(units=3,activation='relu')
            layer.set_weights(np.array([300,200]),np.ndarray([20,20]))
            


class TestSequential:
    def test_post_init(self):
        x_train = np.random.randn(1000,400)
        model = Sequential([
            Dense(units=10, activation='relu',input_shape=x_train.shape),
            Dense(units=15, activation='relu'),
            Dense(units=1, activation='relu')
        ]) 

        w1_shape, b1_shape = (400, 10), (1, 10)
        w2_shape, b2_shape = (10, 15), (1, 15)
        w3_shape, b3_shape = (15, 1), (1, 1)
        
        assert model[0].w.shape == w1_shape
        assert model[0].b.shape == b1_shape
        assert model[1].w.shape == w2_shape
        assert model[1].b.shape == b2_shape
        assert model[2].w.shape == w3_shape
        assert model[2].b.shape == b3_shape

    def test_build(self):
        model = Sequential([
            Dense(units=10, activation='relu'),
            Dense(units=15, activation='relu'),
            Dense(units=1, activation='relu')
        ]) 

        for layer in model:
            assert not hasattr(layer, 'w') or layer.w is None 
            
        model.build(input_shape=(1000, 400))
        
        w1_shape, b1_shape = (400, 10), (1, 10)
        w2_shape, b2_shape = (10, 15), (1, 15)
        w3_shape, b3_shape = (15, 1), (1, 1)

        assert model[0].w.shape == w1_shape
        assert model[0].b.shape == b1_shape
        assert model[1].w.shape == w2_shape
        assert model[1].b.shape == b2_shape
        assert model[2].w.shape == w3_shape
        assert model[2].b.shape == b3_shape 


        
