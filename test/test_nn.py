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
            




