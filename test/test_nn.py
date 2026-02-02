import pytest
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from atom.nn import Dense,Sequential


class TestDense:
    def test_init_validations(self):
        with pytest.raises(ValueError):
             Dense(units=2,activation='foo')
        
