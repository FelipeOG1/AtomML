import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from  atom import Dense,Sequential,BinaryCrossentropy




def sum(x: int,y: int)->int:return x + y

def test_sum():
    assert sum(1,3)  == 4
