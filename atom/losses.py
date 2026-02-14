import numpy as np
from dataclasses import dataclass,field
from atom.constants import EPS

@dataclass
class BinaryCrossentropy:
    y_hat: np.ndarray
    y: np.ndarray
    def compute_loss(self):return -self.y*np.log(self.y_hat) - (1-self.y) * np.log(1-self.y_hat)
    
    def compute_cost(self):return np.mean(self.compute_loss())

@dataclass
class SparseCategoricalCrossentropy:
    y_hat: np.ndarray
    y: np.ndarray

    def compute_loss(self):
        probs = self.y_hat[np.arange(len(self.y)),self.y]
        return -np.log(probs + EPS)
        
    def compute_cost(self):return np.mean(self.compute_loss())





