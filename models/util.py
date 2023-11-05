import numpy as np
from typing import TypeVar

TFloat = TypeVar('TFloat', float, np.ndarray) 

def sigmoid(z: TFloat) -> TFloat:
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z: TFloat) -> TFloat:
    """Derivative of the sigmoid."""
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z: TFloat) -> TFloat:
    return np.exp(z) / np.sum(np.exp(z), axis=1)[:, None]

def softmax_prime(z: TFloat) -> TFloat:
    exp_z = np.exp(z)
    s = np.sum(exp_z, axis=1)
    return np.multiply(exp_z / (s[:, None])**2, (s - exp_z.T).T)