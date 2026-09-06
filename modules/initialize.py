import numpy as np
import numpy.random as nr

def cold_start(shape)-> np.ndarray:
    return np.ones(shape=shape)

def hot_start(shape)-> np.ndarray:
    return nr.randint(2, size=shape)*2 - 1
