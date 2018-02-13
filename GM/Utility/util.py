import numpy as np


def log_det( A ):
    return np.linalg.slogdet( A )[ 1 ]

def invPsd( A ):
    pass