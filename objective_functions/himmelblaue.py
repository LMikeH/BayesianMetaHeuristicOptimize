
import numpy as np
from numba import jit

@jit
def himmelblaue(params):
    x = params[0]
    y = params[1]

    himmel = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    std = himmel/2
    return himmel, std