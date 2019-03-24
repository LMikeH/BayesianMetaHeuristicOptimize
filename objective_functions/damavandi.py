import numpy as np
from numba import jit

@jit
def damavandi(params):
    x = params[0]
    y = params[1]
    pi = np.pi
    val = (1 - np.abs((np.sin(pi*(x - 2))*np.sin(pi*(y-2)))/(pi**2*(x-2)*(y-2)))**5)*(2 + (x-7)**2 + 2*(y-7)**2)
    std = val/4
    return val, std

if __name__ == '__main__':

    print(damavandi([1,1]))