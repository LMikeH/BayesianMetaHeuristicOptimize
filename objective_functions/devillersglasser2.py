
import numpy as np
from numba import jit

@jit
def devillglass2(params):

    x1 = params[0]
    x2 = params[1]
    x3 = params[2]
    x4 = params[3]
    x5 = params[4]
    val = 0
    for i in range(1, 16+1):
        t = .1*(i-1)
        y = 53.81 * 1.27**t * np.tanh(3.012*t + np.sin(2.13*t)) * np.cos(np.exp(.507) * t)

        val += (x1*x2**t*np.tanh(x3*t + np.sin(x4*t))*np.cos(t*np.exp(x5)) - y)**2

    std = val/len(params)**2
    return val, std



if __name__ == '__main__':

    print(devillglass2([-100, -100, -100, -100, -100]))
