import numpy as np
from numba import jit

@jit
def rastrigin(params):
    rastrigin = 0
    for par in params[0]:
        rastrigin += par**2-10*np.cos(2*np.pi*par)+10
    std=rastrigin/(len(params[0]))
    return [[rastrigin]], [[std]]

@jit
def rastrigin_de(params):
    rastrigin = 0
    for par in params:
        rastrigin += par**2-10*np.cos(2*np.pi*par)+10
    std=rastrigin/(len(params))
    return rastrigin

if __name__ == '__main__':
    print(rastrigin([[ 0.59463442, -0.28599058,  0.03983822,  0.36086778, -5.10089233]]))