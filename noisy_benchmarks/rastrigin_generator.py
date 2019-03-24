import numpy as np
from numba import jit

class rastrigin(object):
    def __init__(self, noise_factor):
        self.noise_factor = noise_factor

    @jit
    def sim(self, params):
        """
        This function generates

        :param params:
        :param noise_factor:
        :return:
        """
        rastrigin = 0
        for par in params:
            rastrigin += par**2-10*np.cos(2*np.pi*par) + 10
        return rastrigin + self.noise_factor*(2*np.random.random() - 1)


