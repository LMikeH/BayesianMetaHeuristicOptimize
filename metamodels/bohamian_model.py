

import pybnn
import numpy as np
from pybnn.bohamiann import Bohamiann, get_default_network
import pybnn

class BayesianNN(object):
    def __init__(self):
        self.model = Bohamiann(print_every_n_steps=1000,
                               sampling_method="adaptive_sghmc")
        self.trained = False

    def fit(self, x, y):
        self.model.train(x, y.flatten(),
                    num_steps=10000 + 100*len(x),
                    num_burn_in_steps= 100*len(x),
                    keep_every=200,
                    lr=1e-2,
                    verbose=True,
                    continue_training=self.trained)

    def predict(self, x):
        mean, var = self.model.predict(x)

        return mean, 1.96*np.sqrt(var)