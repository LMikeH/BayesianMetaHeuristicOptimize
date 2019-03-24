
import numpy as np
from noisy_benchmarks.rastrigin_generator import rastrigin as ras
from optimizer.ETA import ETA
from metamodel_tests.gpc_model import gpc_metamodel
from metamodel_tests.neural_nets import nn_ensemble
# from metamodel_tests.mcdropout_model import Learner, dropout_net
from metamodel_tests.sparse_gpr_model import sgpr_model
import torch
import torch.nn as nn
from metamodel_tests.gpy_sgpr import SPGRModel

from metamodel.bohamian_model import BayesianNN

class GBOptimizer(object):
    def __init__(self,
                 bounds,
                 starting_num,
                 points_per_iteration,
                 noise,
                 model):

        self.bnds = bounds
        self.ppi = points_per_iteration
        self.noise = noise
        self.simulation = ras(noise)
        self.model = model
        self.ydata_set = []
        self.xdata_set = []
        self.rand_data_initializer(starting_num)

    def rand_data_initializer(self, num_points):
        xdata = np.array([])
        ydata = np.array([])
        for n in range(num_points):
            sample = []
            for d in range(len(self.bnds)):
                sample.append((self.bnds[d][1]-self.bnds[d][0])
                              *np.random.random()
                              + self.bnds[d][0])
            sample = np.array(sample)
            if n == 0:
                xdata = np.hstack([xdata, sample])
            else:
                xdata = np.vstack([xdata, sample])
            ydata = np.hstack([ydata, self.simulation.sim(sample)])
        self.xdata_set = xdata
        self.ydata_set = ydata.reshape((len(ydata), 1))

    def append_data(self, data):
        print("new data len: {}".format(len(data)))
        self.xdata_set = np.vstack([self.xdata_set, data])
        for x in data:
            self.ydata_set = np.vstack((self.ydata_set,
                                        self.simulation.sim(x)))

    def train_model(self):
        self.model.fit(self.xdata_set, self.ydata_set) #, 10000, 10)

    def run_eta(self):
        self.optimizer = ETA(10, self.bnds, 1, self.model.predict)
        self.optimizer.run()
        new_xdata = self.optimizer.give_new_xdata()
        self.append_data(new_xdata)


if __name__ == '__main__':
    dim = 2
    bnds = []
    for i in range(dim):
        bnds.append([-5.12, 5.12])
    #
    # feature_extractor = FeatureExtractor(dim)
    # model = Learner(feature_extractor)

    model = BayesianNN()

    solver = GBOptimizer(bnds, 200, 100, .001, model)
    solver.train_model()
    solver.run_eta()

    converged = False
    iter = 1
    while converged == False and iter < 10:
        # ensemble = nn_ensemble(dim, 20, 20, 30, 30, 80)
        # solver.model = ensemble
        solver.train_model()
        solver.run_eta()
        converged = solver.optimizer.convergence_check(1.0, 2.0)
        print(len(solver.xdata_set))
        iter += 1

    print("all done! {} {} {}".format(solver.optimizer.best.params,
                                   solver.optimizer.best.cost,
                                   solver.optimizer.best.uncertainty))

    # print('True optimal position gives {:.4f}'.
    #       format(ensemble.predict([0.0, 0.0])[0][0]))
