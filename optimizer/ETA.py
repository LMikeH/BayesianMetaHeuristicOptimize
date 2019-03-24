import numpy as np
from random import randint
import random
from operator import attrgetter
import time
from objective_functions.rastrigin import rastrigin
from objective_functions.himmelblaue import himmelblaue
from objective_functions.devillersglasser2 import devillglass2
from objective_functions.damavandi import damavandi


class sample():
    def __init__(self, params, objective):
        self.params = params
        self.cost = objective(np.array([self.params]))[0]
        self.uncertainty = objective([self.params])[1]

class ETA():
    def __init__(self, size, bounds, precision, objective):
        self.size = size
        self.bounds = bounds
        self.dimensions = len(bounds)
        self.samples = []
        self.CDF = []
        self.noteworthies = []
        self.unworthies = []
        self.precision = precision
        self.radius_factor = 1
        self.gen = 1
        self.best = None
        self.prob = [.25, .50, .25]
        self.objective = objective

    def rand_init_gen(self):
        self.samples = []
        for samp in range(self.size):
            samp_params = []
            for bnd in self.bounds:
                samp_params.append((bnd[1] - bnd[0]) * random.random() + bnd[0])
            self.samples.append(sample(samp_params, self.objective))

    def edge_constrainer(self, params):
        new_params = params
        inx=0
        for bnd in self.bounds:
            if params[inx] < bnd[0]:
                params[inx] = bnd[0]
            elif params[inx] > bnd[1]:
                params[inx] = bnd[1]
            else:
                pass
            inx += 1
        return new_params

    def CDF_gen(self):
        self.samples = sorted(self.samples, key=attrgetter('cost'))
        PDF = []
        new_best = False

        if self.best is None:
            self.best = self.samples[0]
            new_best = True
        elif self.samples[0].cost < self.best.cost - 1e-4:
            self.best = self.samples[0]
            new_best = True

        # Probabilities of spawning areas.
        best_prob = self.prob[0]
        worthy_prob = self.prob[1]
        unworthy_prob = self.prob[2]

        # Check if design is within uncertainty of optimum
        samples = self.samples
        for sampl in samples:
            if sampl == self.best:
                PDF.append(best_prob)
            elif self.best is not None:
                if (sampl.cost - sampl.uncertainty*self.precision) < (self.best.cost):
                    if len(self.noteworthies) < 100:
                        self.noteworthies.append(sampl)
                    else:
                        # pass
                        del self.noteworthies[0]
                        self.noteworthies.append(sampl)
                else:
                    if len(self.unworthies) < 100:
                        self.unworthies.append(sampl)
                    else:
                        del self.unworthies[0]
                        self.unworthies.append(sampl)

        PDF.append(worthy_prob)
        PDF.append(unworthy_prob)
        norm_PDF = np.array(PDF)*.95/np.sum(PDF)
        CDF = []
        for inx in range(len(PDF)):
            CDF.append(np.sum(norm_PDF[:inx+1]))
        self.CDF = CDF
        return self.CDF, new_best

    def noteworthy_check(self):
        new_worthies = []
        old_noteworthies = self.noteworthies
        for samp in old_noteworthies:
            if (samp.cost - samp.uncertainty*self.precision) <= (self.best.cost):
                new_worthies.append(samp)
        self.noteworthies = new_worthies

    def new_gen(self):
        new_pop = []
        m = len(self.bounds)
        n = self.size
        rands = np.random.rand(n)
        for samp in range(self.size):
            parent_params = []
            unworthy = False
            if rands[samp] < self.CDF[0]:
                if self.best is not None:
                    parent_params = self.best.params
                else:
                    for bnd in self.bounds:
                        parent_params.append((bnd[1]-bnd[0])*random.random()+bnd[0])
            elif self.CDF[0] <= rands[samp] < self.CDF[1] and len(self.noteworthies) > 0:
                        parent_params = self.noteworthies[randint(0, len(self.noteworthies)-1)].params
            elif self.CDF[1] <= rands[samp] < self.CDF[2] and len(self.unworthies) > 0:
                parent_params = self.unworthies[randint(0, len(self.unworthies)-1)].params
                unworthy = True
            else:
                for bnd in self.bounds:
                    parent_params.append((bnd[1]-bnd[0])*random.random()+bnd[0])
                unworthy = True

            rand_vect = []
            if len(self.noteworthies) > 0:
                best_params = self.noteworthies[random.randint(0, len(self.noteworthies) - 1)].params
            else:
                best_params = self.best.params
            rand_proj = np.random.rand(len(self.bounds), 2)
            for bnd in range(len(self.bounds)):
                L = (self.bounds[bnd][1] - self.bounds[bnd][0]) #*np.sqrt(m)

                chi = rand_proj[bnd][0]
                eta = rand_proj[bnd][1]
                sig = random.random()

                # **** Magical tunnelling formula ****
                # r = sig * np.tanh(2*chi-1) * delta * np.exp(-m*n*eta)
                r = L*(2*sig-1)*np.exp(-eta*L)
                # **** Magical tunnelling formula ****

                rand_vect.append(r)

            # if unworthy == True:
            #     drift_vector = np.array(self.best.params) - np.array(rand_vect)
            #     drift = drift_vector/(np.array(rand_vect).dot(rand_vect))
            #     rand_vect -= drift

            new_params = self.edge_constrainer(np.array(parent_params) + np.array(rand_vect))
            new_sample = sample(new_params, self.objective)
            new_pop.append(new_sample)
        new_pop.append(self.best)
        self.samples = new_pop

    def give_new_xdata(self):
        new_xdata = []
        samples = [self.best] + self.noteworthies
        for samp in samples:
            new_xdata.append(samp.params)
        return np.array(new_xdata)

    def run(self, convergence=100):
        self.rand_init_gen()
        min_count = 0
        best = 10000
        while min_count < convergence: # and best > 1E-6:# and self.gen <= 3000:
            CDF, new_best = self.CDF_gen()
            if new_best == True:
                min_count = 0
                self.noteworthy_check()
            else:
                min_count += 1
            self.new_gen()
            print(self.gen*self.size,
                  self.best.cost[0],
                  self.best.uncertainty[0],
                  self.best.params,
                  min_count)

            best = self.best.cost
            self.gen += 1

    def convergence_check(self, PI, overlap):
        if self.best.uncertainty > PI:
            print('Uncertainty too large')
            return False
        for elite in self.noteworthies:
            if elite.cost - elite.uncertainty < self.best.cost - overlap - self.best.uncertainty:
                print('Overlap too large',
                      (self.best.cost - self.best.uncertainty) - (elite.cost - elite.uncertainty))
                return False
        return True

if __name__ == '__main__':
    dim = 2
    bounds = []
    for inx in range(dim):
        # bounds.append([1, 60]) # devillersglasser2
        bounds.append([-5.12, 5.12]) # rastrigin
        # bounds.append([0, 14])  # damavandi
    size = 10
    precision = 1E-4#.1/len(bounds)
    clan = ETA(size, bounds, precision, rastrigin)
    clan.run(1000)
    print(clan.best.params)

