#!/usr/bin/env python3

import sys, os
import time
import random
from datetime import datetime
import numpy as np
from copy import deepcopy
import qml
from qml.math import cho_solve
from qml.math import svd_solve
from qml.representations import *
#from qml.wrappers import get_atomic_kernels_gaussian
from qml.kernels import get_local_symmetric_kernels
from qml.kernels import get_local_kernels_gaussian
from qml.kernels import get_local_kernel
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel
import itertools
from time import time

from qml.kernels import get_local_symmetric_kernel_mbdf, get_local_kernel_mbdf

from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping

from tqdm import tqdm

import pandas as pd

from skopt import gp_minimize
from skopt.plots import plot_convergence

from qml.kernels import get_local_symmetric_kernel
from qml.kernels import get_local_kernel

import MBDF

from yaspin import yaspin
from colored import fg


RED = fg('red')
WHITE = fg('white')
GREEN = fg('green')



class CostFunction(object):
    def __init__(self, X, Q, Xt, Qt, train, test, Y_HF, Y_HF_MP2, Y_MP2_CCSD, t_HF, t_MP2, t_CCSD, Y_test):
        self.n_cross = 3
        self.X = X
        self.Q = Q
        self.Xt = Xt
        self.Qt = Qt
        self.test = test
        self.train = train
        self.Y_HF = Y_HF
        self.Y_HF_MP2 = Y_HF_MP2
        self.Y_MP2_CCSD = Y_MP2_CCSD
        self.Y_test = Y_test
        self.t_HF = t_HF
        self.t_MP2 = t_MP2
        self.t_CCSD = t_CCSD
        self.tHF         = np.load("get_sigmas_HF_ae.npz")
        self.sigmas_HF   = self.tHF['sigmas']
        self.tMP2        = np.load("get_sigmas_MP2_ae.npz")
        self.sigmas_MP2  = self.tMP2['sigmas']
        self.tCCSD       = np.load("get_sigmas_CCSD_ae.npz")
        self.sigmas_CCSD = self.tCCSD['sigmas']


    def calc_average_cpu(self, hf, mp, cc):

      cpu = np.sum(hf) + np.sum(mp) + np.sum(cc)

      return cpu

    def opt_sigma(self, train_set_size, dec):
        N = np.array([ 2,  10, 50, 100, 200, 400, 800, 1600, 3200 ])

        # Calculate the absolute differences between each number in array1 and array2
        differences = np.abs(N - train_set_size)

        # Find the index of the minimum difference for each number in array1
        min_indices = np.argmin(differences)

        # Use the index to find the closest number in array2 for each number in array1
        if dec == "HF":
            return self.sigmas_HF[min_indices]
        elif dec == "MP2":
            return self.sigmas_MP2[min_indices]
        elif dec == "CCSD":
            return self.sigmas_CCSD[min_indices]



    def calc_maes(self, X, Q, Xt, Qt, gamma1, gamma2, gamma3):

      N_HF   = int(gamma1)
      N_MP2  = int(gamma2)
      N_CCSD = int(gamma3)

      split = list(range(len(self.train)))
      test = list(range(len(self.test)))

      MAEs = []
      cpus = np.array([])
      #for i in tqdm(range(self.n_cross), desc="Models",  position=0):
      for i in range(self.n_cross):
          random.shuffle(split)

          training_index_HF   = split[:N_HF]
          training_index_MP2  = split[:N_MP2]
          training_index_CCSD = split[:N_CCSD]
          test_index = test

          # HF direct
          sigma_HF   = self.opt_sigma(N_HF, "HF")
          K  = get_local_symmetric_kernel_mbdf(X[training_index_HF],  Q[training_index_HF], sigma_HF)
          Kt = get_local_kernel_mbdf(X[training_index_HF], Xt, Q[training_index_HF], Qt, sigma_HF)

          C = deepcopy(K)
          alpha = svd_solve(C, self.Y_HF[training_index_HF])

          Y_HF_3lvl = np.dot(Kt, alpha)

          # HF -> MP2
          #K  = get_local_symmetric_kernels(Xi[training_index_MP2], Q[training_index_MP2], [sigma2])
          #Kt = get_local_kernel(Xi[training_index_MP2], Xt[test_index], Q[training_index_MP2], Qt[test_index], sigma2)
          sigma_MP2   = self.opt_sigma(N_MP2, "MP2")
          K  = get_local_symmetric_kernel_mbdf(X[training_index_MP2],  Q[training_index_MP2], sigma_MP2)
          Kt = get_local_kernel_mbdf(X[training_index_MP2], Xt, Q[training_index_MP2], Qt, sigma_MP2)

          C = deepcopy(K)
          alpha = svd_solve(C, self.Y_HF_MP2[training_index_MP2])

          Y_HF_MP2 = np.dot(Kt, alpha)

          # MP2 -> CCSD(T)
          #K  = laplacian_kernel(Xi[training_index_CCSD], Xi[training_index_CCSD], sigma3)
          #Kt = laplacian_kernel(Xi[training_index_CCSD], Xt[test_index], sigma3)
          sigma_CCSD   = self.opt_sigma(N_CCSD, "CCSD")
          K  = get_local_symmetric_kernel_mbdf(X[training_index_CCSD],  Q[training_index_CCSD], sigma_CCSD)
          Kt = get_local_kernel_mbdf(X[training_index_CCSD], Xt, Q[training_index_CCSD], Qt, sigma_CCSD)

          C = deepcopy(K)
          alpha = svd_solve(C, self.Y_MP2_CCSD[training_index_CCSD])

          Y_MP2_CCSD = np.dot(Kt, alpha)


          Y_3_level    = Y_HF_3lvl + Y_HF_MP2 + Y_MP2_CCSD
          MAE = np.abs(Y_3_level-self.Y_test)
          MAEs.append(MAE)
          CPU = self.calc_average_cpu(self.t_HF[training_index_HF], self.t_MP2[training_index_MP2], self.t_CCSD[training_index_CCSD])
          cpus = np.append(cpus, CPU)



      maes = np.array(MAEs)
      mae = np.mean(maes)
      stddev = np.std(maes) / np.sqrt(self.n_cross)
      CPUs = np.mean(cpus)

      return mae, stddev, CPUs, sigma_HF, sigma_MP2, sigma_CCSD


    def get_mae(self, parameters):
        #f = open("s2_newCost_CPU_real_2.txt", 'a')
        f = open("cpu.txt", 'a')
        gamma1 = parameters[0]
        gamma2 = parameters[1]
        gamma3 = parameters[2]
#        beta   = parameters[3]

        beta = 0.75
        start_cv = time()
        mae, stddev, CPU, sigma1, sigma2, sigma3 = self.calc_maes(self.X, self.Q, self.Xt, self.Qt, gamma1, gamma2, gamma3)
        end_cv = time()

#        if mae < 0.9:
#            mae = 10.0

        if mae > 2.4: CPU = 1e6
        mae_scaled = (mae - 1.89) / (133.49 - 1.89)
        cost_scaled = (CPU - 2.82) / (8989.93 - 2.82)
        #combined = 0.65*mae_scaled + 0.35*cost_scaled
        combined = beta*mae_scaled + (1-beta)*cost_scaled

        #if mae > 0.7: combined = combined + 0.6
        #if CPU > 145: combined = combined + 0.6
        ##combined = (beta * mae + gamma * CPU) / (beta+gamma)
        #if mae > 1.0:
        #    CPU = 1e6

        print("\nCCSD(T): {} ({}), MP2: {} ({}), HF: {} ({})\n-----------------------------------------------\nCost: {:.4f} +/- {:.4f} kcal/mol       time = {:.4f}\nMAE: {:.4f}, CPU: {:.4f}\n".format(int(gamma3), sigma3, int(gamma2), sigma2, int(gamma1), sigma1, combined, stddev, (end_cv-start_cv), mae, CPU))
        f.write("\nCCSD(T): {} ({}), MP2: {} ({}), HF: {} ({})\n-----------------------------------------------\nCost: {:.4f} +/- {:.4f} kcal/mol       time = {:.4f}\nMAE: {:.4f}, CPU: {:.4f}\n".format(int(gamma3), sigma3, int(gamma2), sigma2, int(gamma1), sigma1, combined, stddev, (end_cv-start_cv), mae, CPU))
        #print("\nCCSD(T): {} ({}), MP2: {} ({}), HF: {} ({})\n-----------------------------------------------\nMAE: {:.4f}, CPU: {:.4f}\n".format(int(gamma3), sigma3, int(gamma2), sigma2, int(gamma1), sigma1, mae, CPU ))
        #f.write("\nCCSD(T): {} ({}), MP2: {} ({}), HF: {} ({})\n-----------------------------------------------\nMAE: {:.4f}, CPU: {:.4f}\n".format(int(gamma3), sigma3, int(gamma2), sigma2, int(gamma1), sigma1, mae, CPU))
        f.close()


        #return combined
        return CPU

def read_costs(f):
 return pd.read_csv(f)


# Function to parse datafile to a dictionary
def main():

    df = read_costs("qm9_ae.txt")
    HF_times = df['t_HF_h'].to_numpy()
    MP_times = df['t_MP2_h'].to_numpy()
    CC_times = df['t_CCSD_h'].to_numpy()
    print(" [ {}OK{} ] Read in Files".format(GREEN, WHITE))


    data       = np.load("qm9_ae.npz", allow_pickle=True)
    Xall       = data['X']
    Qall       = data['Q']
    Y_CCSD     = data['Y_CCSD']
    Y_HF       = data['Y_HF']
    Y_HF_MP2   = data['Y_HF_MP2']
    Y_MP2_CCSD = data['Y_MP2_CCSD']
    idx_train  = np.concatenate((data['idx_train'], data['idx_val']))
    idx_test   = data['idx_test']

    Y_test = Y_CCSD[idx_test]
    Y_HF = Y_HF[idx_train]
    Y_HF_MP2 = Y_HF_MP2[idx_train]
    Y_MP2_CCSD = Y_MP2_CCSD[idx_train]

    HF_times = HF_times[idx_train]
    MP2_times = MP_times[idx_train]
    CCSD_times = CC_times[idx_train]

    #np.savez("qm7b.npz", idx_train=train, idx_val=val, idx_test=test, X=Xall, Q=Qall, Y_CCSD=Y_test, Y_HF=Y_HF, Y_HF_MP2=Y_HF_MP2, Y_MP2_CCSD=Y_MP2_CCSD)
    X  = Xall[idx_train]
    Q  = Qall[idx_train]
    Xt = Xall[idx_test]
    Qt = Qall[idx_test]

    cmd = "rm -f cpu.txt"
    #cmd = "rm -f s2_newCost_CPU_real_2.txt"
    os.system(cmd)
    #cmd = "rm -f s2_newCost_CPU_real.txt"

    cost = CostFunction(X, Q, Xt, Qt, idx_train, idx_test, Y_HF, Y_HF_MP2, Y_MP2_CCSD, HF_times, MP2_times, CCSD_times, Y_test)

    #dims=[[10, 6800], [10, 6800], [10, 6800]] #list of bounds on the parameters, tuple means continuous list means discrete
    #dims=[[10, 6000], [10, 2024], [10, 556], (0.1, 0.9)] #list of bounds on the parameters, tuple means continuous list means discrete
    dims=[[2, 3700], [2, 2024], [2, 556]] #list of bounds on the parameters, tuple means continuous list means discrete

    x0=[3700, 1024, 256] #initial guess point for the parameters
    #x0=[4096, 1024, 256, 0.85] #initial guess point for the parameters
    #x0=[3000, 376, 175] #initial guess point for the parameters

    start = time()
    res=gp_minimize(cost.get_mae, dimensions=dims, x0=x0, n_initial_points=30, n_calls=300, noise=1e-5) #performs GP based bayesian opt
    end = time()
    print("Total time needed: {:.2f}".format((end-start)/60.))
    print(res.x) #values of the best parameters found
#    print(res.y) #value of the best cost
    f = open("cpu.txt", 'a')
    f.write("{},{},{}\n".format(res.x[0], res.x[1], res.x[2]))


if __name__ == "__main__":
  main()

