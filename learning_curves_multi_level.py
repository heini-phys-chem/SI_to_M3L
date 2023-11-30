#!/usr/bin/env python3

import sys
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
from qml.kernels import get_local_symmetric_kernel
from qml.kernels import get_local_kernel
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel

from qml.kernels import get_local_symmetric_kernel_mbdf, get_local_kernel_mbdf
#from qml.kernels import get_local_symmetric_kernel_laplacian_danish
#from qml.kernels import get_local_kernel_laplacian_danish

import itertools
from time import time

from tqdm import tqdm
import pandas as pd

from colored import fg
from yaspin import yaspin

import MBDF

RED = fg('red')
WHITE = fg('white')
GREEN = fg('green')


def read_costs(f):
    return pd.read_csv(f)

@yaspin(text=" Calculate Kernels")
def do_ML(idx_train, train, X, X_test, Yprime, sigma, Q, Q_test):
  # train ML model and return predictions (cross validated w/ # nModels)

  total = list(range(len(idx_train)))
  random.shuffle(total)
  training_index = total[:train]

#  K      = laplacian_kernel(X[training_index], X[training_index], sigma)
#  K_test = laplacian_kernel(X[training_index], X_test, sigma)
  K      = get_local_symmetric_kernel_mbdf(X[training_index],  Q[training_index], sigma)
  K_test = get_local_kernel_mbdf(X_test, X[training_index], Q_test, Q[training_index], sigma)

  Y = Yprime[training_index]
  C = deepcopy(K)
  alpha = svd_solve(C, Y)

  Yss = np.dot((K_test).T, alpha)

  return Yss, training_index

def opt_sigma(N, X, Q, Y):
    idxs = list(range(len(X)))
    nModels = 5
    sigmas = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

    SIGMAS = np.array([])

    for n in N:
        MAEs = np.array([])
        for sigma in sigmas:
            maes = np.array([])
            for nmodel in range(nModels):
                random.shuffle(idxs)
                idxs_n = idxs[:n]

                split = int(len(idxs_n)*0.7)
                train = idxs_n[:split]
                test  = idxs_n[split:]

                X_train = X[train]
                Q_train = Q[train]
                Y_train = Y[train]
                X_test, Q_test, Y_test = X[test], Q[test], Y[test]

                #K      = get_local_symmetric_kernel(X_train, Q_train, sigma)
                #K_test = get_local_kernel(X_train, X_test, Q_train, Q_test, sigma)
                K      = get_local_symmetric_kernel_mbdf(X_train, Q_train, sigma)
                K_test = get_local_kernel_mbdf(X_train, X_test, Q_train, Q_test, sigma)

                C = deepcopy(K)
                alpha = svd_solve(C, Y_train)

                Yss = np.dot(K_test, alpha)

                mae = np.abs(np.mean(Y_test-Yss))
                maes = np.append(maes, mae)
            MAEs = np.append(MAEs, np.mean(maes))
        SIGMAS = np.append(SIGMAS, sigmas[np.argmin(MAEs)])

    return SIGMAS


#def do_LC(X, X_test, total, nModels, mols_train, mols_test, HF_times, MP_times, CC_times, Q, Q_test
def do_LC(X, Q, Y_CCSD, Y_HF, Y_HF_MP2, Y_MP2_CCSD, idx_train, idx_test, HF_times, MP_times, CC_times):

  # direct learning
  #N_CCSD = [ 2,  4,  8,  16,  32,  64,  128,  256,  512]#, 1024, 2048, 4096, 6600]

  # s2
#  N_HF   = [32, 64, 128, 256, 512, 1024, 2048, 3700]
#  N_MP2  = [8, 16, 32, 64, 128, 256, 512, 1024 ]
#  N_CCSD = [2, 4, 8, 16, 32, 64, 128, 256]

  # multi level (BOPT CPU)
#  N_HF   = [ 445, 890, 1780, 2992, 3179, 3366, 3561 ]
#  N_MP2  = [ 113, 226, 453, 760, 807, 855, 906 ]
#  N_CCSD = [ 2, 4, 9, 16, 17, 18, 19 ]


  # fancy guess
  N_HF   = [ 1000, 2000, 3250, 3500 ]
  N_MP2  = [   32,   64,  104,  112 ]
  N_CCSD = [    4,    8,   13,   14 ]
#
#  # b75
#  N_HF   = [ 442, 885, 1771, 2873, 3094, 3315, 3543 ]
#  N_MP2  = [ 20, 41, 83, 136, 147, 157, 167 ]
#  N_CCSD = [ 2, 4, 8, 13, 14, 15, 16 ]
#
#  # b50
#  N_HF   = [ 300, 601, 1203, 1920, 2080, 2240, 2407  ]
#  N_MP2  = [25, 50, 101, 162, 175, 189, 202, 203 ]
#  N_CCSD = [ 1, 3, 7, 12, 13, 14, 15  ]
#
#  #b25
#  N_HF   = [ 462, 925, 1850, 2849, 3256, 3700 ]
#  N_MP2  = [ 33, 67, 135, 210, 240, 270]
#  N_CCSD = [ 1, 2, 4, 7, 8, 9]

#  #b95
#  N_HF   = [1458, 2916, 5833, 11667 ]
#  N_MP2  = [105, 210, 421, 843 ]
#  N_CCSD = [54, 108, 216, 433 ]

#  #b65
#  N_HF   = [ 1256, 2400, 4800, 9600  ]
#  N_MP2  = [  260, 520, 1040, 2080 ]
#  N_CCSD = [  11, 22, 44, 88 ]

#  #b85
#  N_HF   = [ 1534, 3068, 6137, 12260  ]
#  N_MP2  = [ 330, 661, 1322, 2644  ]
#  N_CCSD = [ 44, 89, 179, 360  ]
#
#  #beta
#  N_HF   = [ 300, 600, 1200, 2400, 4800, 9600 ]
#  N_MP2  = [ 10, 20, 40, 80, 160, 320 ]
#  N_CCSD = [ 10, 20, 40, 80, 160, 320 ]

# get timings: beta CPU  b095 85 75 65 05 25 s2
#  N_HF   = [300, 13353, 11667, 6137, 1422, 1256, 1035, 958, 16000]
#  N_MP2  = [ 10,   3534,   843, 1322,  593,  260,  139, 193,  4096]
#  N_CCSD = [ 10,    346,   433,  179,   58,   11,   30,  10,  1024]
#
## get timings fit: 25, 05, 065, 075, 085, 095, cpu, guess, s2
#  N_HF   = [  38188, 31634, 48037, 29554, 31469, 33427, 31239, 33309, 27064 ]
#  N_MP2  = [  7637, 4428, 10408, 13414, 6789, 2421, 8271, 3166, 7023  ]
#  N_CCSD = [  381, 847, 440, 1341, 931, 1242, 813, 962, 1755 ]
#  N_HF = [8188, 17560, 32669, 25456, 53879, 19518]
#  N_MP2  = [612, 1500, 1568, 3241, 1995, 7189]
#  N_CCSD   = [20, 133, 147, 1312, 332, 1797]


  cYprime_HF          = Y_HF[idx_train]
  cYprime_HF_to_MP2   = Y_HF_MP2[idx_train]
  cYprime_MP2_to_CCSD = Y_MP2_CCSD[idx_train]
  cY_test             = Y_CCSD[idx_test]

  e_multi    = np.array([])
  ts_multi   = np.array([])

  #split = list(range(total))
  nModels = 10
  start = time()
  print(" [    ] opt sigmas")
  sigma_HF   = opt_sigma(N_HF, X[idx_train], Q[idx_train], cYprime_HF)
  sigma_MP2  = opt_sigma(N_MP2, X[idx_train], Q[idx_train], cYprime_HF_to_MP2)
  sigma_CCSD = opt_sigma(N_CCSD, X[idx_train], Q[idx_train], cYprime_MP2_to_CCSD)
  end = time()
  print(" [ {}OK{} ] opt sigmas ({:.2f} min)".format(GREEN, WHITE, (end-start)/60.))

  for i in range(nModels):
      MAE_multi   = np.array([])
      times_multi   = np.array([])

      for train in range(len(N_HF)):
      #for train in range(1):
        # Direct Learning QML
        #Y_HF_direct, idxs_HF  = do_ML(split, N_HF[train], X, X_test, cYprime_HF, sigma_HF[-1], ll_HF[-1], Q, Q_test)
#        sigma_HF = opt_sigma(N_HF[train], "HF")
        Y_HF_direct, idxs_HF  = do_ML(idx_train, N_HF[train], X[idx_train], X[idx_test], cYprime_HF, sigma_HF, Q[idx_train], Q[idx_test])
        t_HF = np.sum(HF_times[idxs_HF])

        # Delta learning (HF - MP2)
#        sigma_MP2 = opt_sigma(N_MP2[train], "MP2")
        Y_HF_to_MP2, idxs_HF_to_MP2  = do_ML(idx_train, N_MP2[train],  X[idx_train], X[idx_test], cYprime_HF_to_MP2, sigma_MP2, Q[idx_train], Q[idx_test])
        #t_HF_to_MP2 = np.sum(HF_times[idxs_HF_to_MP2]) + np.sum(MP_times[idxs_HF_to_MP2])
        t_HF_to_MP2 = np.sum(MP_times[idxs_HF_to_MP2])

        # Delta learning (MP2 - CCSD)
#        sigma_CCSD = opt_sigma(N_CCSD[train], "CCSD")
        Y_MP2_to_CCSD, idxs_MP2_to_CCSD  = do_ML(idx_train, N_CCSD[train],  X[idx_train], X[idx_test], cYprime_MP2_to_CCSD, sigma_CCSD, Q[idx_train], Q[idx_test])
        #t_MP2_to_CCSD = np.sum(MP_times[idxs_MP2_to_CCSD]) + np.sum(CC_times[idxs_MP2_to_CCSD])
        t_MP2_to_CCSD = np.sum(CC_times[idxs_MP2_to_CCSD])

        # get energy prediction 2- and 3- levels
        Y_multi  = Y_HF_direct + Y_HF_to_MP2 + Y_MP2_to_CCSD
        t_multi  = t_HF + t_HF_to_MP2 + t_MP2_to_CCSD

        mae_multi   = np.mean(np.abs(Y_multi-cY_test))
        MAE_multi   = np.append(MAE_multi, mae_multi)
        times_multi = np.append(times_multi, t_multi)
        print("N: {:.2f},\tMAE: {:.2f}, CPU: {:.2f}, {}".format(N_CCSD[train], mae_multi, t_multi, i))

      e_multi   = np.append(e_multi, np.asarray(MAE_multi))
      ts_multi  = np.append(ts_multi, np.asarray(times_multi))

  e_multi   = e_multi.reshape(nModels,len(N_HF)).mean(axis=0)
  ts_multi  = ts_multi.reshape(nModels, len(N_HF)).mean(axis=0)

  print("\nN_CCSD,N_MP2,N_HF,e_direct,t_direct")
  for i in range(len(e_multi)):
      print("{},{},{},{:.2f},{:.2f}".format(N_CCSD[i], N_MP2[i], N_HF[i], e_multi[i], ts_multi[i]))

  return True

def main():
  # file containing xyz file names (without extension .xyz) and according proerties
  #mols_train, mols_test = read_files()
  df = read_costs("qm9_ae.txt")
  HF_times = df['t_HF_h'].to_numpy()
  MP_times = df['t_MP2_h'].to_numpy()
  CC_times = df['t_CCSD_h'].to_numpy()
  print(" [ {}OK{} ] Read in Files".format(GREEN, WHITE))

  data       = np.load("qm9_ae.npz", allow_pickle=True)
  X          = data['X']
  Q          = data['Q']
  Y_CCSD     = data['Y_CCSD']
  Y_HF       = data['Y_HF']
  Y_HF_MP2   = data['Y_HF_MP2']
  Y_MP2_CCSD = data['Y_MP2_CCSD']
  idx_train  = np.concatenate((data['idx_train'], data['idx_val']))
  idx_test   = data['idx_test']

  #start = time()
  #X_train, X_test, Q_train, Q_test = get_reps(mols_train, mols_test)
  #end = time()
  #print(" [ {}OK{} ] Calculate Representation ({:.2f} min)\n".format(GREEN, WHITE, (end-start)/60.))

  #total   = len(mols_train)

  random.seed(667)

  start = time()
  #isDone = do_LC(X_train, X_test, total, nModels, mols_train, mols_test, HF_times, MP_times, CC_times, Q_train, Q_test)
  isDone = do_LC(X, Q, Y_CCSD, Y_HF, Y_HF_MP2, Y_MP2_CCSD, idx_train, idx_test, HF_times, MP_times, CC_times)
  end = time()
  if isDone:
      print("\n [ {}OK{} ] Generate Learning Curves ({:.2f} min)\n".format(fg('green'), fg('white'), (end-start)/60.))
  else:
      print("\n [ {}FAILED{} ] Generate Learning Curves ({:.2f} min)\n".format(fg('red'), fg('white'), (end-start)/60.))


if __name__ == "__main__":
  main()

