#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

import qml
from qml.representations import *
from time import time
import random

from qml.kernels import get_local_symmetric_kernel_mbdf, get_local_kernel_mbdf
from copy import deepcopy
from qml.math import svd_solve

import MBDF

from yaspin import yaspin
from colored import fg


RED = fg('red')
WHITE = fg('white')
GREEN = fg('green')

tLDA         = np.load("get_sigmas_LDA.npz")
sigmas_LDA   = tLDA['sigmas']


Hybrid  = [ 'TPSSH', 'B3LYP(VWN5)',
                'O3LYP(VWN5)', 'KMLYP(VWN5)', 'PBE0', 'B3LYP*(VWN5)', 'BHANDH', 'BHANDHLYP',
                'B97-1', 'B97-2', 'MPBE0KCIS', 'MPBE1KCIS', 'B1LYP(VWN5)', 'B1PW91(VWN5)', 'MPW1PW',
                'MPW1K', 'TAU-HCTH-HYBRID', 'X3LYP(VWN5)', 'OPBE0', 'M05', 'M05-2X',
                'M06', 'M06-2X', 'B3LYP-D']


MetaGGA = ['T-MGGA',  'PKZBX-KCISCOR', 'TPSS', 'TPSS-D', 'REVTPSS', 'M06-L', 'TAU-HCTH', 'VS98-X(XC)',
               'VS98-X-ONLY', 'BECKE00', 'BECKE00X(XC)', 'BECKE00-X-ONLY', 'OLAP3',  'BMTAU1', 'MPBEKCIS', ]

GGA     = ['PW91', 'BLYP',   'BLYP-D', 'BP', 'PBE', 'RPBE', 'REVPBE', 'OLYP', 'FT97', 'PBESOL',
                'HCTH/93', 'HCTH/120', 'HCTH/147', 'HCTH/407', 'BOP', 'PBE-D',
                'B97-D', 'B97', 'BP86-D', 'KT1', 'KT2', 'MPBE', 'OPBE', 'OPERDEW', 'MPW',
                'BECKE88X+BR89C', 'XLYP',]

LDA     = ['LDA(VWN)']

@yaspin(text=" Calculate Kernels")
def do_ML(train, X, X_test, Y, sigma, Q, Q_test):
  # train ML model and return predictions (cross validated w/ # nModels)

  total = list(range(len(X)))
  random.shuffle(total)
  training_index = total[:train]

  K      = get_local_symmetric_kernel_mbdf(X[training_index],  Q[training_index], sigma)
  K_test = get_local_kernel_mbdf(X_test, X[training_index], Q_test, Q[training_index], sigma)

  Yp = Y[training_index]
  C = deepcopy(K)
  alpha = svd_solve(C, Yp)

  Yss = np.dot((K_test).T, alpha)

  return Yss

def opt_sigma(train_set_size):
    N = np.array([ 2,  50,  100,  1000,  4000])

    # Calculate the absolute differences between each number in array1 and array2
    differences = np.abs(N - train_set_size)

    # Find the index of the minimum difference for each number in array1
    min_indices = np.argmin(differences)

    # Use the index to find the closest number in array2 for each number in array1
    return sigmas_LDA[min_indices]


def do_LC(X_train, X_test, Q_train, Q_test, Y_LDA, Y_GGA, Y_mGGA, Y_HYBRID, Y_test, func_gga, func_mgga, func_hybrid):

  # direct learning
  N_LDA    = [32, 64, 128, 256, 512, 1024, 2048, 4096]
  N_GGA    = [ 8, 16,  32,  64, 128,  256,  512, 1024]
  N_HYBRID = [ 2,  4,   8,  16,  32,   64,  128,  256]

  e_multi    = np.array([])

  #split = list(range(total))
  nModels = 5

  for i in range(nModels):
      MAE_multi   = np.array([])

      for train in range(len(N_LDA)):
        s_LDA = opt_sigma(N_LDA[train])
        Yp_LDA_direct  = do_ML(N_LDA[train], X_train, X_test, Y_LDA, s_LDA, Q_train, Q_test)

        s_GGA = opt_sigma(N_GGA[train])
        Yp_LDA_GGA    = do_ML(N_GGA[train], X_train, X_test, Y_mGGA, s_GGA, Q_train, Q_test)

        s_HYBRID = opt_sigma(N_HYBRID[train])
        Yp_GGA_HYBRID   = do_ML(N_HYBRID[train], X_train, X_test, Y_HYBRID, s_HYBRID, Q_train, Q_test)

        Y_multi  = Yp_LDA_direct + Yp_LDA_GGA + Yp_GGA_HYBRID

        mae_multi   = np.mean(np.abs(Y_multi-Y_test))
        MAE_multi   = np.append(MAE_multi, mae_multi)

      e_multi   = np.append(e_multi, np.asarray(MAE_multi))

  e_multi   = e_multi.reshape(nModels,len(N_LDA)).mean(axis=0)

  for i in range(len(N_HYBRID)):
    print("{:.2f},{:.4f}".format(N_HYBRID[i], e_multi[i]))

  return True




def read_CCSD(filename):
    lines = open(filename, 'r').readlines()

    names    = np.array([])
    energies = np.array([])


    for i, line in enumerate(lines):
        if i == 0: continue

        tokens = line.split(',')

        name = tokens[0]
        names = np.append(names, name)

        energy = float(tokens[5])
        energies = np.append(energies, energy)

    return names, energies

def read_DFT(filename):
    return pd.read_csv(filename)


def main():
    print("\n [    ] Read data")
    filename_CCSD = "qm9_ae.txt"
    filename_DFT = "molecules_qm9.txt"

#    names, energies = read_CCSD(filename_CCSD)
    random.seed(667)

    df = read_DFT(filename_DFT)

    df['names'] = 'dsgdb9nsd_' + df['index'].astype(str).str.zfill(6)

    print(df)

    array1_length = 15000
    array2_length = 3000

    # Randomly sample rows for array1
    df_train = df.sample(n=array1_length, random_state=42)

    # Remove the sampled rows from the DataFrame
    df = df.drop(df_train.index)

    # Randomly sample rows for array2 from the remaining DataFrame
    df_test = df.sample(n=array2_length, random_state=42)

    names_all = np.concatenate((df_train['names'].to_numpy(), df_test['names'].to_numpy()))

    print(" [ {}OK{} ] Read data".format(GREEN, WHITE))

    mols = []

    spinner = yaspin(text="Calculate Representation", color="yellow")
    spinner.start()
    start = time()

    for name in names_all:
        mol = qml.Compound()
        mol.read_xyz("xyz/" + name + ".xyz")
        mol.name = name
        mols.append(mol)

    coords = np.asarray([mol.coordinates for mol in mols])
    coords *= 1.88973
    Q = np.asarray([mol.nuclear_charges for mol in mols])

    X = MBDF.generate_mbdf(Q,coords,pad=29,cutoff_r=10)
    end = time()
    spinner.stop()
    np.savez("scan_dft.npz", X=X, Q=Q, df_train=df_train, df_test=df_test)


    Y_LDA = df_train['LDA(VWN)_TZP'].to_numpy()
    #Y_LDA *= 23
    X_train, Q_train = X[:-3000], Q[:-3000]
    X_test, Q_test   = X[-3000:], Q[-3000:]

    for hybrid in Hybrid:
        for mgga in MetaGGA:
                gga = mgga

                Y_GGA    = df_train[gga + "_TZP"].to_numpy() -  Y_LDA
                Y_mGGA   = df_train[mgga + "_TZP"].to_numpy()  - Y_LDA

                Y_HYBRID = df_train[hybrid + "_TZP"].to_numpy() - df_train[mgga + "_TZP"].to_numpy()
                Y_test   = df_test[hybrid + "_TZP"].to_numpy()


                isDone = do_LC(X_train, X_test, Q_train, Q_test, Y_LDA, Y_GGA, Y_mGGA, Y_HYBRID, Y_test, gga, mgga, hybrid)
                exit()


if __name__ == '__main__':
    main()

