#!/usr/bin/env python3

#%%
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

c = {"HF": 0.008790413132406904, "MP": 0.08126472507970883 , "CC": 1.52856547479396}
ml = {"HF": 12.827162501165082, "MP": 1.3400759362116055, "CC": 0.32019267235718435}


def cost_and_error(N_HF, N_MP, N_CC):
    cost = N_HF * c["HF"] + N_MP * c["MP"] + N_CC * c["CC"]
    eA = ml["HF"] * N_HF ** (-1 / 2)
    eB = ml["MP"] * N_MP ** (-1 / 2)
    eC = ml["CC"] * N_CC ** (-1 / 2)
    return cost, np.sqrt(eA**2 + eB**2 + eC**2)


def optimal_for_error(thresh):
    lvl = [int(_) for _ in 2 ** np.linspace(0, 15)]

    mcost = 1e100
    opt = None
    for NHF, NMP, NCC in it.product(lvl, lvl, lvl):
        cost, error = cost_and_error(NHF, NMP, NCC)
        if cost < mcost and error < thresh:
            mcost = cost
            opt = (NHF, NMP, NCC)
    return opt

# %%
errors = (64, 32, 16, 8, 4, 2, 1)
values = [optimal_for_error(_) for _ in errors]
values = np.array(values)
print("MAE, N_HF, N_MP2, N_CCSD")
print(np.hstack((np.array(errors)[:, np.newaxis], values)))
