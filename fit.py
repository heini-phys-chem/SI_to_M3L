#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

times = {"HF": 0.016328125, "MP2": 0.039101563, "CCSD": 0.235273438}

def read_data(f):
    # Read in the data from the CSV file
    data = pd.read_csv(f)
    return data

# Define the linear function to fit on a log-log scale
def linear(x, a, b):
    return a * x + b

def logdata(data):
    # Take the logarithm of the data
    data["N_CCSD_log"] = np.log10(data["N_CCSD"])
    data["N_MP2_log"] = np.log10(data["N_MP2"])
    data["N_HF_log"] = np.log10(data["N_HF"])
    data["e_direct_log"] = np.log10(data["e_direct"])

    return data

def fitdata(data, N):
    # Fit the linear function to N vs e_direct
    x = data[N][-3:]
    y = data["e_direct_log"][-3:]
    popt_e, pcov_e = curve_fit(linear, x, y)

    return popt_e

def hitone(popt_e):
    # Find where N hits 1 for e_direct
    n_e = 10**((np.log10(1) - popt_e[1])/popt_e[0])

    return n_e

def main():
    filename = sys.argv[1]
    data = read_data(filename)

    data = logdata(data)
    popt_CCSD = fitdata(data, "N_CCSD_log")
    popt_MP2  = fitdata(data, "N_MP2_log")
    popt_HF   = fitdata(data, "N_HF_log")

    n_ccsd = hitone(popt_CCSD)
    n_mp2  = hitone(popt_MP2)
    n_hf   = hitone(popt_HF)


    print("N at which e_direct = 1: {:.2f}, {:.2f}, {:.2f}".format(n_ccsd, n_mp2, n_hf))
    print("times at which e_direct = 1: {:.2f}, {:.2f}, {:.2f}".format(n_ccsd*times["CCSD"], n_mp2*times["MP2"], n_hf*times["HF"]))
    print("total time: {:.2f}".format(n_ccsd*times["CCSD"]+n_mp2*times["MP2"]+n_hf*times["HF"]))

if __name__ == '__main__':
    main()
