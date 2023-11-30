#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter

def set_axes(ax):
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    #ax.set_xlim([15, 6700])

    # Set x and y axis labels
    ax.set_xlabel(r"$\mathit{N}^\mathrm{CCSD(T)}$")
    ax.set_ylabel("MAE [kcal/mol]")

    # Increase linewidth and tick label size
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=8)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    # Set minor ticks and grid
    ax.minorticks_on()
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

    # Set size of minor ticks
    ax.tick_params(axis='both', which='minor', width=1, length=4)

    # Remove top and right axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Move x-axis down a little
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_position(("axes", -0.05))

    # Move y-axis to the left a little
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))





# Set default font size
mpl.rcParams['font.size'] = 20

filenames_qm7 = ["direct_qm7.txt", "s2_qm7.txt", "results_qm7b.txt", "cpu_qm7.txt", "b25_qm7.txt", "b50_qm7.txt", "b75_qm7.txt" ]
filenames_qm9lcc = ["direct_qm9lcc.txt", "s2_qm9lcc.txt", "results_qm20k.txt", "cpu_qm9lcc.txt", "b25_qm9lcc.txt", "b50_qm9lcc.txt", "b75_qm9lcc.txt" ]
filenames_ae = ["direct_ae.txt", "s2_ae.txt", "results_qm9AE.txt", "cpu_ae.txt", "b25_ae.txt", "b50_ae.txt", "b75_ae.txt" ]
filenames_ea  = ["data/direct_ea.txt", "data/s2_ea.txt", "data/results_qm9ea.txt", "data/cpu_ea.txt", "data/b25_ea.txt", "data/b50_ea.txt", "data/b75_ea.txt" ]
filenames_egp = ["data/direct_egp.txt", "data/s2_egp.txt", "data/results_egp.txt", "data/cpu_egp.txt", "data/b25_egp.txt", "data/b50_egp.txt", "data/b75_egp.txt" ]

# Define line colors, markers, and linestyles for each curve
colors = ["C0", "C3", "C5", "C1", "C2", "C2", "C2", "C2", "C2"]
markers = ["p", "o", "s", "d", "v", "^", "*" ]
linestyles = ["-", "--",  ":", "-.", "-.", "-.", "-.", "-"]
labels = [ "Direct", "M2L", "Guess", "Cost", r'$\beta=0.25$', r'$\beta=0.5$', r'$\beta=0.75$']

# Create figure and axis objects
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30,6))

# Loop over files
ax = axes[0]
for i, filename in enumerate(filenames_qm7):
    # Load data from file
    data = np.loadtxt("data/" + filename, delimiter=",", skiprows=1)

    # Extract columns
    N = data[:,0]
    MAE = data[:,3]

    # Custom xticks according to N
    xticks = [1, 4, 16, 64, 256, 4096]
    yticks = [1, 2, 4, 8, 16]
    ax.set_title("QM7b")

    # Plot data
    ax.loglog(N, MAE, marker=markers[i], linestyle=linestyles[i], color=colors[i], label=labels[i])

set_axes(ax)

# Loop over files
ax = axes[1]
for i, filename in enumerate(filenames_qm9lcc):
    # Load data from file
    data = np.loadtxt("data/" + filename, delimiter=",", skiprows=1)

    # Extract columns
    N = data[:,0]
    MAE = data[:,3]

    # Custom xticks according to N
    xticks = [8, 32, 128, 512, 2048, 16000]
    yticks = [1, 2, 4, 8, 16]

    ax.set_title(r'QM9$^\mathrm{LCCSD(T)}$')
    # Plot data
    ax.loglog(N, MAE, marker=markers[i], linestyle=linestyles[i], color=colors[i], label=labels[i])

set_axes(ax)

# Loop over files
ax = axes[2]
for i, filename in enumerate(filenames_ae):
    # Load data from file
    data = np.loadtxt("data/" + filename, delimiter=",", skiprows=1)

    # Extract columns
    N = data[:,0]
    MAE = data[:,3]

    # Custom xticks according to N
    xticks = [8, 32, 128, 512, 2048]
    yticks = [2, 4, 8, 16]

    ax.set_title(r'QM9$^\mathrm{CCSD(T)}_\mathrm{AE}$')
    # Plot data
    ax.loglog(N, MAE, marker=markers[i], linestyle=linestyles[i], color=colors[i], label=labels[i])

set_axes(ax)

# Loop over files
ax = axes[3]
for i, filename in enumerate(filenames_ea):
    # Load data from file
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    # Extract columns
    N = data[:,0]
    MAE = data[:,3]

    # Custom xticks according to N
    xticks = [2, 8, 32, 128, 512, 2048]
    yticks = [1.2, 1.4, 1.6, 2, 3]

    ax.set_title(r'QM9$^\mathrm{CCSD(T)}_\mathrm{EA}$')

    # Plot data
    ax.loglog(N, MAE, marker=markers[i], linestyle=linestyles[i], color=colors[i], label=labels[i])

set_axes(ax)



# Loop over files
ax = axes[4]
for i, filename in enumerate(filenames_egp):
    # Load data from file
    data = np.loadtxt(filename, delimiter=",", skiprows=1)

    # Extract columns
    N = data[:,0]
    MAE = data[:,3]

    # Custom xticks according to N
    xticks = [2, 8, 32, 128, 512]
    yticks = [4, 8, 16, 32, 64]

    ax.set_title(r'EGP')
    # Plot data
    ax.loglog(N, MAE, marker=markers[i], linestyle=linestyles[i], color=colors[i], label=labels[i])

set_axes(ax)

# Set xticks
set_axes(ax)
# Add legend
ax.legend(ncols=1, fontsize=18)

# Save plot as PNG file
plt.savefig("lc_combined.png", dpi=300, bbox_inches="tight")

# Show plot
#plt.show()

