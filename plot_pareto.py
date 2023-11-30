#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

plt.rcParams['font.size'] = 15

x_qm7b = [87, 74, 514, 917, 243, 232]
y_qm7b = [128, 150.7, 1358, 1159, 149, 151]

x_qm20 = [1239, 813, 290, 643, 1018, 1333]
y_qm20 = [6040, 7127, 7690, 6346, 8436, 6941]

x_ae = [88, 413, 77, 418, 302, 2508]
y_ae = [385, 2061, 400, 1130, 853, 3227]

x_ea = [4225, 1921, 1866, 684, 1987, 75163]
y_ea = [24769, 4951, 9387, 7550, 7128, 128077]

x_egp = [5087, 3529, 455, 917, 268, 6787]
y_egp = [19219, 7601, 1535, 1921, 751, 10382]

def plot_single_pareto(ax, X, Y, color, markers):
    for i, x in enumerate(X):
        #ax.scatter(x, Y[i], edgecolor="white", facecolor=color, s=80, marker=markers[i])
        ax.scatter(x, Y[i], facecolor=color, s=50, marker=markers[i])

def get_ratios(N, cpu):
    s2_n   = N[-1]
    s2_cpu = cpu[-1]

    diff_cpu = 1e6
    diff_n   = 1e6
    for i in range(len(N)-1):
        if diff_cpu > cpu[i]:
            diff_cpu = cpu[i]
            diff_n   = N[i]

    print("CPU:\t", s2_cpu / diff_cpu)
    print("N:\t", s2_n / diff_n)



def main():

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.axhline(y=168, color='k', alpha=0.5, linestyle="--", lw=2)
    ax.text(x=11000, y=200, s="1 week")
    ax.axhline(y=672, color='k', alpha=0.5, linestyle="--", lw=2)
    ax.text(x=10000, y=800, s="1 month")
    ax.axhline(y=8064, color='k', alpha=0.5, linestyle="--", lw=2)
    ax.text(x=10000, y=5000, s="1 year")
    ax.axhline(y=100000, color='k', alpha=0.5, linestyle="--", lw=2)
    ax.text(x=9500, y=60000, s="10 years")

    markers = ["s", "d", "v", "^", "*", "o"]
    plot_single_pareto(ax, x_qm7b, y_qm7b, "C0", markers)
    plot_single_pareto(ax, x_qm20, y_qm20, "C1", markers)
    plot_single_pareto(ax, x_ae, y_ae, "C2", markers)
    plot_single_pareto(ax, x_ea, y_ea, "C3", markers)
    plot_single_pareto(ax, x_egp, y_egp, "C4", markers)

    ax.scatter( [ x_qm7b[-1], x_qm20[-1], x_ae[-1], x_ea[-1], x_egp[-1] ], [ y_qm7b[-1], y_qm20[-1], y_ae[-1], y_ea[-1], y_egp[-1] ], facecolor="None", edgecolor="k", lw=2, s=80)

    ax.set_xticks([100, 1000, 16000])
    ax.set_yticks([100, 1000, 10000, 100000])
    ax.set_xticklabels(["100", "1000", "16000"])
    ax.set_yticklabels(["100", "1000", "10000", "100000" ])


    ax.set_xlabel(r'$N^\mathrm{CCSD(T)}$')
    ax.set_ylabel('Cost [h]]')

    legend_elements = [Line2D([0], [0], color="C0", lw=3, label="QM7b"),
                Line2D([0], [0], color="C1", lw=3, label="QM9 LCCSD(T)"),
                Line2D([0], [0], color="C2", lw=3, label="QM9 CCSD(T) AE"),
                Line2D([0], [0], color="C3", lw=3, label="QM9 CCSD(T) EA"),
                Line2D([0], [0], color="C4", lw=3, label="EGP"),
                Line2D([0], [0], color="k", marker="o", label="M2L"),
                Line2D([0], [0], color="k", marker="d", label=r'Cost'),
                Line2D([0], [0], color="k", marker="s", label=r'Guess'),
                Line2D([0], [0], color="k", marker="v", label=r'$\beta=0.25$'),
                Line2D([0], [0], color="k", marker="^", label=r'$\beta=0.50$'),
                Line2D([0], [0], color="k", marker="*", label=r'$\beta=0.75$'),
                    ]
    ax.legend(handles=legend_elements, ncols=2, fontsize=9)

    plt.tight_layout()

    fig.savefig("pareto_tot.png")


if __name__ == '__main__':
    main()
