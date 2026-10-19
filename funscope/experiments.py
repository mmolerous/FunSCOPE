"""Helpers for the Monte Carlo experiments: reading data, summaries, plots and storage."""

import shelve
from statistics import stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import imse_simpson


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
def data_file(data_dir, prefix, scenario, N, m, ext="csv"):
    """Path of a simulation file, e.g. X_ScenarioB_N_25_m_50.csv."""
    return data_dir / f"{prefix}_Scenario{scenario}_N_{N}_m_{m}.{ext}"


def read_csv(path):
    """Read a CSV written by R as a DataFrame."""
    return pd.read_csv(path, sep=",", decimal=".")


def read_array(path):
    return np.array(read_csv(path))


def read_grid(path):
    """Read the observation grid as a vector."""
    return read_array(path)[:, 0]


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------
def mean_std(values):
    return np.mean(values), stdev(values)


def bias_variance(t, beta_true, beta_est):
    """Mean estimate, integrated squared bias and integrated variance of the beta estimates."""
    beta_est = np.array(beta_est)
    beta_mean = np.mean(beta_est, axis=0)
    isb = imse_simpson(t, beta_true, beta_mean)
    ivar = np.mean([imse_simpson(t, np.array(beta_est[i]), beta_mean) for i in range(len(beta_est))])
    return beta_mean, isb, ivar


def write_report(path, title, raw, averaged):
    """
    Text report with the per-simulation values (`raw`) and the averaged
    metrics (`averaged`); both are lists of (label, value) pairs.
    """
    with open(path, "w") as f:
        print("-------------------------", file=f)
        print(*title, file=f)
        print("-------Not averaged-----------", file=f)
        for label, value in raw:
            print(label, value, file=f)
        print("-------Averaged-----------", file=f)
        for label, value in averaged:
            print(label, value, file=f)


def save_shelf(path, values):
    """Store a dictionary of results with `shelve`."""
    with shelve.open(str(path)) as shelf:
        for key, value in values.items():
            shelf[key] = value


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def _style_axes(ax, ylim):
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel(r"$\beta(t)$", fontsize=14)
    ax.set_ylim(*ylim)
    plt.axhline(y=0, color="black", linestyle="--")
    ax.legend(fontsize=12)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.xaxis.set_tick_params(width=1.2)
    ax.yaxis.set_tick_params(width=1.2)


def plot_band(t, beta_true, beta_est, ylim, path):
    """True beta(t), Monte Carlo mean of the estimates and 95% pointwise band (2.5% and 97.5% percentiles)."""
    beta_est = np.array(beta_est)
    beta_mean = np.mean(beta_est, axis=0)
    beta_upp = np.percentile(beta_est, 97.5, axis=0)
    beta_low = np.percentile(beta_est, 2.5, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, beta_true, label=r"Real $\beta(t)$", color="blue", linewidth=2)
    ax.plot(t, beta_mean, label=r"Estimated $\beta(t)$", color="red", linestyle="--", linewidth=2)
    ax.fill_between(t, beta_low, beta_upp, color="red", alpha=0.2, label="Confidence Interval")
    _style_axes(ax, ylim)
    plt.savefig(path)
    plt.close(fig)
    return beta_low, beta_upp


def plot_some(t, beta_true, beta_est, ylim, path, n=5):
    """True beta(t) and the estimates of the first `n` simulations."""
    beta_est = np.array(beta_est)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, beta_true, label=r"Real $\beta(t)$", color="blue", linewidth=2)
    for i in range(min(n, len(beta_est))):
        ax.plot(t, beta_est[i, :], label=r"Simulated $\beta(t)$", color="red", linestyle="--", linewidth=2)
    _style_axes(ax, ylim)
    plt.savefig(path)
    plt.close(fig)
