from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/dataset/dataset1/106r002p70_spikes.mat"
FIG_DIR = ROOT / "figures"
SESSION = DATA_FILE.stem.removesuffix("_spikes")
STIM_MS = 1280
ORI_STEP_DEG = 22.5
CHUNK_TRIALS = 200
NEURON = "n10"
BOUNDARY_NEURON = "n27"
MS_PER_S = 1000
RATE_LABEL = "mean spikes/s"
Y_MARGIN = 0.15


def load_df(rate=False):
    # h5py sees MATLAB's (neurons, ms, trials) transposed; summing per chunk avoids holding ~3 GB of float64.
    with h5py.File(DATA_FILE, "r") as f:
        stim = f["stim"][()].ravel().astype(int)
        v1 = f[f["spikes"][0, 0]]
        counts = np.empty((v1.shape[0], v1.shape[2]), dtype=int)
        for i in range(0, v1.shape[0], CHUNK_TRIALS):
            counts[i:i + CHUNK_TRIALS] = v1[i:i + CHUNK_TRIALS, :STIM_MS].sum(axis=1)

    df = pd.DataFrame(counts / (STIM_MS / MS_PER_S) if rate else counts, columns=[f"n{j}" for j in range(counts.shape[1])])
    df.insert(0, "orientation", (stim - 1) * ORI_STEP_DEG)
    df.insert(0, "stim", stim)
    df.index.name = "trial"
    return df


def neuron_cols(df):
    return [c for c in df.columns if c.startswith("n")]


def tuning_stats(df):
    grouped = df.groupby("orientation")[neuron_cols(df)]
    return grouped.mean(), grouped.sem()


def neuron_data(df, tuning, n):
    return df["orientation"].to_numpy(), df[n].to_numpy(), tuning.index.to_numpy(), tuning[n].to_numpy()


def style_axis(ax, title, ori, mean_r):
    pad = Y_MARGIN * np.ptp(mean_r)
    ax.set_ylim(mean_r.min() - pad, mean_r.max() + pad)
    ax.set_title(title)
    ax.set_xticks(ori, [f"{o:g}°" for o in ori])
    ax.set_xlabel("orientation (deg)")
    ax.set_ylabel(RATE_LABEL)
    ax.legend(fontsize=8)


def plot_pair(neuron, ori, mean_r, sem, curve, label, color, title, name):
    fig, (ax_mean, ax_fit) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ax_mean.errorbar(ori, mean_r, yerr=sem, marker="o", capsize=3, label="mean ± SEM")
    style_axis(ax_mean, f"{neuron}: mean orientation tuning", ori, mean_r)

    ax_fit.errorbar(ori, mean_r, yerr=sem, fmt="o", color="black", capsize=3, zorder=5, label="mean ± SEM")
    ax_fit.plot(*curve, lw=2.5, color=color, label=label)
    style_axis(ax_fit, f"{neuron}: {title}", ori, mean_r)
    ax_fit.tick_params(labelleft=True)
    plt.tight_layout()
    save(fig, name)
    plt.show()


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
