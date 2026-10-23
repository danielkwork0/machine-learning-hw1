from pathlib import Path

import h5py
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


def load_df():
    # h5py sees MATLAB's (neurons, ms, trials) transposed; summing per chunk avoids holding ~3 GB of float64.
    with h5py.File(DATA_FILE, "r") as f:
        stim = f["stim"][()].ravel().astype(int)
        v1 = f[f["spikes"][0, 0]]
        counts = np.empty((v1.shape[0], v1.shape[2]), dtype=int)
        for i in range(0, v1.shape[0], CHUNK_TRIALS):
            counts[i:i + CHUNK_TRIALS] = v1[i:i + CHUNK_TRIALS, :STIM_MS].sum(axis=1)

    df = pd.DataFrame(counts, columns=[f"n{j}" for j in range(counts.shape[1])])
    df.insert(0, "orientation", (stim - 1) * ORI_STEP_DEG)
    df.insert(0, "stim", stim)
    df.index.name = "trial"
    return df


def neuron_cols(df):
    return [c for c in df.columns if c.startswith("n")]


def tuning_stats(df):
    grouped = df.groupby("orientation")[neuron_cols(df)]
    return grouped.mean(), grouped.sem()


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
