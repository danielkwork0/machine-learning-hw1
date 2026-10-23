import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from common import NEURON, STIM_MS, load_df, save, tuning_stats

DEGREES = {"Linear": 1, "3rd-order": 3, "4th-order": 4, "7th-order": 7, "15th-order": 15}
GRID_POINTS = 400
JITTER_DEG = 4


def fit_polynomials(theta, r, ori, mean_r):
    fits, rows = {}, []
    for name, deg in DEGREES.items():
        p = Polynomial.fit(theta, r, deg)
        fits[name] = p
        rows.append({
            "model": name,
            "degree": deg,
            "n_params": deg + 1,
            "MSE (trials)": np.mean((r - p(theta)) ** 2),
            "MSE (means)": np.mean((mean_r - p(ori)) ** 2),
        })
    return fits, pd.DataFrame(rows).set_index("model")


def main():
    df = load_df()
    tuning, sem = tuning_stats(df)
    theta = df["orientation"].to_numpy()
    r = df[NEURON].to_numpy()
    ori = tuning.index.to_numpy()
    mean_r = tuning[NEURON].to_numpy()
    theta_grid = np.linspace(ori.min(), ori.max(), GRID_POINTS)

    fits, mse_table = fit_polynomials(theta, r, ori, mean_r)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(theta + rng.uniform(-JITTER_DEG, JITTER_DEG, theta.size), r, s=4, alpha=0.08, color="gray", label="single trials")
    ax.errorbar(ori, mean_r, yerr=sem[NEURON], fmt="o", color="black", capsize=3, zorder=5, label="mean ± SEM")
    for name, p in fits.items():
        ax.plot(theta_grid, p(theta_grid), lw=2, label=f"{name}  (MSE {mse_table.loc[name, 'MSE (trials)']:.1f})")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(ori, [f"{o:g}°" for o in ori])
    ax.set_xlabel("orientation (deg)")
    ax.set_ylabel(f"spikes / {STIM_MS} ms")
    ax.set_ylim(bottom=min(p(theta_grid).min() for p in fits.values()) - 5, top=np.percentile(r, 99.5))
    ax.set_title(f"{NEURON}: polynomial fits")
    ax.legend()
    plt.tight_layout()
    save(fig, "task2_polynomial_fits")
    plt.show()

    print("trial-to-trial variance (floor for trial MSE):",
          round(df.groupby("orientation")[NEURON].var(ddof=0).mean(), 2))
    print(mse_table.round(2))


if __name__ == "__main__":
    main()
