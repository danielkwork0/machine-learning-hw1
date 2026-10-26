import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from common import NEURON, load_df, neuron_data, save, style_axis, tuning_stats
from task2_polynomial_fits import GRID_POINTS
from task3_gaussian_fit import fit_gaussian, gaussian
from task4_von_mises import PERIOD_DEG, fit_von_mises, von_mises

NOISY_NEURONS = ["n15", "n62"]
POLY_DEGREE = 4


def fit_models(df, tuning, neuron):
    theta, r, ori, mean_r = neuron_data(df, tuning, neuron)
    poly = Polynomial.fit(theta, r, POLY_DEGREE)
    g, _ = fit_gaussian(theta, r, ori, mean_r)
    vm, _ = fit_von_mises(theta, r, ori, mean_r)
    models = {
        "4th-order": poly,
        "Gaussian": lambda t: gaussian(t, *g),
        "von Mises": lambda t: von_mises(t, *vm),
    }
    mse = {name: np.mean((r - f(theta)) ** 2) for name, f in models.items()}
    print(f"{neuron}  Gaussian: B={g[0]:.2f} A={g[1]:.2f} mu={g[2]:.1f}° sigma={abs(g[3]):.1f}°  |  "
          f"von Mises: B={vm[0]:.2f} A={vm[1]:.3f} mu={vm[2]:.1f}° kappa={vm[3]:.2f}")
    return models, mse


def plot_neuron(df, tuning, sem, neuron, models, mse):
    _, _, ori, mean_r = neuron_data(df, tuning, neuron)
    theta_grid = np.linspace(0, PERIOD_DEG, GRID_POINTS)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(ori, mean_r, yerr=sem[neuron], fmt="o", color="black", capsize=3, zorder=5, label="mean ± SEM")
    ax.plot(PERIOD_DEG, mean_r[0], "o", mfc="none", color="black", zorder=5, label="0° repeated at 180°")
    for (name, f), color, ls in zip(models.items(), ["tab:orange", "tab:purple", "tab:green"], ["--", "-", "-"]):
        ax.plot(theta_grid, f(theta_grid), lw=2, ls=ls, color=color, label=f"{name}  (MSE {mse[name]:.2f})")
    style_axis(ax, f"{neuron}: 4th-order vs Gaussian vs von Mises", ori, mean_r)
    ax.set_xticks([*ori, PERIOD_DEG], [f"{o:g}°" for o in [*ori, PERIOD_DEG]])
    plt.tight_layout()
    save(fig, f"task5_{neuron}_models")
    plt.show()


def main():
    df = load_df(rate=True)
    tuning, sem = tuning_stats(df)
    trial_var = df.groupby("orientation")[tuning.columns].var(ddof=0).mean()

    rows = {}
    for neuron in [NEURON, *NOISY_NEURONS]:
        models, mse = fit_models(df, tuning, neuron)
        rows[neuron] = {**mse, "trial variance (floor)": trial_var[neuron]}
        if neuron in NOISY_NEURONS:
            plot_neuron(df, tuning, sem, neuron, models, mse)

    print(pd.DataFrame(rows).T.round(2))


if __name__ == "__main__":
    main()
