import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from common import BOUNDARY_NEURON, NEURON, load_df, neuron_data, save, style_axis, tuning_stats
from task2_polynomial_fits import GRID_POINTS
from task3_gaussian_fit import fit_gaussian, gaussian

NEURONS = [BOUNDARY_NEURON, NEURON]
PERIOD_DEG = 180
KAPPA_GUESS = 2


def von_mises(theta, B, A, mu, kappa):
    return B + A * np.exp(kappa * np.cos(2 * np.deg2rad(theta - mu)))


def fit_von_mises(theta, r, ori, mean_r):
    p0 = [mean_r.min(), np.ptp(mean_r) / (np.exp(KAPPA_GUESS) - np.exp(-KAPPA_GUESS)), ori[mean_r.argmax()], KAPPA_GUESS]
    params, cov = curve_fit(von_mises, theta, r, p0=p0)
    params[2] %= PERIOD_DEG
    return params, np.sqrt(np.diag(cov))


def plot_von_mises(df, tuning, sem, neuron):
    theta, r, ori, mean_r = neuron_data(df, tuning, neuron)
    theta_grid = np.linspace(0, PERIOD_DEG, GRID_POINTS)

    vm, vm_se = fit_von_mises(theta, r, ori, mean_r)
    B, A, mu, kappa = vm
    vm_mse = np.mean((r - von_mises(theta, *vm)) ** 2)
    vm_mse_means = np.mean((mean_r - von_mises(ori, *vm)) ** 2)
    g, _ = fit_gaussian(theta, r, ori, mean_r)
    g_mse = np.mean((r - gaussian(theta, *g)) ** 2)
    print(f"{neuron} von Mises: B={B:.2f}±{vm_se[0]:.2f}  A={A:.3f}±{vm_se[1]:.3f}  "
          f"mu={mu:.1f}°±{vm_se[2]:.1f}  kappa={kappa:.2f}±{vm_se[3]:.2f}  "
          f"MSE (trials)={vm_mse:.2f}  MSE (means)={vm_mse_means:.2f}  | Gaussian MSE (trials)={g_mse:.2f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(ori, mean_r, yerr=sem[neuron], fmt="o", color="black", capsize=3, zorder=5, label="mean ± SEM")
    ax.plot(PERIOD_DEG, mean_r[0], "o", mfc="none", color="black", zorder=5, label="0° repeated at 180°")
    ax.plot(theta_grid, von_mises(theta_grid, *vm), lw=2.5, color="tab:green",
            label=f"von Mises  (μ = {mu:.1f}°, κ = {kappa:.2f}, MSE {vm_mse:.1f})")
    ax.plot(theta_grid, gaussian(theta_grid, *g), lw=1.5, ls="--", color="tab:purple",
            label=f"Gaussian  (MSE {g_mse:.1f})")
    ax.axvline(mu, color="tab:green", lw=0.8, ls=":")
    style_axis(ax, f"{neuron}: von Mises fit", ori, mean_r)
    ax.set_ylim(top=max(ax.get_ylim()[1], 1.05 * (B + A * np.exp(kappa))))
    ax.set_xticks([*ori, PERIOD_DEG], [f"{o:g}°" for o in [*ori, PERIOD_DEG]])
    plt.tight_layout()
    save(fig, f"task4_von_mises_{neuron}")
    plt.show()


def main():
    df = load_df(rate=True)
    tuning, sem = tuning_stats(df)
    for neuron in NEURONS:
        plot_von_mises(df, tuning, sem, neuron)


if __name__ == "__main__":
    main()
