import matplotlib.pyplot as plt
import numpy as np

from common import BOUNDARY_NEURON, NEURON, RATE_LABEL, load_df, neuron_data, save, tuning_stats
from task2_polynomial_fits import GRID_POINTS
from task3_gaussian_fit import fit_gaussian, gaussian


def plot_mean_tuning(tuning, sem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, n in zip(axes, [NEURON, BOUNDARY_NEURON]):
        ax.errorbar(tuning.index, tuning[n], yerr=sem[n], marker="o", capsize=3)
        ax.set_title(n)
        ax.set_xticks(tuning.index, [f"{o:g}°" for o in tuning.index])
        ax.set_xlabel("orientation (deg)")
        ax.set_ylabel(RATE_LABEL)
    fig.suptitle("Mean orientation tuning (±SEM)")
    plt.tight_layout()
    save(fig, "task4_q2_mean_tuning_n10_n27")
    plt.show()
    print(tuning[[NEURON, BOUNDARY_NEURON]].round(1))


def plot_gaussian(df, tuning, sem):
    theta, r, ori, mean_r = neuron_data(df, tuning, BOUNDARY_NEURON)
    theta_grid = np.linspace(ori.min(), ori.max(), GRID_POINTS)
    params, _ = fit_gaussian(theta, r, ori, mean_r)
    B, A, mu, sigma = params
    mse_trials = np.mean((r - gaussian(theta, *params)) ** 2)
    mse_means = np.mean((mean_r - gaussian(ori, *params)) ** 2)
    print(f"B={B:.2f}  A={A:.2f}  mu={mu:.1f}°  sigma={abs(sigma):.1f}°")
    print(f"MSE (trials) = {mse_trials:.2f}   MSE (means) = {mse_means:.2f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(ori, mean_r, yerr=sem[BOUNDARY_NEURON], fmt="o", color="black", capsize=3, zorder=5, label="mean ± SEM")
    ax.plot(theta_grid, gaussian(theta_grid, *params), lw=2.5, color="tab:purple",
            label=f"Gaussian  (μ = {mu:.1f}°, σ = {abs(sigma):.1f}°, MSE {mse_trials:.1f})")
    ax.axvline(mu, color="tab:purple", lw=0.8, ls=":")
    ax.axhline(B, color="tab:purple", lw=0.8, ls=":")
    ax.annotate(f"B = {B:.1f}", (ori.min(), B), xytext=(4, 4), textcoords="offset points", color="tab:purple")
    ax.set_xticks(ori, [f"{o:g}°" for o in ori])
    ax.set_xlabel("orientation (deg)")
    ax.set_ylabel(RATE_LABEL)
    ax.set_title(f"{BOUNDARY_NEURON}: Gaussian fit to a tuning curve that wraps across 0°/180°")
    ax.legend(loc="upper center")
    plt.tight_layout()
    save(fig, "task4_q2_n27_gaussian")
    plt.show()


def main():
    df = load_df(rate=True)
    tuning, sem = tuning_stats(df)
    plot_mean_tuning(tuning, sem)
    plot_gaussian(df, tuning, sem)


if __name__ == "__main__":
    main()
