import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from common import NEURON, STIM_MS, load_df, save, tuning_stats
from task2_polynomial_fits import GRID_POINTS, fit_polynomials

SIGMA_GUESS_DEG = 30
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def gaussian(theta, B, A, mu, sigma):
    return B + A * np.exp(-(theta - mu) ** 2 / (2 * sigma ** 2))


def fit_gaussian(theta, r, ori, mean_r):
    p0 = [mean_r.min(), np.ptp(mean_r), ori[mean_r.argmax()], SIGMA_GUESS_DEG]
    params, cov = curve_fit(gaussian, theta, r, p0=p0)
    return params, np.sqrt(np.diag(cov))


def main():
    df = load_df()
    tuning, sem = tuning_stats(df)
    theta = df["orientation"].to_numpy()
    r = df[NEURON].to_numpy()
    ori = tuning.index.to_numpy()
    mean_r = tuning[NEURON].to_numpy()
    theta_grid = np.linspace(ori.min(), ori.max(), GRID_POINTS)

    fits, mse_table = fit_polynomials(theta, r, ori, mean_r)
    gauss_params, gauss_se = fit_gaussian(theta, r, ori, mean_r)
    B, A, mu, sigma = gauss_params

    gauss_mse_trials = np.mean((r - gaussian(theta, *gauss_params)) ** 2)
    gauss_mse_means = np.mean((mean_r - gaussian(ori, *gauss_params)) ** 2)
    print(pd.DataFrame({"estimate": gauss_params, "SE": gauss_se}, index=["B", "A", "mu", "sigma"]).round(3))
    print(f"FWHM = 2.355·σ = {FWHM_PER_SIGMA * abs(sigma):.1f}°")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(ori, mean_r, yerr=sem[NEURON], fmt="o", color="black", capsize=3, zorder=5, label="mean ± SEM")
    ax.plot(theta_grid, gaussian(theta_grid, *gauss_params), lw=2.5, color="tab:purple",
            label=f"Gaussian  (MSE {gauss_mse_trials:.1f})")
    ax.plot(theta_grid, fits["4th-order"](theta_grid), lw=1.5, ls="--", color="tab:green",
            label=f"4th-order  (MSE {mse_table.loc['4th-order', 'MSE (trials)']:.1f})")
    ax.plot(theta_grid, fits["7th-order"](theta_grid), lw=1.5, ls="--", color="tab:red",
            label=f"7th-order  (MSE {mse_table.loc['7th-order', 'MSE (trials)']:.1f})")
    ax.axvline(mu, color="tab:purple", lw=0.8, ls=":")
    ax.axhline(B, color="tab:purple", lw=0.8, ls=":")
    ax.annotate(f"μ = {mu:.1f}°", (mu, B + A), xytext=(8, 4), textcoords="offset points", color="tab:purple")
    ax.annotate(f"B = {B:.1f}", (ori.min(), B), xytext=(4, 4), textcoords="offset points", color="tab:purple")
    ax.set_xticks(ori, [f"{o:g}°" for o in ori])
    ax.set_xlabel("orientation (deg)")
    ax.set_ylabel(f"mean spikes / {STIM_MS} ms")
    ax.set_title(f"{NEURON}: Gaussian fit vs polynomials")
    ax.legend()
    plt.tight_layout()
    save(fig, "task3_gaussian_fit")
    plt.show()

    compare = pd.concat([
        mse_table[["n_params", "MSE (trials)", "MSE (means)"]],
        pd.DataFrame({"n_params": [4], "MSE (trials)": [gauss_mse_trials], "MSE (means)": [gauss_mse_means]}, index=["Gaussian"]),
    ])
    print(compare.round(2))


if __name__ == "__main__":
    main()
