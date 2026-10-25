import numpy as np

from common import NEURON, load_df, neuron_data, plot_pair, tuning_stats
from task2_polynomial_fits import GRID_POINTS
from task3_gaussian_fit import fit_gaussian, gaussian


def main():
    df = load_df(rate=True)
    tuning, sem = tuning_stats(df)
    theta, r, ori, mean_r = neuron_data(df, tuning, NEURON)
    theta_grid = np.linspace(ori.min(), ori.max(), GRID_POINTS)

    params, _ = fit_gaussian(theta, r, ori, mean_r)
    B, A, mu, sigma = params
    mse = np.mean((r - gaussian(theta, *params)) ** 2)
    print(f"{NEURON} Gaussian: B={B:.2f}  A={A:.2f}  mu={mu:.1f}°  sigma={abs(sigma):.1f}°  MSE (trials) = {mse:.2f}")
    plot_pair(NEURON, ori, mean_r, sem[NEURON], (theta_grid, gaussian(theta_grid, *params)),
              f"Gaussian  (B = {B:.1f}, A = {A:.1f}, μ = {mu:.1f}°, σ = {abs(sigma):.1f}°, MSE {mse:.1f})",
              "tab:purple", "Gaussian fit", "task4_q4_n10_gaussian")


if __name__ == "__main__":
    main()
