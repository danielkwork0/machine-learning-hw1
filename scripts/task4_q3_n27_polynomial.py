import numpy as np
from numpy.polynomial import Polynomial

from common import BOUNDARY_NEURON, load_df, neuron_data, plot_pair, tuning_stats
from task2_polynomial_fits import GRID_POINTS

POLY_DEGREE = 6


def main():
    df = load_df(rate=True)
    tuning, sem = tuning_stats(df)
    theta, r, ori, mean_r = neuron_data(df, tuning, BOUNDARY_NEURON)
    theta_grid = np.linspace(ori.min(), ori.max(), GRID_POINTS)

    poly = Polynomial.fit(theta, r, POLY_DEGREE)
    mse = np.mean((r - poly(theta)) ** 2)
    print(f"{BOUNDARY_NEURON} {POLY_DEGREE}th-order: MSE (trials) = {mse:.2f}")
    plot_pair(BOUNDARY_NEURON, ori, mean_r, sem[BOUNDARY_NEURON], (theta_grid, poly(theta_grid)),
              f"{POLY_DEGREE}th-order  (MSE {mse:.1f})", "tab:orange",
              f"{POLY_DEGREE}th-order polynomial fit", f"task4_q3_n27_poly{POLY_DEGREE}")


if __name__ == "__main__":
    main()
