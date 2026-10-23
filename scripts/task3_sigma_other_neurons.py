from common import load_df, tuning_stats
from task3_gaussian_fit import fit_gaussian

NEURONS = ["n10", "n69", "n73", "n9", "n54"]


def main():
    df = load_df()
    tuning, _ = tuning_stats(df)
    theta = df["orientation"].to_numpy()
    ori = tuning.index.to_numpy()

    for n in NEURONS:
        (_, _, mu, sigma), _ = fit_gaussian(theta, df[n].to_numpy(), ori, tuning[n].to_numpy())
        print(f"{n}: mu={mu:6.1f}°  sigma={abs(sigma):5.1f}°")


if __name__ == "__main__":
    main()
