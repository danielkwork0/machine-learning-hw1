import matplotlib.pyplot as plt
import numpy as np

from common import NEURON, STIM_MS, load_df, save, tuning_stats

N_EXAMPLES = 8


def main():
    df = load_df()
    tuning, sem = tuning_stats(df)

    oti = (tuning.max() - tuning.min()) / (tuning.max() + tuning.min())
    snr = (tuning.max() - tuning.min()) / sem.mean()
    ranked = snr.sort_values(ascending=False).index
    examples = ranked[np.linspace(0, len(ranked) - 1, N_EXAMPLES).astype(int)]
    examples = [NEURON] + [n for n in examples if n != NEURON]

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for ax, n in zip(axes.flat, examples):
        color = "tab:red" if n == NEURON else "tab:blue"
        ax.errorbar(tuning.index, tuning[n], yerr=sem[n], marker="o", capsize=3, color=color)
        ax.set_title(f"{n}  (OTI {oti[n]:.2f})", color=color)
        ax.set_xticks(tuning.index, [f"{o:g}°" for o in tuning.index], rotation=45)
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"mean spikes / {STIM_MS} ms")
    for ax in axes[-1]:
        ax.set_xlabel("orientation (deg)")
    fig.suptitle("Mean orientation tuning curves (±SEM, 400 trials per orientation), strongly to weakly tuned")
    plt.tight_layout()
    save(fig, "task1_q4_tuning_curves")
    plt.show()

    print(tuning[examples].round(1))


if __name__ == "__main__":
    main()
