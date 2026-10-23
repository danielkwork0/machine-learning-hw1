import matplotlib.pyplot as plt

from common import SESSION, load_df, save


def main():
    df = load_df()
    trials_per_ori = df["orientation"].value_counts().sort_index()
    print(trials_per_ori)
    print("total trials:", trials_per_ori.sum())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(trials_per_ori.index, trials_per_ori.values, width=15)
    ax.set_xticks(trials_per_ori.index)
    ax.set_xlabel("orientation (deg)")
    ax.set_ylabel("number of trials")
    ax.set_title(f"{SESSION}: trials per orientation")
    for x, y in zip(trials_per_ori.index, trials_per_ori.values):
        ax.text(x, y, str(y), ha="center", va="bottom")
    plt.tight_layout()
    save(fig, "task1_q2_trials_per_orientation")
    plt.show()


if __name__ == "__main__":
    main()
