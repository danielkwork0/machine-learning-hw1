import matplotlib.pyplot as plt
import numpy as np

from common import load_df, save


def main():
    df = load_df()
    ori_table = (
        df[["stim", "orientation"]]
        .drop_duplicates()
        .sort_values("stim")
        .reset_index(drop=True)
    )
    print(ori_table.to_string(index=False))
    print("orientations (deg):", ori_table["orientation"].tolist())
    print("spacing (deg):", np.diff(ori_table["orientation"]).tolist())

    fig, axes = plt.subplots(1, len(ori_table), figsize=(12, 2))
    for ax, (_, row) in zip(axes, ori_table.iterrows()):
        th = np.deg2rad(row["orientation"])
        ax.plot([-np.cos(th), np.cos(th)], [-np.sin(th), np.sin(th)], lw=4)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"stim {row['stim']:g}\n{row['orientation']:g}°")
    fig.suptitle("Stimulus orientations")
    plt.tight_layout()
    save(fig, "task1_q3_stimulus_orientations")
    plt.show()


if __name__ == "__main__":
    main()
