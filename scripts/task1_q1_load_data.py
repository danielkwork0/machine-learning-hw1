from common import load_df, neuron_cols


def main():
    df = load_df()
    print(f"df (trials, columns): {df.shape}")
    print(f"V1 neurons recorded: {len(neuron_cols(df))}")
    print(df)


if __name__ == "__main__":
    main()
