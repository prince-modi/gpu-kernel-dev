"""Plots benchmark results."""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot(input_dir="results"):
    """Generates plots for each csv file in input directory."""
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    markers = ["o", "s", "^", "D", "v", "p", "*", "x", "+"]

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        # Assumes first column is the x-axis
        x_axis = str(df.columns[0])

        for i, col in enumerate(col for col in df.columns[1:]):
            plt.plot(
                df[x_axis],
                df[col],
                label=col,
                marker=markers[i % len(markers)],
                markersize=2,
            )

        plt.xlabel(x_axis)
        plt.ylabel("Bandwidth (GB/s)")
        # plt.title(f"RMS Norm Performance: Bandwidth vs {x_axis}")

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        plt.tight_layout()
        file_name = os.path.basename(file_path)
        output_path = f"results/plot_{os.path.splitext(file_name)[0]}.png"
        plt.savefig(output_path)
        plt.clf()


if __name__ == "__main__":
    plot()
