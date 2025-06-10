import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv


def smooth_ema(values: np.ndarray, smoothing: float) -> np.ndarray:
    """
    Exponential moving average smoothing, like TensorBoard's slider.

    Parameters
    ----------
    values : 1-D array of scalar values.
    smoothing : float in [0, 1). 0 = no smoothing, 0.9 = heavy smoothing.

    Returns
    -------
    np.ndarray of smoothed values, same length as `values`.
    """
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in the half-open interval [0, 1)")

    smoothed = np.empty_like(values, dtype=float)
    last = values[0]
    for i, v in enumerate(values):
        last = last * smoothing + v * (1.0 - smoothing)
        smoothed[i] = last
    return smoothed


def plot_runs(csv_paths, draw_raw, smoothing: float = 0.6, figsize=(8, 4)):
    """
    Plot one or many runs exported by TensorBoard as CSV.

    Parameters
    ----------
    csv_paths : str | list[str]
        A single path, list of paths, or a glob pattern (e.g. "logs/*.csv").
    smoothing : float, passed to `smooth_ema`.
    figsize : tuple, matplotlib figure size.
    """
    # Accept glob patterns or explicit lists.
    if isinstance(csv_paths, (str, Path)):
        paths = [Path(p) for p in glob.glob(str(csv_paths))]
    else:
        paths = [Path(p) for p in csv_paths]

    if not paths:
        raise FileNotFoundError("No CSV files matched the given path(s).")

    plt.figure(figsize=figsize)

    for path in paths:
        df = pd.read_csv(path)
        steps = df["Step"].to_numpy()
        values = df["Value"].to_numpy()
        smoothed = smooth_ema(values, smoothing)

        label_base = path.stem.replace("_", " ")
        if draw_raw:
            plt.plot(steps, values, alpha=0.3, linewidth=1, label=f"{label_base} (raw)")
        plt.plot(steps, smoothed, linewidth=2, label=f"{label_base} (smoothed)")

    plt.xlabel("Step")
    plt.ylabel("Value")
    if draw_raw:
        plt.title(f"Training curves ")
    else:    
        plt.title(f"Training curves (smoothing={smoothing})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def mean_value_from_csv(path: str) -> float | None:
    """
    Reads the CSV file at `path`, expects a header with a 'Value' column,
    and returns the mean of all entries in that column. Returns None if empty.
    """
    total = 0.0
    count = 0

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        num_rows = len(rows)
        for i, row in enumerate(rows):
            if i > num_rows/2:
                try:
                    total += float(row['Value'])
                    count += 1
                except (KeyError, ValueError):
                    # Skip rows without a valid 'Value'
                    continue

    return (total / count) if count > 0 else None


# ---------- example usage ----------
if __name__ == "__main__":
    # 1) Single file
    #plot_runs(["./plotting/ppo/run_10/PPO_V2_win_rate.csv"], draw_raw = True, smoothing=0.90)
    print(plot_runs(["./plotting/ppo/run_13/PPO_V3_win_rate.csv", "./plotting/ppo/run_13/PPO_V3_draw_rate.csv"], draw_raw = True))

    # 2) All CSVs in a directory
    # plot_runs("logs/*.csv", smoothing=0.8)