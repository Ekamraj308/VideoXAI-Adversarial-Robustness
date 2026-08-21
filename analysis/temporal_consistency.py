"""
Temporal consistency analysis.

Takes the per-frame drift scores from caption_drift.py and measures how
smoothly (or erratically) each attack's effect changes from frame to
frame within a video - i.e. does the attack degrade the caption steadily,
or does it flicker unpredictably.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def temporal_consistency_analysis(df_frames):
    results = []

    for (vid, atk), group in df_frames.groupby(["Video", "Attack"]):
        group = group.sort_values("Frame")
        scores = group["Score"].values
        diffs = np.diff(scores)

        results.append([
            vid, atk,
            float(np.mean(np.abs(diffs))),   # mean_drift
            float(np.var(scores)),           # temporal_variance
            float(np.std(diffs))             # temporal_smoothness
        ])

    temp_df = pd.DataFrame(results, columns=[
        "Video", "Attack", "Mean_Drift", "Temporal_Variance", "Temporal_Smoothness"
    ])

    sns.set(style="whitegrid", font_scale=1.1)
    numeric_cols = ["Mean_Drift", "Temporal_Variance", "Temporal_Smoothness"]
    agg_df = temp_df.groupby("Attack")[numeric_cols].mean()

    plt.figure(figsize=(6, 4))
    sns.heatmap(agg_df, cmap="magma", annot=True, fmt=".3f")
    plt.title("Global Temporal Consistency (All Videos)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    order = [a for a in ["VBAD", "FGSM", "PGD", "I2V"] if a in agg_df.index]
    agg_df.loc[order, "Temporal_Variance"].plot(kind="bar",
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(order)])
    plt.title("Average Temporal Variance per Attack")
    plt.ylabel("Variance")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    best_stability = agg_df["Temporal_Variance"].idxmin()
    worst_stability = agg_df["Temporal_Variance"].idxmax()
    print(f"Most temporally stable attack: {best_stability}")
    print(f"Most temporally unstable attack: {worst_stability}")

    return temp_df
