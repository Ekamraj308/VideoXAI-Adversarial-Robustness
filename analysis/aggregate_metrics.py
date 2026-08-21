"""
Aggregate per-video metrics across the whole dataset and compare the four
attacks (VBAD, FGSM, PGD, I2V) on:
  - semantic damage (1 - BERTScore, 1 - SBERT, and per-frame damage/eq.10)
  - visual distortion (L-infinity, PSNR)
  - temporal instability (eq. 14)

Also produces the medical-vs-general domain comparison (RQ2), since
results are now saved per dataset under
/content/videoXAI_outputs/<dataset_name>/<video>/summary.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_ROOT = Path('/content/videoXAI_outputs')
ATTACKS = ['vbad', 'fgsm', 'pgd', 'i2v']


def load_all_summaries():
    """
    Walk OUT_ROOT/<dataset_name>/<video>/summary.json and build one row
    per (video, attack), tagged with dataset name and domain.
    """
    rows = []

    dataset_dirs = [p for p in OUT_ROOT.iterdir() if p.is_dir()]

    for dataset_dir in dataset_dirs:
        video_folders = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])

        for video_folder in video_folders:
            summary_path = video_folder / "summary.json"
            if not summary_path.exists():
                continue

            with open(summary_path, 'r') as f:
                summary = json.load(f)

            dataset_name = summary.get("dataset", dataset_dir.name)
            domain = summary.get("domain", "unknown")

            for a in ATTACKS:
                try:
                    row = {
                        "dataset": dataset_name,
                        "domain": domain,
                        "video": video_folder.name,
                        "attack": a,
                        "psnr": float(summary[f'psnr_{a}']),
                        "linf": float(summary[f'linf_{a}']),
                        "sbert": float(summary[f'metrics_{a}']['SBERT_cos']),
                        "bertscore": float(summary[f'metrics_{a}']['BERTScore_F1']),
                    }
                    # Newer runs include per-frame damage + temporal instability;
                    # older runs may not, so handle both gracefully.
                    if f'temporal_instability_{a}' in summary:
                        row["temporal_instability"] = float(summary[f'temporal_instability_{a}'])
                    if f'per_frame_damage_{a}' in summary:
                        row["mean_frame_damage"] = float(np.mean(summary[f'per_frame_damage_{a}']))

                    rows.append(row)
                except Exception:
                    continue

    return pd.DataFrame(rows)


def summarize(df):
    numeric_cols = [c for c in ["psnr", "linf", "sbert", "bertscore",
                                 "temporal_instability", "mean_frame_damage"] if c in df.columns]
    agg = df.groupby("attack")[numeric_cols].mean()
    std = df.groupby("attack")[numeric_cols].std()
    print("\nMean metrics (all datasets pooled):\n", agg)
    return agg, std


def summarize_by_domain(df):
    """
    RQ2: compare medical vs. general domain robustness. Requires the
    'domain' column, which is only present when videos were processed
    through the dataset-aware pipeline (pipeline/run_pipeline.py).
    """
    if "domain" not in df.columns or df["domain"].nunique() < 2:
        print("Not enough domain variety to compare medical vs. general "
              "(need videos from at least two domains processed).")
        return None

    numeric_cols = [c for c in ["psnr", "linf", "sbert", "bertscore",
                                 "temporal_instability", "mean_frame_damage"] if c in df.columns]
    by_domain = df.groupby(["domain", "attack"])[numeric_cols].mean()
    print("\nMean metrics by domain (medical vs. general) x attack:\n", by_domain)
    return by_domain


def plot_visual_and_semantic(agg, std):
    x = np.arange(len(ATTACKS))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, agg.loc[ATTACKS, 'psnr'], width, yerr=std.loc[ATTACKS, 'psnr'], capsize=5, label='PSNR')
    plt.bar(x + width / 2, agg.loc[ATTACKS, 'linf'], width, yerr=std.loc[ATTACKS, 'linf'], capsize=5, label='L-inf')
    plt.xticks(x, [a.upper() for a in ATTACKS])
    plt.title("Visual Impact of Attacks (Mean +/- Std)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, agg.loc[ATTACKS, 'sbert'], width, yerr=std.loc[ATTACKS, 'sbert'], capsize=5, label='SBERT')
    plt.bar(x + width / 2, agg.loc[ATTACKS, 'bertscore'], width, yerr=std.loc[ATTACKS, 'bertscore'], capsize=5, label='BERTScore')
    plt.xticks(x, [a.upper() for a in ATTACKS])
    plt.title("Semantic Impact of Attacks (Mean +/- Std)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_domain_comparison(by_domain):
    """Bar chart comparing mean semantic damage per domain per attack (RQ2)."""
    if by_domain is None or "mean_frame_damage" not in by_domain.columns:
        return

    pivot = by_domain["mean_frame_damage"].unstack("domain")
    pivot.plot(kind="bar", figsize=(8, 5))
    plt.title("Semantic Damage: Medical vs. General Domain (RQ2)")
    plt.ylabel("Mean per-frame semantic damage")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def effectiveness_ranking(agg):
    ranking = agg.copy()
    ranking['effectiveness_score'] = (1 - ranking['sbert']) - ranking['linf'] * 0.5
    ranking = ranking.sort_values(by='effectiveness_score', ascending=False)
    print("\nAttack effectiveness ranking:\n", ranking[['sbert', 'linf', 'psnr', 'effectiveness_score']])
    return ranking


if __name__ == "__main__":
    df = load_all_summaries()
    print(f"Total samples: {len(df)} (videos x attacks, across all datasets)")
    agg, std = summarize(df)
    plot_visual_and_semantic(agg, std)
    effectiveness_ranking(agg)

    by_domain = summarize_by_domain(df)
    plot_domain_comparison(by_domain)
