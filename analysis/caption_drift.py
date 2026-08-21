"""
Per-frame caption drift analysis.

For each frame of each attacked GIF, generates a fresh BLIP caption and
compares it (via SBERT similarity) to the original video's caption. This
shows how caption quality degrades frame-by-frame under each attack,
rather than just looking at one aggregate number per video.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import torch
from PIL import Image

from models.model_setup import processor_blip, model_blip, sbert_model, device
from sentence_transformers import util

ATTACKS = ["vbad", "fgsm", "pgd", "i2v"]
PALETTE = {"VBAD": "#1f77b4", "FGSM": "#ff7f0e", "PGD": "#2ca02c", "I2V": "#d62728"}


def compute_frame_drift(out_root="/content/videoXAI_outputs"):
    root = Path(out_root)
    video_folders = sorted([p for p in root.iterdir() if p.is_dir()])
    if not video_folders:
        raise FileNotFoundError("No video folders found.")

    all_rows = []

    for folder in video_folders:
        summary_path = folder / "summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        orig_caption = summary["orig_caption"]
        orig_emb = sbert_model.encode(orig_caption, convert_to_tensor=True)

        for attack in ATTACKS:
            gif_path = folder / f"{attack}.gif"
            if not gif_path.exists():
                continue

            frames = imageio.mimread(str(gif_path))

            for idx, frame in enumerate(frames):
                pil = Image.fromarray(frame)
                inputs = processor_blip(images=pil, return_tensors="pt").to(device)

                with torch.no_grad():
                    out = model_blip.generate(**inputs, max_new_tokens=25)

                frame_caption = processor_blip.decode(out[0], skip_special_tokens=True)
                frame_emb = sbert_model.encode(frame_caption, convert_to_tensor=True)
                sim = util.cos_sim(orig_emb, frame_emb).item()

                all_rows.append([folder.name, attack.upper(), idx, float(sim)])

    return pd.DataFrame(all_rows, columns=["Video", "Attack", "Frame", "Score"])


def plot_drift(drift_df, max_videos=6):
    sns.set(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(14, 6))
    sns.violinplot(data=drift_df, x="Attack", y="Score", inner="quartile", cut=0, palette=PALETTE)
    plt.title("Per-Frame Caption Drift - All Videos")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=drift_df, x="Attack", y="Score", palette=PALETTE)
    plt.title("Per-Frame Caption Drift - Distribution")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    subset_videos = drift_df["Video"].unique()[:max_videos]
    df_subset = drift_df[drift_df["Video"].isin(subset_videos)]

    sns.catplot(data=df_subset, x="Attack", y="Score", col="Video", kind="violin",
                inner="quartile", height=4, aspect=1, palette=PALETTE)
    plt.show()

    sns.relplot(data=df_subset, x="Frame", y="Score", col="Video", hue="Attack",
                kind="line", height=4, aspect=1.2, palette=PALETTE)
    plt.show()

    print("\nMean drift per attack:\n", drift_df.groupby("Attack")["Score"].mean())
    print("\nStd drift per attack:\n", drift_df.groupby("Attack")["Score"].std())


if __name__ == "__main__":
    df_frames = compute_frame_drift()
    plot_drift(df_frames)
