"""
Pareto trade-off analysis.

Sweeps the epsilon (attack strength) parameter for FGSM and VBAD across
a range of values, and for each setting measures:
  - visual distortion (mean L-infinity, PSNR)
  - semantic damage (1 - SBERT similarity, BERTScore)
  - compute cost (time, energy in kWh via codecarbon, estimated USD cost, CO2)

This shows the trade-off curve between "how strong is the attack" and
"how much distortion/cost does it take to get there."

Requires: pip install codecarbon
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

from data.video_loader import load_video_frames
from models.model_setup import caption_from_frames
from attacks.attacks import fgsm_attack_video, vbad_like_patch_attack
from metrics.metrics import compute_psnr, compute_linf, text_metrics

VIDEO_ROOT = Path('/content/drive/MyDrive/medcraft_videos')
OUT_ROOT = Path('/content/pareto_outputs')
OUT_ROOT.mkdir(exist_ok=True)


def run_sweep(attacks=("fgsm", "vbad"), eps_list=(0.01, 0.05, 0.1, 0.15, 0.2), n_videos=20):
    videos = sorted([str(p) for p in VIDEO_ROOT.glob("*.mp4")])[:n_videos]
    results = []

    for attack in attacks:
        for eps in eps_list:
            print(f"Running {attack.upper()} | epsilon={eps}")

            tracker = EmissionsTracker(log_level="error")
            tracker.start()
            start_time = time.time()

            sbert_scores, bert_scores, psnr_vals, linf_vals = [], [], [], []
            processed = 0

            for v in videos:
                try:
                    frames = load_video_frames(v)
                    orig_cap = caption_from_frames(frames)

                    if attack == "fgsm":
                        adv = fgsm_attack_video(frames, orig_cap, eps=eps)
                    elif attack == "vbad":
                        adv = vbad_like_patch_attack(frames, patch_ratio=eps)
                    else:
                        continue

                    adv_cap = caption_from_frames(adv)
                    txt = text_metrics(orig_cap, adv_cap)

                    sbert_scores.append(txt["SBERT_cos"])
                    bert_scores.append(txt["BERTScore_F1"])
                    psnr_vals.append(np.mean([compute_psnr(frames[i], adv[i]) for i in range(len(frames))]))
                    linf_vals.append(np.mean([compute_linf(frames[i], adv[i]) for i in range(len(frames))]))
                    processed += 1

                except Exception:
                    continue

            emissions = tracker.stop()
            total_time = time.time() - start_time
            energy_kwh = tracker._total_energy.kWh if hasattr(tracker, "_total_energy") else None
            cost_usd = (total_time / 3600) * 0.35  # rough T4 instance estimate

            results.append({
                "attack": attack, "epsilon": eps, "n": processed,
                "mean_sbert": np.mean(sbert_scores),
                "mean_bertscore": np.mean(bert_scores),
                "mean_psnr": np.mean(psnr_vals),
                "mean_linf": np.mean(linf_vals),
                "total_time_s": total_time,
                "energy_kwh": energy_kwh,
                "co2_kg": emissions,
                "cost_usd": cost_usd
            })

    df = pd.DataFrame(results)
    df.to_csv(OUT_ROOT / "pareto_results.csv", index=False)
    return df


def plot_pareto(df_pareto):
    df_pareto = df_pareto.copy()
    df_pareto["semantic_damage"] = 1.0 - df_pareto["mean_sbert"]

    plt.figure(figsize=(8, 6))
    for attack in df_pareto["attack"].unique():
        sub = df_pareto[df_pareto["attack"] == attack].sort_values("epsilon")
        plt.scatter(sub["mean_linf"], sub["semantic_damage"], s=90, label=attack.upper())
        plt.plot(sub["mean_linf"], sub["semantic_damage"], linewidth=2)
        for _, row in sub.iterrows():
            plt.annotate(f"eps={row['epsilon']}", (row["mean_linf"], row["semantic_damage"]),
                         textcoords="offset points", xytext=(5, 5), fontsize=9)

    plt.xlabel("Visual Distortion (mean L-infinity)")
    plt.ylabel("Semantic Damage (1 - mean SBERT)")
    plt.title("Pareto Trade-off: Attacks across Epsilon Settings")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = run_sweep()
    plot_pareto(df)
