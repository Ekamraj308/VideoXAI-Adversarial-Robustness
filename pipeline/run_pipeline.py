"""
Main pipeline: for each configured dataset (Med-M3, MedVideoCap-55K,
Kinetics-400), for each video, generate per-frame captions, run all four
attacks (VBAD, FGSM, PGD, I2V), re-caption the attacked frames, compute
distortion + per-frame semantic-drift + temporal-instability metrics, and
save everything under /content/videoXAI_outputs/<dataset_name>/.

This matches the paper's methodology in two ways that the original
single-folder version did not:
  1. Results are grouped per dataset (medical vs. general domain), so
     RQ2 (medical vs. general robustness) can actually be answered from
     the saved outputs.
  2. Captions are generated per-frame (eq. 3 / eq. 8), and semantic
     damage + temporal instability (eq. 9, 10, 14) are computed directly
     from those per-frame captions, rather than from a single
     middle-frame caption per video.
"""

import json
from pathlib import Path

import numpy as np
import imageio
from tqdm import tqdm

from data.dataset_registry import all_datasets
from models.model_setup import caption_from_frames, caption_each_frame
from attacks.attacks import (
    vbad_like_patch_attack, fgsm_attack_video, pgd_attack_video, i2v_attack_video
)
from metrics.metrics import (
    compute_psnr, compute_linf, text_metrics,
    per_frame_semantic_damage, temporal_instability
)

OUT_ROOT = Path('/content/videoXAI_outputs')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Epsilon (attack strength) settings for each attack
eps_vbad = 0.06   # patch area ratio
eps_fgsm = 0.08   # L-infinity bound
eps_pgd = 0.03    # L-infinity bound
eps_i2v = 0.03    # L-infinity bound


def process_video(v, frames_loader):
    """Run the full attack + evaluation pipeline on a single video path."""
    frames = frames_loader(v)

    # Per-frame clean captions (eq. 3): one caption per sampled frame
    orig_captions_per_frame = caption_each_frame(frames)
    # Single summary caption (middle frame) - kept for quick display only
    orig_cap = caption_from_frames(frames)

    # Run all four attacks
    vbad = vbad_like_patch_attack(frames, patch_ratio=eps_vbad)
    fgsm = fgsm_attack_video(frames, orig_cap, eps=eps_fgsm)
    pgd = pgd_attack_video(frames, orig_cap, eps=eps_pgd)
    i2v = i2v_attack_video(frames, orig_cap, eps=eps_i2v)

    attacked = {"vbad": vbad, "fgsm": fgsm, "pgd": pgd, "i2v": i2v}

    # Per-frame attacked captions (eq. 8), one list per attack
    captions_per_frame = {
        name: caption_each_frame(adv_frames) for name, adv_frames in attacked.items()
    }
    # Single summary caption per attack - kept for quick display only
    captions_summary = {
        name: caption_from_frames(adv_frames) for name, adv_frames in attacked.items()
    }

    # Visual distortion metrics (unchanged from before)
    psnr_vals = {
        f"psnr_{name}": np.mean([compute_psnr(frames[i], adv[i]) for i in range(len(frames))])
        for name, adv in attacked.items()
    }
    linf_vals = {
        f"linf_{name}": np.mean([compute_linf(frames[i], adv[i]) for i in range(len(frames))])
        for name, adv in attacked.items()
    }

    # Video-level semantic drift (BLEU/ROUGE/BERTScore/SBERT), using the
    # middle-frame summary captions - kept for backward-compatible reporting
    metrics_summary = {
        name: text_metrics(orig_cap, captions_summary[name]) for name in attacked
    }

    # Per-frame semantic damage (eq. 9/10) and temporal instability (eq. 14),
    # computed directly from per-frame captions - this is the metric the
    # paper's equations actually describe
    per_frame_damage = {}
    temporal_T = {}
    for name in attacked:
        damages = per_frame_semantic_damage(orig_captions_per_frame, captions_per_frame[name])
        per_frame_damage[name] = damages
        temporal_T[name] = temporal_instability(damages)

    return {
        "frames": frames,
        "attacked": attacked,
        "orig_cap": orig_cap,
        "captions_summary": captions_summary,
        "orig_captions_per_frame": orig_captions_per_frame,
        "captions_per_frame": captions_per_frame,
        "psnr_vals": psnr_vals,
        "linf_vals": linf_vals,
        "metrics_summary": metrics_summary,
        "per_frame_damage": per_frame_damage,
        "temporal_T": temporal_T,
    }


def run(limit_per_dataset=50):
    from data.video_loader import load_video_frames

    for dataset_name, domain, videos in all_datasets(limit=limit_per_dataset):
        print(f"\n=== Dataset: {dataset_name} ({domain} domain) | {len(videos)} videos ===")

        dataset_out = OUT_ROOT / dataset_name
        dataset_out.mkdir(parents=True, exist_ok=True)

        for v in tqdm(videos, desc=dataset_name):
            try:
                print("Processing:", v)
                result = process_video(v, load_video_frames)

                out = dataset_out / Path(v).stem
                out.mkdir(parents=True, exist_ok=True)

                imageio.mimsave(out / "orig.gif", result["frames"], fps=4)
                for name, adv in result["attacked"].items():
                    imageio.mimsave(out / f"{name}.gif", adv, fps=4)

                summary = {
                    "dataset": dataset_name,
                    "domain": domain,
                    "epsilon_vbad_area": eps_vbad,
                    "epsilon_fgsm_linf": eps_fgsm,
                    "epsilon_pgd_linf": eps_pgd,
                    "epsilon_i2v_linf": eps_i2v,
                    "orig_caption": result["orig_cap"],
                    "orig_captions_per_frame": result["orig_captions_per_frame"],
                    **{f"{name}_caption": cap for name, cap in result["captions_summary"].items()},
                    **{f"{name}_captions_per_frame": caps for name, caps in result["captions_per_frame"].items()},
                    **result["psnr_vals"],
                    **result["linf_vals"],
                    **{f"metrics_{name}": m for name, m in result["metrics_summary"].items()},
                    **{f"per_frame_damage_{name}": d for name, d in result["per_frame_damage"].items()},
                    **{f"temporal_instability_{name}": t for name, t in result["temporal_T"].items()},
                }

                with open(out / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

            except Exception as e:
                print("Skipping video:", v)
                print("Error:", e)
                continue

    print("\nFull pipeline completed across all configured datasets.")


if __name__ == "__main__":
    run()
