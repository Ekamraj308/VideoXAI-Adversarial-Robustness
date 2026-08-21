"""
Inspect results saved by pipeline/run_pipeline.py.
Run in Colab to preview GIFs, captions, and metrics for each processed video.
"""

import json
from pathlib import Path
from IPython.display import Image, display

OUT_ROOT = Path('/content/videoXAI_outputs')


def list_output_folders():
    if not OUT_ROOT.exists():
        print("No outputs found at", OUT_ROOT)
        return []

    items = sorted(list(OUT_ROOT.iterdir()))
    video_dirs = [p for p in items if p.is_dir()]

    print(f"Found {len(video_dirs)} video folders")
    for d in video_dirs[:20]:
        print(" >", d.name)

    return video_dirs


def preview_video_summary(video_folder, show_gif=True):
    """Print captions/metrics for one video folder, optionally showing orig.gif."""
    print("\n==============================")
    print("Video:", video_folder.name)
    print("==============================")

    if show_gif:
        gif_path = video_folder / "orig.gif"
        if gif_path.exists():
            display(Image(filename=str(gif_path)))

    summary_path = video_folder / "summary.json"
    if not summary_path.exists():
        print("No summary.json found")
        return

    with open(summary_path, "r") as f:
        summary = json.load(f)

    print("\n--- Captions ---")
    print("Original:", summary.get("orig_caption"))
    print("FGSM    :", summary.get("fgsm_caption"))
    print("PGD     :", summary.get("pgd_caption"))
    print("I2V     :", summary.get("i2v_caption"))

    print("\n--- Metrics ---")
    print("PSNR (FGSM, PGD, I2V):",
          summary.get("psnr_fgsm"), summary.get("psnr_pgd"), summary.get("psnr_i2v"))
    print("L-inf (FGSM, PGD, I2V):",
          summary.get("linf_fgsm"), summary.get("linf_pgd"), summary.get("linf_i2v"))
    print("SBERT similarity (FGSM, PGD, I2V):",
          summary["metrics_fgsm"]["SBERT_cos"],
          summary["metrics_pgd"]["SBERT_cos"],
          summary["metrics_i2v"]["SBERT_cos"])


if __name__ == "__main__":
    folders = list_output_folders()
    for vf in folders:
        preview_video_summary(vf)
