# VideoXAI: Adversarial Robustness Evaluation for Video Classification/Captioning

This repo evaluates how four different adversarial attacks (VBAD, FGSM, PGD, I2V)
degrade video understanding models, measured both visually (how distorted the
frames become) and semantically (how much the generated caption changes).

## Note on "CLIPS" vs "CLIP"

This project does not use CLIPS (the rule-based expert system language). It uses
**CLIP** (OpenAI's `clip-vit-base-patch32`), the image-text embedding model.

CLIP is used as the **optimization target for the attacks**. Concretely:

1. A video's frames are captioned once using BLIP, giving the "ground truth" caption.
2. CLIP embeds that caption (text) and each frame (image) into the same vector space.
3. FGSM, PGD, and I2V all perturb the frame's pixels, within a bounded epsilon,
   in the direction that **reduces the cosine similarity between the CLIP image
   embedding and the CLIP text embedding**.

In other words, CLIP's similarity score is the loss function being minimized —
that's what "CLIP for optimization" refers to. VBAD is the exception: it's a
simple patch-paint attack and does not use CLIP or gradients at all.

## Datasets

The three datasets referenced in the paper and used by `data/dataset_registry.py`:

| Dataset | Domain | Link |
|---|---|---|
| Med-M3 | Medical | https://Med-M3-Dataset.github.io/ |
| MedVideoCap-55K | Medical | https://huggingface.co/datasets/FreedomIntelligence/MedVideoCap-55K |
| Kinetics-400 | General | https://github.com/cvdfoundation/kinetics-dataset |

Notes on getting each one:
- **Med-M3**: project page above has the dataset and access instructions.
- **MedVideoCap-55K**: hosted on Hugging Face; you'll need a Hugging Face account to download via the `datasets` library or direct download.
- **Kinetics-400**: the CVDF-hosted mirror above is the most reliable download path (the original DeepMind project page has broken links). Their repo includes a `k400_downloader.sh` script that pulls the full train/val/test splits — you'll likely want to sample a subset of clips for this project rather than the full ~240K videos.

Once downloaded, put each dataset's videos in its own folder and update the matching `path` in `data/dataset_registry.py` to point there.

## Repo structure

```
videoxai/
├── data/
│   ├── video_loader.py        # loads & samples frames from a video file
│   └── dataset_registry.py    # maps Med-M3 / MedVideoCap-55K / Kinetics-400
│                               # folders to domain labels (medical/general)
├── models/
│   └── model_setup.py         # loads BLIP, CLIP, SBERT; embedding helpers;
│                               # per-frame captioning (caption_each_frame)
├── attacks/
│   └── attacks.py              # VBAD, FGSM, PGD, I2V attack implementations
├── metrics/
│   └── metrics.py              # PSNR, L-infinity, BLEU/ROUGE/BERTScore/SBERT,
│                               # per-frame semantic damage + temporal instability
├── pipeline/
│   └── run_pipeline.py         # main entry point: runs all attacks on all
│                               # videos, per dataset, with per-frame captions
├── analysis/
│   ├── inspect_outputs.py      # view saved GIFs/captions/metrics per video
│   ├── aggregate_metrics.py    # cross-video comparison + RQ2 domain comparison
│   ├── caption_drift.py        # per-frame caption drift (violin/box/line plots)
│   ├── temporal_consistency.py # frame-to-frame stability of each attack
│   └── pareto_tradeoff.py      # epsilon sweep: distortion vs semantic damage vs cost
└── requirements.txt
```

## What changed to match the paper

Three things in the original code didn't match the paper's claims. They're now fixed:

1. **Dataset separation (RQ2).** `data/dataset_registry.py` now defines
   separate folders and domain labels for Med-M3, MedVideoCap-55K
   (medical), and Kinetics-400 (general). `run_pipeline.py` processes
   each dataset separately and tags every `summary.json` with its
   `dataset` and `domain`. `analysis/aggregate_metrics.py` can now
   produce the medical-vs-general comparison the paper's RQ2 describes.
   **You need to update the `path` values in `DATASET_ROOTS`** to point
   at wherever your actual Med-M3 / MedVideoCap-55K / Kinetics-400
   videos live.

2. **I2V's temporal smoothness loss (eq. 7).** `attacks/attacks.py` now
   actually computes `L_temp = sum ||delta_{i+1} - delta_i||_2` and adds
   it (weighted by `lambda_temp`) to I2V's optimization objective, so
   the perturbation is explicitly penalized for changing abruptly
   between frames. Previously I2V just ran the same CLIP-similarity
   loss as PGD, jointly across frames, with no smoothness term.

3. **Per-frame captioning (eq. 3, 8, 9, 10, 14).** `models/model_setup.py`
   adds `caption_each_frame()`, which captions every sampled frame
   individually (`c_i = G_BLIP(f_i)`), matching the paper's equations.
   `run_pipeline.py` now uses this to compute per-frame semantic damage
   (`d_i = 1 - S_i`) and temporal instability
   (`T = mean(|d_{i+1} - d_i|)`) directly, rather than relying only on
   a single middle-frame caption per video. The old middle-frame caption
   is still saved too, for quick display purposes.

## How to run (Colab)

1. Mount Google Drive and place each dataset's videos in its own folder,
   matching the paths in `data/dataset_registry.py` (update those paths
   first if your folders are named/located differently).
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main pipeline:
   ```python
   from pipeline.run_pipeline import run
   run()
   ```
   This processes each configured dataset separately, runs all four
   attacks per video, and saves GIFs + a `summary.json` per video to
   `/content/videoXAI_outputs/<dataset_name>/<video>/`.
4. Use the analysis scripts to inspect and visualize results, e.g.:
   ```python
   from analysis.aggregate_metrics import load_all_summaries, summarize, plot_visual_and_semantic, summarize_by_domain, plot_domain_comparison
   df = load_all_summaries()
   agg, std = summarize(df)
   plot_visual_and_semantic(agg, std)

   by_domain = summarize_by_domain(df)   # RQ2: medical vs. general
   plot_domain_comparison(by_domain)
   ```

## Attack summary

| Attack | Uses CLIP? | Type | What it optimizes |
|---|---|---|---|
| VBAD | No | Patch paint | N/A (fixed patch, no gradient) |
| FGSM | Yes | One-step gradient | Reduce CLIP image-text similarity |
| PGD | Yes | Multi-step gradient, per-frame | Reduce CLIP image-text similarity, projected into L-infinity ball |
| I2V | Yes | Multi-step gradient, joint across all frames | Same as PGD, but one shared perturbation across the whole video |

## Metrics

- **Visual distortion**: PSNR (higher = less distorted), L-infinity (max pixel change)
- **Semantic damage**: BLEU1, ROUGE1, BERTScore, SBERT cosine similarity between
  the original caption and the post-attack caption (lower similarity = more damage)
- **Cost** (in `pareto_tradeoff.py`): wall-clock time, energy (kWh, via CodeCarbon),
  estimated CO2, estimated USD cost
