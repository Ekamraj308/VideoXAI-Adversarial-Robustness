"""
Metrics used to evaluate attacks:
  - PSNR / L-infinity: how visually distorted the attacked frame is
    compared to the original
  - BLEU / ROUGE / BERTScore / SBERT cosine similarity: how much the
    generated caption changed (semantic damage) after the attack
"""

import math
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bertscore_score
from sentence_transformers import util

from models.model_setup import sbert_model

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)


def compute_psnr(a, b):
    """Peak signal-to-noise ratio between two frames (higher = less distortion)."""
    mse = np.mean((a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))


def compute_linf(a, b):
    """L-infinity distance between two frames (max per-pixel change, 0-1 scale)."""
    return float(np.max(np.abs(a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0)))


def text_metrics(ref, hyp):
    """Compare the original caption (ref) against the post-attack caption (hyp)."""
    bleu1 = sentence_bleu([ref.split()], hyp.split(), weights=(1, 0, 0, 0))
    rouge = scorer.score(ref, hyp)["rouge1"].fmeasure
    P, R, F1 = bertscore_score([hyp], [ref], lang="en", verbose=False)
    sbert = util.cos_sim(
        sbert_model.encode(ref, convert_to_tensor=True),
        sbert_model.encode(hyp, convert_to_tensor=True)
    ).item()
    return {
        "BLEU1": float(bleu1),
        "ROUGE1_f": float(rouge),
        "BERTScore_F1": float(F1.mean().item()),
        "SBERT_cos": float(sbert)
    }


def per_frame_semantic_damage(clean_captions, attacked_captions):
    """
    Paper eq. (9)/(10): per-frame semantic similarity and damage.

    S_i = cos(phi(c_i), phi(c_hat_i))   -- SBERT similarity per frame
    d_i = 1 - S_i                        -- semantic damage per frame

    Args:
        clean_captions: list of per-frame captions from the clean video
        attacked_captions: list of per-frame captions from the attacked video
                            (same length, same frame order)

    Returns:
        list of per-frame damage scores d_i
    """
    assert len(clean_captions) == len(attacked_captions), \
        "clean and attacked caption lists must be the same length (one per frame)"

    damages = []
    for c, c_hat in zip(clean_captions, attacked_captions):
        s_i = util.cos_sim(
            sbert_model.encode(c, convert_to_tensor=True),
            sbert_model.encode(c_hat, convert_to_tensor=True)
        ).item()
        damages.append(1.0 - s_i)
    return damages


def temporal_instability(damages):
    """
    Paper eq. (14): average frame-to-frame change in semantic damage.

        T = (1 / (N-1)) * sum_i |d_{i+1} - d_i|

    Lower T = smoother, more temporally consistent adversarial behavior.
    """
    if len(damages) < 2:
        return 0.0
    diffs = [abs(damages[i + 1] - damages[i]) for i in range(len(damages) - 1)]
    return float(np.mean(diffs))
