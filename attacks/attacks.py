"""
Adversarial attack implementations.

FGSM, PGD, and I2V all optimize the same objective: they use CLIP to embed
a video frame and the original caption, then perturb the frame's pixels
(within an epsilon bound) to *reduce* the cosine similarity between the
image embedding and the text embedding. That drop in CLIP similarity is
what "CLIP for optimization" refers to — CLIP's similarity score is the
loss function the attack is minimizing.

VBAD is different: it's a simple patch-based attack (paints a colored
patch onto each frame) and does not use CLIP at all. It exists as a
non-gradient-based baseline for comparison.
"""

import math
import numpy as np
import torch
from PIL import Image

from models.model_setup import (
    device, MEAN, STD, clip_processor, clip_model,
    get_text_embed, get_image_embed,
)


def vbad_like_patch_attack(frames, patch_ratio=0.06, seed=42):
    """
    Baseline patch attack: paints a checkerboard patch at a random
    location in each frame. Does not use CLIP or gradients.
    """
    np.random.seed(seed)
    T, H, W, _ = frames.shape
    out = frames.copy()
    ps = max(4, int(math.sqrt(H * W * patch_ratio)))
    for t in range(T):
        x = np.random.randint(0, max(1, W - ps))
        y = np.random.randint(0, max(1, H - ps))
        for i in range(ps):
            for j in range(ps):
                out[t, y + i, x + j] = [255, 0, 0] if (i + j) % 2 == 0 else [255, 255, 255]
    return out


def fgsm_attack_video(frames_np, caption, eps=0.08):
    """
    Fast Gradient Sign Method: one-step perturbation in the direction that
    most reduces CLIP image-text cosine similarity.
    """
    with torch.no_grad():
        txt = get_text_embed(caption)

    out = []
    for f in frames_np:
        pv = clip_processor(
            images=Image.fromarray(f),
            return_tensors="pt"
        ).to(device)["pixel_values"]

        pv = pv.clone().detach().requires_grad_(True)

        img = get_image_embed(pv)
        loss = torch.cosine_similarity(img, txt, dim=-1).mean()

        loss.backward()

        adv = pv - eps * pv.grad.sign()
        adv = adv.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        adv = ((adv * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)

        out.append(adv)

    return np.array(out)


def pgd_attack_video(frames_np, caption, eps=0.03, iters=6, alpha=0.01):
    """
    Projected Gradient Descent: multi-step version of FGSM. Repeatedly
    steps against the CLIP similarity gradient, then projects the result
    back into the epsilon L-infinity ball around the original pixels.
    """
    with torch.no_grad():
        text_inputs = clip_processor(text=[caption], return_tensors="pt").to(device)
        text_emb = clip_model.get_text_features(**text_inputs)

        if not torch.is_tensor(text_emb) and hasattr(text_emb, "pooler_output"):
            text_emb = text_emb.pooler_output

        text_emb = text_emb / (text_emb.norm(p=2, dim=-1, keepdim=True) + 1e-10)

    text_emb = text_emb.detach()

    out = []

    for f in frames_np:

        pil = Image.fromarray(f.astype(np.uint8))
        inputs = clip_processor(images=pil, return_tensors="pt").to(device)

        pv_orig = inputs["pixel_values"].detach()
        pv = pv_orig.clone().detach().requires_grad_(True)

        for _ in range(iters):

            if pv.grad is not None:
                pv.grad.zero_()

            img_feat = clip_model.get_image_features(pixel_values=pv)

            if not torch.is_tensor(img_feat) and hasattr(img_feat, "pooler_output"):
                img_feat = img_feat.pooler_output

            img_feat = img_feat / (img_feat.norm(p=2, dim=-1, keepdim=True) + 1e-10)

            sim = torch.nn.functional.cosine_similarity(img_feat, text_emb, dim=-1).mean()

            sim.backward()

            grad = pv.grad.sign()
            pv = pv.detach() - alpha * grad

            # L-infinity projection back into the eps ball
            delta = torch.clamp(pv - pv_orig, min=-eps, max=eps)
            pv = torch.clamp(pv_orig + delta, 0.0, 1.0)

            pv = pv.detach().requires_grad_(True)

        adv = pv.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        adv = ((adv * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)

        out.append(adv)

    return np.array(out)


def i2v_attack_video(frames_np, caption, eps=0.03, iters=6, alpha=0.01, lambda_temp=0.5):
    """
    Image-to-Video attack.

    Like PGD, this reduces CLIP image-text cosine similarity. Unlike PGD,
    it optimizes one shared perturbation tensor jointly across all frames,
    AND adds a temporal smoothness penalty on the perturbation itself:

        L_temp = sum_i || delta_{i+1} - delta_i ||_2                (paper eq. 7)

    The total loss being minimized each step is:

        L = cosine_similarity(image, text)  +  lambda_temp * L_temp

    Minimizing L_temp keeps the perturbation from changing abruptly between
    consecutive frames, which is what makes I2V "temporally aware" rather
    than just PGD run frame-independently. lambda_temp controls how much
    weight the smoothness term gets relative to the attack's semantic
    objective.
    """
    with torch.no_grad():
        text_inputs = clip_processor(text=[caption], return_tensors="pt").to(device)
        text_emb = clip_model.get_text_features(**text_inputs)

    if not torch.is_tensor(text_emb) and hasattr(text_emb, "pooler_output"):
        text_emb = text_emb.pooler_output

    text_emb = text_emb / (text_emb.norm(p=2, dim=-1, keepdim=True) + 1e-10)
    text_emb = text_emb.detach()

    frames = torch.tensor(frames_np / 255.0, device=device).permute(0, 3, 1, 2).float()
    delta = torch.zeros_like(frames, requires_grad=True)

    mean_ch = torch.tensor(MEAN.reshape(3), device=device).view(1, 3, 1, 1)
    std_ch = torch.tensor(STD.reshape(3), device=device).view(1, 3, 1, 1)

    for _ in range(iters):

        if delta.grad is not None:
            delta.grad.zero_()

        adv = torch.clamp(frames + delta, 0.0, 1.0)
        adv_norm = (adv - mean_ch) / std_ch

        img_feat = clip_model.get_image_features(pixel_values=adv_norm)

        if not torch.is_tensor(img_feat) and hasattr(img_feat, "pooler_output"):
            img_feat = img_feat.pooler_output

        img_feat = img_feat / (img_feat.norm(p=2, dim=-1, keepdim=True) + 1e-10)

        similarity_loss = torch.nn.functional.cosine_similarity(img_feat, text_emb, dim=-1).mean()

        # Temporal smoothness term (paper eq. 7): penalize large frame-to-frame
        # changes in the perturbation itself.
        temporal_loss = torch.norm(delta[1:] - delta[:-1], p=2, dim=(1, 2, 3)).sum()

        loss = similarity_loss + lambda_temp * temporal_loss
        loss.backward()

        with torch.no_grad():
            delta -= alpha * delta.grad.sign()
            delta.data = torch.clamp(delta.data, -eps, eps)

        delta = delta.detach().requires_grad_(True)

    adv_final = torch.clamp(frames + delta.detach(), 0.0, 1.0)
    return (adv_final.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
