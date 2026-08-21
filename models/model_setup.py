"""
Model loading and embedding helpers.

Loads three pretrained models used throughout the pipeline:
  - BLIP: generates a natural-language caption from a video frame
  - CLIP: embeds images and text into a shared vector space. This is the
    model the adversarial attacks (FGSM, PGD, I2V) optimize against.
  - SBERT: sentence embeddings, used later for caption similarity metrics
"""

import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# BLIP (image captioning)
# -------------------------
processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device).eval()

# -------------------------
# CLIP (image-text embedding model — the attack target)
# -------------------------
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

# Normalization stats used to convert CLIP's normalized tensors back to
# viewable images after an attack perturbs them
fe = getattr(clip_processor, "feature_extractor", None)
if fe is None:
    fe = getattr(clip_processor, "image_processor")

MEAN = np.array(fe.image_mean).reshape(1, 1, 3)
STD = np.array(fe.image_std).reshape(1, 1, 3)

# -------------------------
# SBERT (sentence similarity, used in metrics)
# -------------------------
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def caption_from_frames(frames):
    """Generate a single caption for a video using its middle frame.

    This is a cheap summary caption, kept for quick display/logging.
    It is NOT what the paper's equations describe — see
    caption_each_frame() for the per-frame captioning that matches
    eq. (3)/(8) in the paper.
    """
    mid = frames[len(frames) // 2]
    inputs = processor_blip(images=mid, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_blip.generate(**inputs, max_new_tokens=40)
    return processor_blip.decode(out[0], skip_special_tokens=True)


def caption_each_frame(frames):
    """
    Generate one caption per frame: c_i = G_BLIP(f_i) for each frame f_i.

    This matches the paper's eq. (3) (clean captions) and eq. (8)
    (attacked captions), where captioning happens per-frame rather than
    once per video. Returns a list of caption strings, one per frame.
    """
    captions = []
    for frame in frames:
        inputs = processor_blip(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_blip.generate(**inputs, max_new_tokens=40)
        captions.append(processor_blip.decode(out[0], skip_special_tokens=True))
    return captions


def get_text_embed(text):
    """Embed a text string into CLIP's shared embedding space (normalized)."""
    out = clip_model.get_text_features(
        **clip_processor(text=[text], return_tensors="pt").to(device)
    )
    if not torch.is_tensor(out):
        out = out.pooler_output
    return out / (out.norm(dim=-1, keepdim=True) + 1e-10)


def get_image_embed(pv):
    """Embed pixel values into CLIP's shared embedding space (normalized)."""
    out = clip_model.get_image_features(pixel_values=pv)
    if not torch.is_tensor(out):
        out = out.pooler_output
    return out / (out.norm(dim=-1, keepdim=True) + 1e-10)
