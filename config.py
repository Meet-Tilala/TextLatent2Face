"""Configuration for StyleCLIP text-driven image manipulation.

Centralizes all hyperparameters, model paths, and device settings
so that every module reads from a single source of truth.
"""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class StyleCLIPConfig:
    """Central configuration for the StyleCLIP pipeline."""

    # ── StyleGAN2 architecture ──────────────────────────────────────
    stylegan_size: int = 1024
    style_dim: int = 512
    n_mlp: int = 8
    channel_multiplier: int = 2

    # ── CLIP ────────────────────────────────────────────────────────
    clip_model: str = "ViT-B/32"

    # ── Latent optimisation (CLIP editing) ─────────────────────────
    num_steps: int = 300
    lr: float = 0.01
    l2_lambda: float = 0.3
    clip_lambda: float = 1.0

    # ── Speed / early-stopping ──────────────────────────────────────
    use_fp16: bool = True
    early_stop_patience: int = 50
    early_stop_min_delta: float = 0.0005

    # ── Truncation trick ────────────────────────────────────────────
    truncation: float = 0.7
    truncation_mean: int = 4096

    # ── GAN Inversion ──────────────────────────────────────────────
    inversion_steps: int = 500
    inversion_lr: float = 0.1

    # ── Paths ───────────────────────────────────────────────────────
    output_dir: str = "outputs"
    cache_dir: str = str(Path.home() / ".cache" / "styleclip")

    # ── Device ──────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Pretrained model download ───────────────────────────────────
    stylegan_url: str = (
        "https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT"
    )
    stylegan_filename: str = "stylegan2-ffhq-config-f.pt"
