"""Model downloading and loading utilities.

Handles:
  - Auto-downloading the pretrained StyleGAN2 checkpoint from Google Drive
    (cached at ``~/.cache/styleclip/``).
  - Constructing the Generator and loading state-dict with graceful
    handling of key mismatches (``strict=False``).
"""

import os
import torch
import gdown
from pathlib import Path


def load_stylegan2(config):
    """Load (and optionally download) the pretrained StyleGAN2 generator.

    Args:
        config: ``StyleCLIPConfig`` instance with model paths / device.

    Returns:
        ``Generator`` module with loaded weights, on ``config.device``.
    """
    from models.stylegan2.model import Generator

    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = cache_dir / config.stylegan_filename

    # ── Download if not cached ───────────────────────────────────────
    if not ckpt_path.exists():
        print(f"Downloading StyleGAN2 pretrained model to {ckpt_path} …")
        print("(one-time download, ≈350 MB)")
        gdown.download(config.stylegan_url, str(ckpt_path), quiet=False)
        print("Download complete!")

    # ── Build generator ──────────────────────────────────────────────
    generator = Generator(
        size=config.stylegan_size,
        style_dim=config.style_dim,
        n_mlp=config.n_mlp,
        channel_multiplier=config.channel_multiplier,
    ).to(config.device)

    # ── Load weights ─────────────────────────────────────────────────
    print(f"Loading StyleGAN2 weights from {ckpt_path} …")
    checkpoint = torch.load(
        str(ckpt_path), map_location=config.device, weights_only=False
    )

    # The Rosinality checkpoint stores the EMA generator under "g_ema"
    if "g_ema" in checkpoint:
        state_dict = checkpoint["g_ema"]
    elif "g" in checkpoint:
        state_dict = checkpoint["g"]
    else:
        state_dict = checkpoint

    generator.load_state_dict(state_dict, strict=False)
    print(
        f"StyleGAN2 loaded  "
        f"(resolution {config.stylegan_size}×{config.stylegan_size})"
    )
    return generator
