# TextLatent2Face — StyleCLIP Face Editor

**An advanced PyTorch implementation** that combines **StyleGAN2** (image synthesis) and **CLIP** (text–image alignment) to perform **text-driven editing on your own uploaded photos**, preserving identity while enabling creative semantic transformations.

**TextLatent2Face** is a modular PyTorch implementation of **StyleCLIP** that combines **StyleGAN2** (image synthesis) with **CLIP** (text–image alignment) to edit **your own uploaded photos** using natural-language prompts.

> **Example:** Upload a portrait → type *"a face with blonde hair"* → get an edited version with blonde hair, with identity preserved.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐
│  Uploaded     │     │  Text Prompt │
│  Photo       │     └──────┬───────┘
└──────┬───────┘            │
       │                    ▼
       ▼             ┌──────────────┐
┌──────────────┐     │  CLIP Text   │
│  GAN         │     │  Encoder     │
│  Inversion   │     └──────┬───────┘
│  (MSE+LPIPS) │            │
└──────┬───────┘            │
       │                    │
 ┌─────▼──────┐             │
 │  W+ Latent │ ◄── Optimised iteratively
 └─────┬──────┘             │
       │                    │
 ┌─────▼──────┐             │
 │  StyleGAN2 │             │
 │  Generator  │             │
 └─────┬──────┘             │
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│  Generated   │     │  CLIP Image  │
│  Image       │     │  Encoder     │
└──────────────┘     └──────┬───────┘
                            │
                            ▼
┌──────────────────────────────────┐
│  Loss = CLIP_loss + λ · L2_loss  │
│  Backprop through latent only    │
└──────────────────────────────────┘
```

The pipeline works in two stages:
1. **GAN Inversion**: The uploaded photo is projected into StyleGAN2's W+ latent space by optimising a latent vector to minimise MSE + LPIPS perceptual loss.
2. **CLIP-Guided Editing**: The inverted W+ latent is further optimised so the generated image matches the text prompt (CLIP loss) while staying close to the reconstruction (L2 regularisation).

---

## Project Structure

```
CV/
├── config.py                  # All hyperparameters & paths
├── models/
│   ├── stylegan2/
│   │   ├── op/
│   │   │   ├── fused_act.py   # Fused bias + LeakyReLU
│   │   │   └── upfirdn2d.py   # FIR up/downsampling
│   │   └── model.py           # StyleGAN2 generator
│   └── clip_loss.py           # CLIP loss functions
├── core/
│   ├── inverter.py            # GAN inversion (image → W+ latent)
│   ├── optimizer.py           # Latent optimisation loop
│   └── manipulator.py         # High-level manipulation API
├── utils/
│   ├── image_utils.py         # Image I/O & visualisation
│   └── model_loader.py        # Checkpoint downloading
├── main.py                    # CLI entry point
├── app.py                     # Gradio web UI
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The CLIP package is installed from GitHub. If you have issues, install it separately:
> ```bash
> pip install git+https://github.com/openai/CLIP.git
> ```

### 2. Pretrained models

Models are **downloaded automatically** on first run:

| Model | Size | Source |
|-------|------|--------|
| StyleGAN2 (FFHQ 1024×1024) | ~350 MB | Rosinality |
| CLIP (ViT-B/32) | ~340 MB | OpenAI |
| LPIPS (VGG) | ~80 MB | Perceptual loss |

Cached at `~/.cache/styleclip/`.

### 3. Hardware

- **GPU recommended**: CUDA-capable GPU with ≥ 6 GB VRAM.
- **CPU mode**: Works but is significantly slower. Reduce `--steps` and `--inversion-steps`.

---

## Usage

### CLI

```bash
# Edit an uploaded photo — make the person smile
python main.py --image photo.jpg --target "a smiling face"

# Directional edit — more stable, better identity preservation
python main.py --image photo.jpg --target "a face with blonde hair" --source "a face with dark hair"

# Quick edit (fewer steps)
python main.py --image photo.jpg --target "a face with glasses" --steps 100

# Stronger edit (lower L2 regularisation)
python main.py --image photo.jpg --target "an angry face" --l2-lambda 0.4

# CPU mode
python main.py --image photo.jpg --target "a smiling face" --device cpu --no-fp16 --steps 50
```

### Web UI (Gradio)

```bash
python app.py
# → Open http://localhost:7860
```

Features:
- Photo upload widget
- Target & source text inputs
- Adjustable inversion & editing steps, L2 weight, CLIP weight
- GAN reconstruction + edited result side-by-side
- Live loss curves (inversion + editing)

---

## Configuration

All defaults are in `config.py`. Override via CLI args or by modifying the config dataclass:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_steps` | 200 | CLIP editing iterations |
| `inversion_steps` | 300 | GAN inversion iterations |
| `inversion_lr` | 0.05 | Learning rate for inversion |
| `lr` | 0.1 | Learning rate for CLIP editing |
| `l2_lambda` | 0.8 | L2 regularisation weight |
| `clip_lambda` | 1.0 | CLIP loss weight |
| `truncation` | 0.7 | Truncation ψ (0–1) |
| `use_fp16` | True | Mixed-precision on GPU |
| `early_stop_patience` | 20 | Stop if no improvement |

---

## Speed Optimisations

| Technique | Effect |
|-----------|--------|
| Mixed precision (FP16) | ~2× faster on GPU |
| Early stopping | Saves 30–50% of editing steps |
| Frozen generator | No gradient for G's ~30M params |
| Pre-encoded text | CLIP text encoding done once |
| Configurable steps | 50 steps ≈ 5s on a modern GPU |

---

## How It Works

1. **Upload** a real photo (any size — resized internally to 1024×1024).
2. **Invert** the photo into StyleGAN2's W+ latent space by optimising a latent to minimise MSE + LPIPS reconstruction loss.
3. **Optimise** the inverted W+ to minimise:
   - **CLIP loss**: cosine distance between the generated image and the target text in CLIP's shared embedding space.
   - **L2 loss**: keeps the latent close to the inverted point (identity preservation).
4. **Generate** the final edited image from the optimised W+.

The **directional CLIP loss** variant measures whether the *direction* of change in image space aligns with the *direction* of change in text space, yielding more disentangled edits.

---

## Citation

This project is based on:

```bibtex
@InProceedings{Patashnik_2021_ICCV,
    author    = {Patashnik, Or and Wu, Zongze and Shechtman, Eli
                 and Cohen-Or, Daniel and Lischinski, Dani},
    title     = {StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery},
    booktitle = {ICCV},
    year      = {2021},
}
```

StyleGAN2 generator based on [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).
CLIP from [openai/CLIP](https://github.com/openai/CLIP).
