"""GAN Inversion — project a real image into StyleGAN2's W+ latent space.

Given a PIL image, iteratively optimises a W+ latent vector and
per-layer noise maps so that the StyleGAN2 generator reproduces the
image as faithfully as possible.

Based on the NVIDIA StyleGAN2 projector approach:
  - Optimise W+ latent directly (no warmup phase needed)
  - Co-optimise per-layer noise with soft regularisation
  - LPIPS perceptual loss as primary objective
  - MSE as secondary pixel-level loss
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import lpips

from utils.image_utils import pil_to_tensor


class GanInverter:
    """Projects real images into StyleGAN2's W+ latent space."""

    def __init__(self, config, generator, mean_latent):
        self.config = config
        self.device = config.device
        self.generator = generator
        self.mean_latent = mean_latent

        # Perceptual loss (pretrained VGG, frozen)
        self.lpips_fn = lpips.LPIPS(net="vgg").to(self.device).eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def _get_noise_tensors(self):
        """Create optimisable noise tensors in correct layer order."""
        noises = []
        for i in range(self.generator.num_layers):
            buf = getattr(self.generator.noises, f"noise_{i}")
            n = torch.randn_like(buf).requires_grad_(True)
            noises.append(n)
        return noises

    @staticmethod
    def _crop_face(image):
        """Detect and crop face from image for better FFHQ alignment.

        Falls back to centre crop if no face is found or OpenCV
        is not available.
        """
        try:
            import cv2
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) > 0:
                # Pick the largest detected face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cx, cy = x + w // 2, y + h // 2
                # Expand bbox by 80 % so we include hair, chin, etc.
                size = int(max(w, h) * 1.8)
                half = size // 2

                # Clamp to image bounds
                img_h, img_w = img_np.shape[:2]
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(img_w, cx + half)
                y2 = min(img_h, cy + half)

                cropped = image.crop((x1, y1, x2, y2))
                print(f"  Face detected — cropped to {cropped.size}")
                return cropped

        except Exception as e:
            print(f"  Face detection skipped ({e})")

        # Fallback: centre-crop to square
        w, h = image.size
        short = min(w, h)
        left = (w - short) // 2
        top = (h - short) // 2
        return image.crop((left, top, left + short, top + short))

    def invert(self, image, num_steps=None, callback=None):
        """Find the W+ latent and noise maps that reconstruct *image*.

        Args:
            image:     PIL.Image.Image (RGB, any size).
            num_steps: Override ``config.inversion_steps``.
            callback:  ``fn(step, loss_val, current_image)`` per iteration.

        Returns:
            dict with ``latent``, ``noise``, ``reconstructed_image``,
            ``loss_history``.
        """
        steps = num_steps or self.config.inversion_steps

        # ── Pre-process: crop face, resize to generator resolution ────
        image = image.convert("RGB")
        image = self._crop_face(image)
        target_size = self.config.stylegan_size
        image = image.resize(
            (target_size, target_size), Image.LANCZOS
        )

        target = pil_to_tensor(image, device=self.device)

        # ── Initialise W+ from mean latent ───────────────────────────
        n_latent = self.generator.n_latent
        latent = (
            self.mean_latent.clone()
            .unsqueeze(0)
            .repeat(1, n_latent, 1)
            .detach()
            .requires_grad_(True)
        )

        # ── Initialise noise (explicit ordering) ─────────────────────
        noise_list = self._get_noise_tensors()

        # ── Optimizer: separate groups for latent vs noise ────────────
        optimizer = torch.optim.Adam(
            [
                {"params": [latent], "lr": self.config.inversion_lr},
                {"params": noise_list, "lr": 0.07},
            ],
        )

        # ── Mixed precision ──────────────────────────────────────────
        use_amp = self.config.use_fp16 and self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        loss_history = []
        best_loss = float("inf")
        best_latent = latent.clone().detach()
        best_noise = [n.clone().detach() for n in noise_list]

        pbar = tqdm(range(steps), desc="Inverting image")

        for step in pbar:
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                img_gen, _ = self.generator(
                    [latent],
                    input_is_latent=True,
                    noise=noise_list,
                    randomize_noise=False,
                )

                # ── Losses ───────────────────────────────────────────
                # LPIPS is the PRIMARY loss (perceptual quality)
                perc_loss = self.lpips_fn(img_gen, target).mean()
                # MSE is SECONDARY (pixel accuracy)
                mse_loss = F.mse_loss(img_gen, target)
                # Soft noise regularisation toward N(0,1)
                noise_reg = sum(
                    (n * n).mean() for n in noise_list
                ) * 1e-5

                loss = perc_loss + 0.1 * mse_loss + noise_reg

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_latent = latent.clone().detach()
                best_noise = [n.clone().detach() for n in noise_list]

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                mse=f"{mse_loss.item():.4f}",
                lpips=f"{perc_loss.item():.4f}",
            )

            if callback is not None:
                with torch.no_grad():
                    callback(step, loss_val, img_gen)

        # ── Final reconstruction from best latent + noise ────────────
        with torch.no_grad():
            reconstructed, _ = self.generator(
                [best_latent],
                input_is_latent=True,
                noise=best_noise,
                randomize_noise=False,
            )

        return {
            "latent": best_latent,
            "noise": best_noise,
            "reconstructed_image": reconstructed,
            "loss_history": loss_history,
        }
