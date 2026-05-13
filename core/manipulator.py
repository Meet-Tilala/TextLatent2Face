"""High-level StyleCLIP manipulation pipeline.

This is the main user-facing API.  It wires together:
  - StyleGAN2 generator  (image synthesis)
  - CLIP loss            (text–image alignment)
  - GAN Inverter         (real-image → W+ projection)
  - LatentOptimizer      (iterative latent editing)

Typical usage::

    from core.manipulator import StyleCLIPManipulator
    from PIL import Image

    m = StyleCLIPManipulator()
    result = m.manipulate(Image.open("photo.jpg"), "a smiling face")
    result["edited_image"].show()
"""

import os
import sys
import torch

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import StyleCLIPConfig
from models.clip_loss import CLIPLoss
from core.optimizer import LatentOptimizer
from core.inverter import GanInverter
from utils.model_loader import load_stylegan2
from utils.image_utils import tensor_to_pil, save_comparison


class StyleCLIPManipulator:
    """Orchestrates text-driven image manipulation using StyleGAN2 + CLIP."""

    def __init__(self, config: StyleCLIPConfig = None):
        self.config = config or StyleCLIPConfig()
        self.device = self.config.device

        print(f"Initialising StyleCLIP on device: {self.device}")

        # ── Load models ──────────────────────────────────────────────
        self.generator = self._load_generator()
        self.clip_loss = CLIPLoss(
            clip_model=self.config.clip_model,
            device=self.device,
        )
        self.optimizer = LatentOptimizer(self.config)

        # ── Mean latent for truncation / inversion init ──────────────
        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(
                self.config.truncation_mean
            )

        # ── GAN inverter ─────────────────────────────────────────────
        self.inverter = GanInverter(
            self.config, self.generator, self.mean_latent
        )

        print("StyleCLIP ready!\n")

    # ── Internal helpers ─────────────────────────────────────────────

    def _load_generator(self):
        """Load and freeze the StyleGAN2 generator."""
        generator = load_stylegan2(self.config)
        generator.eval()

        for param in generator.parameters():
            param.requires_grad = False

        return generator

    # ── Public API ───────────────────────────────────────────────────

    def generate_image(self, latent):
        """Generate an image from a W+ latent code (no grad).

        Returns:
            ``[1, 3, H, W]`` tensor in [-1, 1].
        """
        with torch.no_grad():
            image, _ = self.generator(
                [latent], input_is_latent=True, randomize_noise=False
            )
        return image

    def invert(self, image, num_steps=None):
        """Project a real image into W+ latent space.

        Args:
            image:     PIL.Image.Image (RGB, any size).
            num_steps: Override ``config.inversion_steps``.

        Returns:
            dict with ``latent``, ``reconstructed_image`` (tensor),
            ``loss_history``.
        """
        return self.inverter.invert(image, num_steps=num_steps)

    def manipulate(
        self,
        image,
        target_text,
        source_text=None,
        num_steps=None,
        inversion_steps=None,
    ):
        """Upload a photo, invert it, then edit according to a text prompt.

        Args:
            image:           PIL.Image.Image to edit.
            target_text:     What the result should look like
                             (e.g. ``"a face with blonde hair"``).
            source_text:     (optional) Description of the *current* image
                             for directional editing.
            num_steps:       Override ``config.num_steps`` for CLIP editing.
            inversion_steps: Override ``config.inversion_steps``.

        Returns:
            dict with ``original_image``, ``reconstructed_image``,
            ``edited_image`` (PIL), ``latent_inverted``,
            ``latent_edited`` (tensors),
            ``inversion_loss_history``, ``edit_loss_history``.
        """
        # Temporarily override steps if requested
        original_steps = self.config.num_steps
        if num_steps is not None:
            self.config.num_steps = num_steps

        # ── Step 1: Invert the uploaded image ────────────────────────
        print("Step 1/2: Inverting uploaded image into latent space...")
        inv_result = self.invert(image, num_steps=inversion_steps)

        latent_init = inv_result["latent"]
        inverted_noise = inv_result["noise"]
        reconstructed = inv_result["reconstructed_image"]

        # ── Step 2: CLIP-guided editing ──────────────────────────────
        print("Step 2/2: Applying text-guided edit...")
        target_emb = self.clip_loss.encode_text(target_text)
        source_emb = (
            self.clip_loss.encode_text(source_text) if source_text else None
        )

        edit_result = self.optimizer.optimize(
            generator=self.generator,
            clip_loss_fn=self.clip_loss,
            latent_init=latent_init,
            target_text_embedding=target_emb,
            source_text_embedding=source_emb,
            original_image=reconstructed if source_text else None,
            noise=inverted_noise,
        )

        # Restore steps
        self.config.num_steps = original_steps

        return {
            "original_image": image,
            "reconstructed_image": tensor_to_pil(reconstructed),
            "edited_image": tensor_to_pil(edit_result["image"]),
            "latent_inverted": latent_init,
            "latent_edited": edit_result["latent"],
            "inversion_loss_history": inv_result["loss_history"],
            "edit_loss_history": edit_result["loss_history"],
        }

    def manipulate_and_save(
        self,
        image,
        target_text,
        source_text=None,
        output_path=None,
        **kwargs,
    ):
        """Manipulate and save a comparison image.

        Same arguments as ``manipulate()``, plus *output_path*.

        Returns:
            Same dict as ``manipulate()``, with an extra ``output_path`` key.
        """
        result = self.manipulate(
            image, target_text, source_text=source_text, **kwargs
        )

        if output_path is None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            safe_name = target_text.replace(" ", "_")[:50]
            output_path = os.path.join(
                self.config.output_dir, f"edit_{safe_name}.png"
            )

        save_comparison(
            result["reconstructed_image"],
            result["edited_image"],
            output_path,
            title_left="Reconstructed",
            title_right=target_text,
        )

        result["output_path"] = output_path
        print(f"Comparison saved to: {output_path}")
        return result
