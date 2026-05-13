"""Latent-code optimiser for CLIP-guided StyleGAN manipulation.

This module contains the inner optimisation loop that iteratively
adjusts a W+ latent vector so that the generated image matches a
target text description (measured via CLIP).

Speed optimisations
───────────────────
• Mixed-precision (FP16) via ``torch.amp`` on CUDA.
• Early stopping when the loss plateaus.
• CLIP text embedding is pre-computed once and reused.
• Generator weights are frozen — only the latent is updated.
"""

import torch
from tqdm import tqdm


class LatentOptimizer:
    """Optimises a W+ latent vector to match a text description."""

    def __init__(self, config):
        self.config = config
        self.device = config.device

    def optimize(
        self,
        generator,
        clip_loss_fn,
        latent_init,
        target_text_embedding,
        source_text_embedding=None,
        original_image=None,
        noise=None,
        callback=None,
    ):
        """Run the optimisation loop.

        Args:
            generator:              Frozen StyleGAN2 generator.
            clip_loss_fn:           ``CLIPLoss`` instance.
            latent_init:            Starting W+ latent ``[1, n_latent, D]``.
            target_text_embedding:  Pre-encoded target text  ``[1, D]``.
            source_text_embedding:  (optional) Pre-encoded source text.
            original_image:         (optional) Image from *latent_init* —
                                    needed when using directional loss.
            noise:                  (optional) List of per-layer noise tensors
                                    from GAN inversion. Kept fixed during
                                    editing to preserve reconstruction.
            callback:               ``fn(step, loss_dict, image)`` called
                                    every iteration.

        Returns:
            dict with ``latent``, ``image``, ``loss_history``.
        """
        # ── Trainable latent ─────────────────────────────────────────
        latent = latent_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=self.config.lr)

        # ── Mixed precision ──────────────────────────────────────────
        use_amp = self.config.use_fp16 and self.device == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # ── Tracking  ────────────────────────────────────────────────
        loss_history = []
        best_loss = float("inf")
        patience_counter = 0
        best_latent = latent.clone().detach()

        pbar = tqdm(range(self.config.num_steps), desc="Optimising latent")

        for step in pbar:
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                # Forward: latent → image (use inverted noise if available)
                if noise is not None:
                    img_gen, _ = generator(
                        [latent],
                        input_is_latent=True,
                        noise=noise,
                        randomize_noise=False,
                    )
                else:
                    img_gen, _ = generator(
                        [latent],
                        input_is_latent=True,
                        randomize_noise=False,
                    )

                # CLIP loss (directional when both source text & image given)
                if source_text_embedding is not None and original_image is not None:
                    c_loss = clip_loss_fn.directional_loss(
                        original_image, img_gen,
                        source_text_embedding, target_text_embedding,
                    )
                else:
                    c_loss = clip_loss_fn(img_gen, target_text_embedding)

                # L2 regularisation — keep the latent near its starting point
                l2_loss = ((latent - latent_init.detach()) ** 2).mean()

                loss = (
                    self.config.clip_lambda * c_loss
                    + self.config.l2_lambda * l2_loss
                )

            # ── Backward + step ──────────────────────────────────────
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # ── Logging ──────────────────────────────────────────────
            loss_val = loss.item()
            loss_dict = {
                "total": loss_val,
                "clip": c_loss.item(),
                "l2": l2_loss.item(),
            }
            loss_history.append(loss_dict)

            pbar.set_postfix(
                total=f"{loss_val:.4f}",
                clip=f"{c_loss.item():.4f}",
                l2=f"{l2_loss.item():.4f}",
            )

            # ── Early stopping ───────────────────────────────────────
            if loss_val < best_loss - self.config.early_stop_min_delta:
                best_loss = loss_val
                best_latent = latent.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stop_patience:
                print(
                    f"\n  Early stopping at step {step + 1} "
                    f"(no improvement for {self.config.early_stop_patience} steps)"
                )
                break

            if callback is not None:
                with torch.no_grad():
                    callback(step, loss_dict, img_gen)

        # ── Final image from best latent ─────────────────────────────
        with torch.no_grad():
            if noise is not None:
                final_image, _ = generator(
                    [best_latent],
                    input_is_latent=True,
                    noise=noise,
                    randomize_noise=False,
                )
            else:
                final_image, _ = generator(
                    [best_latent],
                    input_is_latent=True,
                    randomize_noise=False,
                )

        return {
            "latent": best_latent,
            "image": final_image,
            "loss_history": loss_history,
        }
