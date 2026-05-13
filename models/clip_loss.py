"""CLIP-based losses for text-guided image manipulation.

Provides two loss functions:
  1. **Global CLIP loss** — cosine distance between a generated image and
     a target text prompt in CLIP embedding space.
  2. **Directional CLIP loss** — measures whether the *change* in image
     embeddings is aligned with the *change* in text embeddings.  This
     is more stable for editing because it ignores the absolute position
     and focuses on the direction of the edit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPLoss(nn.Module):
    """CLIP-based loss for aligning generated images with text descriptions."""

    def __init__(self, clip_model="ViT-B/32", device="cuda"):
        super().__init__()

        self.device = device
        self.model, _ = clip.load(clip_model, device=device)
        self.model.eval()

        # Freeze all CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # CLIP normalisation stats (ImageNet-derived)
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

        self.target_size = 224

    # ── Encoding helpers ─────────────────────────────────────────────

    def encode_text(self, text):
        """Tokenise + encode text → normalised CLIP embedding.

        Args:
            text: A single string describing the desired appearance.

        Returns:
            ``[1, D]`` float tensor (detached, unit-normalised).
        """
        tokens = clip.tokenize([text]).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach()

    def encode_image(self, image):
        """Resize + normalise + encode an image → normalised CLIP embedding.

        Args:
            image: ``[B, 3, H, W]`` in **[-1, 1]** (StyleGAN output range).

        Returns:
            ``[B, D]`` float tensor (unit-normalised).
        """
        # [-1, 1] → [0, 1]
        image = (image + 1.0) / 2.0
        image = image.clamp(0, 1)

        # Resize to CLIP input resolution
        image = F.interpolate(
            image,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        # Normalise with CLIP stats
        image = (image - self.mean.to(image.device)) / self.std.to(image.device)

        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    # ── Loss functions ───────────────────────────────────────────────

    def forward(self, image, text_embedding):
        """Global CLIP loss: cosine distance between image and text.

        Args:
            image:          ``[B, 3, H, W]`` generated image in [-1, 1].
            text_embedding: Pre-computed target text embedding from
                            ``encode_text()``.

        Returns:
            Scalar loss ∈ [0, 2].  Lower → image matches text better.
        """
        image_features = self.encode_image(image)
        similarity = (image_features * text_embedding).sum(dim=-1)
        return (1 - similarity).mean()

    def directional_loss(self, img_orig, img_edit, text_source, text_target):
        """Directional CLIP loss for stable editing.

        Instead of pushing the image toward an absolute text target, this
        measures whether the *direction* of change in image space is
        aligned with the *direction* of change in text space::

            Δimage = CLIP(edited) − CLIP(original)
            Δtext  = CLIP(target) − CLIP(source)
            loss   = 1 − cos(Δimage, Δtext)

        This yields more disentangled edits and better identity
        preservation.

        Args:
            img_orig:    Original generated image  ``[B, 3, H, W]``.
            img_edit:    Edited generated image     ``[B, 3, H, W]``.
            text_source: Source text embedding (from ``encode_text``).
            text_target: Target text embedding (from ``encode_text``).

        Returns:
            Scalar directional cosine-distance loss.
        """
        img_feat_orig = self.encode_image(img_orig)
        img_feat_edit = self.encode_image(img_edit)

        delta_img = img_feat_edit - img_feat_orig
        delta_img = delta_img / (delta_img.norm(dim=-1, keepdim=True) + 1e-8)

        delta_text = text_target - text_source
        delta_text = delta_text / (delta_text.norm(dim=-1, keepdim=True) + 1e-8)

        similarity = (delta_img * delta_text).sum(dim=-1)
        return (1 - similarity).mean()
