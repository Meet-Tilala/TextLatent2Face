"""Image conversion, saving, and visualisation utilities."""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def tensor_to_pil(tensor):
    """Convert a ``[-1, 1]`` image tensor to a PIL Image.

    Args:
        tensor: ``[B, 3, H, W]`` or ``[3, H, W]`` float tensor in [-1, 1].

    Returns:
        PIL.Image.Image (uses the first image when batched).
    """
    if tensor.ndim == 4:
        tensor = tensor[0]

    tensor = (tensor.clamp(-1, 1) + 1) / 2.0 * 255.0
    tensor = tensor.byte()
    array = tensor.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def pil_to_tensor(image, device="cpu"):
    """Convert a PIL Image to a ``[-1, 1]`` tensor.

    Args:
        image:  PIL.Image.Image (RGB).
        device: Target torch device.

    Returns:
        ``[1, 3, H, W]`` float tensor in [-1, 1].
    """
    array = np.array(image).astype(np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / 127.5 - 1.0
    return tensor.to(device)


def save_comparison(
    original,
    edited,
    path,
    title_left="Original",
    title_right="Edited",
):
    """Create and save a side-by-side comparison image.

    Args:
        original:    PIL Image of the original.
        edited:      PIL Image of the edited version.
        path:        File path to save the result.
        title_left:  Label for the left pane.
        title_right: Label for the right pane.
    """
    w, h = original.size
    gap = 20
    header = 40
    canvas_w = w * 2 + gap
    canvas_h = h + header

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
    canvas.paste(original, (0, header))
    canvas.paste(edited, (w + gap, header))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Centre-ish labels above each pane
    draw.text((w // 2 - 40, 8), title_left, fill=(200, 200, 200), font=font)
    draw.text(
        (w + gap + w // 2 - 40, 8), title_right, fill=(200, 200, 200), font=font
    )

    canvas.save(path)


def create_grid(images, nrow=4, padding=4):
    """Arrange a list of PIL Images into a grid.

    Args:
        images:  List of PIL Images (all same size).
        nrow:    Number of images per row.
        padding: Pixel gap between images.

    Returns:
        PIL.Image.Image containing the grid.
    """
    n = len(images)
    w, h = images[0].size
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol

    grid_w = ncol * w + (ncol - 1) * padding
    grid_h = nrow_actual * h + (nrow_actual - 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))

    for idx, img in enumerate(images):
        row = idx // ncol
        col = idx % ncol
        x = col * (w + padding)
        y = row * (h + padding)
        grid.paste(img, (x, y))

    return grid
