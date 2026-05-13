"""Up/down-sampling with FIR (Finite Impulse Response) filtering.

Pure PyTorch implementation of the upfirdn2d operation used in
StyleGAN2 for anti-aliased resampling.  Avoids the need to compile
custom CUDA extensions.
"""

import torch
import torch.nn.functional as F


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Apply upsampling, FIR filtering, and downsampling in one pass.

    Args:
        input:  Feature map  ``[N, C, H, W]``.
        kernel: 2-D FIR filter ``[kH, kW]``.
        up:     Integer upsampling factor.
        down:   Integer downsampling factor.
        pad:    ``(pad_before, pad_after)`` applied symmetrically to H and W.

    Returns:
        Filtered and resampled tensor.
    """
    return _upfirdn2d_native(
        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )


def _upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    """Native PyTorch fallback for the upfirdn2d CUDA kernel."""

    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    # ── Upsample by inserting zeros ──────────────────────────────────
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    # ── Spatial padding ──────────────────────────────────────────────
    out = F.pad(
        out,
        [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)],
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    # ── Apply FIR filter via depthwise convolution ───────────────────
    out = out.permute(0, 3, 1, 2)  # [N, minor, H', W']
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )

    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)

    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)

    # ── Downsample by strided slicing ────────────────────────────────
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    return out.view(-1, channel, out_h, out_w)
