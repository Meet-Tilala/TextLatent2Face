"""Fused bias + LeakyReLU activation.

Pure PyTorch implementation — no custom CUDA kernels required.
The scale factor sqrt(2) is the activation gain used in equalized
learning-rate layers (He initialisation correction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedLeakyReLU(nn.Module):
    """Learnable bias followed by LeakyReLU with gain scaling.

    Combines ``input + bias`` and ``leaky_relu`` into a single module so
    that the bias lives *before* the non-linearity (matches the original
    CUDA fused kernel behaviour).
    """

    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    """Functional fused bias + leaky-relu + gain scaling."""

    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim),
                negative_slope=negative_slope,
            )
            * scale
        )
    else:
        return F.leaky_relu(input, negative_slope=negative_slope) * scale
