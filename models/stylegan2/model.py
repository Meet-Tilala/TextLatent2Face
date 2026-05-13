"""StyleGAN2 Generator — inference-only implementation.

Based on the Rosinality PyTorch port of StyleGAN2.  Only the generator
is included (no discriminator) because we only need forward synthesis
for latent-space manipulation.

Key building blocks
───────────────────
PixelNorm          → normalise latent vectors
EqualLinear        → FC layer with equalized learning rate
ModulatedConv2d    → style-modulated + demodulated convolution
StyledConv         → ModulatedConv2d + noise injection + activation
ToRGB              → 1×1 conv projecting features → RGB
Generator          → full mapping + synthesis network

Weight compatibility
────────────────────
State-dict keys exactly match ``rosinality/stylegan2-pytorch`` so that
pretrained checkpoints can be loaded with ``load_state_dict``.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


# ═══════════════════════════════════════════════════════════════════
#  Helper modules
# ═══════════════════════════════════════════════════════════════════

class PixelNorm(nn.Module):
    """Per-pixel feature normalisation (used in the mapping network)."""

    def forward(self, input):
        return input * torch.rsqrt(
            torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8
        )


def make_kernel(k):
    """Create a 2-D FIR kernel from a 1-D sequence via outer product."""
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Upsample(nn.Module):
    """2× spatial upsampling with FIR anti-aliasing."""

    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = (pad0, pad1)

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=self.factor, pad=self.pad)


class Blur(nn.Module):
    """FIR blur filter (applied after transposed convolution to reduce aliasing)."""

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)


# ═══════════════════════════════════════════════════════════════════
#  Equalized-LR layers
# ═══════════════════════════════════════════════════════════════════

class EqualLinear(nn.Module):
    """Fully-connected layer with equalized learning rate.

    The weight is scaled at runtime by ``1 / sqrt(fan_in) * lr_mul``
    so that the effective learning rate is uniform regardless of layer
    width (He initialisation applied at forward time).
    """

    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_dim).fill_(float(bias_init))
            )
        else:
            self.bias = None

        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


# ═══════════════════════════════════════════════════════════════════
#  Modulated convolution
# ═══════════════════════════════════════════════════════════════════

class ModulatedConv2d(nn.Module):
    """Style-modulated and (optionally) demodulated convolution.

    This is the core StyleGAN2 primitive.  A ``style`` vector scales the
    convolution weights per input channel, and demodulation normalises
    them so the output statistics stay unit-variance.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.demodulate = demodulate

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        # Modulation: style → per-channel scale  (bias_init=1 → identity at init)
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        # ── Modulate weights by style ────────────────────────────────
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        # ── Demodulate (normalise output statistics) ─────────────────
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel,
                self.kernel_size, self.kernel_size,
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel,
                self.kernel_size, self.kernel_size,
            )
            out = F.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


# ═══════════════════════════════════════════════════════════════════
#  Noise injection
# ═══════════════════════════════════════════════════════════════════

class NoiseInjection(nn.Module):
    """Injects per-pixel Gaussian noise scaled by a learnable weight."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


# ═══════════════════════════════════════════════════════════════════
#  Constant input + styled conv + ToRGB
# ═══════════════════════════════════════════════════════════════════

class ConstantInput(nn.Module):
    """Learned constant 4×4 tensor that seeds the synthesis network."""

    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        return self.input.repeat(batch, 1, 1, 1)


class StyledConv(nn.Module):
    """Style-modulated conv → noise injection → fused leaky-ReLU."""

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=(1, 3, 3, 1),
        demodulate=True,
    ):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channel, out_channel, kernel_size, style_dim,
            upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate,
        )
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    """Project feature map → 3-channel RGB with optional skip upsampling."""

    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


# ═══════════════════════════════════════════════════════════════════
#  Full Generator
# ═══════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """StyleGAN2 generator: mapping network + synthesis network.

    Latent spaces
    ─────────────
    Z   random normal input  (``style_dim``-dimensional)
    W   output of the mapping network (shared across layers)
    W+  per-layer W vectors  (``[B, n_latent, style_dim]``)

    The generator supports both W and W+ as input via the
    ``input_is_latent`` flag.
    """

    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=(1, 3, 3, 1),
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim

        # ── Mapping network  Z → W  (PixelNorm + 8 FC) ──────────────
        layers = [PixelNorm()]
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )
        self.style = nn.Sequential(*layers)

        # ── Channel widths per resolution ────────────────────────────
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.n_latent = self.log_size * 2 - 2

        # ── Initial 4×4 block ────────────────────────────────────────
        self.input = ConstantInput(channels[4])
        self.conv1 = StyledConv(
            channels[4], channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(channels[4], style_dim, upsample=False)

        # ── Progressive upsampling blocks ────────────────────────────
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        in_channel = channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel, out_channel, 3, style_dim,
                    upsample=True, blur_kernel=blur_kernel,
                )
            )
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim,
                    blur_kernel=blur_kernel,
                )
            )
            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

    # ── Convenience methods ──────────────────────────────────────────

    def mean_latent(self, n_latent):
        """Compute mean W vector for the truncation trick."""
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)
        return latent

    def get_latent(self, input):
        """Map a Z-space vector through the mapping network → W."""
        return self.style(input)

    # ── Forward synthesis ────────────────────────────────────────────

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        """Synthesise an image from latent codes.

        Args:
            styles: List of Z vectors **or** a single W / W+ tensor.
            return_latents: Also return the W+ tensor used.
            inject_index: Layer index for style mixing (training only).
            truncation: Truncation psi (0–1).  Lower ⇒ closer to mean.
            truncation_latent: Mean W (from ``mean_latent``).
            input_is_latent: If ``True``, *styles* is already in W / W+.
            noise: Explicit per-layer noise list.  ``None`` → automatic.
            randomize_noise: If ``True`` and *noise* is ``None``, sample
                fresh noise each call; otherwise use stored buffers.

        Returns:
            ``(image, latent)`` — image is ``[B, 3, size, size]`` in [-1, 1].
        """
        # ── Map Z → W if needed ─────────────────────────────────────
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        # ── Build noise list ─────────────────────────────────────────
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}")
                    for i in range(self.num_layers)
                ]

        # ── Truncation trick ─────────────────────────────────────────
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            styles = style_t

        # ── Expand single W to W+ ───────────────────────────────────
        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.n_latent - inject_index, 1
            )
            latent = torch.cat([latent, latent2], 1)

        # ── Synthesis ────────────────────────────────────────────────
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2],
            self.convs[1::2],
            noise[1::2],
            noise[2::2],
            self.to_rgbs,
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None
