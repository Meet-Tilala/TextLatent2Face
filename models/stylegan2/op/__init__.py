"""StyleGAN2 custom operations — pure PyTorch fallbacks (no CUDA compilation)."""

from .fused_act import FusedLeakyReLU, fused_leaky_relu
from .upfirdn2d import upfirdn2d
