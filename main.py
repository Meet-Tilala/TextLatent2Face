"""CLI entry point for StyleCLIP text-driven image editing.

Examples
────────
  # Edit an uploaded photo
  python main.py --image photo.jpg --target "a smiling face"

  # Directional edit (more stable, better identity preservation)
  python main.py --image photo.jpg --target "a face with blonde hair" --source "a face with dark hair"

  # Quick edit with fewer steps
  python main.py --image photo.jpg --target "a face with glasses" --steps 100

  # Lower regularisation → stronger edit
  python main.py --image photo.jpg --target "an angry face" --l2-lambda 0.5
"""

import argparse
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from config import StyleCLIPConfig
from core.manipulator import StyleCLIPManipulator


def parse_args():
    parser = argparse.ArgumentParser(
        description="StyleCLIP — Text-Driven Image Editing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to the input image to edit.",
    )
    parser.add_argument(
        "--target", "-t", type=str, required=True,
        help="Target text description for the edit.",
    )
    parser.add_argument(
        "--source", "-s", type=str, default=None,
        help="Source text (enables directional CLIP loss for stable edits).",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="CLIP editing optimisation steps  (default: 200).",
    )
    parser.add_argument(
        "--inversion-steps", type=int, default=300,
        help="GAN inversion steps  (default: 300).",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Learning rate  (default: 0.1).",
    )
    parser.add_argument(
        "--l2-lambda", type=float, default=0.8,
        help="L2 regularisation weight  (default: 0.8).",
    )
    parser.add_argument(
        "--clip-lambda", type=float, default=1.0,
        help="CLIP loss weight  (default: 1.0).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path  (auto-generated if omitted).",
    )
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Disable mixed-precision (FP16) inference.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on  (auto-detected if omitted).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input image
    if not os.path.isfile(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    config = StyleCLIPConfig(
        num_steps=args.steps,
        lr=args.lr,
        l2_lambda=args.l2_lambda,
        clip_lambda=args.clip_lambda,
        inversion_steps=args.inversion_steps,
        use_fp16=not args.no_fp16,
    )

    if args.device:
        config.device = args.device

    # ── Banner ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  StyleCLIP — Text-Driven Image Editing")
    print("=" * 60)
    print(f"  Image:       {args.image}")
    print(f"  Target:      {args.target}")
    if args.source:
        print(f"  Source:      {args.source}")
    print(f"  Inv. steps:  {config.inversion_steps}")
    print(f"  Edit steps:  {config.num_steps}")
    print(f"  Device:      {config.device}")
    print(f"  FP16:        {config.use_fp16}")
    print("=" * 60 + "\n")

    # ── Load image ───────────────────────────────────────────────────
    input_image = Image.open(args.image).convert("RGB")

    # ── Run ──────────────────────────────────────────────────────────
    manipulator = StyleCLIPManipulator(config)
    result = manipulator.manipulate_and_save(
        image=input_image,
        target_text=args.target,
        source_text=args.source,
        output_path=args.output,
    )

    # ── Summary ──────────────────────────────────────────────────────
    edit_history = result["edit_loss_history"]
    final = edit_history[-1]
    print(f"\nFinal losses — CLIP: {final['clip']:.4f}  L2: {final['l2']:.4f}")
    print(f"Inversion steps: {len(result['inversion_loss_history'])}")
    print(f"Editing steps:   {len(edit_history)}")
    print("Done!")


if __name__ == "__main__":
    main()
