"""Gradio web interface for StyleCLIP text-driven image manipulation.

Upload a photo, describe the desired edit, and get the result.

Launch with::

    python app.py

Then open http://localhost:7860 in your browser.
"""

import sys
import os

# Make sure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO

import gradio as gr
from PIL import Image

from config import StyleCLIPConfig
from core.manipulator import StyleCLIPManipulator

# ── Global state (loaded once on first call) ─────────────────────────
_manipulator = None


def get_manipulator():
    """Lazy-initialise the manipulator so the UI loads quickly."""
    global _manipulator
    if _manipulator is None:
        config = StyleCLIPConfig()
        _manipulator = StyleCLIPManipulator(config)
    return _manipulator


def create_loss_plot(inversion_history, edit_history):
    """Render the optimisation loss curves as a PIL image."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))

    # ── Inversion loss ───────────────────────────────────────────
    ax1.plot(
        range(len(inversion_history)),
        inversion_history,
        color="#F59E0B",
        linewidth=2,
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("GAN Inversion (MSE + LPIPS)")
    ax1.grid(True, alpha=0.3)

    # ── CLIP editing loss ────────────────────────────────────────
    clip_losses = [h["clip"] for h in edit_history]
    total_losses = [h["total"] for h in edit_history]

    ax2.plot(
        range(len(edit_history)),
        clip_losses,
        label="CLIP Loss",
        color="#7C3AED",
        linewidth=2,
    )
    ax2.plot(
        range(len(edit_history)),
        total_losses,
        label="Total Loss",
        color="#2563EB",
        alpha=0.5,
        linewidth=1.5,
    )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Text-Guided Editing")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def run_styleclip(
    input_image,
    target_text,
    source_text,
    num_steps,
    l2_lambda,
    clip_lambda,
    inversion_steps,
):
    """Callback wired to the Gradio UI."""
    if input_image is None:
        raise gr.Error("Please upload a photo first!")
    if not target_text or not target_text.strip():
        raise gr.Error("Please enter a target description!")

    manipulator = get_manipulator()

    # Update config
    manipulator.config.l2_lambda = l2_lambda
    manipulator.config.clip_lambda = clip_lambda

    source = source_text.strip() if source_text and source_text.strip() else None

    result = manipulator.manipulate(
        image=input_image,
        target_text=target_text,
        source_text=source,
        num_steps=int(num_steps),
        inversion_steps=int(inversion_steps),
    )

    loss_plot = create_loss_plot(
        result["inversion_loss_history"],
        result["edit_loss_history"],
    )

    return (
        result["reconstructed_image"],
        result["edited_image"],
        loss_plot,
    )


# ═════════════════════════════════════════════════════════════════════
#  Build UI
# ═════════════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks() as demo:

        gr.Markdown(
            "#  StyleCLIP\n"
            "### Text-Driven Image Editing with StyleGAN2 + CLIP but in Anchored latent space\n"

        )

        with gr.Row():
            # ── Left column: controls ────────────────────────────────
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label=" Upload Photo",
                    type="pil",
                    height=280,
                )

                target_text = gr.Textbox(
                    label="Target Description",
                    placeholder="e.g. 'a face with blonde hair'",
                    lines=2,
                )
                source_text = gr.Textbox(
                    label="Source Description (optional — directional editing)",
                    placeholder="e.g. 'a face with dark hair'",
                    lines=2,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    num_steps = gr.Slider(
                        50, 500, value=300, step=10,
                        label="CLIP Editing Steps",
                    )
                    inversion_steps = gr.Slider(
                        100, 1000, value=500, step=50,
                        label="Inversion Steps (↑ = better reconstruction)",
                    )
                    l2_lambda = gr.Slider(
                        0.0, 2.0, value=0.3, step=0.05,
                        label="L2 Regularisation (↑ = less change)",
                    )
                    clip_lambda = gr.Slider(
                        0.1, 3.0, value=1.0, step=0.1,
                        label="CLIP Loss Weight",
                    )

                run_btn = gr.Button(
                    " Edit Photo", variant="primary", size="lg"
                )

            # ── Right column: outputs ────────────────────────────────
            with gr.Column(scale=2):
                with gr.Row():
                    reconstructed_img = gr.Image(
                        label="GAN Reconstruction", type="pil"
                    )
                    edited_img = gr.Image(
                        label="Edited Result", type="pil"
                    )

                loss_img = gr.Image(label="Loss Curves", type="pil")

        # ── Example prompts ──────────────────────────────────────────


        run_btn.click(
            fn=run_styleclip,
            inputs=[
                input_image, target_text, source_text,
                num_steps, l2_lambda, clip_lambda, inversion_steps,
            ],
            outputs=[reconstructed_img, edited_img, loss_img],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue"),
    )
