from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.serving import (
    generate_tac_completion,
    load_tac_checkpoint_for_generation,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a Gradio GUI for TAC checkpoint generation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=["auto", "tac", "vanilla"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser


def create_demo(
    checkpoint: str | Path,
    *,
    model_type: str = "auto",
    device: str | torch.device = "auto",
    precision: str = "fp32",
):
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError(
            "Gradio is not installed. Install it with `pip install gradio` to use the TAC GUI."
        ) from exc

    resolved_device = _select_device(str(device))
    model, metadata = load_tac_checkpoint_for_generation(
        checkpoint,
        model_type=model_type,
        device=resolved_device,
    )

    def generate(prompt, max_new_tokens, temperature, top_k, top_p, seed):
        result = generate_tac_completion(
            model,
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            device=resolved_device,
            precision=precision,
            seed=None if seed in (None, "") else int(seed),
        )
        return result["completion"], _format_metadata(metadata, result)

    with gr.Blocks(title="TAC Checkpoint Generator") as demo:
        gr.Markdown("# TAC Checkpoint Generator")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=8)
            output = gr.Textbox(label="Completion", lines=8)
        with gr.Row():
            max_new_tokens = gr.Slider(1, 512, value=128, step=1, label="Max new tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_k = gr.Slider(1, 256, value=50, step=1, label="Top-K")
            top_p = gr.Slider(0.05, 1.0, value=0.9, step=0.05, label="Top-P")
            seed = gr.Number(value=0, precision=0, label="Seed")
        generate_button = gr.Button("Generate", variant="primary")
        metadata_box = gr.JSON(label="Run metadata")
        generate_button.click(
            generate,
            inputs=[prompt, max_new_tokens, temperature, top_k, top_p, seed],
            outputs=[output, metadata_box],
        )
    return demo


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    demo = create_demo(
        args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        precision=args.precision,
    )
    demo.launch(server_name=args.host, server_port=args.port)


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _format_metadata(metadata, result):
    return {
        "model_type": metadata["model_type"],
        "checkpoint_step": metadata["checkpoint_step"],
        "best_eval_loss": metadata["best_eval_loss"],
        "tokenizer": metadata["tokenizer"],
        "generated_token_count": result["generated_token_count"],
        "tokens_per_second": result["tokens_per_second"],
        "truncated_prompt_token_count": result["truncated_prompt_token_count"],
        "context_window": result["context_window"],
    }


if __name__ == "__main__":
    main()
