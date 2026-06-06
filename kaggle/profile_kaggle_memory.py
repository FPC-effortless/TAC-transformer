from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle.train_best_tac_agentic import MODEL_SCALES
from tac_transformer import TACTransformerLM, best_tac_config
from tac_transformer.optimization import TACOptimizerConfig, build_tac_optimizer
from tac_transformer.training import count_parameters


DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate and optionally measure Kaggle TAC training memory."
    )
    parser.add_argument("--scales", nargs="+", choices=sorted(MODEL_SCALES), default=["base"])
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--content-store-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--run-forward",
        action="store_true",
        help="Run one forward/backward step and report CUDA peak allocation when CUDA is available.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/benchmarks/kaggle_memory_profile_2026_05_30.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    rows = [
        profile_scale(args=args, scale_name=scale_name, device=device)
        for scale_name in args.scales
    ]
    result = {"device": str(device), "precision": args.precision, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def profile_scale(
    *,
    args: argparse.Namespace,
    scale_name: str,
    device: torch.device,
) -> dict[str, Any]:
    scale = MODEL_SCALES[scale_name]
    config = best_tac_config(
        vocab_size=args.vocab_size,
        d_model=scale["d_model"],
        n_heads=scale["n_heads"],
        n_layers=scale["n_layers"],
        n_programs=scale["n_programs"],
        max_seq_len=scale["seq_len"],
        content_store_size=args.content_store_size,
    )
    model = TACTransformerLM(config)
    counts = count_parameters(model)
    dtype_bytes = DTYPE_BYTES[args.precision]
    state_bytes = estimate_identity_state_bytes(
        batch_size=scale["batch_size"],
        n_layers=scale["n_layers"],
        n_programs=scale["n_programs"],
        d_model=scale["d_model"],
        content_store_size=args.content_store_size,
        dtype_bytes=dtype_bytes,
    )
    optimizer_bytes = counts["trainable"] * 8
    parameter_bytes = counts["trainable"] * dtype_bytes
    gradient_bytes = counts["trainable"] * dtype_bytes
    measured = None
    if args.run_forward and device.type == "cuda":
        measured = measure_cuda_step(
            model=model,
            config=config,
            batch_size=scale["batch_size"],
            device=device,
            precision=args.precision,
        )
    return {
        "scale": scale_name,
        "seq_len": scale["seq_len"],
        "batch_size_per_device": scale["batch_size"],
        "grad_accum_steps": scale["grad_accum_steps"],
        "config": {
            "d_model": scale["d_model"],
            "n_heads": scale["n_heads"],
            "n_layers": scale["n_layers"],
            "n_programs": scale["n_programs"],
            "memory_read_type": config.memory_read_type,
            "content_store_size": config.content_store_size,
        },
        "parameter_counts": counts,
        "estimated_bytes": {
            "parameters": parameter_bytes,
            "gradients": gradient_bytes,
            "adamw_states": optimizer_bytes,
            "identity_state_total": state_bytes["total"],
            "identity_content_store": state_bytes["content_store"],
            "identity_program_memory": state_bytes["program_memory"],
            "identity_stability_and_metadata": state_bytes["metadata"],
            "parameters_plus_gradients_plus_adamw": parameter_bytes
            + gradient_bytes
            + optimizer_bytes,
        },
        "estimated_mib": {
            key: value / (1024**2)
            for key, value in {
                "parameters": parameter_bytes,
                "gradients": gradient_bytes,
                "adamw_states": optimizer_bytes,
                "identity_state_total": state_bytes["total"],
                "identity_content_store": state_bytes["content_store"],
                "identity_program_memory": state_bytes["program_memory"],
                "identity_stability_and_metadata": state_bytes["metadata"],
                "parameters_plus_gradients_plus_adamw": parameter_bytes
                + gradient_bytes
                + optimizer_bytes,
            }.items()
        },
        "measured_cuda": measured,
    }


def estimate_identity_state_bytes(
    *,
    batch_size: int,
    n_layers: int,
    n_programs: int,
    d_model: int,
    content_store_size: int,
    dtype_bytes: int,
) -> dict[str, int]:
    program_memory = batch_size * n_programs * d_model * dtype_bytes
    all_program_memory = 3 * program_memory
    stability = batch_size * n_programs * dtype_bytes
    program_age = batch_size * n_programs * 2 * dtype_bytes
    content_store = batch_size * content_store_size * d_model * 2 * dtype_bytes
    content_mask = batch_size * content_store_size
    per_layer = all_program_memory + stability + program_age + content_store + content_mask
    return {
        "program_memory": n_layers * all_program_memory,
        "metadata": n_layers * (stability + program_age + content_mask),
        "content_store": n_layers * content_store,
        "total": n_layers * per_layer,
    }


def measure_cuda_step(
    *,
    model: TACTransformerLM,
    config,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.to(device)
    model.train()
    optimizer = build_tac_optimizer(
        model,
        TACOptimizerConfig(learning_rate=1e-4),
    )
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, config.max_seq_len),
        device=device,
    )
    labels = torch.randint(
        0,
        config.vocab_size,
        (batch_size, config.max_seq_len),
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    autocast_enabled = precision in {"fp16", "bf16"}
    with torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        output = model(input_ids, labels=labels)
        loss = output.loss
        if loss is None:
            raise RuntimeError("expected model loss")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "peak_allocated_bytes": float(peak),
        "peak_allocated_mib": float(peak / (1024**2)),
    }


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
