from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import kaggle_fast_tac_config
from tac_transformer.training import (
    estimate_tac_parameter_count,
    estimate_vanilla_parameter_count,
    parameter_matched_baseline_config,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac_v02_100m_kaggle_feasibility")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a TAC v0.2 100M+ Kaggle training profile without starting "
            "a long-running training job."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-min-params", type=int, default=100_000_000)
    parser.add_argument("--target-max-params", type=int, default=160_000_000)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[8192, 12000, 16000])
    parser.add_argument("--d-models", type=int, nargs="+", default=[384, 448, 512, 576, 640])
    parser.add_argument("--n-layers", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--n-programs", type=int, default=24)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--attention-window-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_estimate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tac_v02_100m_kaggle_feasibility.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(format_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def run_estimate(args: argparse.Namespace) -> dict[str, Any]:
    candidates = []
    for vocab_size in args.vocab_sizes:
        for d_model in args.d_models:
            n_heads = _default_heads(d_model)
            if d_model % n_heads != 0:
                continue
            for n_layers in args.n_layers:
                config = kaggle_fast_tac_config(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    n_programs=args.n_programs,
                    max_seq_len=args.seq_len,
                    attention_window_size=args.attention_window_size,
                )
                tac_params = estimate_tac_parameter_count(config)
                if not args.target_min_params <= tac_params <= args.target_max_params:
                    continue
                vanilla_config = parameter_matched_baseline_config(config)
                candidates.append(
                    {
                        "vocab_size": vocab_size,
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                        "n_programs": args.n_programs,
                        "seq_len": args.seq_len,
                        "attention_window_size": args.attention_window_size,
                        "tac_params": tac_params,
                        "parameter_matched_vanilla_params": estimate_vanilla_parameter_count(
                            vanilla_config
                        ),
                        "parameter_matched_vanilla_d_model": vanilla_config.d_model,
                    }
                )

    candidates.sort(key=lambda row: (abs(row["tac_params"] - 112_000_000), row["d_model"]))
    recommended = candidates[0] if candidates else None
    command = _training_command(
        recommended,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        steps=args.steps,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
    )

    status = "feasible_for_kaggle_pilot" if recommended else "blocked_no_candidate"
    return {
        "schema": "tac_v02_100m_kaggle_feasibility.v1",
        "method": "static_parameter_estimate",
        "target": {
            "min_params": args.target_min_params,
            "max_params": args.target_max_params,
            "claim_boundary": (
                "Feasibility means the model shape reaches 100M+ parameters and "
                "can be launched as a bounded Kaggle pilot. It is not evidence "
                "that TAC mechanisms survive scale."
            ),
        },
        "recommended": recommended,
        "candidate_count": len(candidates),
        "top_candidates": candidates[:8],
        "kaggle_pilot": {
            "recommended_first_run_steps": args.steps,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_sequences": args.batch_size * args.grad_accum_steps,
            "eval_every": args.eval_every,
            "checkpoint_every": args.checkpoint_every,
            "precision": "fp16",
            "resume_required_for_long_runs": True,
            "run_vanilla_control": True,
            "run_reset_state_control": True,
            "run_program_knockout_control": True,
        },
        "launch_command": command,
        "decision": {
            "status": status,
            "can_train_100m_plus_on_kaggle": bool(recommended),
            "next_step": (
                "Run the pilot command on Kaggle, inspect memory/throughput and "
                "loss stability, then launch matched TAC/reset/vanilla controls "
                "only if the pilot is healthy."
                if recommended
                else "Broaden the shape search or reduce non-backbone TAC overhead."
            ),
        },
    }


def _default_heads(d_model: int) -> int:
    return max(4, d_model // 64)


def _training_command(
    recommended: dict[str, Any] | None,
    *,
    batch_size: int,
    grad_accum_steps: int,
    steps: int,
    eval_every: int,
    checkpoint_every: int,
) -> str | None:
    if recommended is None:
        return None
    return "\n".join(
        [
            "!python kaggle/train_best_tac_agentic.py \\",
            "  --preset kaggle_fast_tac \\",
            "  --scale smoke \\",
            f"  --vocab-size {recommended['vocab_size']} \\",
            f"  --d-model {recommended['d_model']} \\",
            f"  --n-heads {recommended['n_heads']} \\",
            f"  --n-layers {recommended['n_layers']} \\",
            f"  --n-programs {recommended['n_programs']} \\",
            f"  --seq-len {recommended['seq_len']} \\",
            f"  --attention-window-size {recommended['attention_window_size']} \\",
            f"  --steps {steps} \\",
            f"  --batch-size {batch_size} \\",
            f"  --grad-accum-steps {grad_accum_steps} \\",
            f"  --eval-every {eval_every} \\",
            "  --eval-batches 2 \\",
            f"  --checkpoint-every {checkpoint_every} \\",
            "  --precision fp16 \\",
            "  --device auto \\",
            "  --max-seconds 21600 \\",
            "  --stop-buffer-seconds 1200 \\",
            "  --skip-end-specialization-on-time-stop \\",
            "  --output-dir /kaggle/working/tac_v02_100m_pilot",
        ]
    )


def format_markdown(result: dict[str, Any]) -> str:
    recommended = result["recommended"]
    lines = [
        "# TAC v0.2 100M+ Kaggle Feasibility",
        "",
        f"Decision: `{result['decision']['status']}`",
        "",
    ]
    if recommended is not None:
        lines.extend(
            [
                "Recommended pilot shape:",
                "",
                "| Field | Value |",
                "|---|---:|",
                f"| TAC params | {recommended['tac_params']} |",
                f"| Vocab size | {recommended['vocab_size']} |",
                f"| d_model | {recommended['d_model']} |",
                f"| n_heads | {recommended['n_heads']} |",
                f"| n_layers | {recommended['n_layers']} |",
                f"| n_programs | {recommended['n_programs']} |",
                f"| seq_len | {recommended['seq_len']} |",
                f"| attention_window_size | {recommended['attention_window_size']} |",
                "",
                "Launch command:",
                "",
                "```python",
                result["launch_command"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "Boundary:",
            "",
            "- This is a feasibility and launch-shape estimate.",
            "- It does not validate scale survival.",
            "- TAC v0.2 still requires carried-state, reset-state, vanilla, and knockout controls.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
