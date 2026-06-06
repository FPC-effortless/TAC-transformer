from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.phase_d_benchmarks import (
    load_jsonl,
    run_phase_d_checkpoint_predictions,
    score_phase_d_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a TAC or vanilla checkpoint over Phase D benchmark tasks."
    )
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--control-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--model-type",
        choices=["auto", "tac", "vanilla"],
        default="auto",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--answer-extraction",
        choices=["raw", "first_line", "first_token"],
        default="first_token",
    )
    parser.add_argument("--score-output-json", type=Path, default=None)
    parser.add_argument("--score-output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    device = _select_device(args.device)
    payload = run_phase_d_checkpoint_predictions(
        tasks_jsonl=args.tasks_jsonl,
        checkpoint_path=args.checkpoint,
        control_id=args.control_id,
        seed=args.seed,
        output_jsonl=args.output_jsonl,
        model_type=args.model_type,
        device=device,
        precision=args.precision,
        max_new_tokens=args.max_new_tokens,
        answer_extraction=args.answer_extraction,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    score_payload = None
    if args.score_output_json is not None or args.score_output_jsonl is not None:
        score_rows = score_phase_d_predictions(
            load_jsonl(args.tasks_jsonl),
            payload["rows"],
            control_id=args.control_id,
            seed=args.seed,
        )
        score_payload = {
            "phase": "D",
            "schema": "tac_control_v1_phase_d_scored_predictions.v1",
            "tasks_jsonl": str(args.tasks_jsonl),
            "predictions_jsonl": str(args.output_jsonl),
            "checkpoint": str(args.checkpoint),
            "control_id": args.control_id,
            "seed": args.seed,
            "rows": score_rows,
        }
        if args.score_output_json is not None:
            args.score_output_json.parent.mkdir(parents=True, exist_ok=True)
            args.score_output_json.write_text(
                json.dumps(score_payload, indent=2),
                encoding="utf-8",
            )
        if args.score_output_jsonl is not None:
            args.score_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.score_output_jsonl.open("w", encoding="utf-8") as handle:
                for row in score_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "prediction_count": payload["prediction_count"],
                "output_jsonl": str(args.output_jsonl),
                "score_rows": None if score_payload is None else len(score_payload["rows"]),
            },
            indent=2,
        ),
        flush=True,
    )


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
