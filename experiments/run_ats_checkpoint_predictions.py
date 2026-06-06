from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    run_ats_checkpoint_predictions,
    score_ats_transfer_predictions,
)


DEFAULT_SUITE_JSON = Path(
    "runs/benchmarks/ats_transfer_suite_2026_06_05/ats_transfer_suite.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TAC or vanilla checkpoint over an ATS transfer suite."
    )
    parser.add_argument("--suite-json", type=Path, default=DEFAULT_SUITE_JSON)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--control-id", required=True)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=["auto", "tac", "vanilla"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--answer-extraction",
        choices=["raw", "first_line", "first_token"],
        default="first_token",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    suite = json.loads(args.suite_json.read_text(encoding="utf-8"))
    examples = suite["examples"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_jsonl = args.output_dir / "ats_checkpoint_predictions.jsonl"
    payload = run_ats_checkpoint_predictions(
        examples=examples,
        checkpoint_path=args.checkpoint,
        control_id=args.control_id,
        seed=args.seed,
        output_jsonl=predictions_jsonl,
        model_type=args.model_type,
        device=args.device,
        precision=args.precision,
        max_new_tokens=args.max_new_tokens,
        answer_extraction=args.answer_extraction,
    )
    score_rows = score_ats_transfer_predictions(examples, payload["rows"])
    _write_jsonl(args.output_dir / "ats_checkpoint_score_rows.jsonl", score_rows)
    report = {
        "schema": "ats_checkpoint_prediction_run.v1",
        "suite_json": str(args.suite_json),
        "checkpoint": payload["checkpoint"],
        "checkpoint_step": payload["checkpoint_step"],
        "model_type": payload["model_type"],
        "control_id": args.control_id,
        "seed": args.seed,
        "prediction_count": payload["prediction_count"],
        "score_rows": score_rows,
    }
    (args.output_dir / "ats_checkpoint_prediction_run.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
