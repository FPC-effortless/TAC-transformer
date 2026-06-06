from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.phase_d_benchmarks import load_jsonl, score_phase_d_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Phase D predictions into benchmark rows."
    )
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--control-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    args = parser.parse_args()

    rows = score_phase_d_predictions(
        load_jsonl(args.tasks_jsonl),
        load_jsonl(args.predictions_jsonl),
        control_id=args.control_id,
        seed=args.seed,
    )
    payload = {
        "phase": "D",
        "schema": "tac_control_v1_phase_d_scored_predictions.v1",
        "tasks_jsonl": str(args.tasks_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "control_id": args.control_id,
        "seed": args.seed,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "output_json": str(args.output_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
