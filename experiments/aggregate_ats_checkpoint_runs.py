from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    aggregate_ats_checkpoint_run_results,
    format_ats_checkpoint_run_markdown,
)


DEFAULT_INPUT_DIRS = [
    Path("runs/kaggle_outputs"),
    Path("runs/benchmarks"),
]
DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/ats_checkpoint_comparison_2026_06_05")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate TAC and vanilla ATS checkpoint prediction runs."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        default=None,
        help="Directory to scan recursively for ats_checkpoint_prediction_run.json.",
    )
    parser.add_argument(
        "--run-json",
        action="append",
        type=Path,
        default=None,
        help="Explicit ats_checkpoint_prediction_run.json path.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tac-control-id", default="tac_base_ats_5k")
    parser.add_argument("--vanilla-control-id", default="vanilla_base_ats_5k")
    parser.add_argument("--min-train-score", type=float, default=0.95)
    parser.add_argument("--min-tac-test-score", type=float, default=0.95)
    parser.add_argument("--min-tac-test-advantage", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs = [_load_json(path) for path in _discover_run_jsons(args)]
    aggregate = aggregate_ats_checkpoint_run_results(
        runs,
        tac_control_id=args.tac_control_id,
        vanilla_control_id=args.vanilla_control_id,
        min_train_score=args.min_train_score,
        min_tac_test_score=args.min_tac_test_score,
        min_tac_test_advantage=args.min_tac_test_advantage,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ats_checkpoint_run_aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(args.output_dir / "ats_checkpoint_score_rows.jsonl", aggregate["score_rows"])
    (args.output_dir / "RESULTS.md").write_text(
        format_ats_checkpoint_run_markdown(aggregate),
        encoding="utf-8",
    )
    print(json.dumps(aggregate["decision"], indent=2), flush=True)


def _discover_run_jsons(args: argparse.Namespace) -> list[Path]:
    paths = list(args.run_json or [])
    input_dirs = args.input_dir
    if input_dirs is None:
        input_dirs = [] if args.run_json else DEFAULT_INPUT_DIRS
    for directory in input_dirs:
        if directory.exists():
            paths.extend(directory.rglob("ats_checkpoint_prediction_run.json"))
    unique = sorted({path.resolve() for path in paths if path.exists()})
    return unique


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
