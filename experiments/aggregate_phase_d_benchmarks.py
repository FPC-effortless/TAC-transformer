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
    aggregate_phase_d_benchmark_results,
    format_phase_d_benchmark_results_markdown,
)


DEFAULT_PHASE_B_JSON = Path(
    "runs/benchmarks/tac_control_v1_phase_b_2026_06_04/phase_b_seed_results.json"
)
DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac_control_v1_phase_d_2026_06_04")
DEFAULT_ROWS_DIR = Path(
    "runs/benchmarks/tac_control_v1_phase_d_predictions_2026_06_04"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate TAC-Control-v1 Phase D benchmark rows."
    )
    parser.add_argument("--rows", type=Path, nargs="*", default=None)
    parser.add_argument("--rows-dir", type=Path, default=DEFAULT_ROWS_DIR)
    parser.add_argument("--phase-b-json", type=Path, default=DEFAULT_PHASE_B_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    phase_b = _read_json(args.phase_b_json) if args.phase_b_json.exists() else {}
    row_sources = (
        discover_phase_d_row_sources(args.rows_dir)
        if args.rows is None
        else list(args.rows)
    )
    rows = []
    for path in row_sources:
        rows.extend(_read_rows(path))

    result = aggregate_phase_d_benchmark_results(
        rows,
        phase_b_decision=phase_b.get("decision", {}),
    )
    result["phase_b_source"] = str(args.phase_b_json)
    result["row_sources"] = [str(path) for path in row_sources]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_d_benchmark_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_phase_d_benchmark_results_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2), flush=True)


def discover_phase_d_row_sources(rows_dir: Path = DEFAULT_ROWS_DIR) -> list[Path]:
    """Discover scored Phase D row files produced by the matrix runner."""

    if not rows_dir.exists():
        return []
    candidates = []
    combined = rows_dir / "phase_d_benchmark_rows.jsonl"
    if combined.exists():
        candidates.append(combined)
    candidates.extend(sorted(rows_dir.rglob("*score*.jsonl")))
    seen = set()
    unique = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    payload = _read_json(path)
    if isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload.get("benchmarks"), list):
        return [row for row in payload["benchmarks"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


if __name__ == "__main__":
    main()
