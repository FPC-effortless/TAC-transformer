from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    aggregate_phase_b_seed_results,
    format_phase_b_seed_results_markdown,
    summarize_phase_b_seed_result,
)


DEFAULT_INPUT_DIR = Path("runs/kaggle_results/tac_control_v1_phase_b_2026_06_04")
DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac_control_v1_phase_b_2026_06_04")
DEFAULT_STATUS_JSON = (
    DEFAULT_INPUT_DIR / "phase_b_kaggle_status.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate pulled TAC-Control-v1 Phase B seed outputs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    args = parser.parse_args()

    status = _read_json(args.status_json) if args.status_json.exists() else {}
    seed_rows = []
    for seed_dir in _candidate_seed_dirs(args.input_dir):
        final_summary_path = _find_first(seed_dir, "final_summary.json")
        if final_summary_path is None:
            continue
        run_dir = final_summary_path.parent
        seed_rows.append(
            summarize_phase_b_seed_result(
                seed=_seed_from_path(seed_dir),
                final_summary=_read_json(final_summary_path),
                metrics_rows=_read_jsonl(run_dir / "metrics.jsonl"),
                run_manifest=(
                    _read_json(run_dir / "run_manifest.json")
                    if (run_dir / "run_manifest.json").exists()
                    else None
                ),
                specialization_report=_read_optional_specialization_report(run_dir),
            )
        )

    if seed_rows:
        result = aggregate_phase_b_seed_results(seed_rows)
    else:
        result = {
            "phase": "B",
            "decision": {
                "status": "pending",
                "reason": "No completed Phase B seed final_summary.json artifacts were found.",
                "ready_for_phase_d": False,
                "required_pass_count": 2,
            },
            "summary": {
                "seed_count": 0,
                "passed_seed_count": 0,
                "failed_seed_count": 0,
                "pending_seed_count": 0,
                "seeds": [],
            },
            "aggregates": {},
            "seeds": [],
        }
    result["status"] = status

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_b_seed_results.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_phase_b_seed_results_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2), flush=True)


def _candidate_seed_dirs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    direct = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_dir() and path.name.startswith("seed_")
    ]
    if direct:
        return _dedupe_seed_dirs(direct)
    return _dedupe_seed_dirs(
        [path.parent for path in sorted(input_dir.rglob("final_summary.json"))]
    )


def _dedupe_seed_dirs(paths: list[Path]) -> list[Path]:
    by_seed: dict[int, Path] = {}
    for path in paths:
        seed = _seed_from_path(path)
        if seed <= 0:
            continue
        current = by_seed.get(seed)
        if current is None or _seed_dir_score(path) > _seed_dir_score(current):
            by_seed[seed] = path
    return [by_seed[seed] for seed in sorted(by_seed)]


def _seed_dir_score(path: Path) -> tuple[int, int, str]:
    final_summary_path = _find_first(path, "final_summary.json")
    if final_summary_path is None:
        return (-1, 0, str(path))
    final_summary = _read_json(final_summary_path)
    completed_steps = _coerce_int(final_summary.get("completed_steps"))
    completed_target = int(
        completed_steps >= _coerce_int(final_summary.get("target_steps"))
        and _coerce_int(final_summary.get("target_steps")) > 0
    )
    return (completed_steps, completed_target, str(path))


def _seed_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.match(r"^seed_(\d+)(?:\D.*)?$", part)
        if match:
            return int(match.group(1))
    return 0


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _find_first(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _read_optional_specialization_report(run_dir: Path) -> dict[str, Any] | None:
    reports = sorted(run_dir.rglob("program_specialization.json"))
    for report in reports:
        if "step_010000" in str(report).replace("\\", "/"):
            return _read_json(report)
    if reports:
        return _read_json(reports[-1])
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
