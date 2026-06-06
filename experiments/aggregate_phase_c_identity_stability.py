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
    aggregate_phase_c_identity_stability_results,
    format_phase_c_identity_stability_markdown,
    summarize_phase_c_identity_seed,
)


DEFAULT_INPUT_DIR = Path("runs/kaggle_results/tac_control_v1_phase_b_2026_06_04")
DEFAULT_PHASE_B_JSON = Path(
    "runs/benchmarks/tac_control_v1_phase_b_2026_06_04/phase_b_seed_results.json"
)
DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac_control_v1_phase_c_2026_06_04")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate TAC-Control-v1 Phase C identity-stability evidence."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--phase-b-json", type=Path, default=DEFAULT_PHASE_B_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    phase_b = _read_json(args.phase_b_json) if args.phase_b_json.exists() else {}
    seed_rows = []
    for seed_dir in _candidate_seed_dirs(args.input_dir):
        report_path = _read_optional_specialization_report_path(seed_dir)
        if report_path is None:
            continue
        seed_rows.append(
            summarize_phase_c_identity_seed(
                seed=_seed_from_path(seed_dir),
                specialization_report=_read_json(report_path),
            )
        )

    result = aggregate_phase_c_identity_stability_results(
        seed_rows,
        phase_b_decision=phase_b.get("decision", {}),
    )
    result["phase_b_source"] = str(args.phase_b_json)
    result["input_dir"] = str(args.input_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_c_identity_stability.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_phase_c_identity_stability_markdown(result),
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
        [path.parent for path in sorted(input_dir.rglob("program_specialization.json"))]
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


def _seed_dir_score(path: Path) -> tuple[int, str]:
    final_summary_path = next(iter(sorted(path.rglob("final_summary.json"))), None)
    if final_summary_path is None:
        return (-1, str(path))
    final_summary = _read_json(final_summary_path)
    try:
        completed_steps = int(final_summary.get("completed_steps", 0))
    except (TypeError, ValueError):
        completed_steps = 0
    return (completed_steps, str(path))


def _seed_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.match(r"^seed_(\d+)(?:\D.*)?$", part)
        if match:
            return int(match.group(1))
    return 0


def _read_optional_specialization_report_path(seed_dir: Path) -> Path | None:
    reports = sorted(seed_dir.rglob("program_specialization.json"))
    for report in reports:
        if "step_010000" in str(report).replace("\\", "/"):
            return report
    return reports[-1] if reports else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
