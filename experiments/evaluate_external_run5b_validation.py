from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.capability import (
    aggregate_external_run5b_validation,
    format_external_run5b_validation_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an external Run 5B TAC artifact against vanilla gates."
    )
    parser.add_argument("--tac-final-summary", type=Path, required=True)
    parser.add_argument("--same-backbone-final-summary", type=Path, required=True)
    parser.add_argument("--parameter-matched-final-summary", type=Path, required=True)
    parser.add_argument("--tac-manifest", type=Path)
    parser.add_argument("--specialization-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-same-backbone-loss-gap", type=float, default=0.15)
    parser.add_argument("--max-parameter-matched-loss-gap", type=float, default=0.25)
    parser.add_argument("--max-program-memory-cosine", type=float, default=0.85)
    args = parser.parse_args()

    tac_summary = _read_json(args.tac_final_summary)
    same_backbone = _read_json(args.same_backbone_final_summary)
    parameter_matched = _read_json(args.parameter_matched_final_summary)
    tac_manifest = _read_json(args.tac_manifest) if args.tac_manifest else None
    specialization = (
        _read_json(args.specialization_report) if args.specialization_report else None
    )

    result = aggregate_external_run5b_validation(
        tac_summary,
        same_backbone,
        parameter_matched,
        tac_manifest=tac_manifest,
        specialization_report=specialization,
        max_same_backbone_loss_gap=args.max_same_backbone_loss_gap,
        max_parameter_matched_loss_gap=args.max_parameter_matched_loss_gap,
        max_program_memory_cosine=args.max_program_memory_cosine,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_validation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_external_run5b_validation_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps({"decision": result["decision"]}, indent=2), flush=True)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
