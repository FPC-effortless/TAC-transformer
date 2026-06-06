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
    audit_tac_control_v1,
    build_phase_b_replication_plan,
    build_phase_c_identity_stability_protocol,
    build_phase_d_benchmark_protocol,
    format_tac_research_plan_markdown,
)


DEFAULT_MANIFEST = Path(
    "runs/kaggle_results/"
    "tac_run5b_program_conditioned_creb_k6_20k_2026_06_04/"
    "best_tac_agentic_run5b_program_conditioned/run_manifest.json"
)
DEFAULT_TAC_SUMMARY = Path(
    "runs/benchmarks/"
    "external_run5b_program_conditioned_creb_k6_step10000_fair_token_validation_2026_06_04/"
    "tac_step10000_summary.json"
)
DEFAULT_EXTERNAL_VALIDATION = Path(
    "runs/benchmarks/"
    "external_run5b_program_conditioned_creb_k6_step10000_fair_token_validation_2026_06_04/"
    "external_validation.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "runs/benchmarks/tac_control_v1_next_stage_2026_06_04"
)
DEFAULT_DOCS_OUTPUT = Path("docs/tac_control_v1_research_contract.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advance TAC Run 5B into a frozen next-stage research contract."
    )
    parser.add_argument("--tac-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tac-summary", type=Path, default=DEFAULT_TAC_SUMMARY)
    parser.add_argument(
        "--external-validation",
        type=Path,
        default=DEFAULT_EXTERNAL_VALIDATION,
        help="Optional external validation JSON carrying specialization evidence.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--docs-output",
        default=str(DEFAULT_DOCS_OUTPUT),
        help="Optional Markdown copy for docs; pass an empty string to skip.",
    )
    parser.add_argument("--phase-b-seeds", type=int, nargs="+", default=[11, 23, 37])
    args = parser.parse_args()

    manifest = _read_json(args.tac_manifest)
    tac_summary = _read_json(args.tac_summary)
    external_validation = (
        _read_json(args.external_validation)
        if args.external_validation and args.external_validation.exists()
        else {}
    )

    phase_a = audit_tac_control_v1(
        manifest,
        tac_summary,
        external_validation=external_validation,
    )
    phase_b = build_phase_b_replication_plan(seeds=args.phase_b_seeds)
    phase_c = build_phase_c_identity_stability_protocol()
    phase_d = build_phase_d_benchmark_protocol()
    markdown = format_tac_research_plan_markdown(phase_a, phase_b, phase_c, phase_d)
    result = {
        "schema": "tac_next_stage_research_contract.v1",
        "inputs": {
            "tac_manifest": str(args.tac_manifest),
            "tac_summary": str(args.tac_summary),
            "external_validation": (
                str(args.external_validation) if args.external_validation else None
            ),
        },
        "phase_a_freeze": phase_a,
        "phase_b_replication": phase_b,
        "phase_c_identity_stability_protocol": phase_c,
        "phase_d_benchmark_protocol": phase_d,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "tac_next_stage_research_contract.json", result)
    _write_json(args.output_dir / "phase_a_freeze.json", phase_a)
    _write_json(args.output_dir / "phase_b_replication_plan.json", phase_b)
    _write_json(args.output_dir / "phase_c_identity_stability_protocol.json", phase_c)
    _write_json(args.output_dir / "phase_d_benchmark_protocol.json", phase_d)
    (args.output_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")

    docs_output = _docs_output_path(args.docs_output)
    if docs_output is not None:
        docs_output.parent.mkdir(parents=True, exist_ok=True)
        docs_output.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "decision": phase_a["decision"],
                "output_dir": str(args.output_dir),
                "docs_output": str(docs_output) if docs_output else None,
            },
            indent=2,
        ),
        flush=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _docs_output_path(value: str | None) -> Path | None:
    if value is None:
        return None
    if not value:
        return None
    return Path(value)


if __name__ == "__main__":
    main()
