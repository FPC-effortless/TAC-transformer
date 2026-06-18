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

from tac_transformer.capability import aggregate_external_run5b_validation


EXPECTED_KERNEL_ID = "jeffkolo/tac-run5b-best-capability-fast-20k-2026-06-06"
EXPECTED_PRESET = "run5b_best_capability_fast"
EXPECTED_CODE_DATASET = "jeffkolo/tac-run5b-best-capability-fast-code-2026-06-06"
EXPECTED_DATA_DATASET = "jeffkolo/tac-run5b-capability-data-2026-06-03"
EXPECTED_RESUME_DATASET = "jeffkolo/tac-run5b-fast-resume-12031-20260606"
EXPECTED_KERNEL_RUN_VERSIONS = (2, 3)
REQUIRED_OUTPUTS = ("final_summary.json", "metrics.jsonl", "best.pt", "last.pt")


def build_run5b_best_capability_external_status(
    *,
    source_dir: Path,
    output_dir: Path,
    kaggle_status: str | None = None,
    same_backbone_final_summary: Path | None = None,
    parameter_matched_final_summary: Path | None = None,
) -> dict[str, Any]:
    source = _summarize_source_pull(source_dir)
    outputs = _summarize_outputs(output_dir)
    evaluation: dict[str, Any] | None = None

    if (
        source["passes"]
        and not outputs["missing_required"]
        and same_backbone_final_summary is not None
        and parameter_matched_final_summary is not None
    ):
        evaluation = aggregate_external_run5b_validation(
            _read_json(Path(outputs["required_paths"]["final_summary.json"])),
            _read_json(same_backbone_final_summary),
            _read_json(parameter_matched_final_summary),
            tac_manifest=_read_optional_json(outputs["run_manifest_path"]),
            specialization_report=_read_optional_json(outputs["specialization_report_path"]),
        )

    decision = _decision(source, outputs, evaluation, kaggle_status)
    return {
        "schema": "tac.run5b_best_capability_external_status.v1",
        "kernel": {
            "id": EXPECTED_KERNEL_ID,
            "expected_preset": EXPECTED_PRESET,
        },
        "decision": decision,
        "source": source,
        "outputs": outputs,
        "evaluation": evaluation,
        "phase_unblocks": _phase_unblocks(decision, evaluation),
    }


def format_run5b_best_capability_external_status_markdown(result: dict[str, Any]) -> str:
    decision = result["decision"]
    source = result["source"]
    outputs = result["outputs"]
    lines = [
        "# External Run 5B Best-Capability Fast Status",
        "",
        f"Decision: `{decision['status']}`",
        "",
        f"Reason: {decision['reason']}",
        "",
        "## Source",
        "",
        f"- Kernel run version: `{source.get('kernel_run_version')}`",
        f"- Preset: `{source.get('preset')}`",
        f"- Source passes: `{source.get('passes')}`",
        "",
        "## Outputs",
        "",
        f"- Required artifacts present: `{not outputs['missing_required']}`",
        f"- Missing required: `{', '.join(outputs['missing_required']) or 'none'}`",
        f"- Specialization report: `{outputs.get('specialization_report_path') or 'missing'}`",
        "",
        "## Phase Unblocks",
        "",
    ]
    for key, value in result["phase_unblocks"].items():
        lines.append(f"- {key}: `{value}`")
    if decision["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in decision["blockers"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor Run 5B best-capability fast external validation artifacts."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--kaggle-status", default=None)
    parser.add_argument("--same-backbone-final-summary", type=Path)
    parser.add_argument("--parameter-matched-final-summary", type=Path)
    args = parser.parse_args()

    status = build_run5b_best_capability_external_status(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        kaggle_status=args.kaggle_status,
        same_backbone_final_summary=args.same_backbone_final_summary,
        parameter_matched_final_summary=args.parameter_matched_final_summary,
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "external_run5b_best_capability_status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    (args.artifact_dir / "RESULTS.md").write_text(
        format_run5b_best_capability_external_status_markdown(status),
        encoding="utf-8",
    )
    print(json.dumps({"decision": status["decision"]}, indent=2), flush=True)


def _summarize_source_pull(source_dir: Path) -> dict[str, Any]:
    metadata = _read_optional_json(source_dir / "kernel-metadata.json") or {}
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(source_dir.glob("*.py"))
    )
    kernel_run_version = _extract_int(source_text, r"kernel_run_version[\"']?\s*[:=]\s*(\d+)")
    preset = EXPECTED_PRESET if EXPECTED_PRESET in source_text else None
    flags = {
        "aux_loss_cadence_4": (
            "--aux-loss-cadence" in source_text and _has_adjacent_value(source_text, "aux-loss-cadence", "4")
        ),
        "precision_fp32": (
            "--precision" in source_text and _has_adjacent_value(source_text, "precision", "fp32")
        ),
        "min_healthy_gradient_norm": "--min-healthy-gradient-norm" in source_text
        and "1e-12" in source_text,
        "fail_on_unhealthy_optimization": "--fail-on-unhealthy-optimization" in source_text,
    }
    dataset_sources = set(metadata.get("dataset_sources") or [])
    blockers: list[str] = []
    if metadata.get("id") != EXPECTED_KERNEL_ID:
        blockers.append("kernel metadata id does not match expected Run 5B best-capability kernel")
    if EXPECTED_CODE_DATASET not in dataset_sources:
        blockers.append("expected code dataset is missing from kernel metadata")
    if EXPECTED_DATA_DATASET not in dataset_sources:
        blockers.append("expected data dataset is missing from kernel metadata")
    if kernel_run_version not in EXPECTED_KERNEL_RUN_VERSIONS:
        blockers.append("source script does not confirm an expected kernel_run_version")
    if kernel_run_version == 3 and EXPECTED_RESUME_DATASET not in dataset_sources:
        blockers.append("resumed v3 source is missing the expected resume dataset")
    if preset != EXPECTED_PRESET:
        blockers.append("source script does not confirm run5b_best_capability_fast preset")
    for flag, passes in flags.items():
        if not passes:
            blockers.append(f"source script missing expected flag: {flag}")
    return {
        "source_dir": str(source_dir),
        "passes": not blockers,
        "blockers": blockers,
        "metadata_id": metadata.get("id"),
        "machine_shape": metadata.get("machine_shape"),
        "dataset_sources": sorted(dataset_sources),
        "kernel_run_version": kernel_run_version,
        "preset": preset,
        "flags": flags,
    }


def _summarize_outputs(output_dir: Path) -> dict[str, Any]:
    required_paths = {
        name: str(path)
        for name in REQUIRED_OUTPUTS
        if (path := _find_file(output_dir, name)) is not None
    }
    missing = [name for name in REQUIRED_OUTPUTS if name not in required_paths]
    specialization = _find_specialization_report(output_dir)
    run_manifest = _find_file(output_dir, "run_manifest.json")
    return {
        "output_dir": str(output_dir),
        "required_paths": required_paths,
        "missing_required": missing,
        "run_manifest_path": str(run_manifest) if run_manifest else None,
        "specialization_report_path": str(specialization) if specialization else None,
        "file_count": sum(1 for _ in output_dir.rglob("*")) if output_dir.exists() else 0,
    }


def _decision(
    source: dict[str, Any],
    outputs: dict[str, Any],
    evaluation: dict[str, Any] | None,
    kaggle_status: str | None,
) -> dict[str, Any]:
    blockers = list(source["blockers"])
    status_unavailable = bool(kaggle_status and "500" in kaggle_status)
    if blockers:
        return {
            "status": "blocked",
            "reason": "; ".join(blockers),
            "blockers": blockers,
            "capability_claim_allowed": False,
        }
    if outputs["missing_required"]:
        reason = "required completed output artifacts are missing"
        if status_unavailable:
            reason = "Kaggle status unavailable and required completed output artifacts are missing"
        return {
            "status": "external_pending",
            "reason": reason,
            "blockers": [f"missing {name}" for name in outputs["missing_required"]],
            "capability_claim_allowed": False,
        }
    if evaluation is None:
        return {
            "status": "outputs_ready",
            "reason": "completed outputs are present but fair-baseline validation has not run",
            "blockers": [],
            "capability_claim_allowed": False,
        }
    return {
        "status": evaluation["decision"]["status"],
        "reason": evaluation["decision"]["reason"],
        "blockers": list(evaluation["decision"].get("hard_blockers") or []),
        "capability_claim_allowed": evaluation["decision"]["status"] == "promote",
    }


def _phase_unblocks(decision: dict[str, Any], evaluation: dict[str, Any] | None) -> dict[str, str]:
    if decision["status"] == "external_pending":
        return {
            "phase_b": "still_blocked_external_pending",
            "phase_d": "still_blocked_external_pending",
            "ats_checkpoint_scoring": "still_blocked_external_pending",
            "long_horizon_checkpoint_validation": "still_blocked_external_pending",
        }
    if evaluation is None or decision["status"] != "promote":
        return {
            "phase_b": "requires_iteration",
            "phase_d": "still_blocked",
            "ats_checkpoint_scoring": "still_blocked",
            "long_horizon_checkpoint_validation": "still_blocked",
        }
    return {
        "phase_b": "candidate_unblocked_for_seed_replication_audit",
        "phase_d": "candidate_unblocked_pending_phase_b_gate",
        "ats_checkpoint_scoring": "candidate_unblocked",
        "long_horizon_checkpoint_validation": "candidate_unblocked",
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return _read_json(resolved)


def _find_file(root: Path, name: str) -> Path | None:
    if not root.exists():
        return None
    direct = root / name
    if direct.exists():
        return direct
    return next(root.rglob(name), None)


def _find_specialization_report(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.rglob("*.json")
        if "specialization" in path.name.lower() or "knockout" in path.name.lower()
    ]
    return sorted(candidates)[0] if candidates else None


def _extract_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _has_adjacent_value(text: str, flag_name: str, expected: str) -> bool:
    pattern = rf"--{re.escape(flag_name)}[\"']?\s*,\s*[\"']{re.escape(expected)}[\"']"
    return re.search(pattern, text) is not None


if __name__ == "__main__":
    main()
