from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_GLOBS = (
    "kaggle/benchmark_real003_structure_to_behavior.py",
    "kaggle/benchmark_tac_scm_real*.py",
    "tacm/experiments/benchmark_tac_scm_ssa*.py",
    "tacm/scm_ssa_research_flow.py",
    "tacm/scm_ssa_trainable.py",
    "experiments/benchmark_memory_advantage_model_version.py",
    "experiments/benchmark_long_horizon_memory_advantage.py",
    "experiments/benchmark_cpu_research_tac_version.py",
    "experiments/benchmark_local_tac_efficiency_matrix.py",
)


def audit_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    scripted_decision = bool(re.search(r"def\s+_decision\s*\(", text))
    hard_coded_rates = bool(re.search(r"rates\s*=\s*\{[^}]*tac_scm", text, re.DOTALL))
    variant_name_selector = bool(re.search(r"if\s+variant\s*(?:==|in)\s*", text))
    constructs_tac_config_only = "tac_scm_v02_config(" in text and not re.search(r"TACTransformerLM|forward\(|model\(", text)
    executable_patch_verification = "unittest" in text and ("subprocess.run" in text or "TextTestRunner" in text)
    model_inference = bool(re.search(r"TACTransformerLM|torch\.nn|optimizer|loss\.backward|model\(", text))
    executable_repair_candidates = "RepairCandidate" in text or "apply_patch" in text or "patches" in text and executable_patch_verification
    synthetic_oracle_fields = bool(re.search(r"oracle|gold|target_slot|hidden_", text))
    deterministic_variant_multipliers = bool(re.search(r"variant.*multiplier|active.*floor|random.*gap|_variant", text, re.IGNORECASE))
    invalid_for_model_claim = scripted_decision or hard_coded_rates or (variant_name_selector and constructs_tac_config_only)
    invalid_for_real_repair_claim = invalid_for_model_claim or (path.name.startswith("benchmark_tac_scm_real") and "repair" in text.lower() and not executable_repair_candidates)
    support_level = _support_level(
        invalid_for_model_claim=invalid_for_model_claim,
        executable_patch_verification=executable_patch_verification,
        model_inference=model_inference,
        synthetic_oracle_fields=synthetic_oracle_fields,
        deterministic_variant_multipliers=deterministic_variant_multipliers,
        path=path,
    )
    return {
        "path": str(path),
        "scripted_decision_function": scripted_decision,
        "hard_coded_tac_rates": hard_coded_rates,
        "variant_name_selector": variant_name_selector,
        "constructs_tac_config_without_model_inference": constructs_tac_config_only,
        "model_or_training_inference_present": model_inference,
        "executable_patch_verification": executable_patch_verification,
        "executable_repair_candidates": executable_repair_candidates,
        "synthetic_or_oracle_fields_present": synthetic_oracle_fields,
        "deterministic_variant_multipliers_or_controls": deterministic_variant_multipliers,
        "invalid_for_tac_model_advantage_claim": invalid_for_model_claim,
        "invalid_for_real_repair_claim": invalid_for_real_repair_claim,
        "support_level": support_level,
    }


def run_integrity_audit(paths: Iterable[Path]) -> dict[str, Any]:
    reports = sorted((audit_file(path) for path in paths if path.exists()), key=lambda row: row["path"])
    return {
        "audited_files": len(reports),
        "invalid_for_tac_model_advantage_claim": [row["path"] for row in reports if row["invalid_for_tac_model_advantage_claim"]],
        "invalid_for_real_repair_claim": [row["path"] for row in reports if row["invalid_for_real_repair_claim"]],
        "support_level_counts": _support_level_counts(reports),
        "reports": reports,
    }


def discover_default_targets(root: Path = Path(".")) -> list[Path]:
    paths: set[Path] = set()
    for pattern in DEFAULT_GLOBS:
        paths.update(root.glob(pattern))
    return sorted(paths)


def _support_level(
    *,
    invalid_for_model_claim: bool,
    executable_patch_verification: bool,
    model_inference: bool,
    synthetic_oracle_fields: bool,
    deterministic_variant_multipliers: bool,
    path: Path,
) -> str:
    if invalid_for_model_claim:
        return "invalid_for_claim"
    if path.name.endswith("_real.py") and executable_patch_verification:
        return "valid_harness_no_tac_advantage"
    if model_inference and not deterministic_variant_multipliers:
        return "candidate_model_evidence_requires_manual_review"
    if executable_patch_verification:
        return "executable_harness_requires_manual_review"
    if synthetic_oracle_fields or deterministic_variant_multipliers:
        return "synthetic_mechanism_only"
    return "requires_manual_review"


def _support_level_counts(reports: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in reports:
        level = row["support_level"]
        counts[level] = counts.get(level, 0) + 1
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit TAC-SCM benchmarks for scripted outcome selectors")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = args.paths or discover_default_targets()
    result = run_integrity_audit(paths)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
