from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_TARGETS = (
    "kaggle/benchmark_tac_scm_real006.py",
    "kaggle/benchmark_tac_scm_real008.py",
    "kaggle/benchmark_tac_scm_real008b.py",
    "kaggle/benchmark_tac_scm_real009.py",
    "kaggle/benchmark_tac_scm_real010.py",
    "kaggle/benchmark_tac_scm_real010_real.py",
)


def audit_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    scripted_decision = bool(re.search(r"def\s+_decision\s*\(", text))
    hard_coded_rates = bool(re.search(r"rates\s*=\s*\{[^}]*tac_scm", text, re.DOTALL))
    variant_name_selector = bool(re.search(r"if\s+variant\s*(?:==|in)\s*", text))
    constructs_tac_config_only = "tac_scm_v02_config(" in text and not re.search(r"TACTransformerLM|forward\(|model\(", text)
    executable_patch_verification = "unittest" in text and ("subprocess.run" in text or "TextTestRunner" in text)
    invalid_for_model_claim = scripted_decision or hard_coded_rates or (variant_name_selector and constructs_tac_config_only)
    return {
        "path": str(path),
        "scripted_decision_function": scripted_decision,
        "hard_coded_tac_rates": hard_coded_rates,
        "variant_name_selector": variant_name_selector,
        "constructs_tac_config_without_model_inference": constructs_tac_config_only,
        "executable_patch_verification": executable_patch_verification,
        "invalid_for_tac_model_advantage_claim": invalid_for_model_claim,
    }


def run_integrity_audit(paths: Iterable[Path]) -> dict[str, Any]:
    reports = [audit_file(path) for path in paths if path.exists()]
    return {
        "audited_files": len(reports),
        "invalid_for_tac_model_advantage_claim": [row["path"] for row in reports if row["invalid_for_tac_model_advantage_claim"]],
        "reports": reports,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit TAC-SCM benchmarks for scripted outcome selectors")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = args.paths or [Path(path) for path in DEFAULT_TARGETS]
    result = run_integrity_audit(paths)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
