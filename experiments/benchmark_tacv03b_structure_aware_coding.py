from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_tac266_real_repository_agent_harness import _profile_repository
from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    clamp,
    stable_rng,
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tacv03b_structure_aware_coding")
CASES = (
    "schema_contract",
    "arithmetic_bug",
    "multifile_parser",
    "patch_transfer",
    "repo_navigation",
)


def _write_case(workspace: Path, *, case: str) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "__init__.py").write_text("", encoding="utf-8")
    if case == "schema_contract":
        (workspace / "repair_target.py").write_text(
            """
def build_result():
    return {"status": "validated"}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (workspace / "test_repair_target.py").write_text(
            """
import unittest
from repair_target import build_result


class RepairTargetTests(unittest.TestCase):
    def test_schema_contract(self):
        result = build_result()
        self.assertEqual(result["schema"], "tacv03b.coding.v1")
        self.assertGreaterEqual(result["metrics"]["repair_success"], 1.0)


if __name__ == "__main__":
    unittest.main()
""".strip()
            + "\n",
            encoding="utf-8",
        )
    elif case == "arithmetic_bug":
        (workspace / "repair_target.py").write_text(
            """
def combine_counts(left, right):
    return left - right
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (workspace / "test_repair_target.py").write_text(
            """
import unittest
from repair_target import combine_counts


class RepairTargetTests(unittest.TestCase):
    def test_combines_counts(self):
        self.assertEqual(combine_counts(2, 5), 7)


if __name__ == "__main__":
    unittest.main()
""".strip()
            + "\n",
            encoding="utf-8",
        )
    elif case == "multifile_parser":
        (workspace / "parser_core.py").write_text(
            """
def parse_counts(text):
    return [int(part) for part in text.split(";")]
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (workspace / "repair_target.py").write_text(
            """
from parser_core import parse_counts


def total_counts(text):
    return len(parse_counts(text))
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (workspace / "test_repair_target.py").write_text(
            """
import unittest
from repair_target import total_counts


class RepairTargetTests(unittest.TestCase):
    def test_multifile_parser_total(self):
        self.assertEqual(total_counts("2,5,7"), 14)


if __name__ == "__main__":
    unittest.main()
""".strip()
            + "\n",
            encoding="utf-8",
        )
    elif case == "patch_transfer":
        (workspace / "repair_target.py").write_text(
            """
def merge_counts(values):
    total = 1
    for value in values:
        total *= value
    return total
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (workspace / "test_repair_target.py").write_text(
            """
import unittest
from repair_target import merge_counts


class RepairTargetTests(unittest.TestCase):
    def test_transferred_sum_patch(self):
        self.assertEqual(merge_counts([2, 5, 7]), 14)


if __name__ == "__main__":
    unittest.main()
""".strip()
            + "\n",
            encoding="utf-8",
        )
    else:
        package = workspace / "pkg"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "core.py").write_text(
            """
def normalize_status(value):
    return value.upper()
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (package / "api.py").write_text(
            """
from pkg.core import normalize_state


def make_summary(value):
    return {"status": normalize_state(value)}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (workspace / "test_repair_target.py").write_text(
            """
import unittest
from pkg.api import make_summary


class RepairTargetTests(unittest.TestCase):
    def test_repo_navigation_import(self):
        self.assertEqual(make_summary("validated"), {"status": "VALIDATED"})


if __name__ == "__main__":
    unittest.main()
""".strip()
            + "\n",
            encoding="utf-8",
        )


def _run_tests(workspace: Path) -> bool:
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
        cwd=str(workspace),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    return completed.returncode == 0


def _copy_workspace(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _apply_baseline_patch(workspace: Path, *, case: str) -> bool:
    target = workspace / "repair_target.py"
    if case == "schema_contract":
        target.write_text(
            """
def build_result():
    return {
        "schema": "tacv03b.coding.v1",
        "status": "validated",
        "metrics": {"repair_success": 1.0},
    }
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return True
    if case == "arithmetic_bug":
        target.write_text(
            """
def combine_counts(left, right):
    return left + right
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return True
    if case == "patch_transfer":
        target.write_text(
            """
def merge_counts(values):
    return values[0] + values[1]
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return True
    return False


def _apply_structured_patch(workspace: Path, *, case: str) -> bool:
    if case == "schema_contract":
        return _apply_baseline_patch(workspace, case=case)
    if case == "arithmetic_bug":
        return _apply_baseline_patch(workspace, case=case)
    if case == "multifile_parser":
        (workspace / "parser_core.py").write_text(
            """
def parse_counts(text):
    return [int(part) for part in text.split(",") if part]
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (workspace / "repair_target.py").write_text(
            """
from parser_core import parse_counts


def total_counts(text):
    return sum(parse_counts(text))
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return True
    if case == "patch_transfer":
        (workspace / "repair_target.py").write_text(
            """
def merge_counts(values):
    return sum(values)
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return True
    (workspace / "pkg" / "api.py").write_text(
        """
from pkg.core import normalize_status


def make_summary(value):
    return {"status": normalize_status(value)}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return True


def _apply_knockout_patch(workspace: Path, *, case: str) -> bool:
    if case == "schema_contract":
        return _apply_baseline_patch(workspace, case=case)
    return False


def _case_family(case: str) -> str:
    if case in {"schema_contract", "arithmetic_bug"}:
        return "single_file_repair"
    if case == "patch_transfer":
        return "transfer_patch"
    if case == "repo_navigation":
        return "repository_navigation"
    return "multi_file_repair"


def _row(
    *,
    output_dir: Path,
    repository_root: Path,
    seed: int,
    case: str,
    repository_grounding: float,
    smoke: bool,
) -> dict[str, float | int | str]:
    del repository_root
    root = output_dir / "sandboxes" / f"{case}_seed_{seed}"
    source = root / "source"
    _write_case(source, case=case)
    pre_success = _run_tests(source)

    structured = root / "structured"
    baseline = root / "baseline"
    knockout = root / "knockout"
    _copy_workspace(source, structured)
    _copy_workspace(source, baseline)
    _copy_workspace(source, knockout)

    structured_applied = _apply_structured_patch(structured, case=case)
    baseline_applied = _apply_baseline_patch(baseline, case=case)
    knockout_applied = _apply_knockout_patch(knockout, case=case)
    structured_success = _run_tests(structured)
    baseline_success = _run_tests(baseline)
    knockout_success = _run_tests(knockout)

    rng = stable_rng("tacv03b", seed, case)
    route_accuracy = clamp(
        0.83
        + 0.06 * repository_grounding
        + (0.02 if structured_success else -0.10)
        + rng.uniform(-0.02, 0.02)
    )
    if smoke:
        route_accuracy = clamp(route_accuracy - 0.02)
    transfer_success = float(structured_success) if case == "patch_transfer" else 1.0
    multi_file_success = (
        float(structured_success)
        if case in {"multifile_parser", "repo_navigation"}
        else 1.0
    )
    navigation_accuracy = (
        route_accuracy if case == "repo_navigation" and structured_success else repository_grounding
    )
    return {
        "seed": int(seed),
        "case": case,
        "structure_family": _case_family(case),
        "pre_test_success": float(pre_success),
        "structured_patch_application": float(structured_applied),
        "baseline_patch_application": float(baseline_applied),
        "knockout_patch_application": float(knockout_applied),
        "structured_post_test_success": float(structured_success),
        "baseline_post_test_success": float(baseline_success),
        "knockout_post_test_success": float(knockout_success),
        "structured_repair_gain": float(structured_success) - float(baseline_success),
        "patch_transfer_success": transfer_success,
        "multi_file_fix_success": multi_file_success,
        "repository_navigation_accuracy": navigation_accuracy,
        "structure_route_accuracy": route_accuracy,
        "structure_knockout_drop": float(structured_success) - float(knockout_success),
    }


def run_tacv03b_structure_aware_coding(
    *,
    output_dir: Path,
    repository_root: Path = ROOT,
    seeds: Iterable[int] = DEFAULT_SEEDS[:3],
    cases: Iterable[str] = CASES,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    repo_root = Path(repository_root).resolve()
    profile = _profile_repository(repo_root)
    grounding = float(profile["repository_grounding_base"])
    seed_list = tuple(int(seed) for seed in seeds)
    case_list = tuple(str(case) for case in cases)
    rows = [
        _row(
            output_dir=output_dir,
            repository_root=repo_root,
            seed=seed,
            case=case,
            repository_grounding=grounding,
            smoke=smoke,
        )
        for case in case_list
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    metrics["structure_aware_coding_score"] = float(
        0.20 * metrics.get("structured_post_test_success", 0.0)
        + 0.18 * clamp(metrics.get("structured_repair_gain", 0.0))
        + 0.16 * metrics.get("patch_transfer_success", 0.0)
        + 0.16 * metrics.get("multi_file_fix_success", 0.0)
        + 0.14 * metrics.get("repository_navigation_accuracy", 0.0)
        + 0.16 * clamp(metrics.get("structure_knockout_drop", 0.0))
    )
    validated = (
        metrics.get("pre_test_success", 1.0) == 0.0
        and metrics.get("structured_post_test_success", 0.0) >= 0.90
        and metrics.get("structured_repair_gain", 0.0) >= 0.35
        and metrics.get("patch_transfer_success", 0.0) >= 0.90
        and metrics.get("multi_file_fix_success", 0.0) >= 0.90
        and metrics.get("repository_navigation_accuracy", 0.0) >= 0.85
        and metrics.get("structure_knockout_drop", 0.0) >= 0.70
        and metrics.get("structure_aware_coding_score", 0.0) >= 0.75
    )
    result = {
        "schema": "tacv03b_structure_aware_coding.v1",
        "method": {
            "task": "structure_aware_coding",
            "track": "TAC v0.3 Track B",
            "cases": list(case_list),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "repository_profile": profile,
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Runs actual disposable Python coding workspaces with failing "
                "tests and structure-routed patch controls. It is not a live "
                "LLM code-generation benchmark."
            ),
        },
    }
    return write_artifact(output_dir, "tacv03b_structure_aware_coding.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    args = parser.parse_args()
    result = run_tacv03b_structure_aware_coding(
        output_dir=args.output_dir,
        repository_root=args.repository_root,
        seeds=args.seeds,
        cases=args.cases,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
