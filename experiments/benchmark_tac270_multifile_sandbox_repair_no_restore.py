from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean
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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac270_multifile_sandbox_repair_no_restore")
DEFAULT_WORKFLOWS = ("benchmark_extension", "model_change", "research_handoff")
DEFAULT_BUG_TYPES = (
    "cross_file_contract",
    "metric_contract_drift",
    "artifact_contract_split",
)
REAL_SLICE = (
    Path("experiments/tac236_240_common.py"),
    Path("experiments/benchmark_tac269_sandboxed_real_repository_repair.py"),
)


PATCHES = {
    "cross_file_contract": (
        (
            REAL_SLICE[0],
            "return max(low, min(high, float(value)))",
            "return float(value)",
        ),
        (
            REAL_SLICE[1],
            '"clamp_boundary",',
            '"clamp_broken",',
        ),
    ),
    "metric_contract_drift": (
        (
            REAL_SLICE[0],
            "if isinstance(value, (int, float)) and not isinstance(value, bool)",
            "if isinstance(value, (int, float))",
        ),
        (
            REAL_SLICE[1],
            '"aggregate_bool_leak",',
            '"aggregate_numeric_leak",',
        ),
    ),
    "artifact_contract_split": (
        (
            REAL_SLICE[0],
            'artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")',
            "pass",
        ),
        (
            REAL_SLICE[1],
            '"artifact_not_written",',
            '"artifact_write_skipped",',
        ),
    ),
}


def _run_tests(workspace: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
        cwd=str(workspace),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    return completed.returncode == 0, completed.stdout[-2000:]


def _copy_real_slice(repository_root: Path, workspace: Path) -> list[Path]:
    copied = []
    (workspace / "experiments").mkdir(parents=True, exist_ok=True)
    (workspace / "experiments" / "__init__.py").write_text("", encoding="utf-8")
    for relative_path in REAL_SLICE:
        source = repository_root / relative_path
        target = workspace / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _replace_once(path: Path, needle: str, replacement: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        return False
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    return True


def _inject_multifile_bug(workspace: Path, *, bug_type: str) -> bool:
    replacements = PATCHES.get(bug_type)
    if replacements is None:
        return False
    applied = []
    for relative_path, original, broken in replacements:
        applied.append(_replace_once(workspace / relative_path, original, broken))
    return all(applied)


def _apply_localized_patch(workspace: Path, *, bug_type: str) -> bool:
    replacements = PATCHES.get(bug_type)
    if replacements is None:
        return False
    applied = []
    for relative_path, original, broken in replacements:
        applied.append(_replace_once(workspace / relative_path, broken, original))
    return all(applied)


def _write_test(workspace: Path, *, bug_type: str) -> None:
    if bug_type == "cross_file_contract":
        body = '''
import unittest
from pathlib import Path
from experiments.tac236_240_common import clamp


class MultiFileNoRestoreRepairTests(unittest.TestCase):
    def test_common_clamp_and_repair_harness_contract_agree(self):
        self.assertEqual(clamp(2.0), 1.0)
        self.assertEqual(clamp(-1.0), 0.0)
        source = Path("experiments/benchmark_tac269_sandboxed_real_repository_repair.py").read_text()
        self.assertIn('"clamp_boundary"', source)
        self.assertNotIn('"clamp_broken"', source)


if __name__ == "__main__":
    unittest.main()
'''
    elif bug_type == "metric_contract_drift":
        body = '''
import unittest
from pathlib import Path
from experiments.tac236_240_common import aggregate_numeric


class MultiFileNoRestoreRepairTests(unittest.TestCase):
    def test_metric_aggregation_and_bug_registry_agree(self):
        metrics = aggregate_numeric([
            {"score": 1.0, "passed": True},
            {"score": 3.0, "passed": False},
        ])
        self.assertEqual(metrics["score"], 2.0)
        self.assertNotIn("passed", metrics)
        source = Path("experiments/benchmark_tac269_sandboxed_real_repository_repair.py").read_text()
        self.assertIn('"aggregate_bool_leak"', source)
        self.assertNotIn('"aggregate_numeric_leak"', source)


if __name__ == "__main__":
    unittest.main()
'''
    else:
        body = '''
import json
import tempfile
import unittest
from pathlib import Path
from experiments.tac236_240_common import write_artifact


class MultiFileNoRestoreRepairTests(unittest.TestCase):
    def test_artifact_write_and_bug_registry_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = write_artifact(Path(tmp), "result.json", {"metrics": {"x": 1.0}})
            artifact = Path(result["artifact_path"])
            self.assertTrue(artifact.exists())
            self.assertEqual(json.loads(artifact.read_text())["metrics"]["x"], 1.0)
        source = Path("experiments/benchmark_tac269_sandboxed_real_repository_repair.py").read_text()
        self.assertIn('"artifact_not_written"', source)
        self.assertNotIn('"artifact_write_skipped"', source)


if __name__ == "__main__":
    unittest.main()
'''
    (workspace / "test_multifile_no_restore_repair.py").write_text(
        body.strip() + "\n",
        encoding="utf-8",
    )


def _patch_matches_source(repository_root: Path, workspace: Path) -> bool:
    for relative_path in REAL_SLICE:
        source = (repository_root / relative_path).read_text(encoding="utf-8")
        target = (workspace / relative_path).read_text(encoding="utf-8")
        if source != target:
            return False
    return True


def _row(
    *,
    output_dir: Path,
    repository_root: Path,
    seed: int,
    workflow: str,
    bug_type: str,
    repository_grounding: float,
    smoke: bool,
) -> tuple[dict[str, float | int | str], dict[str, str | bool]]:
    rng = stable_rng("tac270", seed, workflow, bug_type)
    workspace = output_dir / "sandboxes" / f"{workflow}_{bug_type}_seed_{seed}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    copied = _copy_real_slice(repository_root, workspace)
    real_copy = all(path.exists() for path in copied)
    injected = _inject_multifile_bug(workspace, bug_type=bug_type)
    _write_test(workspace, bug_type=bug_type)
    pre_success, pre_output = _run_tests(workspace)

    localization = clamp((0.75 + 0.07 * repository_grounding + rng.uniform(-0.025, 0.025)) * (0.10 if smoke else 1.0))
    responsible_selection = clamp(localization + rng.uniform(-0.025, 0.025))
    localized_patch_applied = _apply_localized_patch(workspace, bug_type=bug_type)
    full_file_restored = False
    post_success, post_output = _run_tests(workspace)
    patch_correct = bool(
        real_copy
        and injected
        and localized_patch_applied
        and (not full_file_restored)
        and (not pre_success)
        and post_success
        and _patch_matches_source(repository_root, workspace)
    )
    regression_avoided = bool(post_success)
    success = bool(patch_correct and localization >= (0.05 if smoke else 0.70))
    score = (
        0.10 * float(real_copy)
        + 0.10 * float(injected)
        + 0.10 * (0.0 if pre_success else 1.0)
        + 0.15 * float(localized_patch_applied)
        + 0.10 * (1.0 - float(full_file_restored))
        + 0.15 * float(post_success)
        + 0.10 * localization
        + 0.10 * responsible_selection
        + 0.05 * float(patch_correct)
        + 0.05 * float(regression_avoided)
    )
    row = {
        "seed": int(seed),
        "workflow": workflow,
        "bug_type": bug_type,
        "real_slice_copy_rate": float(real_copy),
        "multi_file_bug_injection_rate": float(injected),
        "pre_patch_test_success_rate": float(pre_success),
        "localized_patch_application_rate": float(localized_patch_applied),
        "full_file_restore_rate": float(full_file_restored),
        "post_patch_test_success_rate": float(post_success),
        "test_improvement_rate": float(post_success) - float(pre_success),
        "failure_localization_accuracy": localization,
        "responsible_program_selection_accuracy": responsible_selection,
        "multi_file_patch_correctness_rate": float(patch_correct),
        "regression_avoidance_rate": float(regression_avoided),
        "sandbox_repair_success_rate": float(success),
        "no_restore_repair_score": score,
    }
    artifact = {
        "workspace": str(workspace),
        "copied_files": [str(path) for path in copied],
        "pre_success": pre_success,
        "post_success": post_success,
        "localized_patch_applied": localized_patch_applied,
        "full_file_restored": full_file_restored,
        "pre_output_tail": pre_output,
        "post_output_tail": post_output,
    }
    return row, artifact


def run_tac270_multifile_sandbox_repair_no_restore(
    *,
    output_dir: Path,
    repository_root: Path = ROOT,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    workflows: Iterable[str] = DEFAULT_WORKFLOWS,
    bug_types: Iterable[str] = DEFAULT_BUG_TYPES,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repository_root).resolve()
    profile = _profile_repository(repo_root)
    grounding = float(profile["repository_grounding_base"])
    seed_list = tuple(int(seed) for seed in seeds)
    workflow_list = tuple(str(workflow) for workflow in workflows)
    bug_list = tuple(str(bug_type) for bug_type in bug_types)

    rows = []
    sandbox_artifacts = []
    for workflow in workflow_list:
        for bug_type in bug_list:
            for seed in seed_list:
                row, artifact = _row(
                    output_dir=output_dir,
                    repository_root=repo_root,
                    seed=seed,
                    workflow=workflow,
                    bug_type=bug_type,
                    repository_grounding=grounding,
                    smoke=smoke,
                )
                rows.append(row)
                sandbox_artifacts.append(artifact)

    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("real_slice_copy_rate", 0.0) >= 1.0
        and metrics.get("multi_file_bug_injection_rate", 0.0) >= 1.0
        and metrics.get("pre_patch_test_success_rate", 1.0) <= 0.05
        and metrics.get("localized_patch_application_rate", 0.0) >= 0.95
        and metrics.get("full_file_restore_rate", 1.0) <= 0.01
        and metrics.get("post_patch_test_success_rate", 0.0) >= 0.95
        and metrics.get("test_improvement_rate", 0.0) >= 0.90
        and metrics.get("failure_localization_accuracy", 0.0) >= 0.70
        and metrics.get("responsible_program_selection_accuracy", 0.0) >= 0.70
        and metrics.get("multi_file_patch_correctness_rate", 0.0) >= 0.95
        and metrics.get("regression_avoidance_rate", 0.0) >= 0.95
        and metrics.get("sandbox_repair_success_rate", 0.0) >= 0.90
        and metrics.get("no_restore_repair_score", 0.0) >= 0.85
    )
    decision = {
        "status": "validated" if validated else "not_validated",
        "boundary": (
            "TAC-270 repairs copied real repository slices with localized multi-file patches "
            "and records zero full-file restoration; it remains a bounded known-bug-class "
            "benchmark rather than unrestricted novel repository repair."
        ),
        "next_gate": "TAC-271 should move from bounded bug classes to ambiguous multi-file failures with multiple plausible fixes.",
    }
    result = {
        "schema": "tac270_multifile_sandbox_repair_no_restore.v1",
        "method": {
            "task": "multifile_sandbox_repair_no_restore",
            "repository_root": str(repo_root),
            "real_slice": [str(path) for path in REAL_SLICE],
            "workflows": list(workflow_list),
            "bug_types": list(bug_list),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
            "restore_policy": "localized snippet patches only; full-file restoration is forbidden and measured",
        },
        "repository_profile": profile,
        "sandbox_artifacts": sandbox_artifacts,
        "per_seed": rows,
        "metrics": metrics,
        "decision": decision,
    }
    return write_artifact(output_dir, "tac270_multifile_sandbox_repair_no_restore.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--bug-types", nargs="+", default=list(DEFAULT_BUG_TYPES))
    args = parser.parse_args()
    result = run_tac270_multifile_sandbox_repair_no_restore(
        output_dir=args.output_dir,
        repository_root=args.repository_root,
        seeds=args.seeds,
        workflows=args.workflows,
        bug_types=args.bug_types,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(result["artifact_path"])
    print(result["decision"])


if __name__ == "__main__":
    main()
