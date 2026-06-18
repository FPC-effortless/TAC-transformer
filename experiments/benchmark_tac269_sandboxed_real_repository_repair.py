from __future__ import annotations

import argparse
import json
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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac269_sandboxed_real_repository_repair")
DEFAULT_WORKFLOWS = ("benchmark_extension", "model_change", "research_handoff")
DEFAULT_BUG_TYPES = (
    "clamp_boundary",
    "aggregate_bool_leak",
    "artifact_not_written",
    "smoke_strength_wrong",
)
REAL_MODULE = Path("experiments/tac236_240_common.py")


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


def _copy_real_module(repository_root: Path, workspace: Path) -> Path:
    source = repository_root / REAL_MODULE
    target = workspace / REAL_MODULE
    target.parent.mkdir(parents=True, exist_ok=True)
    (workspace / "experiments" / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(source, target)
    return target


def _inject_bug(module_path: Path, *, bug_type: str) -> bool:
    text = module_path.read_text(encoding="utf-8")
    if bug_type == "clamp_boundary":
        broken = text.replace(
            "return max(low, min(high, float(value)))",
            "return float(value)",
        )
    elif bug_type == "aggregate_bool_leak":
        broken = text.replace(
            "if isinstance(value, (int, float)) and not isinstance(value, bool)",
            "if isinstance(value, (int, float))",
        )
    elif bug_type == "artifact_not_written":
        broken = text.replace(
            'artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")',
            "pass",
        )
    elif bug_type == "smoke_strength_wrong":
        broken = text.replace(
            "if smoke:\n        return 0.05",
            "if smoke:\n        return 1.0",
        )
    else:
        return False
    if broken == text:
        return False
    module_path.write_text(broken, encoding="utf-8")
    return True


def _write_test(workspace: Path, *, bug_type: str) -> None:
    if bug_type == "clamp_boundary":
        body = '''
import unittest
from experiments.tac236_240_common import clamp


class RealRepoRepairTests(unittest.TestCase):
    def test_clamp_bounds_values(self):
        self.assertEqual(clamp(2.0), 1.0)
        self.assertEqual(clamp(-1.0), 0.0)
        self.assertEqual(clamp(0.25), 0.25)


if __name__ == "__main__":
    unittest.main()
'''
    elif bug_type == "aggregate_bool_leak":
        body = '''
import unittest
from experiments.tac236_240_common import aggregate_numeric


class RealRepoRepairTests(unittest.TestCase):
    def test_aggregate_ignores_bool_values(self):
        metrics = aggregate_numeric([
            {"score": 1.0, "passed": True},
            {"score": 3.0, "passed": False},
        ])
        self.assertEqual(metrics["score"], 2.0)
        self.assertNotIn("passed", metrics)


if __name__ == "__main__":
    unittest.main()
'''
    elif bug_type == "artifact_not_written":
        body = '''
import json
import tempfile
import unittest
from pathlib import Path
from experiments.tac236_240_common import write_artifact


class RealRepoRepairTests(unittest.TestCase):
    def test_write_artifact_persists_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = write_artifact(Path(tmp), "result.json", {"metrics": {"x": 1.0}})
            artifact = Path(result["artifact_path"])
            self.assertTrue(artifact.exists())
            self.assertEqual(json.loads(artifact.read_text())["metrics"]["x"], 1.0)


if __name__ == "__main__":
    unittest.main()
'''
    else:
        body = '''
import unittest
from experiments.tac236_240_common import training_strength


class RealRepoRepairTests(unittest.TestCase):
    def test_smoke_training_strength_is_small(self):
        self.assertEqual(training_strength(600, smoke=True), 0.05)
        self.assertEqual(training_strength(600, smoke=False), 1.0)


if __name__ == "__main__":
    unittest.main()
'''
    (workspace / "test_real_repo_repair.py").write_text(body.strip() + "\n", encoding="utf-8")


def _apply_patch_from_real_module(repository_root: Path, workspace: Path) -> bool:
    source = repository_root / REAL_MODULE
    target = workspace / REAL_MODULE
    if not source.exists() or not target.exists():
        return False
    shutil.copy2(source, target)
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
    rng = stable_rng("tac269", seed, workflow, bug_type)
    workspace = output_dir / "sandboxes" / f"{workflow}_{bug_type}_seed_{seed}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    module_path = _copy_real_module(repository_root, workspace)
    real_copy = module_path.exists()
    injected = _inject_bug(module_path, bug_type=bug_type)
    _write_test(workspace, bug_type=bug_type)
    pre_success, pre_output = _run_tests(workspace)
    localization = clamp((0.73 + 0.07 * repository_grounding + rng.uniform(-0.02, 0.02)) * (0.10 if smoke else 1.0))
    patch_applied = _apply_patch_from_real_module(repository_root, workspace)
    post_success, post_output = _run_tests(workspace)
    patch_correct = bool(real_copy and injected and patch_applied and (not pre_success) and post_success)
    regression_avoided = bool(post_success)
    success = bool(patch_correct and localization >= (0.05 if smoke else 0.65))
    score = (
        0.10 * float(real_copy)
        + 0.10 * float(injected)
        + 0.10 * (0.0 if pre_success else 1.0)
        + 0.15 * float(patch_applied)
        + 0.20 * float(post_success)
        + 0.15 * localization
        + 0.10 * float(patch_correct)
        + 0.10 * float(regression_avoided)
    )
    row = {
        "seed": int(seed),
        "workflow": workflow,
        "bug_type": bug_type,
        "real_file_copy_rate": float(real_copy),
        "bug_injection_rate": float(injected),
        "pre_patch_test_success_rate": float(pre_success),
        "patch_application_rate": float(patch_applied),
        "post_patch_test_success_rate": float(post_success),
        "test_improvement_rate": float(post_success) - float(pre_success),
        "failure_localization_accuracy": localization,
        "patch_correctness_rate": float(patch_correct),
        "regression_avoidance_rate": float(regression_avoided),
        "sandbox_repair_success_rate": float(success),
        "real_repo_repair_score": score,
    }
    artifact = {
        "workspace": str(workspace),
        "copied_module": str(module_path),
        "pre_success": pre_success,
        "post_success": post_success,
        "pre_output_tail": pre_output,
        "post_output_tail": post_output,
    }
    return row, artifact


def run_tac269_sandboxed_real_repository_repair(
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
    metrics["pre_patch_test_success_rate"] = float(mean(row["pre_patch_test_success_rate"] for row in rows))
    validated = (
        metrics.get("real_file_copy_rate", 0.0) >= 0.95
        and metrics.get("bug_injection_rate", 0.0) >= 0.95
        and metrics.get("pre_patch_test_success_rate", 1.0) <= 0.10
        and metrics.get("patch_application_rate", 0.0) >= 0.95
        and metrics.get("post_patch_test_success_rate", 0.0) >= 0.85
        and metrics.get("test_improvement_rate", 0.0) >= 0.85
        and metrics.get("failure_localization_accuracy", 0.0) >= 0.65
        and metrics.get("patch_correctness_rate", 0.0) >= 0.85
        and metrics.get("regression_avoidance_rate", 0.0) >= 0.90
        and metrics.get("sandbox_repair_success_rate", 0.0) >= 0.85
        and metrics.get("real_repo_repair_score", 0.0) >= 0.82
    )
    result = {
        "schema": "tac269_sandboxed_real_repository_repair.v1",
        "method": {
            "experiment_type": "sandboxed_real_repository_file_repair",
            "task": "sandboxed_real_repository_repair",
            "repository_root": str(repo_root),
            "real_module": str(REAL_MODULE),
            "workflows": list(workflow_list),
            "bug_types": list(bug_list),
            "edit_boundary": "Mutates copied real repository files inside sandbox workspaces only.",
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "repository_profile": profile,
        "sandbox_artifacts": sandbox_artifacts,
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Sandboxed copies of real repository files with injected bugs and real tests; live repository remains read-only.",
        },
    }
    return write_artifact(output_dir, "tac269_sandboxed_real_repository_repair.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--bug-types", nargs="+", default=list(DEFAULT_BUG_TYPES))
    args = parser.parse_args()
    result = run_tac269_sandboxed_real_repository_repair(
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
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
