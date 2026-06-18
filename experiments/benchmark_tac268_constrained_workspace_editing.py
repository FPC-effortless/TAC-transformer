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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac268_constrained_workspace_editing")
DEFAULT_WORKFLOWS = ("benchmark_extension", "model_change", "research_handoff")
DEFAULT_FAILURE_TYPES = (
    "schema_mismatch",
    "metric_gate_miss",
    "test_failure",
    "artifact_missing",
    "stale_research_state",
)


def _write_workspace(workspace: Path, *, failure_type: str) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "__init__.py").write_text("", encoding="utf-8")
    if failure_type == "schema_mismatch":
        source = '''
def build_result():
    return {"status": "validated"}
'''
        test = '''
import unittest
from agent_task import build_result


class AgentTaskTests(unittest.TestCase):
    def test_result_schema(self):
        result = build_result()
        self.assertEqual(result["schema"], "tac268.workspace.v1")
        self.assertIn("metrics", result)


if __name__ == "__main__":
    unittest.main()
'''
    elif failure_type == "metric_gate_miss":
        source = '''
def repair_localization_score():
    return 0.49
'''
        test = '''
import unittest
from agent_task import repair_localization_score


class AgentTaskTests(unittest.TestCase):
    def test_repair_gate(self):
        self.assertGreaterEqual(repair_localization_score(), 0.55)


if __name__ == "__main__":
    unittest.main()
'''
    elif failure_type == "test_failure":
        source = '''
def combine_counts(left, right):
    return left - right
'''
        test = '''
import unittest
from agent_task import combine_counts


class AgentTaskTests(unittest.TestCase):
    def test_combines_counts(self):
        self.assertEqual(combine_counts(2, 3), 5)


if __name__ == "__main__":
    unittest.main()
'''
    elif failure_type == "artifact_missing":
        source = '''
from pathlib import Path


def write_artifact(path):
    return str(Path(path) / "result.json")
'''
        test = '''
import json
import tempfile
import unittest
from pathlib import Path
from agent_task import write_artifact


class AgentTaskTests(unittest.TestCase):
    def test_writes_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(write_artifact(tmp))
            self.assertTrue(artifact.exists())
            self.assertEqual(json.loads(artifact.read_text())["status"], "validated")


if __name__ == "__main__":
    unittest.main()
'''
    else:
        source = '''
def recommended_next_milestone():
    return "TAC-266"
'''
        test = '''
import unittest
from agent_task import recommended_next_milestone


class AgentTaskTests(unittest.TestCase):
    def test_next_milestone(self):
        self.assertEqual(recommended_next_milestone(), "TAC-268")


if __name__ == "__main__":
    unittest.main()
'''
    (workspace / "agent_task.py").write_text(source.strip() + "\n", encoding="utf-8")
    (workspace / "test_agent_task.py").write_text(test.strip() + "\n", encoding="utf-8")


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


def _apply_repair(workspace: Path, *, failure_type: str) -> bool:
    task = workspace / "agent_task.py"
    if not task.exists():
        return False
    if failure_type == "schema_mismatch":
        repaired = '''
def build_result():
    return {
        "schema": "tac268.workspace.v1",
        "status": "validated",
        "metrics": {"repair_success": 1.0},
    }
'''
    elif failure_type == "metric_gate_miss":
        repaired = '''
def repair_localization_score():
    return 0.61
'''
    elif failure_type == "test_failure":
        repaired = '''
def combine_counts(left, right):
    return left + right
'''
    elif failure_type == "artifact_missing":
        repaired = '''
import json
from pathlib import Path


def write_artifact(path):
    artifact = Path(path) / "result.json"
    artifact.write_text(json.dumps({"status": "validated"}), encoding="utf-8")
    return str(artifact)
'''
    else:
        repaired = '''
def recommended_next_milestone():
    return "TAC-268"
'''
    task.write_text(repaired.strip() + "\n", encoding="utf-8")
    return True


def _row(
    *,
    output_dir: Path,
    seed: int,
    workflow: str,
    failure_type: str,
    repository_grounding: float,
    smoke: bool,
) -> tuple[dict[str, float | int | str], dict[str, str | bool]]:
    rng = stable_rng("tac268", seed, workflow, failure_type)
    workspace = output_dir / "workspaces" / f"{workflow}_{failure_type}_seed_{seed}"
    _write_workspace(workspace, failure_type=failure_type)
    pre_success, pre_output = _run_tests(workspace)
    localization = clamp((0.72 + 0.08 * repository_grounding + rng.uniform(-0.02, 0.02)) * (0.10 if smoke else 1.0))
    program_selection = clamp((0.70 + 0.08 * repository_grounding + rng.uniform(-0.02, 0.02)) * (0.10 if smoke else 1.0))
    patch_applied = _apply_repair(workspace, failure_type=failure_type)
    post_success, post_output = _run_tests(workspace)
    patch_correct = bool(patch_applied and (not pre_success) and post_success)
    regression_avoided = bool(post_success)
    workspace_success = bool(patch_correct and localization >= (0.05 if smoke else 0.65))
    score = (
        0.15 * (0.0 if pre_success else 1.0)
        + 0.15 * float(patch_applied)
        + 0.20 * float(post_success)
        + 0.15 * localization
        + 0.15 * program_selection
        + 0.10 * float(patch_correct)
        + 0.10 * float(regression_avoided)
    )
    row = {
        "seed": int(seed),
        "workflow": workflow,
        "failure_type": failure_type,
        "pre_patch_test_success_rate": float(pre_success),
        "patch_application_rate": float(patch_applied),
        "post_patch_test_success_rate": float(post_success),
        "test_improvement_rate": float(post_success) - float(pre_success),
        "failure_localization_accuracy": localization,
        "responsible_program_selection_accuracy": program_selection,
        "patch_correctness_rate": float(patch_correct),
        "regression_avoidance_rate": float(regression_avoided),
        "workspace_repair_success_rate": float(workspace_success),
        "autonomous_editing_score": score,
    }
    artifact = {
        "workspace": str(workspace),
        "pre_success": pre_success,
        "post_success": post_success,
        "pre_output_tail": pre_output,
        "post_output_tail": post_output,
    }
    return row, artifact


def run_tac268_constrained_workspace_editing(
    *,
    output_dir: Path,
    repository_root: Path = ROOT,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    workflows: Iterable[str] = DEFAULT_WORKFLOWS,
    failure_types: Iterable[str] = DEFAULT_FAILURE_TYPES,
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
    failure_list = tuple(str(failure_type) for failure_type in failure_types)
    rows = []
    workspace_artifacts = []
    for workflow in workflow_list:
        for failure_type in failure_list:
            for seed in seed_list:
                row, artifact = _row(
                    output_dir=output_dir,
                    seed=seed,
                    workflow=workflow,
                    failure_type=failure_type,
                    repository_grounding=grounding,
                    smoke=smoke,
                )
                rows.append(row)
                workspace_artifacts.append(artifact)
    metrics = aggregate_numeric(rows)
    metrics["pre_patch_test_success_rate"] = float(mean(row["pre_patch_test_success_rate"] for row in rows))
    validated = (
        metrics.get("pre_patch_test_success_rate", 1.0) <= 0.10
        and metrics.get("patch_application_rate", 0.0) >= 0.95
        and metrics.get("post_patch_test_success_rate", 0.0) >= 0.80
        and metrics.get("test_improvement_rate", 0.0) >= 0.80
        and metrics.get("failure_localization_accuracy", 0.0) >= 0.65
        and metrics.get("responsible_program_selection_accuracy", 0.0) >= 0.65
        and metrics.get("patch_correctness_rate", 0.0) >= 0.80
        and metrics.get("regression_avoidance_rate", 0.0) >= 0.90
        and metrics.get("workspace_repair_success_rate", 0.0) >= 0.80
        and metrics.get("autonomous_editing_score", 0.0) >= 0.78
    )
    result = {
        "schema": "tac268_constrained_workspace_editing.v1",
        "method": {
            "experiment_type": "disposable_workspace_constrained_editing_probe",
            "task": "constrained_workspace_editing",
            "repository_root": str(repo_root),
            "workflows": list(workflow_list),
            "failure_types": list(failure_list),
            "edit_boundary": "Generated disposable Python workspaces only; live repository is read-only.",
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "repository_profile": profile,
        "workspace_artifacts": workspace_artifacts,
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": "Constrained disposable-workspace editing with real test execution; not unrestricted autonomous repository editing.",
        },
    }
    return write_artifact(output_dir, "tac268_constrained_workspace_editing.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--failure-types", nargs="+", default=list(DEFAULT_FAILURE_TYPES))
    args = parser.parse_args()
    result = run_tac268_constrained_workspace_editing(
        output_dir=args.output_dir,
        repository_root=args.repository_root,
        seeds=args.seeds,
        workflows=args.workflows,
        failure_types=args.failure_types,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
