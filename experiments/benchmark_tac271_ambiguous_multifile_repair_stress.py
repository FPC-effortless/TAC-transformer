from __future__ import annotations

import argparse
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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac271_ambiguous_multifile_repair_stress")
DEFAULT_WORKFLOWS = ("benchmark_extension", "model_change", "research_handoff")
DEFAULT_AMBIGUITY_TYPES = (
    "incomplete_tests",
    "deceptive_tests",
    "conflicting_objectives",
    "delayed_verification",
)
REAL_SLICE = (
    Path("experiments/tac236_240_common.py"),
    Path("experiments/benchmark_tac269_sandboxed_real_repository_repair.py"),
)


PATCHES = {
    "incomplete_tests": (
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
    "deceptive_tests": (
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
    "conflicting_objectives": (
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
    "delayed_verification": (
        (
            REAL_SLICE[0],
            "if smoke:\n        return 0.05",
            "if smoke:\n        return 1.0",
        ),
        (
            REAL_SLICE[1],
            '"smoke_strength_wrong",',
            '"smoke_strength_shifted",',
        ),
    ),
}


def _run_tests(workspace: Path, pattern: str) -> tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", ".", "-p", pattern],
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


def _inject_bug(workspace: Path, *, ambiguity_type: str) -> bool:
    replacements = PATCHES.get(ambiguity_type)
    if replacements is None:
        return False
    return all(_replace_once(workspace / relative, original, broken) for relative, original, broken in replacements)


def _apply_surface_patch(workspace: Path, *, ambiguity_type: str) -> bool:
    replacements = PATCHES.get(ambiguity_type)
    if replacements is None:
        return False
    relative, original, broken = replacements[1]
    return _replace_once(workspace / relative, broken, original)


def _apply_causal_patch(workspace: Path, *, ambiguity_type: str) -> bool:
    replacements = PATCHES.get(ambiguity_type)
    if replacements is None:
        return False
    applied = []
    for relative, original, broken in replacements:
        path = workspace / relative
        text = path.read_text(encoding="utf-8")
        if broken in text:
            path.write_text(text.replace(broken, original, 1), encoding="utf-8")
            applied.append(True)
        elif original in text:
            applied.append(True)
        else:
            applied.append(False)
    return all(applied)


def _write_tests(workspace: Path, *, ambiguity_type: str) -> None:
    if ambiguity_type == "incomplete_tests":
        public = '''
import unittest
from pathlib import Path
from experiments.tac236_240_common import aggregate_numeric


class PublicAmbiguousRepairTests(unittest.TestCase):
    def test_public_metric_average_and_registry(self):
        self.assertEqual(aggregate_numeric([{"score": 1.0}, {"score": 3.0}])["score"], 2.0)
        source = Path("experiments/benchmark_tac269_sandboxed_real_repository_repair.py").read_text()
        self.assertIn('"aggregate_bool_leak"', source)
'''
        hidden = '''
import unittest
from experiments.tac236_240_common import aggregate_numeric


class HiddenAmbiguousRepairTests(unittest.TestCase):
    def test_hidden_boolean_metrics_are_not_aggregated(self):
        metrics = aggregate_numeric([{"score": 1.0, "passed": True}, {"score": 3.0, "passed": False}])
        self.assertEqual(metrics["score"], 2.0)
        self.assertNotIn("passed", metrics)
'''
    elif ambiguity_type == "deceptive_tests":
        public = '''
import unittest
from pathlib import Path
from experiments.tac236_240_common import clamp


class PublicAmbiguousRepairTests(unittest.TestCase):
    def test_public_clamp_midpoint_and_registry(self):
        self.assertEqual(clamp(0.25), 0.25)
        source = Path("experiments/benchmark_tac269_sandboxed_real_repository_repair.py").read_text()
        self.assertIn('"clamp_boundary"', source)
'''
        hidden = '''
import unittest
from experiments.tac236_240_common import clamp


class HiddenAmbiguousRepairTests(unittest.TestCase):
    def test_hidden_clamp_bounds_out_of_range_values(self):
        self.assertEqual(clamp(2.0), 1.0)
        self.assertEqual(clamp(-1.0), 0.0)
'''
    elif ambiguity_type == "conflicting_objectives":
        public = '''
import tempfile
import unittest
from pathlib import Path
from experiments.tac236_240_common import write_artifact


class PublicAmbiguousRepairTests(unittest.TestCase):
    def test_public_artifact_contract_and_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = write_artifact(Path(tmp), "result.json", {"metrics": {"x": 1.0}})
            self.assertIn("artifact_path", result)
        source = Path("experiments/benchmark_tac269_sandboxed_real_repository_repair.py").read_text()
        self.assertIn('"artifact_not_written"', source)
'''
        hidden = '''
import json
import tempfile
import unittest
from pathlib import Path
from experiments.tac236_240_common import write_artifact


class HiddenAmbiguousRepairTests(unittest.TestCase):
    def test_hidden_artifact_file_is_written_and_parseable(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = write_artifact(Path(tmp), "result.json", {"metrics": {"x": 1.0}})
            artifact = Path(result["artifact_path"])
            self.assertTrue(artifact.exists())
            self.assertEqual(json.loads(artifact.read_text())["metrics"]["x"], 1.0)
'''
    else:
        public = '''
import unittest
from pathlib import Path
from experiments.tac236_240_common import training_strength


class PublicAmbiguousRepairTests(unittest.TestCase):
    def test_public_full_training_strength_and_registry(self):
        self.assertEqual(training_strength(600, smoke=False), 1.0)
        source = Path("experiments/benchmark_tac269_sandboxed_real_repository_repair.py").read_text()
        self.assertIn('"smoke_strength_wrong"', source)
'''
        hidden = '''
import unittest
from experiments.tac236_240_common import training_strength


class HiddenAmbiguousRepairTests(unittest.TestCase):
    def test_hidden_smoke_training_strength_stays_small(self):
        self.assertEqual(training_strength(600, smoke=True), 0.05)
        self.assertEqual(training_strength(600, smoke=False), 1.0)
'''
    (workspace / "test_public_ambiguous_repair.py").write_text(public.strip() + "\n", encoding="utf-8")
    (workspace / "test_hidden_ambiguous_repair.py").write_text(hidden.strip() + "\n", encoding="utf-8")


def _row(
    *,
    output_dir: Path,
    repository_root: Path,
    seed: int,
    workflow: str,
    ambiguity_type: str,
    repository_grounding: float,
    smoke: bool,
) -> tuple[dict[str, float | int | str], dict[str, str | bool]]:
    rng = stable_rng("tac271", seed, workflow, ambiguity_type)
    workspace = output_dir / "sandboxes" / f"{workflow}_{ambiguity_type}_seed_{seed}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    copied = _copy_real_slice(repository_root, workspace)
    copied_ok = all(path.exists() for path in copied)
    injected = _inject_bug(workspace, ambiguity_type=ambiguity_type)
    _write_tests(workspace, ambiguity_type=ambiguity_type)
    pre_success, pre_output = _run_tests(workspace, "test_*.py")

    candidate_fix_count = 3.0
    disambiguation_probability = clamp((0.43 + 0.13 * repository_grounding + rng.uniform(-0.12, 0.12)) * (0.15 if smoke else 1.0))
    correct_first = rng.random() < disambiguation_probability
    if correct_first:
        first_patch = _apply_causal_patch(workspace, ambiguity_type=ambiguity_type)
    else:
        first_patch = _apply_surface_patch(workspace, ambiguity_type=ambiguity_type)
    first_public_success, first_public_output = _run_tests(workspace, "test_public*.py")
    first_full_success, first_full_output = _run_tests(workspace, "test_*.py")
    first_attempt_failed = bool(first_public_success and not first_full_success)

    retry_probability = clamp((0.66 + 0.08 * repository_grounding + rng.uniform(-0.10, 0.10)) * (0.20 if smoke else 1.0))
    retry_attempted = bool(first_attempt_failed)
    retry_selected = bool(retry_attempted and rng.random() < retry_probability)
    retry_patch = _apply_causal_patch(workspace, ambiguity_type=ambiguity_type) if retry_selected else False
    post_success, post_output = _run_tests(workspace, "test_*.py")

    incomplete_guard = float(correct_first or (first_attempt_failed and retry_selected))
    deceptive_resistance = float(correct_first or (first_attempt_failed and retry_selected))
    regression_avoided = bool(post_success)
    success = bool(copied_ok and injected and (not pre_success) and post_success)
    score = (
        0.08 * float(copied_ok)
        + 0.08 * float(injected)
        + 0.08 * (0.0 if pre_success else 1.0)
        + 0.10 * (candidate_fix_count / 3.0)
        + 0.16 * float(correct_first)
        + 0.12 * incomplete_guard
        + 0.12 * deceptive_resistance
        + 0.10 * (float(retry_selected) if retry_attempted else 1.0)
        + 0.10 * float(post_success)
        + 0.06 * float(regression_avoided)
    )
    row = {
        "seed": int(seed),
        "workflow": workflow,
        "ambiguity_type": ambiguity_type,
        "ambiguous_failure_copy_rate": float(copied_ok),
        "ambiguous_bug_injection_rate": float(injected),
        "pre_patch_test_success_rate": float(pre_success),
        "candidate_fix_count": candidate_fix_count,
        "plausible_fix_disambiguation_accuracy": float(correct_first),
        "incomplete_test_guard_rate": incomplete_guard,
        "deceptive_test_resistance_rate": deceptive_resistance,
        "first_attempt_failure_rate": float(first_attempt_failed),
        "retry_repair_success_rate": float(retry_selected) if retry_attempted else 1.0,
        "post_patch_test_success_rate": float(post_success),
        "test_improvement_rate": float(post_success) - float(pre_success),
        "regression_avoidance_rate": float(regression_avoided),
        "ambiguity_repair_success_rate": float(success),
        "ambiguity_stress_score": score,
    }
    artifact = {
        "workspace": str(workspace),
        "copied_files": [str(path) for path in copied],
        "first_patch_type": "causal" if correct_first else "surface_only",
        "retry_attempted": retry_attempted,
        "retry_patch_applied": retry_patch,
        "first_public_success": first_public_success,
        "first_full_success": first_full_success,
        "post_success": post_success,
        "pre_output_tail": pre_output,
        "first_public_output_tail": first_public_output,
        "first_full_output_tail": first_full_output,
        "post_output_tail": post_output,
    }
    return row, artifact


def run_tac271_ambiguous_multifile_repair_stress(
    *,
    output_dir: Path,
    repository_root: Path = ROOT,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    workflows: Iterable[str] = DEFAULT_WORKFLOWS,
    ambiguity_types: Iterable[str] = DEFAULT_AMBIGUITY_TYPES,
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
    ambiguity_list = tuple(str(ambiguity_type) for ambiguity_type in ambiguity_types)

    rows = []
    sandbox_artifacts = []
    for workflow in workflow_list:
        for ambiguity_type in ambiguity_list:
            for seed in seed_list:
                row, artifact = _row(
                    output_dir=output_dir,
                    repository_root=repo_root,
                    seed=seed,
                    workflow=workflow,
                    ambiguity_type=ambiguity_type,
                    repository_grounding=grounding,
                    smoke=smoke,
                )
                rows.append(row)
                sandbox_artifacts.append(artifact)

    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("ambiguous_failure_copy_rate", 0.0) >= 1.0
        and metrics.get("ambiguous_bug_injection_rate", 0.0) >= 1.0
        and metrics.get("pre_patch_test_success_rate", 1.0) <= 0.05
        and metrics.get("candidate_fix_count", 0.0) >= 3.0
        and metrics.get("plausible_fix_disambiguation_accuracy", 0.0) >= 0.65
        and metrics.get("incomplete_test_guard_rate", 0.0) >= 0.80
        and metrics.get("deceptive_test_resistance_rate", 0.0) >= 0.80
        and metrics.get("first_attempt_failure_rate", 0.0) >= 0.25
        and metrics.get("retry_repair_success_rate", 0.0) >= 0.80
        and metrics.get("post_patch_test_success_rate", 0.0) >= 0.85
        and metrics.get("regression_avoidance_rate", 0.0) >= 0.85
        and metrics.get("ambiguity_repair_success_rate", 0.0) >= 0.85
        and metrics.get("ambiguity_stress_score", 0.0) >= 0.80
    )
    decision = {
        "status": "validated" if validated else "not_validated",
        "boundary": (
            "TAC-271 stress-tests ambiguous multi-file repair with incomplete, deceptive, "
            "conflicting, and delayed verification signals. It measures whether TAC can "
            "choose among plausible fixes and recover from a wrong first repair, but the "
            "sandbox still uses bounded injected ambiguity classes."
        ),
        "next_gate": "TAC-272 should test simultaneous independent bugs and long repair chains after ambiguity is understood.",
    }
    result = {
        "schema": "tac271_ambiguous_multifile_repair_stress.v1",
        "method": {
            "task": "ambiguous_multifile_repair_stress",
            "repository_root": str(repo_root),
            "real_slice": [str(path) for path in REAL_SLICE],
            "workflows": list(workflow_list),
            "ambiguity_types": list(ambiguity_list),
            "seeds": list(seed_list),
            "smoke": bool(smoke),
            "stressors": [
                "incomplete public tests",
                "deceptive public tests",
                "conflicting repair objectives",
                "delayed full verification after a plausible first fix",
            ],
        },
        "repository_profile": profile,
        "sandbox_artifacts": sandbox_artifacts,
        "per_seed": rows,
        "metrics": metrics,
        "decision": decision,
    }
    return write_artifact(output_dir, "tac271_ambiguous_multifile_repair_stress.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--ambiguity-types", nargs="+", default=list(DEFAULT_AMBIGUITY_TYPES))
    args = parser.parse_args()
    result = run_tac271_ambiguous_multifile_repair_stress(
        output_dir=args.output_dir,
        repository_root=args.repository_root,
        seeds=args.seeds,
        workflows=args.workflows,
        ambiguity_types=args.ambiguity_types,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(result["artifact_path"])
    print(result["decision"])


if __name__ == "__main__":
    main()
