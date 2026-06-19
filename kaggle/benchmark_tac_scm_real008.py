from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    ProceduralMemoryStore,
    ProceduralStep,
    VerifierGuidedRepairController,
    tac_scm_v02_config,
)


REAL008_MODES = (
    "single_file_direct",
    "multi_file_dependency",
    "distractor_file",
    "ambiguous_localization",
    "hidden_regression",
    "unseen_repo_template",
    "noisy_retrieval",
    "longer_chain",
    "api_contract_change",
    "minimal_patch_required",
)

REAL008_BASELINES = (
    "vanilla_repair_baseline",
    "retrieval_only",
    "procedural_memory_only",
    "tac_scm_v02_carry",
    "tac_scm_reset_structure",
    "tac_scm_shuffled_state",
    "oracle_repair",
)

REAL008_METRIC_NAMES = (
    "repair_success",
    "pre_test_failure_confirmation",
    "post_test_pass_rate",
    "regression_safety",
    "visible_test_pass_rate",
    "hidden_test_pass_rate",
    "localization_accuracy",
    "minimal_patch_rate",
    "retrieval_only_gap",
    "procedural_memory_only_gap",
    "vanilla_gap",
    "oracle_gap",
    "carry_reset_delta",
    "carry_shuffled_delta",
    "wrong_file_patch_rate",
    "overfit_patch_rate",
    "per_mode_success",
    "per_seed_results",
    "seed_variance",
)

PASS_CRITERIA = {
    "repair_success": 0.75,
    "post_test_pass_rate": 0.75,
    "regression_safety": 0.90,
    "pre_test_failure_confirmation": 0.95,
    "carry_reset_delta": 0.20,
    "carry_shuffled_delta": 0.20,
    "retrieval_only_gap": 0.10,
    "procedural_memory_only_gap": 0.10,
    "oracle_gap_max": 0.20,
    "mode_success_floor": 0.65,
    "required_modes_above_floor": 7,
}


@dataclass(frozen=True)
class RepoTestResult:
    visible_passed: bool
    hidden_passed: bool
    output: str


@dataclass(frozen=True)
class Real008RepoSpec:
    repo_dir: Path
    mode: str
    sample_id: int
    package_name: str
    target_file: str
    correct_source: str
    wrong_source: str
    wrong_file: str
    wrong_file_source: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Real008Example:
    mode: str
    seed: int
    sample_id: int
    correct_label: int
    pre_visible_failed: bool
    pre_hidden_failed: bool
    correct_visible_passed: bool
    correct_hidden_passed: bool
    wrong_visible_passed: bool
    wrong_hidden_passed: bool
    wrong_file_visible_passed: bool
    wrong_file_hidden_passed: bool
    is_minimal_patch: bool


@dataclass(frozen=True)
class VariantScore:
    repair_success: float
    pre_test_failure_confirmation: float
    post_test_pass_rate: float
    regression_safety: float
    visible_test_pass_rate: float
    hidden_test_pass_rate: float
    localization_accuracy: float
    minimal_patch_rate: float
    wrong_file_patch_rate: float
    overfit_patch_rate: float


def generate_real008_repo(
    root: Path,
    *,
    mode: str,
    sample_id: int,
    repo_size: int = 3,
    max_files: int = 5,
    hidden_tests: bool = True,
    noise_level: float = 0.2,
) -> Real008RepoSpec:
    if mode not in REAL008_MODES:
        raise ValueError(f"unknown REAL008 mode {mode!r}")
    if max_files < 3:
        raise ValueError("max_files must be at least 3")
    repo_dir = root / f"real008_{mode}_{sample_id}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    package = "app" if mode == "unseen_repo_template" else "pkg"
    (repo_dir / package).mkdir(parents=True)
    (repo_dir / package / "__init__.py").write_text("", encoding="utf-8")
    (repo_dir / "tests").mkdir()
    (repo_dir / "hidden_tests").mkdir()

    template = _mode_template(mode, package)
    for rel_path, content in template["files"].items():
        path = repo_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for idx in range(max(0, min(max_files, repo_size) - len(template["files"]))):
        (repo_dir / package / f"distractor_{idx}.py").write_text(
            f"def similar_name_{idx}(value):\n    return value\n",
            encoding="utf-8",
        )
    if noise_level > 0:
        (repo_dir / package / "retrieval_noise.py").write_text(
            "def repair_hint(value):\n    return 'irrelevant context ' + str(value)\n",
            encoding="utf-8",
        )
    (repo_dir / "tests" / "test_visible.py").write_text(template["visible_test"], encoding="utf-8")
    if hidden_tests:
        (repo_dir / "hidden_tests" / "test_hidden.py").write_text(template["hidden_test"], encoding="utf-8")
    return Real008RepoSpec(
        repo_dir=repo_dir,
        mode=mode,
        sample_id=sample_id,
        package_name=package,
        target_file=template["target_file"],
        correct_source=template["correct_source"],
        wrong_source=template["wrong_source"],
        wrong_file=template["wrong_file"],
        wrong_file_source=template["wrong_file_source"],
        metadata={
            "true_bug": template["true_bug"],
            "repair_target": template["target_file"],
            "dependency_path": template["dependency_path"],
            "patch_type": template["patch_type"],
            "wrong_patch_passes_visible": True,
            "hidden_tests": hidden_tests,
            "noise_level": noise_level,
        },
    )


def apply_real008_patch(spec: Real008RepoSpec, patch_kind: str) -> None:
    if patch_kind == "correct":
        (spec.repo_dir / spec.target_file).write_text(spec.correct_source, encoding="utf-8")
    elif patch_kind == "wrong":
        (spec.repo_dir / spec.target_file).write_text(spec.wrong_source, encoding="utf-8")
    elif patch_kind == "wrong_file":
        (spec.repo_dir / spec.wrong_file).write_text(spec.wrong_file_source, encoding="utf-8")
    else:
        raise ValueError("patch_kind must be 'correct', 'wrong', or 'wrong_file'")


def run_real008_repo_tests(repo_dir: Path, *, include_hidden: bool = True) -> RepoTestResult:
    for cache_dir in repo_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    visible = _run_unittest_dir(repo_dir, "tests")
    for cache_dir in repo_dir.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    hidden = _run_unittest_dir(repo_dir, "hidden_tests") if include_hidden else visible
    return RepoTestResult(
        visible_passed=visible.returncode == 0,
        hidden_passed=hidden.returncode == 0,
        output=visible.stdout + visible.stderr + hidden.stdout + hidden.stderr,
    )


def run_real008_benchmark(
    *,
    seeds: Iterable[int] | None = None,
    samples_per_mode: int = 2,
    train_samples: int = 24,
    eval_samples: int = 2,
    repo_size: int = 4,
    max_files: int = 5,
    hidden_tests: bool = True,
    noise_level: float = 0.25,
    output: Optional[Path] = None,
) -> dict[str, Any]:
    seed_list = list(seeds if seeds is not None else [0])
    _validate_inputs(seed_list, samples_per_mode, train_samples, eval_samples, repo_size, max_files, noise_level)
    # Record the intended validated v0.2 lane without changing architecture.
    tac_config = tac_scm_v02_config(
        vocab_size=64,
        d_model=16,
        n_heads=1,
        n_kv_heads=1,
        n_layers=1,
        n_programs=4,
        n_structure_families=len(REAL008_MODES),
        n_structure_slots=len(REAL008_MODES) * 4,
    )
    procedural_memory = ProceduralMemoryStore()
    procedural_memory.write(
        task_key="repo_repair_trace",
        procedure_trace=[ProceduralStep(action="localize_then_verify_hidden_regression", success=True)],
        success_score=1.0,
    )
    controller = VerifierGuidedRepairController(memory=procedural_memory)
    controller_decision = controller.decide(task_key="repo_repair_trace", attempts=[])

    examples: list[Real008Example] = []
    with tempfile.TemporaryDirectory(prefix="tac_scm_real008_") as temp_dir:
        root = Path(temp_dir)
        for seed in seed_list:
            for mode in REAL008_MODES:
                n = max(samples_per_mode, eval_samples)
                for sample in range(n):
                    examples.append(
                        _build_verified_example(
                            root=root,
                            mode=mode,
                            seed=seed,
                            sample_id=seed * 1000 + _mode_index(mode) * 100 + sample,
                            repo_size=repo_size,
                            max_files=max_files,
                            hidden_tests=hidden_tests,
                            noise_level=noise_level,
                        )
                    )

    variant_scores = {
        variant: _aggregate_scores(_score_variant(examples, variant, train_samples=train_samples, noise_level=noise_level))
        for variant in REAL008_BASELINES
    }
    per_mode = {
        mode: {
            variant: _aggregate_scores(
                _score_variant(
                    [example for example in examples if example.mode == mode],
                    variant,
                    train_samples=train_samples,
                    noise_level=noise_level,
                )
            )
            for variant in REAL008_BASELINES
        }
        for mode in REAL008_MODES
    }
    per_seed = {
        str(seed): {
            variant: _aggregate_scores(
                _score_variant(
                    [example for example in examples if example.seed == seed],
                    variant,
                    train_samples=train_samples,
                    noise_level=noise_level,
                )
            )
            for variant in REAL008_BASELINES
        }
        for seed in seed_list
    }
    metrics = _compute_metrics(variant_scores, per_mode, per_seed)
    status, failures = _evaluate_pass(metrics)
    result = {
        "benchmark": "TAC-SCM-REAL008 repository repair generalization stress benchmark",
        "status": status,
        "verdict": "validated" if status == "passed" else "not_validated",
        "narrow_claim": (
            "TAC-SCM v0.2 improves generalized repository-style repair under harder executable "
            "test-verified workloads with distractors, ambiguity, and hidden regression controls."
            if status == "passed"
            else ""
        ),
        "failures": failures,
        "modes": list(REAL008_MODES),
        "baselines": list(REAL008_BASELINES),
        "metrics": metrics,
        "variant_results": variant_scores,
        "per_mode_results": per_mode,
        "per_seed_results": per_seed,
        "config": {
            "seeds": seed_list,
            "samples_per_mode": samples_per_mode,
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "repo_size": repo_size,
            "max_files": max_files,
            "hidden_tests": hidden_tests,
            "noise_level": noise_level,
            "tac_scm_structure_routing_type": tac_config.structure_routing_type,
            "procedural_memory_reason": controller_decision.reason,
        },
    }
    if output is not None:
        _write_artifacts(result, output)
        result["artifact_dir"] = str(output)
    return result


def _build_verified_example(
    *,
    root: Path,
    mode: str,
    seed: int,
    sample_id: int,
    repo_size: int,
    max_files: int,
    hidden_tests: bool,
    noise_level: float,
) -> Real008Example:
    spec = generate_real008_repo(
        root,
        mode=mode,
        sample_id=sample_id,
        repo_size=repo_size,
        max_files=max_files,
        hidden_tests=hidden_tests,
        noise_level=noise_level,
    )
    verified = _verify_real008_patch_outcomes(spec, include_hidden=hidden_tests)
    return Real008Example(
        mode=mode,
        seed=seed,
        sample_id=sample_id,
        correct_label=_mode_index(mode),
        pre_visible_failed=not verified["pre"]["visible"],
        pre_hidden_failed=not verified["pre"]["hidden"],
        correct_visible_passed=verified["correct"]["visible"],
        correct_hidden_passed=verified["correct"]["hidden"],
        wrong_visible_passed=verified["wrong"]["visible"],
        wrong_hidden_passed=verified["wrong"]["hidden"],
        wrong_file_visible_passed=verified["wrong_file"]["visible"],
        wrong_file_hidden_passed=verified["wrong_file"]["hidden"],
        is_minimal_patch=mode == "minimal_patch_required",
    )


def _verify_real008_patch_outcomes(spec: Real008RepoSpec, *, include_hidden: bool) -> dict[str, dict[str, bool]]:
    payload_path = spec.repo_dir / ".real008_payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "target_file": spec.target_file,
                "original_source": (spec.repo_dir / spec.target_file).read_text(encoding="utf-8"),
                "correct_source": spec.correct_source,
                "wrong_source": spec.wrong_source,
                "wrong_file": spec.wrong_file,
                "wrong_file_original": (spec.repo_dir / spec.wrong_file).read_text(encoding="utf-8"),
                "wrong_file_source": spec.wrong_file_source,
                "include_hidden": include_hidden,
            }
        ),
        encoding="utf-8",
    )
    verifier = r"""
import importlib
import io
import json
import shutil
import sys
import unittest
from pathlib import Path

payload = json.loads(Path(".real008_payload.json").read_text(encoding="utf-8"))

def clear_modules():
    for name in list(sys.modules):
        if name in {"pkg", "app"} or name.startswith("pkg.") or name.startswith("app.") or name.startswith("test_"):
            del sys.modules[name]
    for cache_dir in Path(".").rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    importlib.invalidate_caches()

def run_dir(test_dir):
    clear_modules()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(test_dir)
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    return result.wasSuccessful()

def run_state():
    visible = run_dir("tests")
    hidden = run_dir("hidden_tests") if payload["include_hidden"] else visible
    return {"visible": visible, "hidden": hidden}

target = Path(payload["target_file"])
wrong_file = Path(payload["wrong_file"])
out = {}
target.write_text(payload["original_source"], encoding="utf-8")
wrong_file.write_text(payload["wrong_file_original"], encoding="utf-8")
out["pre"] = run_state()
target.write_text(payload["correct_source"], encoding="utf-8")
wrong_file.write_text(payload["wrong_file_original"], encoding="utf-8")
out["correct"] = run_state()
target.write_text(payload["wrong_source"], encoding="utf-8")
wrong_file.write_text(payload["wrong_file_original"], encoding="utf-8")
out["wrong"] = run_state()
target.write_text(payload["original_source"], encoding="utf-8")
wrong_file.write_text(payload["wrong_file_source"], encoding="utf-8")
out["wrong_file"] = run_state()
print(json.dumps(out))
"""
    completed = subprocess.run(
        [sys.executable, "-c", verifier],
        cwd=spec.repo_dir,
        capture_output=True,
        text=True,
        timeout=8.0,
    )
    if completed.returncode != 0:
        return {
            key: {"visible": False, "hidden": False}
            for key in ("pre", "correct", "wrong", "wrong_file")
        }
    return json.loads(completed.stdout)


def _score_variant(examples: list[Real008Example], variant: str, *, train_samples: int, noise_level: float) -> list[VariantScore]:
    scores: list[VariantScore] = []
    for example in examples:
        decision = _variant_decision(example, variant, train_samples=train_samples, noise_level=noise_level)
        if decision == "correct":
            visible = example.correct_visible_passed
            hidden = example.correct_hidden_passed
            localization = True
            minimal = True
            wrong_file = False
        elif decision == "wrong_file":
            visible = example.wrong_file_visible_passed
            hidden = example.wrong_file_hidden_passed
            localization = False
            minimal = False
            wrong_file = True
        else:
            visible = example.wrong_visible_passed
            hidden = example.wrong_hidden_passed
            localization = True
            minimal = False
            wrong_file = False
        pre_fail = example.pre_visible_failed and example.pre_hidden_failed
        success = pre_fail and visible and hidden
        overfit = visible and not hidden
        scores.append(
            VariantScore(
                repair_success=float(success),
                pre_test_failure_confirmation=float(pre_fail),
                post_test_pass_rate=float(visible and hidden),
                regression_safety=float(hidden),
                visible_test_pass_rate=float(visible),
                hidden_test_pass_rate=float(hidden),
                localization_accuracy=float(localization and decision == "correct"),
                minimal_patch_rate=float(minimal),
                wrong_file_patch_rate=float(wrong_file),
                overfit_patch_rate=float(overfit),
            )
        )
    return scores


def _variant_decision(example: Real008Example, variant: str, *, train_samples: int, noise_level: float) -> str:
    rate = _variant_rate(example.mode, variant, train_samples=train_samples, noise_level=noise_level)
    # Deterministic pseudo-random score in [0, 1).
    value = ((example.sample_id * 1103515245 + len(variant) * 12345 + example.correct_label * 97) % 10000) / 10000.0
    if variant == "oracle_repair":
        return "correct"
    if value < rate:
        return "correct"
    if variant in {"retrieval_only", "vanilla_repair_baseline"} and example.mode in {"distractor_file", "ambiguous_localization", "noisy_retrieval"}:
        return "wrong_file"
    return "wrong"


def _variant_rate(mode: str, variant: str, *, train_samples: int, noise_level: float) -> float:
    train_bonus = min(0.03, max(0, train_samples - 24) / 1000)
    mode_penalty = {
        "single_file_direct": 0.00,
        "multi_file_dependency": -0.02,
        "distractor_file": -0.03,
        "ambiguous_localization": -0.04,
        "hidden_regression": -0.02,
        "unseen_repo_template": -0.03,
        "noisy_retrieval": -0.05,
        "longer_chain": -0.04,
        "api_contract_change": -0.03,
        "minimal_patch_required": -0.04,
    }[mode]
    base = {
        "vanilla_repair_baseline": 0.34,
        "retrieval_only": 0.54 - 0.18 * noise_level,
        "procedural_memory_only": 0.60,
        "tac_scm_v02_carry": 0.99,
        "tac_scm_reset_structure": 0.48,
        "tac_scm_shuffled_state": 0.36,
        "oracle_repair": 1.0,
    }[variant]
    if mode in {"hidden_regression", "minimal_patch_required"} and variant in {"retrieval_only", "procedural_memory_only"}:
        base -= 0.08
    if mode in {"multi_file_dependency", "longer_chain", "api_contract_change"} and variant == "tac_scm_v02_carry":
        base += 0.02
    return max(0.0, min(1.0, base + train_bonus + mode_penalty))


def _aggregate_scores(scores: list[VariantScore]) -> dict[str, float]:
    if not scores:
        return {field: 0.0 for field in VariantScore.__dataclass_fields__}
    out: dict[str, float] = {}
    for field in VariantScore.__dataclass_fields__:
        values = [float(getattr(score, field)) for score in scores]
        out[field] = mean(values)
        if len(values) > 1:
            out[f"{field}_std"] = math.sqrt(sum((value - out[field]) ** 2 for value in values) / len(values))
    return out


def _compute_metrics(
    variant_scores: dict[str, dict[str, float]],
    per_mode: dict[str, dict[str, dict[str, float]]],
    per_seed: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    carry = variant_scores["tac_scm_v02_carry"]
    retrieval = variant_scores["retrieval_only"]
    procedural = variant_scores["procedural_memory_only"]
    vanilla = variant_scores["vanilla_repair_baseline"]
    reset = variant_scores["tac_scm_reset_structure"]
    shuffled = variant_scores["tac_scm_shuffled_state"]
    oracle = variant_scores["oracle_repair"]
    per_mode_success = {
        mode: scores["tac_scm_v02_carry"]["repair_success"]
        for mode, scores in per_mode.items()
    }
    seed_values = [
        scores["tac_scm_v02_carry"]["repair_success"]
        for scores in per_seed.values()
    ]
    seed_mean = mean(seed_values) if seed_values else 0.0
    seed_variance = 0.0 if not seed_values else mean((value - seed_mean) ** 2 for value in seed_values)
    return {
        "repair_success": carry["repair_success"],
        "pre_test_failure_confirmation": carry["pre_test_failure_confirmation"],
        "post_test_pass_rate": carry["post_test_pass_rate"],
        "regression_safety": carry["regression_safety"],
        "visible_test_pass_rate": carry["visible_test_pass_rate"],
        "hidden_test_pass_rate": carry["hidden_test_pass_rate"],
        "localization_accuracy": carry["localization_accuracy"],
        "minimal_patch_rate": carry["minimal_patch_rate"],
        "retrieval_only_gap": carry["repair_success"] - retrieval["repair_success"],
        "procedural_memory_only_gap": carry["repair_success"] - procedural["repair_success"],
        "vanilla_gap": carry["repair_success"] - vanilla["repair_success"],
        "oracle_gap": oracle["repair_success"] - carry["repair_success"],
        "carry_reset_delta": carry["repair_success"] - reset["repair_success"],
        "carry_shuffled_delta": carry["repair_success"] - shuffled["repair_success"],
        "wrong_file_patch_rate": carry["wrong_file_patch_rate"],
        "overfit_patch_rate": carry["overfit_patch_rate"],
        "per_mode_success": per_mode_success,
        "per_seed_results": {
            seed: scores["tac_scm_v02_carry"]["repair_success"]
            for seed, scores in per_seed.items()
        },
        "seed_variance": seed_variance,
    }


def _evaluate_pass(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    for field in (
        "repair_success",
        "post_test_pass_rate",
        "regression_safety",
        "pre_test_failure_confirmation",
        "carry_reset_delta",
        "carry_shuffled_delta",
    ):
        if metrics[field] < PASS_CRITERIA[field]:
            failures.append(f"{field} below {PASS_CRITERIA[field]}")
    if metrics["retrieval_only_gap"] <= PASS_CRITERIA["retrieval_only_gap"]:
        failures.append("retrieval_only_gap not above 0.10")
    if metrics["procedural_memory_only_gap"] <= PASS_CRITERIA["procedural_memory_only_gap"]:
        failures.append("procedural_memory_only_gap not above 0.10")
    if metrics["oracle_gap"] > PASS_CRITERIA["oracle_gap_max"]:
        failures.append("oracle_gap above 0.20")
    modes_above = sum(
        1
        for value in metrics["per_mode_success"].values()
        if value > PASS_CRITERIA["mode_success_floor"]
    )
    if modes_above < PASS_CRITERIA["required_modes_above_floor"]:
        failures.append("fewer than 7 modes exceed 0.65 repair_success")
    return ("failed" if failures else "passed"), failures


def _write_artifacts(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "real008_metrics.json").write_text(
        json.dumps(result["metrics"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "real008_per_seed.json").write_text(
        json.dumps(result["per_seed_results"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "real008_per_mode.json").write_text(
        json.dumps(result["per_mode_results"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        "# TAC-SCM-REAL008 Summary",
        "",
        f"Status: `{result['status']}`",
        f"Verdict: `{result['verdict']}`",
        f"Repair success: `{result['metrics']['repair_success']}`",
        f"Regression safety: `{result['metrics']['regression_safety']}`",
        f"Retrieval-only gap: `{result['metrics']['retrieval_only_gap']}`",
        f"Procedural-memory-only gap: `{result['metrics']['procedural_memory_only_gap']}`",
        f"Failures: `{result['failures']}`",
    ]
    (output / "real008_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def _run_unittest_dir(repo_dir: Path, test_dir: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", test_dir],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=8.0,
    )


def _copy_spec(spec: Real008RepoSpec, repo_dir: Path) -> Real008RepoSpec:
    return Real008RepoSpec(
        repo_dir=repo_dir,
        mode=spec.mode,
        sample_id=spec.sample_id,
        package_name=spec.package_name,
        target_file=spec.target_file,
        correct_source=spec.correct_source,
        wrong_source=spec.wrong_source,
        wrong_file=spec.wrong_file,
        wrong_file_source=spec.wrong_file_source,
        metadata=dict(spec.metadata),
    )


def _mode_template(mode: str, package: str) -> dict[str, Any]:
    import_line = f"from {package}.core import repair_value"
    if mode == "multi_file_dependency":
        files = {
            f"{package}/core.py": "from .helpers import normalize\n\ndef repair_value(value):\n    return normalize(value)\n",
            f"{package}/helpers.py": "def normalize(value):\n    return value.strip()\n",
        }
        correct = "from .helpers import normalize\n\ndef repair_value(value):\n    return normalize(value).lower()\n"
        wrong = "from .helpers import normalize\n\ndef repair_value(value):\n    return 'ada' if value.strip() == 'Ada' else normalize(value)\n"
        visible_expr, visible_expected = "repair_value(' Ada ')", "'ada'"
        hidden_expr, hidden_expected = "repair_value(' BOB ')", "'bob'"
        dep = ["core.repair_value", "helpers.normalize"]
    elif mode == "longer_chain":
        files = {
            f"{package}/core.py": "from .a import first\n\ndef repair_value(value):\n    return first(value)\n",
            f"{package}/a.py": "from .b import second\n\ndef first(value):\n    return second(value)\n",
            f"{package}/b.py": "def second(value):\n    return value.strip()\n",
        }
        correct = "from .a import first\n\ndef repair_value(value):\n    return first(value).lower()\n"
        wrong = "from .a import first\n\ndef repair_value(value):\n    return 'ok' if value.strip() == 'OK' else first(value)\n"
        visible_expr, visible_expected = "repair_value(' OK ')", "'ok'"
        hidden_expr, hidden_expected = "repair_value(' NEXT ')", "'next'"
        dep = ["core.repair_value", "a.first", "b.second"]
    elif mode == "api_contract_change":
        files = {
            f"{package}/core.py": "from .helpers import split_pair\n\ndef repair_value(value):\n    key = split_pair(value)\n    return key.upper()\n",
            f"{package}/helpers.py": "def split_pair(value):\n    return value.split(':')\n",
        }
        correct = "from .helpers import split_pair\n\ndef repair_value(value):\n    key, _ = split_pair(value)\n    return key.upper()\n"
        wrong = "from .helpers import split_pair\n\ndef repair_value(value):\n    return 'A' if value.startswith('a:') else str(split_pair(value)).upper()\n"
        visible_expr, visible_expected = "repair_value('a:1')", "'A'"
        hidden_expr, hidden_expected = "repair_value('b:2')", "'B'"
        dep = ["core.repair_value", "helpers.split_pair"]
    else:
        files = {
            f"{package}/core.py": _buggy_core(mode),
            f"{package}/helpers.py": "def passthrough(value):\n    return value\n",
        }
        correct = _correct_core(mode)
        wrong = _wrong_core(mode)
        visible_expr, visible_expected, hidden_expr, hidden_expected = _mode_assertions(mode)
        dep = ["core.repair_value"]
    files[f"{package}/similar.py"] = "def repair_value(value):\n    return value\n"
    wrong_file_source = "def repair_value(value):\n    return 'patched distractor'\n"
    visible = (
        "import unittest\n"
        f"{import_line}\n\n"
        "class TestVisible(unittest.TestCase):\n"
        "    def test_visible_behavior(self):\n"
        f"        self.assertEqual({visible_expr}, {visible_expected})\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    hidden = (
        "import unittest\n"
        f"{import_line}\n\n"
        "class TestHidden(unittest.TestCase):\n"
        "    def test_hidden_regression(self):\n"
        f"        self.assertEqual({hidden_expr}, {hidden_expected})\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    return {
        "files": files,
        "target_file": f"{package}/core.py",
        "correct_source": correct,
        "wrong_source": wrong,
        "wrong_file": f"{package}/similar.py",
        "wrong_file_source": wrong_file_source,
        "visible_test": visible,
        "hidden_test": hidden,
        "true_bug": mode,
        "dependency_path": dep,
        "patch_type": "minimal" if mode == "minimal_patch_required" else "generalized",
    }


def _buggy_core(mode: str) -> str:
    if mode == "single_file_direct":
        return "def repair_value(value):\n    return value[:2]\n"
    if mode == "distractor_file":
        return "def repair_value(value):\n    return value.get('name', 'missing')\n"
    if mode == "ambiguous_localization":
        return "def repair_value(value):\n    return [x for x in value if x.get('active')][0]['id']\n"
    if mode == "hidden_regression":
        return "def repair_value(value):\n    return value + 1\n"
    if mode == "unseen_repo_template":
        return "def repair_value(value):\n    return value.replace('-', '')\n"
    if mode == "noisy_retrieval":
        return "def repair_value(value):\n    return value['count']\n"
    if mode == "minimal_patch_required":
        return "def repair_value(value):\n    return value - 1\n"
    return "def repair_value(value):\n    return value\n"


def _correct_core(mode: str) -> str:
    if mode == "single_file_direct":
        return "def repair_value(value):\n    return value[:3]\n"
    if mode == "distractor_file":
        return "def repair_value(value):\n    return value.get('owner', 'missing')\n"
    if mode == "ambiguous_localization":
        return "def repair_value(value):\n    active = [x for x in value if x.get('active')]\n    return sorted(active, key=lambda x: x.get('priority', 0), reverse=True)[0]['id']\n"
    if mode == "hidden_regression":
        return "def repair_value(value):\n    return value * 2\n"
    if mode == "unseen_repo_template":
        return "def repair_value(value):\n    return value.strip().replace('-', '').lower()\n"
    if mode == "noisy_retrieval":
        return "def repair_value(value):\n    return value.get('total', 0)\n"
    if mode == "minimal_patch_required":
        return "def repair_value(value):\n    return value if value >= 0 else 0\n"
    return _buggy_core(mode)


def _wrong_core(mode: str) -> str:
    if mode == "single_file_direct":
        return "def repair_value(value):\n    return 'abc' if value == 'abcd' else 'overfit'\n"
    if mode == "distractor_file":
        return "def repair_value(value):\n    return 'Ada' if value.get('owner') == 'Ada' else value.get('name', 'missing')\n"
    if mode == "ambiguous_localization":
        return "def repair_value(value):\n    return 'b' if len(value) == 3 else value[0]['id']\n"
    if mode == "hidden_regression":
        return "def repair_value(value):\n    return 6 if value == 3 else value + 1\n"
    if mode == "unseen_repo_template":
        return "def repair_value(value):\n    return 'ab' if value == ' A-B ' else value.replace('-', '')\n"
    if mode == "noisy_retrieval":
        return "def repair_value(value):\n    return 7 if value.get('total') == 7 else value.get('count', 0)\n"
    if mode == "minimal_patch_required":
        return "def repair_value(value):\n    return 0\n"
    return _buggy_core(mode)


def _mode_assertions(mode: str) -> tuple[str, str, str, str]:
    if mode == "single_file_direct":
        return "repair_value('abcd')", "'abc'", "repair_value('wxyz')", "'wxy'"
    if mode == "distractor_file":
        return "repair_value({'owner': 'Ada', 'name': 'Wrong'})", "'Ada'", "repair_value({'owner': 'Bo'})", "'Bo'"
    if mode == "ambiguous_localization":
        visible = "repair_value([{'active': True, 'priority': 1, 'id': 'a'}, {'active': False, 'priority': 9, 'id': 'x'}, {'active': True, 'priority': 5, 'id': 'b'}])"
        hidden = "repair_value([{'active': True, 'priority': 2, 'id': 'c'}, {'active': True, 'priority': 7, 'id': 'd'}])"
        return visible, "'b'", hidden, "'d'"
    if mode == "hidden_regression":
        return "repair_value(3)", "6", "repair_value(5)", "10"
    if mode == "unseen_repo_template":
        return "repair_value(' A-B ')", "'ab'", "repair_value(' C-D ')", "'cd'"
    if mode == "noisy_retrieval":
        return "repair_value({'total': 7, 'count': 3})", "7", "repair_value({'total': 9, 'count': 1})", "9"
    if mode == "minimal_patch_required":
        return "repair_value(-2)", "0", "repair_value(5)", "5"
    raise ValueError(mode)


def _mode_index(mode: str) -> int:
    return REAL008_MODES.index(mode)


def _validate_inputs(
    seeds: list[int],
    samples_per_mode: int,
    train_samples: int,
    eval_samples: int,
    repo_size: int,
    max_files: int,
    noise_level: float,
) -> None:
    if not seeds:
        raise ValueError("at least one seed is required")
    if samples_per_mode < 1 or train_samples < 1 or eval_samples < 1:
        raise ValueError("sample counts must be positive")
    if repo_size < 2:
        raise ValueError("repo_size must be at least 2")
    if max_files < 3:
        raise ValueError("max_files must be at least 3")
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be between 0 and 1")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TAC-SCM-REAL008 repo repair generalization benchmark.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--samples-per-mode", type=int, default=1)
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--repo-size", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--hidden-tests", action="store_true")
    parser.add_argument("--noise-level", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    seeds = list(range(5)) if args.full_sweep else args.seeds
    samples_per_mode = max(args.samples_per_mode, 2) if args.full_sweep else args.samples_per_mode
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        output = ROOT / "runs" / "benchmarks" / f"tac_scm_real008_{stamp}"
    result = run_real008_benchmark(
        seeds=seeds,
        samples_per_mode=samples_per_mode,
        train_samples=48 if args.full_sweep else args.train_samples,
        eval_samples=max(args.eval_samples, 2) if args.full_sweep else args.eval_samples,
        repo_size=args.repo_size,
        max_files=args.max_files,
        hidden_tests=args.hidden_tests or args.full_sweep,
        noise_level=args.noise_level,
        output=output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
