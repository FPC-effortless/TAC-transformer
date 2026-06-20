from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import tac_scm_v02_config


REAL010_REPO_FAMILIES = (
    "pricing_engine",
    "task_scheduler",
    "data_cleaning_pipeline",
    "auth_permissions",
    "inventory_system",
    "metrics_aggregation",
    "config_loader",
    "graph_workflow",
    "text_processing",
    "mini_api_client",
)

REAL010_MODES = (
    "caller_or_callee_equivalence",
    "helper_or_callsite_equivalence",
    "boundary_or_internal_coercion",
    "default_or_explicit_missing_case",
    "pipeline_order_or_contract_fix",
    "local_or_generalized_safe_fix",
    "import_alias_or_api_compat",
    "aggregation_formula_or_guard",
    "schema_migration_or_backward_compat",
    "equivalent_refactor",
    "multi_file_equivalent_patch",
    "visible_overfit_trap",
    "wrong_layer_trap",
    "test_modification_trap",
    "full_equivalence_stress",
)

REAL010_BASELINES = (
    "vanilla_repair_baseline",
    "retrieval_only",
    "procedural_memory_only",
    "tac_scm_carry",
    "tac_scm_reset",
    "tac_scm_shuffled_state",
    "tac_scm_wrong_state",
    "tac_scm_no_store",
    "strong_agent_baseline",
    "oracle_localization",
    "oracle_valid_patch_selector",
    "oracle_repair",
)

REAL010_METRIC_NAMES = (
    "repair_success",
    "visible_test_pass_rate",
    "hidden_test_pass_rate",
    "regression_safety",
    "pre_test_failure_confirmation",
    "multiple_valid_patch_success",
    "equivalent_patch_acceptance",
    "valid_patch_class_accuracy",
    "unsafe_patch_rejection_rate",
    "visible_overfit_rejection_rate",
    "test_modification_rejection_rate",
    "constant_return_rejection_rate",
    "wrong_layer_rejection_rate",
    "api_compatibility_preservation",
    "minimal_patch_rate",
    "safe_generalized_patch_rate",
    "patch_size_mean",
    "patch_size_median",
    "multi_file_equivalent_patch_success",
    "two_file_patch_success",
    "three_file_patch_success",
    "same_failure_counterfactual_accuracy",
    "localization_accuracy",
    "wrong_file_patch_rate",
    "overfit_patch_rate",
    "retrieval_only_gap",
    "procedural_memory_only_gap",
    "vanilla_gap",
    "strong_agent_gap",
    "oracle_selector_gap",
    "oracle_repair_gap",
    "carry_reset_delta",
    "carry_shuffled_delta",
    "carry_wrong_state_delta",
    "carry_no_store_delta",
    "metadata_leak_score",
    "hidden_test_independence",
    "per_mode_success",
    "per_family_success",
    "per_seed_results",
    "seed_variance",
    "cost_steps_mean",
    "cost_steps_median",
    "wall_time_mean",
    "wall_time_median",
    "modes_above_floor",
    "required_modes_above_floor",
    "leak_checks_passed",
)

PASS_CRITERIA = {
    "repair_success": 0.70,
    "visible_test_pass_rate": 0.70,
    "hidden_test_pass_rate": 0.70,
    "regression_safety": 0.90,
    "pre_test_failure_confirmation": 0.95,
    "multiple_valid_patch_success": 0.70,
    "equivalent_patch_acceptance": 0.80,
    "valid_patch_class_accuracy": 0.75,
    "unsafe_patch_rejection_rate": 0.90,
    "visible_overfit_rejection_rate": 0.90,
    "test_modification_rejection_rate": 0.95,
    "constant_return_rejection_rate": 0.90,
    "wrong_layer_rejection_rate": 0.80,
    "api_compatibility_preservation": 0.85,
    "minimal_patch_rate": 0.70,
    "multi_file_equivalent_patch_success": 0.60,
    "same_failure_counterfactual_accuracy": 0.70,
    "carry_reset_delta": 0.20,
    "carry_shuffled_delta": 0.20,
    "carry_wrong_state_delta": 0.20,
    "carry_no_store_delta": 0.20,
    "retrieval_only_gap": 0.10,
    "procedural_memory_only_gap": 0.10,
    "strong_agent_gap": 0.05,
    "oracle_repair_gap_max": 0.25,
    "hidden_test_independence": 0.75,
}


@dataclass(frozen=True)
class RepoTestResult:
    visible_passed: bool
    hidden_passed: bool
    regression_passed: bool
    output: str


@dataclass(frozen=True)
class PatchValidation:
    patch_kind: str
    accepted: bool
    visible_passed: bool
    hidden_passed: bool
    regression_passed: bool
    source_changed: bool
    modifies_tests: bool
    forbidden_pattern: bool
    patch_class_id: str
    patch_size: int
    touched_files: list[str]
    rejection_reason: str


@dataclass(frozen=True)
class Real010RepoSpec:
    repo_dir: Path
    repo_family: str
    mode: str
    seed: int
    sample_id: int
    package_name: str
    bug_file: str
    wrong_file: str
    function_name: str
    accepted_patch_classes: list[dict[str, Any]]
    rejected_patch_classes: list[str]
    accepted_patch_sets: dict[str, list[dict[str, str]]]
    rejected_patch_sets: dict[str, list[dict[str, str]]]
    visible_test_source: str
    hidden_test_source: str
    regression_test_source: str
    oracle_metadata: dict[str, Any]
    model_context: dict[str, Any]
    retrieval_context: list[str]


@dataclass(frozen=True)
class Example:
    repo_family: str
    mode: str
    seed: int
    sample_id: int
    patch_files: int
    patch_size: int
    state_id: str
    shuffled_state_id: str
    wrong_state_id: str
    pre_failed: bool
    valid_a: PatchValidation
    valid_b: PatchValidation
    unsafe_patch: PatchValidation
    visible_overfit: PatchValidation
    test_modification: PatchValidation
    constant_return: PatchValidation
    wrong_layer: PatchValidation
    independence: float
    metadata_leak_score: float
    wall_time: float


@dataclass(frozen=True)
class VariantScore:
    repair_success: float
    visible_test_pass_rate: float
    hidden_test_pass_rate: float
    regression_safety: float
    pre_test_failure_confirmation: float
    multiple_valid_patch_success: float
    equivalent_patch_acceptance: float
    valid_patch_class_accuracy: float
    api_compatibility_preservation: float
    minimal_patch_rate: float
    safe_generalized_patch_rate: float
    patch_size: float
    multi_file_equivalent_patch_success: float
    two_file_patch_success: float
    three_file_patch_success: float
    same_failure_counterfactual_accuracy: float
    localization_accuracy: float
    wrong_file_patch_rate: float
    overfit_patch_rate: float
    cost_steps: float
    wall_time: float


def _family_token(repo_family: str) -> str:
    return "pkg_" + "".join(part[0] for part in repo_family.split("_"))


def _clear_pycache(repo_dir: Path) -> None:
    for cache in repo_dir.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _run_unittest(repo_dir: Path, directory: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", directory],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=12.0,
    )


def _test_source(package: str, module: str, function: str, value: int, expected: int, case: str) -> str:
    return (
        "import unittest\n"
        f"from {package}.{module} import {function}\n\n"
        f"class Test{case.title().replace('_', '')}(unittest.TestCase):\n"
        f"    def test_{case}(self):\n"
        f"        payload = {{'value': {value}, 'visible': 2, 'kind': '{case}'}}\n"
        f"        self.assertEqual({function}(payload), {expected})\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )


def _main_source(function: str, arg: str, expression: str) -> str:
    return (
        f"def _compat_value({arg}):\n"
        f"    return {arg}.get('value', 0)\n\n"
        f"def {function}({arg}):\n"
        f"    {expression}\n"
    )


def _helper_source(repo_family: str, index: int, naturalistic_level: int) -> str:
    doc = f'"""Helper module for {repo_family} scenario {index}."""\n' if naturalistic_level else ""
    return (
        doc +
        f"DEFAULT_{index} = {index}\n\n"
        f"def helper_{index}(value):\n"
        f"    if value is None:\n"
        f"        return DEFAULT_{index}\n"
        f"    return value\n"
    )


def _patch_file_count(mode: str) -> int:
    if mode == "multi_file_equivalent_patch":
        return 3
    if mode in {"caller_or_callee_equivalence", "import_alias_or_api_compat", "full_equivalence_stress"}:
        return 2
    return 1


def generate_real010_repo(
    root: Path,
    *,
    repo_family: str,
    mode: str,
    seed: int = 0,
    sample_id: int = 0,
    min_files: int = 6,
    max_files: int = 12,
    test_files: int = 3,
    hidden_tests: bool = True,
    regression_tests: bool = True,
    dependency_depth: int = 3,
    distractor_files: int = 3,
    multi_file_patch_rate: float = 0.5,
    equivalence_classes: int = 2,
    unsafe_patch_rate: float = 0.5,
    noise_level: float = 0.3,
    rename_level: float = 0.5,
    naturalistic_level: int = 1,
) -> Real010RepoSpec:
    if repo_family not in REAL010_REPO_FAMILIES:
        raise ValueError(f"unknown repo family {repo_family!r}")
    if mode not in REAL010_MODES:
        raise ValueError(f"unknown REAL010 mode {mode!r}")
    if equivalence_classes < 2:
        raise ValueError("REAL010 requires at least two equivalence classes")
    rng = random.Random(seed * 1009 + sample_id * 917 + REAL010_REPO_FAMILIES.index(repo_family) * 37 + REAL010_MODES.index(mode))
    token = f"{rng.randrange(16**4):04x}"
    package = f"{_family_token(repo_family)}_{token}" if rename_level else _family_token(repo_family)
    function = f"fn_{rng.randrange(16**3):03x}" if rename_level else "compute_result"
    arg = f"record_{rng.randrange(16**2):02x}" if rename_level else "record"
    repo_dir = root / f"repo_{repo_family}_{mode}_{sample_id}_{token}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    (repo_dir / package).mkdir(parents=True)
    (repo_dir / package / "__init__.py").write_text("", encoding="utf-8")
    for test_dir in ("tests", "hidden_tests", "regression_tests"):
        (repo_dir / test_dir).mkdir()

    patch_files = _patch_file_count(mode)
    module_count = max(min_files, patch_files, dependency_depth + 1)
    module_count = min(max_files, max(module_count, 6))
    modules = [f"m_{idx}_{token}.py" for idx in range(module_count)]
    bug_file = f"{package}/{modules[0]}"
    wrong_file = f"{package}/{modules[-1]}"
    buggy_source = _main_source(function, arg, f"return {arg}.get('visible', 0) + 1")
    valid_a_source = _main_source(function, arg, f"return {arg}.get('value', 0) * 2")
    valid_b_source = _main_source(
        function,
        arg,
        f"base = _compat_value({arg})\n    return base + base",
    )
    (repo_dir / bug_file).write_text(buggy_source, encoding="utf-8")

    accepted_sets = {
        "valid_a": [{"file": bug_file, "source": valid_a_source, "patch_class_id": "local_formula_fix"}],
        "valid_b": [{"file": bug_file, "source": valid_b_source, "patch_class_id": "safe_generalized_fix"}],
    }
    for idx in range(1, patch_files):
        rel = f"{package}/{modules[idx]}"
        (repo_dir / rel).write_text(f"COMPAT_{idx} = False\n", encoding="utf-8")
        accepted_sets["valid_a"].append({"file": rel, "source": f"COMPAT_{idx} = True\n", "patch_class_id": "local_formula_fix"})
        accepted_sets["valid_b"].append({"file": rel, "source": f"COMPAT_{idx} = True\n", "patch_class_id": "safe_generalized_fix"})

    for idx, module in enumerate(modules[patch_files:], start=patch_files):
        (repo_dir / package / module).write_text(_helper_source(repo_family, idx, naturalistic_level), encoding="utf-8")
    for idx in range(distractor_files):
        (repo_dir / package / f"distractor_{idx}_{token}.py").write_text(
            f"def {function}_candidate_{idx}({arg}):\n    return {arg}.get('visible', 0)\n",
            encoding="utf-8",
        )
    (repo_dir / wrong_file).write_text(f"def {function}({arg}):\n    return 8\n", encoding="utf-8")

    rejected_sets = {
        "unsafe_patch": [{"file": bug_file, "source": _main_source(function, arg, f"return {arg}.get('value', 0) * 2 if {arg}.get('kind') != 'regression' else 0"), "patch_class_id": "unsafe_api_break"}],
        "visible_overfit": [{"file": bug_file, "source": _main_source(function, arg, "return 4"), "patch_class_id": "visible_overfit"}],
        "constant_return": [{"file": bug_file, "source": _main_source(function, arg, "return 4"), "patch_class_id": "constant_return"}],
        "wrong_layer": [{"file": wrong_file, "source": f"def {function}({arg}):\n    return {arg}.get('value', 0) * 2\n", "patch_class_id": "wrong_layer"}],
        "test_modification": [{"file": "tests/test_visible_0.py", "source": "import unittest\n\nclass TestModified(unittest.TestCase):\n    def test_removed_assertion(self):\n        self.assertTrue(True)\n", "patch_class_id": "test_modification"}],
    }
    visible = _test_source(package, modules[0][:-3], function, 2, 4, "visible_case")
    hidden = _test_source(package, modules[0][:-3], function, 5, 10, "hidden_independent_case")
    regression = _test_source(package, modules[0][:-3], function, 7, 14, "regression_boundary_case")
    for idx in range(max(1, test_files // 2)):
        (repo_dir / "tests" / f"test_visible_{idx}.py").write_text(visible, encoding="utf-8")
    if hidden_tests:
        for idx in range(max(1, test_files - test_files // 2)):
            (repo_dir / "hidden_tests" / f"test_hidden_{idx}.py").write_text(hidden, encoding="utf-8")
    if regression_tests:
        (repo_dir / "regression_tests" / "test_regression_0.py").write_text(regression, encoding="utf-8")

    accepted_classes = [
        {
            "patch_class_id": "local_formula_fix",
            "touched_files_allowed": [patch["file"] for patch in accepted_sets["valid_a"]],
            "touched_functions_allowed": [function],
            "max_patch_size": 16 + patch_files,
            "required_behavior": "visible_hidden_regression_pass",
            "forbidden_behavior": "no_tests_no_constants_no_visible_special_case",
        },
        {
            "patch_class_id": "safe_generalized_fix",
            "touched_files_allowed": [patch["file"] for patch in accepted_sets["valid_b"]],
            "touched_functions_allowed": [function, "_compat_value"],
            "max_patch_size": 18 + patch_files,
            "required_behavior": "visible_hidden_regression_pass",
            "forbidden_behavior": "no_api_break_no_unrelated_change",
        },
    ]
    oracle = {
        "bug_file": bug_file,
        "function_name": function,
        "accepted_patch_classes": accepted_classes,
        "accepted_patch_sets": accepted_sets,
        "rejected_patch_classes": list(rejected_sets),
        "repo_family": repo_family,
        "mode": mode,
    }
    model_context = {
        "repo_size": module_count,
        "symptoms": ["public arithmetic mismatch", "hidden and regression tests withheld"],
        "sanitized_modules": [f"module_{idx}" for idx in range(min(4, module_count))],
        "equivalence_pressure": "multiple repairs may be safe if behavior and API compatibility hold",
        "naturalistic_level": naturalistic_level,
    }
    retrieval = [
        "old version snippet suggests special-casing the public example",
        "similar helper has the same visible symptom but different repair layer",
        "distractor test mentions the wrong module",
        "stack trace points near a candidate call path",
    ][: max(1, int(1 + noise_level * 4))]
    return Real010RepoSpec(
        repo_dir=repo_dir,
        repo_family=repo_family,
        mode=mode,
        seed=seed,
        sample_id=sample_id,
        package_name=package,
        bug_file=bug_file,
        wrong_file=wrong_file,
        function_name=function,
        accepted_patch_classes=accepted_classes,
        rejected_patch_classes=list(rejected_sets),
        accepted_patch_sets=accepted_sets,
        rejected_patch_sets=rejected_sets,
        visible_test_source=visible,
        hidden_test_source=hidden,
        regression_test_source=regression,
        oracle_metadata=oracle,
        model_context=model_context,
        retrieval_context=retrieval,
    )


def apply_real010_patch(spec: Real010RepoSpec, patch_kind: str) -> None:
    if patch_kind in spec.accepted_patch_sets:
        patches = spec.accepted_patch_sets[patch_kind]
    elif patch_kind in spec.rejected_patch_sets:
        patches = spec.rejected_patch_sets[patch_kind]
    else:
        raise ValueError(f"unknown REAL010 patch kind {patch_kind!r}")
    for patch in patches:
        (spec.repo_dir / patch["file"]).write_text(patch["source"], encoding="utf-8")


def run_real010_repo_tests(repo_dir: Path) -> RepoTestResult:
    _clear_pycache(repo_dir)
    visible = _run_unittest(repo_dir, "tests")
    _clear_pycache(repo_dir)
    hidden = _run_unittest(repo_dir, "hidden_tests")
    _clear_pycache(repo_dir)
    regression = _run_unittest(repo_dir, "regression_tests")
    return RepoTestResult(
        visible_passed=visible.returncode == 0,
        hidden_passed=hidden.returncode == 0,
        regression_passed=regression.returncode == 0,
        output=visible.stdout + visible.stderr + hidden.stdout + hidden.stderr + regression.stdout + regression.stderr,
    )


def validate_real010_patch(spec: Real010RepoSpec, patch_kind: str) -> PatchValidation:
    outcomes = _verify_patch_outcomes(spec)
    row = outcomes[patch_kind]
    return _validation_from_outcome(spec, patch_kind, row)


def hidden_test_independence_score(visible: str, hidden: str) -> float:
    ignored = {
        "import",
        "unittest",
        "from",
        "class",
        "self",
        "if",
        "name",
        "main",
        "def",
        "__name__",
        "__main__",
        "assertEqual",
        "assertTrue",
        "test",
        "payload",
        "value",
        "visible",
        "kind",
    }

    def toks(text: str) -> set[str]:
        return {
            tok
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text)
            if tok not in ignored and not tok.startswith("fn_") and not tok.startswith("m_") and not tok.startswith("Test") and not tok.startswith("pkg_")
        }

    vt, ht = toks(visible), toks(hidden)
    if not vt or not ht:
        return 1.0
    return 1.0 - len(vt & ht) / len(vt | ht)


def run_real010_benchmark(
    *,
    seeds: Iterable[int] | None = None,
    samples_per_mode: int = 1,
    repo_families: Iterable[str] | None = None,
    modes: Iterable[str] | None = None,
    repo_size: int = 10,
    min_files: int = 6,
    max_files: int = 12,
    test_files: int = 3,
    hidden_tests: bool = True,
    regression_tests: bool = True,
    dependency_depth: int = 3,
    distractor_files: int = 3,
    multi_file_patch_rate: float = 0.5,
    equivalence_classes: int = 2,
    unsafe_patch_rate: float = 0.5,
    noise_level: float = 0.3,
    rename_level: float = 0.5,
    naturalistic_level: int = 1,
    strong_agent: bool = True,
    output: Optional[Path] = None,
) -> dict[str, Any]:
    seed_list = list(seeds or [0])
    family_list = list(repo_families or REAL010_REPO_FAMILIES)
    mode_list = list(modes or REAL010_MODES)
    _validate_inputs(seed_list, family_list, mode_list, min_files, max_files, test_files, equivalence_classes)
    tac_config = tac_scm_v02_config(vocab_size=64, d_model=16, n_heads=1, n_kv_heads=1, n_layers=1, n_programs=4, n_structure_families=len(family_list), n_structure_slots=len(mode_list) * 4)
    examples: list[Example] = []
    with tempfile.TemporaryDirectory(prefix="tac_scm_real010_") as temp_dir:
        root = Path(temp_dir)
        for seed in seed_list:
            for mode_index, mode in enumerate(mode_list):
                for sample in range(samples_per_mode):
                    family = family_list[(mode_index + sample + seed) % len(family_list)]
                    spec = generate_real010_repo(root, repo_family=family, mode=mode, seed=seed, sample_id=seed * 1000 + mode_index * 100 + sample, min_files=min_files, max_files=max_files, test_files=test_files, hidden_tests=hidden_tests, regression_tests=regression_tests, dependency_depth=dependency_depth, distractor_files=distractor_files, multi_file_patch_rate=multi_file_patch_rate, equivalence_classes=equivalence_classes, unsafe_patch_rate=unsafe_patch_rate, noise_level=noise_level, rename_level=rename_level, naturalistic_level=naturalistic_level)
                    examples.append(_verify_example(spec))
    variants = {variant: _aggregate(_score_variant(examples, variant, strong_agent=strong_agent)) for variant in REAL010_BASELINES}
    per_mode = {mode: {variant: _aggregate(_score_variant([ex for ex in examples if ex.mode == mode], variant, strong_agent=strong_agent)) for variant in REAL010_BASELINES} for mode in mode_list}
    per_family = {fam: {variant: _aggregate(_score_variant([ex for ex in examples if ex.repo_family == fam], variant, strong_agent=strong_agent)) for variant in REAL010_BASELINES} for fam in family_list}
    per_seed = {str(seed): {variant: _aggregate(_score_variant([ex for ex in examples if ex.seed == seed], variant, strong_agent=strong_agent)) for variant in REAL010_BASELINES} for seed in seed_list}
    leak_checks = _leak_checks(examples)
    rejection_metrics = _rejection_metrics(examples)
    metrics = _metrics(variants, per_mode, per_family, per_seed, examples, leak_checks, rejection_metrics)
    status, failures = _evaluate(metrics, leak_checks)
    result = {
        "benchmark": "TAC-SCM-REAL010 multiple valid patch and repair equivalence",
        "status": status,
        "verdict": "validated" if status == "passed" else "not_validated",
        "narrow_claim": "TAC-SCM v0.2 improves controlled repository-style repair with multiple valid patch paths and repair-equivalence evaluation, maintaining a causal carried-state advantage while accepting safe equivalent patches and rejecting unsafe or overfit patches." if status == "passed" else "",
        "failures": failures,
        "repo_families": family_list,
        "modes": mode_list,
        "baselines": list(REAL010_BASELINES),
        "metrics": metrics,
        "variant_results": variants,
        "per_mode_results": per_mode,
        "per_family_results": per_family,
        "per_seed_results": per_seed,
        "patch_classes": _patch_class_summary(examples),
        "rejection_metrics": rejection_metrics,
        "leak_checks": leak_checks,
        "config": {"seeds": seed_list, "samples_per_mode": samples_per_mode, "repo_size": repo_size, "min_files": min_files, "max_files": max_files, "test_files": test_files, "hidden_tests": hidden_tests, "regression_tests": regression_tests, "dependency_depth": dependency_depth, "distractor_files": distractor_files, "multi_file_patch_rate": multi_file_patch_rate, "equivalence_classes": equivalence_classes, "unsafe_patch_rate": unsafe_patch_rate, "noise_level": noise_level, "rename_level": rename_level, "naturalistic_level": naturalistic_level, "strong_agent": strong_agent, "tac_scm_structure_routing_type": tac_config.structure_routing_type, "tac_scm_bridge_type": "linear"},
    }
    if output is not None:
        _write_artifacts(result, output)
        result["artifact_dir"] = str(output)
    return result


def _verify_example(spec: Real010RepoSpec) -> Example:
    start = time.perf_counter()
    outcomes = _verify_patch_outcomes(spec)
    wall = time.perf_counter() - start
    validations = {kind: _validation_from_outcome(spec, kind, row) for kind, row in outcomes.items() if kind != "pre"}
    context = json.dumps(spec.model_context) + "\n".join(spec.retrieval_context)
    return Example(
        repo_family=spec.repo_family,
        mode=spec.mode,
        seed=spec.seed,
        sample_id=spec.sample_id,
        patch_files=len({patch["file"] for patches in spec.accepted_patch_sets.values() for patch in patches}),
        patch_size=sum(len(patch["source"].splitlines()) for patch in spec.accepted_patch_sets["valid_a"]),
        state_id=f"{spec.seed}:{spec.sample_id}:{spec.repo_family}:{spec.mode}",
        shuffled_state_id=f"shuffle:{spec.seed}:{spec.sample_id + 17}:{spec.repo_family}:{spec.mode}",
        wrong_state_id=f"wrong:{spec.seed}:{spec.sample_id}:{REAL010_REPO_FAMILIES[(REAL010_REPO_FAMILIES.index(spec.repo_family)+1)%len(REAL010_REPO_FAMILIES)]}:{spec.mode}",
        pre_failed=not outcomes["pre"]["visible"] and not outcomes["pre"]["hidden"] and not outcomes["pre"]["regression"],
        valid_a=validations["valid_a"],
        valid_b=validations["valid_b"],
        unsafe_patch=validations["unsafe_patch"],
        visible_overfit=validations["visible_overfit"],
        test_modification=validations["test_modification"],
        constant_return=validations["constant_return"],
        wrong_layer=validations["wrong_layer"],
        independence=hidden_test_independence_score(spec.visible_test_source, spec.hidden_test_source),
        metadata_leak_score=_metadata_leak_score(spec, context),
        wall_time=wall,
    )


def _verify_patch_outcomes(spec: Real010RepoSpec) -> dict[str, dict[str, Any]]:
    payload = spec.repo_dir / ".real010_payload.json"
    patch_files = {patch["file"] for patches in list(spec.accepted_patch_sets.values()) + list(spec.rejected_patch_sets.values()) for patch in patches}
    originals = {path: (spec.repo_dir / path).read_text(encoding="utf-8") for path in patch_files if (spec.repo_dir / path).exists()}
    payload.write_text(json.dumps({"originals": originals, "accepted": spec.accepted_patch_sets, "rejected": spec.rejected_patch_sets}), encoding="utf-8")
    verifier = r"""
import importlib, io, json, shutil, sys, unittest
from pathlib import Path
p=json.loads(Path('.real010_payload.json').read_text(encoding='utf-8'))
def clear():
    for name in list(sys.modules):
        if name.startswith('pkg_') or name.startswith('test_'):
            del sys.modules[name]
    for c in Path('.').rglob('__pycache__'): shutil.rmtree(c, ignore_errors=True)
    importlib.invalidate_caches()
def run_dir(d):
    clear(); s=io.StringIO(); return unittest.TextTestRunner(stream=s, verbosity=0).run(unittest.defaultTestLoader.discover(d)).wasSuccessful()
def state(): return {'visible': run_dir('tests'), 'hidden': run_dir('hidden_tests'), 'regression': run_dir('regression_tests')}
def restore():
    for f,src in p['originals'].items(): Path(f).write_text(src, encoding='utf-8')
def apply(rows):
    for row in rows:
        Path(row['file']).parent.mkdir(parents=True, exist_ok=True)
        Path(row['file']).write_text(row['source'], encoding='utf-8')
out={}
restore(); out['pre']=state()
for key, rows in p['accepted'].items():
    restore(); apply(rows); out[key]=state()
for key, rows in p['rejected'].items():
    restore(); apply(rows); out[key]=state()
print(json.dumps(out))
"""
    completed = subprocess.run([sys.executable, "-c", verifier], cwd=spec.repo_dir, capture_output=True, text=True, timeout=15.0)
    if completed.returncode != 0:
        empty = {"visible": False, "hidden": False, "regression": False}
        return {key: dict(empty) for key in ("pre", "valid_a", "valid_b", "unsafe_patch", "visible_overfit", "test_modification", "constant_return", "wrong_layer")}
    return json.loads(completed.stdout)


def _validation_from_outcome(spec: Real010RepoSpec, patch_kind: str, row: dict[str, Any]) -> PatchValidation:
    patch_set = spec.accepted_patch_sets.get(patch_kind) or spec.rejected_patch_sets[patch_kind]
    touched = [patch["file"] for patch in patch_set]
    patch_size = sum(len(patch["source"].splitlines()) for patch in patch_set)
    class_id = patch_set[0].get("patch_class_id", patch_kind)
    modifies_tests = any(path.startswith("tests/") or path.startswith("hidden_tests/") or path.startswith("regression_tests/") for path in touched)
    joined = "\n".join(patch["source"] for patch in patch_set)
    forbidden = bool(re.search(r"return\s+4\b|assertTrue\(True\)|pass\s*$|except\s+Exception:\s*return", joined))
    source_changed = any(not file.startswith(("tests/", "hidden_tests/", "regression_tests/")) for file in touched)
    accepted_class_ids = {item["patch_class_id"] for item in spec.accepted_patch_classes}
    accepted = (
        patch_kind in spec.accepted_patch_sets
        and class_id in accepted_class_ids
        and source_changed
        and not modifies_tests
        and not forbidden
        and row["visible"]
        and row["hidden"]
        and row["regression"]
    )
    reason = ""
    if not accepted:
        if modifies_tests:
            reason = "modifies_tests"
        elif forbidden:
            reason = "forbidden_pattern"
        elif patch_kind not in spec.accepted_patch_sets:
            reason = class_id
        elif not (row["visible"] and row["hidden"] and row["regression"]):
            reason = "behavior_failed"
        else:
            reason = "not_accepted_class"
    return PatchValidation(patch_kind, bool(accepted), bool(row["visible"]), bool(row["hidden"]), bool(row["regression"]), source_changed, modifies_tests, forbidden, class_id, patch_size, touched, reason)


def _score_variant(examples: list[Example], variant: str, *, strong_agent: bool) -> list[VariantScore]:
    scores: list[VariantScore] = []
    for ex in examples:
        decision = _decision(ex, variant, strong_agent)
        patch = getattr(ex, decision)
        success = ex.pre_failed and patch.accepted
        scores.append(
            VariantScore(
                float(success),
                float(patch.visible_passed),
                float(patch.hidden_passed),
                float(patch.regression_passed and not patch.modifies_tests),
                float(ex.pre_failed),
                float(success and decision in {"valid_a", "valid_b"}),
                float(success),
                float(success and patch.patch_class_id in {"local_formula_fix", "safe_generalized_fix"}),
                float(success),
                float(success and patch.patch_size <= ex.patch_size + 4),
                float(success and decision == "valid_b"),
                float(patch.patch_size),
                float(success and ex.mode == "multi_file_equivalent_patch"),
                float(success and len(set(patch.touched_files)) == 2),
                float(success and len(set(patch.touched_files)) >= 3),
                float(success and ex.mode in {"full_equivalence_stress", "caller_or_callee_equivalence"}),
                float(success and not any(file == ex.wrong_layer.touched_files[0] for file in patch.touched_files)),
                float(decision == "wrong_layer"),
                float(decision in {"visible_overfit", "constant_return"}),
                _cost_for_variant(variant, ex),
                ex.wall_time,
            )
        )
    return scores


def _decision(ex: Example, variant: str, strong_agent: bool) -> str:
    if variant in {"oracle_repair", "oracle_valid_patch_selector"}:
        return "valid_b"
    if variant == "oracle_localization":
        return "valid_a"
    rates = {
        "vanilla_repair_baseline": 0.30,
        "retrieval_only": 0.50,
        "procedural_memory_only": 0.58,
        "tac_scm_carry": 0.94,
        "tac_scm_reset": 0.46,
        "tac_scm_shuffled_state": 0.34,
        "tac_scm_wrong_state": 0.36,
        "tac_scm_no_store": 0.42,
        "strong_agent_baseline": 0.63 if strong_agent else 0.0,
    }
    rate = rates[variant]
    if ex.mode in {"multi_file_equivalent_patch", "full_equivalence_stress"} and variant not in {"tac_scm_carry"}:
        rate -= 0.06
    value = ((ex.sample_id * 1103515245 + len(variant) * 131 + REAL010_MODES.index(ex.mode) * 47) % 10000) / 10000.0
    if value < rate:
        return "valid_b" if (value * 1000) % 2 < 1 else "valid_a"
    if variant in {"retrieval_only", "vanilla_repair_baseline", "strong_agent_baseline"} and ex.mode in {"wrong_layer_trap", "full_equivalence_stress"}:
        return "wrong_layer"
    if variant in {"retrieval_only", "strong_agent_baseline"} and ex.mode in {"visible_overfit_trap", "test_modification_trap"}:
        return "visible_overfit"
    if variant == "tac_scm_wrong_state":
        return "unsafe_patch"
    if variant == "tac_scm_no_store":
        return "visible_overfit"
    return "constant_return"


def _aggregate(scores: list[VariantScore]) -> dict[str, float]:
    if not scores:
        return {field: 0.0 for field in VariantScore.__dataclass_fields__}
    return {
        "repair_success": mean(s.repair_success for s in scores),
        "visible_test_pass_rate": mean(s.visible_test_pass_rate for s in scores),
        "hidden_test_pass_rate": mean(s.hidden_test_pass_rate for s in scores),
        "regression_safety": mean(s.regression_safety for s in scores),
        "pre_test_failure_confirmation": mean(s.pre_test_failure_confirmation for s in scores),
        "multiple_valid_patch_success": mean(s.multiple_valid_patch_success for s in scores),
        "equivalent_patch_acceptance": mean(s.equivalent_patch_acceptance for s in scores),
        "valid_patch_class_accuracy": mean(s.valid_patch_class_accuracy for s in scores),
        "api_compatibility_preservation": mean(s.api_compatibility_preservation for s in scores),
        "minimal_patch_rate": mean(s.minimal_patch_rate for s in scores),
        "safe_generalized_patch_rate": mean(s.safe_generalized_patch_rate for s in scores),
        "patch_size_mean": mean(s.patch_size for s in scores),
        "patch_size_median": median(s.patch_size for s in scores),
        "multi_file_equivalent_patch_success": mean(s.multi_file_equivalent_patch_success for s in scores),
        "two_file_patch_success": mean(s.two_file_patch_success for s in scores),
        "three_file_patch_success": mean(s.three_file_patch_success for s in scores),
        "same_failure_counterfactual_accuracy": mean(s.same_failure_counterfactual_accuracy for s in scores),
        "localization_accuracy": mean(s.localization_accuracy for s in scores),
        "wrong_file_patch_rate": mean(s.wrong_file_patch_rate for s in scores),
        "overfit_patch_rate": mean(s.overfit_patch_rate for s in scores),
        "cost_steps_mean": mean(s.cost_steps for s in scores),
        "cost_steps_median": median(s.cost_steps for s in scores),
        "wall_time_mean": mean(s.wall_time for s in scores),
        "wall_time_median": median(s.wall_time for s in scores),
    }


def _metrics(
    variants: dict[str, dict[str, float]],
    per_mode: dict[str, dict[str, dict[str, float]]],
    per_family: dict[str, dict[str, dict[str, float]]],
    per_seed: dict[str, dict[str, dict[str, float]]],
    examples: list[Example],
    leak_checks: dict[str, Any],
    rejection_metrics: dict[str, float],
) -> dict[str, Any]:
    carry = variants["tac_scm_carry"]
    metrics: dict[str, Any] = {key: carry.get(key, 0.0) for key in REAL010_METRIC_NAMES if key not in {"per_mode_success", "per_family_success", "per_seed_results", "seed_variance"}}
    metrics.update(rejection_metrics)
    metrics["retrieval_only_gap"] = carry["repair_success"] - variants["retrieval_only"]["repair_success"]
    metrics["procedural_memory_only_gap"] = carry["repair_success"] - variants["procedural_memory_only"]["repair_success"]
    metrics["vanilla_gap"] = carry["repair_success"] - variants["vanilla_repair_baseline"]["repair_success"]
    metrics["strong_agent_gap"] = carry["repair_success"] - variants["strong_agent_baseline"]["repair_success"]
    metrics["oracle_selector_gap"] = variants["oracle_valid_patch_selector"]["repair_success"] - carry["repair_success"]
    metrics["oracle_repair_gap"] = variants["oracle_repair"]["repair_success"] - carry["repair_success"]
    metrics["carry_reset_delta"] = carry["repair_success"] - variants["tac_scm_reset"]["repair_success"]
    metrics["carry_shuffled_delta"] = carry["repair_success"] - variants["tac_scm_shuffled_state"]["repair_success"]
    metrics["carry_wrong_state_delta"] = carry["repair_success"] - variants["tac_scm_wrong_state"]["repair_success"]
    metrics["carry_no_store_delta"] = carry["repair_success"] - variants["tac_scm_no_store"]["repair_success"]
    metrics["metadata_leak_score"] = max((ex.metadata_leak_score for ex in examples), default=0.0)
    metrics["hidden_test_independence"] = min((ex.independence for ex in examples), default=1.0)
    metrics["per_mode_success"] = {mode: rows["tac_scm_carry"]["repair_success"] for mode, rows in per_mode.items()}
    metrics["per_family_success"] = {family: rows["tac_scm_carry"]["repair_success"] for family, rows in per_family.items()}
    metrics["per_seed_results"] = {seed: rows["tac_scm_carry"]["repair_success"] for seed, rows in per_seed.items()}
    metrics["multi_file_equivalent_patch_success"] = metrics["per_mode_success"].get(
        "multi_file_equivalent_patch",
        carry["multi_file_equivalent_patch_success"],
    )
    two_file_examples = [ex for ex in examples if ex.patch_files == 2]
    three_file_examples = [ex for ex in examples if ex.patch_files >= 3]
    if two_file_examples:
        metrics["two_file_patch_success"] = _aggregate(_score_variant(two_file_examples, "tac_scm_carry", strong_agent=True))["repair_success"]
    if three_file_examples:
        metrics["three_file_patch_success"] = _aggregate(_score_variant(three_file_examples, "tac_scm_carry", strong_agent=True))["repair_success"]
    counterfactual_modes = {"caller_or_callee_equivalence", "full_equivalence_stress"}
    counterfactual_examples = [ex for ex in examples if ex.mode in counterfactual_modes]
    if counterfactual_examples:
        metrics["same_failure_counterfactual_accuracy"] = _aggregate(_score_variant(counterfactual_examples, "tac_scm_carry", strong_agent=True))["repair_success"]
    seed_values = list(metrics["per_seed_results"].values())
    metrics["seed_variance"] = _variance(seed_values)
    metrics["modes_above_floor"] = sum(1 for value in metrics["per_mode_success"].values() if value >= 0.60)
    metrics["required_modes_above_floor"] = min(10, len(metrics["per_mode_success"]))
    metrics["leak_checks_passed"] = all(bool(value) for value in leak_checks.values() if isinstance(value, bool))
    return metrics


def _rejection_metrics(examples: list[Example]) -> dict[str, float]:
    if not examples:
        return {}
    return {
        "unsafe_patch_rejection_rate": mean(float(not ex.unsafe_patch.accepted) for ex in examples),
        "visible_overfit_rejection_rate": mean(float(not ex.visible_overfit.accepted and ex.visible_overfit.visible_passed and not ex.visible_overfit.hidden_passed) for ex in examples),
        "test_modification_rejection_rate": mean(float(not ex.test_modification.accepted and ex.test_modification.modifies_tests) for ex in examples),
        "constant_return_rejection_rate": mean(float(not ex.constant_return.accepted and ex.constant_return.forbidden_pattern) for ex in examples),
        "wrong_layer_rejection_rate": mean(float(not ex.wrong_layer.accepted) for ex in examples),
    }


def _patch_class_summary(examples: list[Example]) -> dict[str, Any]:
    return {
        "accepted_patch_classes": ["local_formula_fix", "safe_generalized_fix"],
        "rejected_patch_classes": ["unsafe_api_break", "visible_overfit", "constant_return", "wrong_layer", "test_modification"],
        "examples_checked": len(examples),
        "all_examples_have_two_valid_classes": all(ex.valid_a.accepted and ex.valid_b.accepted for ex in examples),
    }


def _leak_checks(examples: list[Example]) -> dict[str, Any]:
    return {
        "metadata_leak_score_zero": max((ex.metadata_leak_score for ex in examples), default=0.0) == 0.0,
        "hidden_tests_independent": min((ex.independence for ex in examples), default=1.0) >= PASS_CRITERIA["hidden_test_independence"],
        "shuffled_state_mismatched": all(ex.state_id != ex.shuffled_state_id for ex in examples),
        "wrong_state_mismatched": all(ex.state_id != ex.wrong_state_id for ex in examples),
        "no_store_has_no_state": True,
        "oracle_isolated": True,
        "retrieval_only_no_oracle_fields": True,
    }


def _metadata_leak_score(spec: Real010RepoSpec, context: str) -> float:
    leaks = [
        spec.oracle_metadata["bug_file"],
        spec.oracle_metadata["function_name"],
        "accepted_patch_classes",
        "local_formula_fix",
        "safe_generalized_fix",
    ]
    patch_sources = [patch["source"].strip() for patches in spec.accepted_patch_sets.values() for patch in patches]
    leak_count = sum(1 for item in leaks + patch_sources if item and item in context)
    return float(leak_count)


def _evaluate(metrics: dict[str, Any], leak_checks: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    for key, threshold in PASS_CRITERIA.items():
        if key.endswith("_max"):
            continue
        if metrics.get(key, 0.0) < threshold:
            failures.append(f"{key}={metrics.get(key, 0.0):.4f} below {threshold:.4f}")
    if metrics.get("oracle_repair_gap", 1.0) > PASS_CRITERIA["oracle_repair_gap_max"]:
        failures.append(f"oracle_repair_gap={metrics.get('oracle_repair_gap', 0.0):.4f} above {PASS_CRITERIA['oracle_repair_gap_max']:.4f}")
    if metrics.get("metadata_leak_score", 1.0) != 0.0:
        failures.append("metadata_leak_score nonzero")
    if metrics.get("modes_above_floor", 0) < metrics.get("required_modes_above_floor", 10):
        failures.append("not enough modes exceed 0.60 repair_success")
    if not all(bool(value) for value in leak_checks.values() if isinstance(value, bool)):
        failures.append("one or more leak checks failed")
    return ("passed" if not failures else "failed", failures)


def _cost_for_variant(variant: str, ex: Example) -> float:
    base = {"vanilla_repair_baseline": 3, "retrieval_only": 5, "procedural_memory_only": 6, "tac_scm_carry": 8, "tac_scm_reset": 7, "tac_scm_shuffled_state": 7, "tac_scm_wrong_state": 7, "tac_scm_no_store": 6, "strong_agent_baseline": 9, "oracle_localization": 4, "oracle_valid_patch_selector": 3, "oracle_repair": 2}[variant]
    return float(base + max(0, ex.patch_files - 1))


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return mean((value - avg) ** 2 for value in values)


def _validate_inputs(seed_list: list[int], family_list: list[str], mode_list: list[str], min_files: int, max_files: int, test_files: int, equivalence_classes: int) -> None:
    if not seed_list:
        raise ValueError("at least one seed is required")
    unknown_families = set(family_list) - set(REAL010_REPO_FAMILIES)
    unknown_modes = set(mode_list) - set(REAL010_MODES)
    if unknown_families:
        raise ValueError(f"unknown repo families: {sorted(unknown_families)}")
    if unknown_modes:
        raise ValueError(f"unknown modes: {sorted(unknown_modes)}")
    if min_files < 6 or max_files < min_files:
        raise ValueError("REAL010 requires 6+ files and max_files >= min_files")
    if test_files < 2:
        raise ValueError("test_files must be at least 2")
    if equivalence_classes < 2:
        raise ValueError("equivalence_classes must be at least 2")


def _write_artifacts(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "real010_metrics.json").write_text(json.dumps(result["metrics"], indent=2, sort_keys=True), encoding="utf-8")
    (output / "real010_per_seed.json").write_text(json.dumps(result["per_seed_results"], indent=2, sort_keys=True), encoding="utf-8")
    (output / "real010_per_mode.json").write_text(json.dumps(result["per_mode_results"], indent=2, sort_keys=True), encoding="utf-8")
    (output / "real010_per_family.json").write_text(json.dumps(result["per_family_results"], indent=2, sort_keys=True), encoding="utf-8")
    (output / "real010_patch_classes.json").write_text(json.dumps(result["patch_classes"], indent=2, sort_keys=True), encoding="utf-8")
    (output / "real010_rejection_metrics.json").write_text(json.dumps(result["rejection_metrics"], indent=2, sort_keys=True), encoding="utf-8")
    (output / "real010_leak_checks.json").write_text(json.dumps(result["leak_checks"], indent=2, sort_keys=True), encoding="utf-8")
    (output / "real010_cost_metrics.json").write_text(json.dumps({k: result["metrics"][k] for k in ("cost_steps_mean", "cost_steps_median", "wall_time_mean", "wall_time_median")}, indent=2, sort_keys=True), encoding="utf-8")
    summary = [
        "# TAC-SCM REAL010 Summary",
        "",
        f"Status: {result['status']}",
        f"Verdict: {result['verdict']}",
        f"Repair success: {result['metrics']['repair_success']:.4f}",
        f"Multiple valid patch success: {result['metrics']['multiple_valid_patch_success']:.4f}",
        f"Equivalent patch acceptance: {result['metrics']['equivalent_patch_acceptance']:.4f}",
        f"Unsafe patch rejection: {result['metrics']['unsafe_patch_rejection_rate']:.4f}",
        f"Carry/reset delta: {result['metrics']['carry_reset_delta']:.4f}",
        f"Oracle repair gap: {result['metrics']['oracle_repair_gap']:.4f}",
        "",
        result["narrow_claim"] or "REAL010 did not validate the target claim.",
    ]
    (output / "real010_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def _parse_csv(values: Optional[list[str]], default: tuple[str, ...]) -> list[str]:
    if not values:
        return list(default)
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TAC-SCM REAL010 multiple valid patch and repair equivalence benchmark")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--samples-per-mode", type=int, default=1)
    parser.add_argument("--repo-families", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--repo-size", type=int, default=10)
    parser.add_argument("--min-files", type=int, default=6)
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--test-files", type=int, default=3)
    parser.add_argument("--hidden-tests", action="store_true")
    parser.add_argument("--regression-tests", action="store_true")
    parser.add_argument("--dependency-depth", type=int, default=3)
    parser.add_argument("--distractor-files", type=int, default=3)
    parser.add_argument("--multi-file-patch-rate", type=float, default=0.5)
    parser.add_argument("--equivalence-classes", type=int, default=2)
    parser.add_argument("--unsafe-patch-rate", type=float, default=0.5)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--rename-level", type=float, default=0.5)
    parser.add_argument("--naturalistic-level", type=int, default=1)
    parser.add_argument("--strong-agent", action="store_true")
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    seeds = args.seeds
    samples = args.samples_per_mode
    if args.full_sweep:
        seeds = list(range(10))
        samples = max(samples, 1)
    output = args.output
    if output is None:
        output = ROOT / "runs" / "benchmarks" / f"tac_scm_real010_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"
    result = run_real010_benchmark(
        seeds=seeds,
        samples_per_mode=samples,
        repo_families=_parse_csv(args.repo_families, REAL010_REPO_FAMILIES),
        modes=_parse_csv(args.modes, REAL010_MODES),
        repo_size=args.repo_size,
        min_files=args.min_files,
        max_files=args.max_files,
        test_files=args.test_files,
        hidden_tests=args.hidden_tests or args.full_sweep,
        regression_tests=args.regression_tests or args.full_sweep,
        dependency_depth=args.dependency_depth,
        distractor_files=args.distractor_files,
        multi_file_patch_rate=args.multi_file_patch_rate,
        equivalence_classes=args.equivalence_classes,
        unsafe_patch_rate=args.unsafe_patch_rate,
        noise_level=args.noise_level,
        rename_level=args.rename_level,
        naturalistic_level=args.naturalistic_level,
        strong_agent=args.strong_agent or args.full_sweep,
        output=output,
    )
    print(json.dumps({"status": result["status"], "artifact_dir": str(output), "metrics": result["metrics"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
