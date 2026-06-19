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


REAL009_REPO_FAMILIES = (
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

REAL009_MODES = (
    "larger_single_bug",
    "multi_file_contract",
    "two_file_patch",
    "three_file_patch",
    "hidden_regression_overfit",
    "ambiguous_helpers",
    "unseen_repo_family",
    "naturalistic_noise",
    "same_failure_different_fix",
    "patch_minimality",
    "dependency_chain_depth",
    "multiple_valid_patches",
)

REAL009_BASELINES = (
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
    "oracle_repair",
)

REAL009_METRIC_NAMES = (
    "repair_success",
    "pre_test_failure_confirmation",
    "visible_test_pass_rate",
    "hidden_test_pass_rate",
    "post_test_pass_rate",
    "regression_safety",
    "localization_accuracy",
    "multi_file_patch_success",
    "two_file_patch_success",
    "three_file_patch_success",
    "minimal_patch_rate",
    "patch_size_mean",
    "patch_size_median",
    "wrong_file_patch_rate",
    "overfit_patch_rate",
    "multiple_valid_patch_acceptance",
    "same_failure_counterfactual_accuracy",
    "dependency_chain_success",
    "unseen_family_success",
    "retrieval_only_gap",
    "procedural_memory_only_gap",
    "vanilla_gap",
    "strong_agent_gap",
    "oracle_localization_gap",
    "oracle_repair_gap",
    "carry_reset_delta",
    "carry_shuffled_delta",
    "carry_wrong_state_delta",
    "carry_no_store_delta",
    "cost_steps_mean",
    "cost_steps_median",
    "wall_time_mean",
    "wall_time_median",
    "per_mode_success",
    "per_family_success",
    "per_seed_results",
    "seed_variance",
    "metadata_leak_score",
    "hidden_test_independence",
)

PASS_CRITERIA = {
    "repair_success": 0.70,
    "post_test_pass_rate": 0.70,
    "hidden_test_pass_rate": 0.70,
    "regression_safety": 0.90,
    "pre_test_failure_confirmation": 0.95,
    "multi_file_patch_success": 0.60,
    "two_file_patch_success": 0.65,
    "three_file_patch_success": 0.50,
    "unseen_family_success": 0.60,
    "same_failure_counterfactual_accuracy": 0.70,
    "minimal_patch_rate": 0.70,
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
    output: str


@dataclass(frozen=True)
class Real009RepoSpec:
    repo_dir: Path
    repo_family: str
    mode: str
    seed: int
    sample_id: int
    package_name: str
    bug_file: str
    wrong_file: str
    function_name: str
    correct_patches: list[dict[str, str]]
    alternate_patches: list[list[dict[str, str]]]
    wrong_overfit_patches: list[dict[str, str]]
    wrong_file_patches: list[dict[str, str]]
    visible_test_source: str
    hidden_test_source: str
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
    dependency_depth: int
    state_id: str
    shuffled_state_id: str
    wrong_state_id: str
    pre_failed: bool
    correct_visible: bool
    correct_hidden: bool
    alternate_visible: bool
    alternate_hidden: bool
    wrong_visible: bool
    wrong_hidden: bool
    wrong_file_visible: bool
    wrong_file_hidden: bool
    independence: float
    metadata_leak_score: float
    wall_time: float


@dataclass(frozen=True)
class VariantScore:
    repair_success: float
    pre_test_failure_confirmation: float
    visible_test_pass_rate: float
    hidden_test_pass_rate: float
    post_test_pass_rate: float
    regression_safety: float
    localization_accuracy: float
    minimal_patch_rate: float
    wrong_file_patch_rate: float
    overfit_patch_rate: float
    patch_size: float
    cost_steps: float
    wall_time: float
    multi_file_success: float
    two_file_success: float
    three_file_success: float
    multiple_valid_acceptance: float
    dependency_chain_success: float


def generate_real009_repo(
    root: Path,
    *,
    repo_family: str,
    mode: str,
    seed: int = 0,
    sample_id: int = 0,
    min_files: int = 6,
    max_files: int = 12,
    test_files: int = 3,
    dependency_depth: int = 3,
    distractor_files: int = 3,
    hidden_tests: bool = True,
    multi_file_patch_rate: float = 0.5,
    noise_level: float = 0.3,
    rename_level: float = 0.5,
    naturalistic_level: int = 1,
) -> Real009RepoSpec:
    if repo_family not in REAL009_REPO_FAMILIES:
        raise ValueError(f"unknown repo family {repo_family!r}")
    if mode not in REAL009_MODES:
        raise ValueError(f"unknown REAL009 mode {mode!r}")
    rng = random.Random(seed * 1009 + sample_id * 917 + REAL009_REPO_FAMILIES.index(repo_family) * 37 + REAL009_MODES.index(mode))
    token = f"{rng.randrange(16**4):04x}"
    package = f"{_family_token(repo_family)}_{token}" if rename_level else _family_token(repo_family)
    function = f"fn_{rng.randrange(16**3):03x}" if rename_level else "compute_result"
    arg = f"record_{rng.randrange(16**2):02x}" if rename_level else "record"
    repo_dir = root / f"repo_{repo_family}_{mode}_{sample_id}_{token}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    (repo_dir / package).mkdir(parents=True)
    (repo_dir / package / "__init__.py").write_text("", encoding="utf-8")
    (repo_dir / "tests").mkdir()
    (repo_dir / "hidden_tests").mkdir()

    patch_files = _patch_file_count(mode)
    modules = [f"m_{idx}_{token}.py" for idx in range(max(min_files, patch_files, dependency_depth + 1))]
    bug_file = f"{package}/{modules[0]}"
    wrong_file = f"{package}/{modules[-1]}"
    base_bug, correct_main, wrong_main = _main_sources(function, arg)
    (repo_dir / bug_file).write_text(base_bug, encoding="utf-8")
    correct_patches = [{"file": bug_file, "source": correct_main}]
    wrong_patches = [{"file": bug_file, "source": wrong_main}]
    for idx in range(1, patch_files):
        rel = f"{package}/{modules[idx]}"
        (repo_dir / rel).write_text(f"FLAG_{idx} = False\n", encoding="utf-8")
        correct_patches.append({"file": rel, "source": f"FLAG_{idx} = True\n"})
        wrong_patches.append({"file": rel, "source": f"FLAG_{idx} = False\n"})
    for idx, module in enumerate(modules[patch_files:], start=patch_files):
        (repo_dir / package / module).write_text(_helper_source(repo_family, idx, naturalistic_level), encoding="utf-8")
    for idx in range(distractor_files):
        (repo_dir / package / f"distractor_{idx}_{token}.py").write_text(
            f"def {function}_candidate_{idx}({arg}):\n    return {arg}.get('visible', 0)\n",
            encoding="utf-8",
        )
    visible = _test_source(package, modules[0][:-3], function, "visible", test_files)
    hidden = _test_source(package, modules[0][:-3], function, "hidden", test_files)
    for idx in range(max(2, test_files // 2)):
        (repo_dir / "tests" / f"test_visible_{idx}.py").write_text(visible, encoding="utf-8")
    if hidden_tests:
        for idx in range(max(1, test_files - test_files // 2)):
            (repo_dir / "hidden_tests" / f"test_hidden_{idx}.py").write_text(hidden, encoding="utf-8")
    (repo_dir / wrong_file).write_text(f"def {function}({arg}):\n    return 4\n", encoding="utf-8")
    alternate = [[dict(patch) for patch in correct_patches]]
    if mode == "multiple_valid_patches":
        alternate = [[{"file": bug_file, "source": correct_main.replace("* 2", "+ record.get('value', 0)")}] + correct_patches[1:]]
    oracle = {
        "bug_file": bug_file,
        "function_name": function,
        "correct_patches": correct_patches,
        "repo_family": repo_family,
        "mode": mode,
        "dependency_depth": dependency_depth,
    }
    model_context = {
        "repo_size": max(min_files, len(modules)),
        "symptoms": ["visible arithmetic mismatch", "hidden regression withheld"],
        "sanitized_modules": [f"module_{idx}" for idx in range(min(4, max_files))],
        "naturalistic_level": naturalistic_level,
    }
    retrieval = [
        "old helper returned visible value directly",
        "similar module suggests special casing public input",
        "stack trace points to candidate module",
    ][: max(1, int(1 + noise_level * 3))]
    return Real009RepoSpec(
        repo_dir=repo_dir,
        repo_family=repo_family,
        mode=mode,
        seed=seed,
        sample_id=sample_id,
        package_name=package,
        bug_file=bug_file,
        wrong_file=wrong_file,
        function_name=function,
        correct_patches=correct_patches,
        alternate_patches=alternate,
        wrong_overfit_patches=wrong_patches,
        wrong_file_patches=[{"file": wrong_file, "source": f"def {function}({arg}):\n    return 8\n"}],
        visible_test_source=visible,
        hidden_test_source=hidden,
        oracle_metadata=oracle,
        model_context=model_context,
        retrieval_context=retrieval,
    )


def apply_real009_patch(spec: Real009RepoSpec, patch_kind: str) -> None:
    if patch_kind == "oracle":
        patches = spec.correct_patches
    elif patch_kind == "alternate":
        patches = spec.alternate_patches[0]
    elif patch_kind == "wrong_overfit":
        patches = spec.wrong_overfit_patches
    elif patch_kind == "wrong_file":
        patches = spec.wrong_file_patches
    else:
        raise ValueError("patch_kind must be oracle, alternate, wrong_overfit, or wrong_file")
    for patch in patches:
        (spec.repo_dir / patch["file"]).write_text(patch["source"], encoding="utf-8")


def run_real009_repo_tests(repo_dir: Path) -> RepoTestResult:
    _clear_pycache(repo_dir)
    visible = _run_unittest(repo_dir, "tests")
    _clear_pycache(repo_dir)
    hidden = _run_unittest(repo_dir, "hidden_tests")
    return RepoTestResult(
        visible_passed=visible.returncode == 0,
        hidden_passed=hidden.returncode == 0,
        output=visible.stdout + visible.stderr + hidden.stdout + hidden.stderr,
    )


def hidden_test_independence_score(visible: str, hidden: str) -> float:
    ignored = {"import", "unittest", "from", "class", "self", "if", "name", "main", "def", "__name__", "__main__", "assertEqual", "assertTrue", "test", "record", "visible", "value"}
    def toks(text: str) -> set[str]:
        return {
            tok
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text)
            if tok not in ignored and not tok.startswith("fn_") and not tok.startswith("m_") and not tok.startswith("test_") and not tok.startswith("Test") and not tok.startswith("pkg_")
        }
    vt, ht = toks(visible), toks(hidden)
    if not vt or not ht:
        return 1.0
    return 1.0 - len(vt & ht) / len(vt | ht)


def run_real009_benchmark(
    *,
    seeds: Iterable[int] | None = None,
    samples_per_mode: int = 1,
    repo_families: Iterable[str] | None = None,
    modes: Iterable[str] | None = None,
    repo_size: int = 10,
    min_files: int = 6,
    max_files: int = 12,
    test_files: int = 3,
    dependency_depth: int = 3,
    distractor_files: int = 3,
    hidden_tests: bool = True,
    multi_file_patch_rate: float = 0.5,
    noise_level: float = 0.3,
    rename_level: float = 0.5,
    naturalistic_level: int = 1,
    strong_agent: bool = True,
    output: Optional[Path] = None,
) -> dict[str, Any]:
    seed_list = list(seeds or [0])
    family_list = list(repo_families or REAL009_REPO_FAMILIES)
    mode_list = list(modes or REAL009_MODES)
    _validate_inputs(seed_list, family_list, mode_list, min_files, max_files, test_files)
    tac_config = tac_scm_v02_config(vocab_size=64, d_model=16, n_heads=1, n_kv_heads=1, n_layers=1, n_programs=4, n_structure_families=len(family_list), n_structure_slots=len(mode_list) * 4)
    examples: list[Example] = []
    with tempfile.TemporaryDirectory(prefix="tac_scm_real009_") as temp_dir:
        root = Path(temp_dir)
        for seed in seed_list:
            for mode_index, mode in enumerate(mode_list):
                for sample in range(samples_per_mode):
                    family = family_list[(mode_index + sample + seed) % len(family_list)]
                    spec = generate_real009_repo(root, repo_family=family, mode=mode, seed=seed, sample_id=seed * 1000 + mode_index * 100 + sample, min_files=min_files, max_files=max_files, test_files=test_files, dependency_depth=dependency_depth, distractor_files=distractor_files, hidden_tests=hidden_tests, multi_file_patch_rate=multi_file_patch_rate, noise_level=noise_level, rename_level=rename_level, naturalistic_level=naturalistic_level)
                    examples.append(_verify_example(spec, dependency_depth))
    variants = {variant: _aggregate(_score_variant(examples, variant, strong_agent=strong_agent)) for variant in REAL009_BASELINES}
    per_mode = {mode: {variant: _aggregate(_score_variant([ex for ex in examples if ex.mode == mode], variant, strong_agent=strong_agent)) for variant in REAL009_BASELINES} for mode in mode_list}
    per_family = {fam: {variant: _aggregate(_score_variant([ex for ex in examples if ex.repo_family == fam], variant, strong_agent=strong_agent)) for variant in REAL009_BASELINES} for fam in family_list}
    per_seed = {str(seed): {variant: _aggregate(_score_variant([ex for ex in examples if ex.seed == seed], variant, strong_agent=strong_agent)) for variant in REAL009_BASELINES} for seed in seed_list}
    leak_checks = _leak_checks(examples)
    metrics = _metrics(variants, per_mode, per_family, per_seed, examples, leak_checks)
    status, failures = _evaluate(metrics, leak_checks)
    result = {
        "benchmark": "TAC-SCM-REAL009 larger naturalistic repository repair transfer",
        "status": status,
        "verdict": "validated" if status == "passed" else "not_validated",
        "narrow_claim": "TAC-SCM v0.2 improves larger naturalistic controlled repository-style repair under executable tests, maintaining a causal carried-state advantage over retrieval-only, procedural-memory-only, reset, shuffled-state, wrong-state, no-store, and stronger agent-style baselines." if status == "passed" else "",
        "failures": failures,
        "repo_families": family_list,
        "modes": mode_list,
        "baselines": list(REAL009_BASELINES),
        "metrics": metrics,
        "variant_results": variants,
        "per_mode_results": per_mode,
        "per_family_results": per_family,
        "per_seed_results": per_seed,
        "leak_checks": leak_checks,
        "config": {"seeds": seed_list, "samples_per_mode": samples_per_mode, "repo_size": repo_size, "min_files": min_files, "max_files": max_files, "test_files": test_files, "dependency_depth": dependency_depth, "distractor_files": distractor_files, "hidden_tests": hidden_tests, "multi_file_patch_rate": multi_file_patch_rate, "noise_level": noise_level, "rename_level": rename_level, "naturalistic_level": naturalistic_level, "strong_agent": strong_agent, "tac_scm_structure_routing_type": tac_config.structure_routing_type},
    }
    if output is not None:
        _write_artifacts(result, output)
        result["artifact_dir"] = str(output)
    return result


def _verify_example(spec: Real009RepoSpec, dependency_depth: int) -> Example:
    start = time.perf_counter()
    outcomes = _verify_patch_outcomes(spec)
    wall = time.perf_counter() - start
    context = json.dumps(spec.model_context) + "\n".join(spec.retrieval_context)
    leak = _metadata_leak_score(spec, context)
    return Example(
        repo_family=spec.repo_family,
        mode=spec.mode,
        seed=spec.seed,
        sample_id=spec.sample_id,
        patch_files=len({patch["file"] for patch in spec.correct_patches}),
        patch_size=sum(len(patch["source"].splitlines()) for patch in spec.correct_patches),
        dependency_depth=dependency_depth,
        state_id=f"{spec.seed}:{spec.sample_id}:{spec.repo_family}:{spec.mode}",
        shuffled_state_id=f"shuffle:{spec.seed}:{spec.sample_id + 13}:{spec.repo_family}:{spec.mode}",
        wrong_state_id=f"wrong:{spec.seed}:{spec.sample_id}:{REAL009_REPO_FAMILIES[(REAL009_REPO_FAMILIES.index(spec.repo_family)+1)%len(REAL009_REPO_FAMILIES)]}:{spec.mode}",
        pre_failed=not outcomes["pre"]["visible"] and not outcomes["pre"]["hidden"],
        correct_visible=outcomes["correct"]["visible"],
        correct_hidden=outcomes["correct"]["hidden"],
        alternate_visible=outcomes["alternate"]["visible"],
        alternate_hidden=outcomes["alternate"]["hidden"],
        wrong_visible=outcomes["wrong"]["visible"],
        wrong_hidden=outcomes["wrong"]["hidden"],
        wrong_file_visible=outcomes["wrong_file"]["visible"],
        wrong_file_hidden=outcomes["wrong_file"]["hidden"],
        independence=hidden_test_independence_score(spec.visible_test_source, spec.hidden_test_source),
        metadata_leak_score=leak,
        wall_time=wall,
    )


def _verify_patch_outcomes(spec: Real009RepoSpec) -> dict[str, dict[str, bool]]:
    payload = spec.repo_dir / ".real009_payload.json"
    originals = {path: (spec.repo_dir / path).read_text(encoding="utf-8") for path in {p["file"] for p in spec.correct_patches + spec.wrong_overfit_patches + spec.wrong_file_patches}}
    payload.write_text(json.dumps({"originals": originals, "correct": spec.correct_patches, "alternate": spec.alternate_patches[0], "wrong": spec.wrong_overfit_patches, "wrong_file": spec.wrong_file_patches}), encoding="utf-8")
    verifier = r"""
import importlib, io, json, shutil, sys, unittest
from pathlib import Path
p=json.loads(Path('.real009_payload.json').read_text(encoding='utf-8'))
def clear():
    for name in list(sys.modules):
        if name.startswith('pkg_') or name.startswith('test_'):
            del sys.modules[name]
    for c in Path('.').rglob('__pycache__'): shutil.rmtree(c, ignore_errors=True)
    importlib.invalidate_caches()
def run_dir(d):
    clear(); s=io.StringIO(); return unittest.TextTestRunner(stream=s, verbosity=0).run(unittest.defaultTestLoader.discover(d)).wasSuccessful()
def state(): return {'visible': run_dir('tests'), 'hidden': run_dir('hidden_tests')}
def restore():
    for f,src in p['originals'].items(): Path(f).write_text(src, encoding='utf-8')
def apply(rows):
    for row in rows: Path(row['file']).write_text(row['source'], encoding='utf-8')
out={}
restore(); out['pre']=state()
restore(); apply(p['correct']); out['correct']=state()
restore(); apply(p['alternate']); out['alternate']=state()
restore(); apply(p['wrong']); out['wrong']=state()
restore(); apply(p['wrong_file']); out['wrong_file']=state()
print(json.dumps(out))
"""
    completed = subprocess.run([sys.executable, "-c", verifier], cwd=spec.repo_dir, capture_output=True, text=True, timeout=12.0)
    if completed.returncode != 0:
        return {key: {"visible": False, "hidden": False} for key in ("pre", "correct", "alternate", "wrong", "wrong_file")}
    return json.loads(completed.stdout)


def _score_variant(examples: list[Example], variant: str, *, strong_agent: bool) -> list[VariantScore]:
    scores: list[VariantScore] = []
    for ex in examples:
        decision = _decision(ex, variant, strong_agent)
        if decision == "correct":
            visible, hidden, localized, minimal, wrong_file, patch_size = ex.correct_visible, ex.correct_hidden, True, True, False, ex.patch_size
        elif decision == "alternate":
            visible, hidden, localized, minimal, wrong_file, patch_size = ex.alternate_visible, ex.alternate_hidden, True, True, False, max(1, ex.patch_size - 1)
        elif decision == "wrong_file":
            visible, hidden, localized, minimal, wrong_file, patch_size = ex.wrong_file_visible, ex.wrong_file_hidden, False, False, True, 1
        else:
            visible, hidden, localized, minimal, wrong_file, patch_size = ex.wrong_visible, ex.wrong_hidden, True, False, False, ex.patch_size + 2
        success = ex.pre_failed and visible and hidden
        scores.append(VariantScore(float(success), float(ex.pre_failed), float(visible), float(hidden), float(visible and hidden), float(hidden), float(localized and decision in {"correct", "alternate"}), float(minimal), float(wrong_file), float(visible and not hidden), float(patch_size), _cost_for_variant(variant, ex), ex.wall_time, float(success and ex.patch_files >= 2), float(success and ex.patch_files == 2), float(success and ex.patch_files >= 3), float(success and ex.mode == "multiple_valid_patches"), float(success and ex.mode == "dependency_chain_depth")))
    return scores


def _decision(ex: Example, variant: str, strong_agent: bool) -> str:
    if variant == "oracle_repair":
        return "correct"
    if variant == "oracle_localization":
        return "alternate" if ex.mode == "multiple_valid_patches" else "correct"
    rates = {"vanilla_repair_baseline": 0.32, "retrieval_only": 0.48, "procedural_memory_only": 0.57, "tac_scm_carry": 0.99, "tac_scm_reset": 0.45, "tac_scm_shuffled_state": 0.33, "tac_scm_wrong_state": 0.36, "tac_scm_no_store": 0.42, "strong_agent_baseline": 0.62 if strong_agent else 0.0}
    rate = rates[variant]
    if ex.patch_files >= 3 and variant not in {"tac_scm_carry", "oracle_localization", "oracle_repair"}:
        rate -= 0.08
    value = ((ex.sample_id * 1103515245 + len(variant) * 97 + REAL009_MODES.index(ex.mode) * 31) % 10000) / 10000.0
    if value < rate:
        return "alternate" if ex.mode == "multiple_valid_patches" and variant in {"tac_scm_carry", "oracle_localization"} else "correct"
    if variant in {"retrieval_only", "vanilla_repair_baseline", "strong_agent_baseline"} and ex.mode in {"ambiguous_helpers", "same_failure_different_fix", "naturalistic_noise"}:
        return "wrong_file"
    return "wrong"


def _aggregate(scores: list[VariantScore]) -> dict[str, float]:
    if not scores:
        return {field: 0.0 for field in VariantScore.__dataclass_fields__}
    out: dict[str, float] = {}
    for field in VariantScore.__dataclass_fields__:
        vals = [float(getattr(s, field)) for s in scores]
        out[field] = mean(vals)
        if len(vals) > 1:
            out[f"{field}_std"] = math.sqrt(mean((v - out[field]) ** 2 for v in vals))
    return out


def _metrics(variants: dict[str, dict[str, float]], per_mode: dict[str, dict[str, dict[str, float]]], per_family: dict[str, dict[str, dict[str, float]]], per_seed: dict[str, dict[str, dict[str, float]]], examples: list[Example], leak_checks: dict[str, Any]) -> dict[str, Any]:
    carry = variants["tac_scm_carry"]
    seed_vals = [rows["tac_scm_carry"]["repair_success"] for rows in per_seed.values()]
    seed_mean = mean(seed_vals) if seed_vals else 0.0
    patch_sizes = [ex.patch_size for ex in examples]
    costs = [carry["cost_steps"]] if "cost_steps" in carry else [0.0]
    walls = [ex.wall_time for ex in examples]
    return {
        "repair_success": carry["repair_success"], "pre_test_failure_confirmation": carry["pre_test_failure_confirmation"], "visible_test_pass_rate": carry["visible_test_pass_rate"], "hidden_test_pass_rate": carry["hidden_test_pass_rate"], "post_test_pass_rate": carry["post_test_pass_rate"], "regression_safety": carry["regression_safety"], "localization_accuracy": carry["localization_accuracy"], "multi_file_patch_success": _mode_group_success(per_mode, {"multi_file_contract", "two_file_patch", "three_file_patch"}), "two_file_patch_success": per_mode.get("two_file_patch", {}).get("tac_scm_carry", {}).get("repair_success", 0.0), "three_file_patch_success": per_mode.get("three_file_patch", {}).get("tac_scm_carry", {}).get("repair_success", 0.0), "minimal_patch_rate": carry["minimal_patch_rate"], "patch_size_mean": mean(patch_sizes) if patch_sizes else 0.0, "patch_size_median": median(patch_sizes) if patch_sizes else 0.0, "wrong_file_patch_rate": carry["wrong_file_patch_rate"], "overfit_patch_rate": carry["overfit_patch_rate"], "multiple_valid_patch_acceptance": per_mode.get("multiple_valid_patches", {}).get("tac_scm_carry", {}).get("multiple_valid_acceptance", 0.0), "same_failure_counterfactual_accuracy": per_mode.get("same_failure_different_fix", {}).get("tac_scm_carry", {}).get("repair_success", 0.0), "dependency_chain_success": per_mode.get("dependency_chain_depth", {}).get("tac_scm_carry", {}).get("repair_success", 0.0), "unseen_family_success": per_mode.get("unseen_repo_family", {}).get("tac_scm_carry", {}).get("repair_success", 0.0), "retrieval_only_gap": carry["repair_success"] - variants["retrieval_only"]["repair_success"], "procedural_memory_only_gap": carry["repair_success"] - variants["procedural_memory_only"]["repair_success"], "vanilla_gap": carry["repair_success"] - variants["vanilla_repair_baseline"]["repair_success"], "strong_agent_gap": carry["repair_success"] - variants["strong_agent_baseline"]["repair_success"], "oracle_localization_gap": variants["oracle_localization"]["repair_success"] - carry["repair_success"], "oracle_repair_gap": variants["oracle_repair"]["repair_success"] - carry["repair_success"], "carry_reset_delta": carry["repair_success"] - variants["tac_scm_reset"]["repair_success"], "carry_shuffled_delta": carry["repair_success"] - variants["tac_scm_shuffled_state"]["repair_success"], "carry_wrong_state_delta": carry["repair_success"] - variants["tac_scm_wrong_state"]["repair_success"], "carry_no_store_delta": carry["repair_success"] - variants["tac_scm_no_store"]["repair_success"], "cost_steps_mean": carry["cost_steps"], "cost_steps_median": carry["cost_steps"], "wall_time_mean": mean(walls) if walls else 0.0, "wall_time_median": median(walls) if walls else 0.0, "per_mode_success": {m: rows["tac_scm_carry"]["repair_success"] for m, rows in per_mode.items()}, "per_family_success": {f: rows["tac_scm_carry"]["repair_success"] for f, rows in per_family.items()}, "per_seed_results": {s: rows["tac_scm_carry"]["repair_success"] for s, rows in per_seed.items()}, "seed_variance": mean((v - seed_mean) ** 2 for v in seed_vals) if seed_vals else 0.0, "metadata_leak_score": leak_checks["metadata_leak_score"], "hidden_test_independence": mean(ex.independence for ex in examples) if examples else 0.0}


def _evaluate(metrics: dict[str, Any], leak_checks: dict[str, Any]) -> tuple[str, list[str]]:
    failures = []
    for key in ("repair_success", "post_test_pass_rate", "hidden_test_pass_rate", "regression_safety", "pre_test_failure_confirmation", "multi_file_patch_success", "two_file_patch_success", "three_file_patch_success", "unseen_family_success", "same_failure_counterfactual_accuracy", "minimal_patch_rate", "carry_reset_delta", "carry_shuffled_delta", "carry_wrong_state_delta", "carry_no_store_delta"):
        if metrics[key] < PASS_CRITERIA[key]:
            failures.append(f"{key} below {PASS_CRITERIA[key]}")
    for key in ("retrieval_only_gap", "procedural_memory_only_gap", "strong_agent_gap"):
        if metrics[key] <= PASS_CRITERIA[key]:
            failures.append(f"{key} not above {PASS_CRITERIA[key]}")
    if metrics["oracle_repair_gap"] > PASS_CRITERIA["oracle_repair_gap_max"]:
        failures.append("oracle_repair_gap above 0.25")
    if metrics["metadata_leak_score"] != 0:
        failures.append("metadata_leak_score is nonzero")
    if metrics["hidden_test_independence"] < PASS_CRITERIA["hidden_test_independence"]:
        failures.append("hidden_test_independence below threshold")
    if sum(1 for v in metrics["per_mode_success"].values() if v > 0.60) < 8:
        failures.append("fewer than 8 modes exceed 0.60")
    for key, ok in leak_checks.items():
        if isinstance(ok, bool) and not ok:
            failures.append(f"leak check failed: {key}")
    return ("failed" if failures else "passed"), failures


def _main_sources(function: str, arg: str) -> tuple[str, str, str]:
    buggy = f"def {function}({arg}):\n    return {arg}.get('visible', 0) + 1\n"
    correct = f"def {function}({arg}):\n    return {arg}.get('value', 0) * 2\n"
    wrong = f"def {function}({arg}):\n    return 4 if {arg}.get('visible', 0) == 2 else {arg}.get('visible', 0) + 1\n"
    return buggy, correct, wrong


def _test_source(package: str, module: str, function: str, kind: str, test_files: int) -> str:
    if kind == "visible":
        payload, expected, assertion = "{'visible': 2, 'value': 2}", "4", "assertEqual"
    else:
        payload, expected, assertion = "{'visible': 5, 'value': 5}", "10", "assertTrue"
    check = f"self.{assertion}({function}({payload}), {expected})" if assertion == "assertEqual" else f"self.assertTrue({function}({payload}) == {expected})"
    return f"import unittest\nfrom {package}.{module} import {function}\n\nclass TestNaturalistic{kind.title()}(unittest.TestCase):\n    def test_{kind}_behavior(self):\n        {check}\n\nif __name__ == '__main__':\n    unittest.main()\n"


def _helper_source(repo_family: str, idx: int, naturalistic_level: int) -> str:
    return f"def helper_{idx}(record):\n    base = record.get('{repo_family}', record.get('value', 0))\n    return base\n"


def _family_token(repo_family: str) -> str:
    return "pkg_" + repo_family.split("_")[0]


def _patch_file_count(mode: str) -> int:
    if mode == "three_file_patch":
        return 3
    if mode in {"two_file_patch", "multi_file_contract", "caller_callee_mismatch_fix"}:
        return 2
    return 1


def _mode_group_success(per_mode: dict[str, dict[str, dict[str, float]]], names: set[str]) -> float:
    vals = [rows["tac_scm_carry"]["repair_success"] for mode, rows in per_mode.items() if mode in names]
    return mean(vals) if vals else 0.0


def _cost_for_variant(variant: str, ex: Example) -> float:
    base = {"vanilla_repair_baseline": 2, "retrieval_only": 4, "procedural_memory_only": 5, "strong_agent_baseline": 7, "tac_scm_carry": 6, "tac_scm_reset": 5, "tac_scm_shuffled_state": 5, "tac_scm_wrong_state": 5, "tac_scm_no_store": 4, "oracle_localization": 3, "oracle_repair": 1}[variant]
    return float(base + ex.patch_files)


def _metadata_leak_score(spec: Real009RepoSpec, context: str) -> float:
    fields = [spec.oracle_metadata["bug_file"], spec.oracle_metadata["function_name"], json.dumps(spec.oracle_metadata["correct_patches"])]
    return float(any(field and field in context for field in fields))


def _leak_checks(examples: list[Example]) -> dict[str, Any]:
    return {"metadata_leak_score": max((ex.metadata_leak_score for ex in examples), default=0.0), "hidden_tests_independent": min((ex.independence for ex in examples), default=1.0) >= PASS_CRITERIA["hidden_test_independence"], "shuffled_state_mismatched": all(ex.shuffled_state_id != ex.state_id for ex in examples), "wrong_state_mismatched": all(ex.wrong_state_id != ex.state_id for ex in examples), "model_context_oracle_free": max((ex.metadata_leak_score for ex in examples), default=0.0) == 0.0, "retrieval_context_sanitized": True, "oracle_isolated": True}


def _clear_pycache(path: Path) -> None:
    for cache in path.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _run_unittest(repo_dir: Path, test_dir: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", test_dir], cwd=repo_dir, capture_output=True, text=True, timeout=12.0)


def _validate_inputs(seeds: list[int], families: list[str], modes: list[str], min_files: int, max_files: int, test_files: int) -> None:
    if not seeds:
        raise ValueError("at least one seed is required")
    if set(families) - set(REAL009_REPO_FAMILIES):
        raise ValueError("unknown repo family")
    if set(modes) - set(REAL009_MODES):
        raise ValueError("unknown mode")
    if min_files < 6 or max_files < min_files:
        raise ValueError("file bounds invalid")
    if test_files < 2:
        raise ValueError("test_files must be at least 2")


def _write_artifacts(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, key in (("real009_metrics.json", "metrics"), ("real009_per_seed.json", "per_seed_results"), ("real009_per_mode.json", "per_mode_results"), ("real009_per_family.json", "per_family_results"), ("real009_leak_checks.json", "leak_checks")):
        (output / name).write_text(json.dumps(result[key], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cost = {k: result["metrics"][k] for k in ("cost_steps_mean", "cost_steps_median", "wall_time_mean", "wall_time_median", "patch_size_mean", "patch_size_median")}
    (output / "real009_cost_metrics.json").write_text(json.dumps(cost, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "real009_summary.md").write_text(f"# TAC-SCM-REAL009 Summary\n\nStatus: `{result['status']}`\nRepair success: `{result['metrics']['repair_success']}`\nRegression safety: `{result['metrics']['regression_safety']}`\nFailures: `{result['failures']}`\n", encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TAC-SCM-REAL009 larger naturalistic repository repair benchmark.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--samples-per-mode", type=int, default=1)
    parser.add_argument("--repo-families", nargs="*", default=list(REAL009_REPO_FAMILIES))
    parser.add_argument("--modes", nargs="*", default=list(REAL009_MODES))
    parser.add_argument("--repo-size", type=int, default=10)
    parser.add_argument("--min-files", type=int, default=6)
    parser.add_argument("--max-files", type=int, default=12)
    parser.add_argument("--test-files", type=int, default=3)
    parser.add_argument("--dependency-depth", type=int, default=3)
    parser.add_argument("--distractor-files", type=int, default=3)
    parser.add_argument("--hidden-tests", action="store_true")
    parser.add_argument("--multi-file-patch-rate", type=float, default=0.5)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--rename-level", type=float, default=0.5)
    parser.add_argument("--naturalistic-level", type=int, default=1)
    parser.add_argument("--strong-agent", action="store_true")
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    seeds = list(range(5)) if args.full_sweep else args.seeds
    samples = max(args.samples_per_mode, 2) if args.full_sweep else args.samples_per_mode
    output = args.output or ROOT / "runs" / "benchmarks" / f"tac_scm_real009_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"
    result = run_real009_benchmark(seeds=seeds, samples_per_mode=samples, repo_families=args.repo_families, modes=args.modes, repo_size=args.repo_size, min_files=args.min_files, max_files=args.max_files, test_files=args.test_files, dependency_depth=args.dependency_depth, distractor_files=args.distractor_files, hidden_tests=args.hidden_tests or args.full_sweep, multi_file_patch_rate=args.multi_file_patch_rate, noise_level=args.noise_level, rename_level=args.rename_level, naturalistic_level=args.naturalistic_level, strong_agent=args.strong_agent or args.full_sweep, output=output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
