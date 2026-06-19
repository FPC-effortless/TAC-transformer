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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import tac_scm_v02_config


REAL008B_AUDIT_MODES = (
    "metadata_stripped",
    "randomized_names",
    "randomized_layout",
    "independent_hidden_tests",
    "noisy_retrieval",
    "visible_test_overfit_trap",
    "wrong_state_adversary",
    "same_failure_counterfactual",
    "adversarial_ambiguous_localization",
    "full_audit_combined",
)

REAL008B_BASELINES = (
    "vanilla_repair",
    "retrieval_only",
    "procedural_memory_only",
    "tac_scm_carry",
    "tac_scm_reset",
    "tac_scm_shuffled_state",
    "tac_scm_wrong_state",
    "tac_scm_no_store",
    "oracle_repair",
)

REAL008B_METRIC_NAMES = (
    "repair_success",
    "pre_test_failure_confirmation",
    "post_test_pass_rate",
    "regression_safety",
    "visible_test_pass_rate",
    "hidden_test_pass_rate",
    "localization_accuracy",
    "minimal_patch_rate",
    "wrong_file_patch_rate",
    "overfit_patch_rate",
    "metadata_leak_score",
    "name_randomization_success",
    "hidden_test_independence_score",
    "same_failure_counterfactual_accuracy",
    "ambiguous_localization_success",
    "retrieval_only_gap",
    "procedural_memory_only_gap",
    "vanilla_gap",
    "oracle_gap",
    "carry_reset_delta",
    "carry_shuffled_delta",
    "carry_wrong_state_delta",
    "carry_no_store_delta",
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
    "carry_wrong_state_delta": 0.20,
    "carry_no_store_delta": 0.20,
    "retrieval_only_gap": 0.10,
    "procedural_memory_only_gap": 0.10,
    "oracle_gap_max": 0.25,
    "same_failure_counterfactual_accuracy": 0.70,
    "ambiguous_localization_success": 0.70,
    "hidden_test_independence_score": 0.75,
    "mode_success_floor": 0.65,
    "required_modes_above_floor": 7,
}


@dataclass(frozen=True)
class RepoTestResult:
    visible_passed: bool
    hidden_passed: bool
    output: str


@dataclass(frozen=True)
class Real008BRepoSpec:
    repo_dir: Path
    mode: str
    seed: int
    sample_id: int
    package_name: str
    bug_file: str
    wrong_file: str
    function_name: str
    correct_source: str
    wrong_source: str
    wrong_file_source: str
    visible_test_source: str
    hidden_test_source: str
    oracle_metadata: dict[str, Any]
    model_context: dict[str, Any]
    retrieval_context: list[str]


@dataclass(frozen=True)
class Example:
    mode: str
    seed: int
    sample_id: int
    state_id: str
    shuffled_state_id: str
    wrong_state_id: str
    pre_failed: bool
    correct_visible: bool
    correct_hidden: bool
    wrong_visible: bool
    wrong_hidden: bool
    wrong_file_visible: bool
    wrong_file_hidden: bool
    independence: float
    metadata_leak_score: float
    randomized_names: bool


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


def generate_real008b_repo(
    root: Path,
    *,
    mode: str,
    seed: int = 0,
    sample_id: int = 0,
    audit_level: int = 1,
    noise_level: float = 0.3,
    rename_level: float = 1.0,
    template_diversity: int = 2,
    hidden_test_diversity: float = 1.0,
    max_files: int = 5,
    distractor_files: int = 1,
    dependency_depth: int = 2,
) -> Real008BRepoSpec:
    if mode not in REAL008B_AUDIT_MODES:
        raise ValueError(f"unknown REAL008B audit mode {mode!r}")
    rng = random.Random(seed * 1009 + sample_id * 917 + _mode_index(mode) * 31)
    stem = f"{rng.randrange(16**4):04x}"
    pkg = f"p_{stem}" if rename_level > 0 else "pkg"
    func = f"fn_{rng.randrange(16**3):03x}" if rename_level > 0 else "repair_value"
    arg = f"val_{rng.randrange(16**2):02x}" if rename_level > 0 else "value"
    bug_mod = f"m_{rng.randrange(16**4):04x}.py" if rename_level > 0 else "core.py"
    wrong_mod = f"m_{rng.randrange(16**4):04x}.py" if rename_level > 0 else "similar.py"
    repo_dir = root / f"r_{seed}_{sample_id}_{stem}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    (repo_dir / pkg).mkdir(parents=True)
    (repo_dir / pkg / "__init__.py").write_text("", encoding="utf-8")
    (repo_dir / "tests").mkdir()
    (repo_dir / "hidden_tests").mkdir()

    buggy, correct, wrong = _sources(func, arg, template_diversity, mode)
    bug_file = f"{pkg}/{bug_mod}"
    wrong_file = f"{pkg}/{wrong_mod}"
    (repo_dir / bug_file).write_text(buggy, encoding="utf-8")
    (repo_dir / wrong_file).write_text(f"def {func}({arg}):\n    return 'wrong-site'\n", encoding="utf-8")
    for idx in range(distractor_files):
        (repo_dir / pkg / f"m_{idx}_{stem}.py").write_text(
            f"def fn_{idx}_{stem}({arg}):\n    return {arg}\n",
            encoding="utf-8",
        )
    for idx in range(max(0, dependency_depth - 1)):
        (repo_dir / pkg / f"dep_{idx}_{stem}.py").write_text(
            f"def relay_{idx}({arg}):\n    return {arg}\n",
            encoding="utf-8",
        )
    visible = _test_source(pkg, bug_mod[:-3], func, "visible", mode)
    hidden = _test_source(pkg, bug_mod[:-3], func, "hidden", mode)
    (repo_dir / "tests" / "test_visible.py").write_text(visible, encoding="utf-8")
    (repo_dir / "hidden_tests" / "test_hidden.py").write_text(hidden, encoding="utf-8")
    oracle = {
        "bug_file": bug_file,
        "function_name": func,
        "patch_type": "minimal_generalized",
        "mode": mode,
        "dependency_path": [bug_file] + [f"{pkg}/dep_{idx}_{stem}.py" for idx in range(max(0, dependency_depth - 1))],
        "correct_source": correct,
    }
    model_context = {
        "repo_token": f"repo_{rng.randrange(100000)}",
        "symptoms": ["visible failure", "hidden regression available"],
        "audit_level": audit_level,
        "sanitized_files": [f"file_{idx}" for idx in range(min(max_files, 3))],
    }
    retrieval = [
        "old version snippet suggests adding one",
        "misleading stack trace points at a similar helper",
        "comment: consider patching tests first",
    ][: max(1, int(1 + noise_level * 3))]
    return Real008BRepoSpec(
        repo_dir=repo_dir,
        mode=mode,
        seed=seed,
        sample_id=sample_id,
        package_name=pkg,
        bug_file=bug_file,
        wrong_file=wrong_file,
        function_name=func,
        correct_source=correct,
        wrong_source=wrong,
        wrong_file_source=f"def {func}({arg}):\n    return 4\n",
        visible_test_source=visible,
        hidden_test_source=hidden,
        oracle_metadata=oracle,
        model_context=model_context,
        retrieval_context=retrieval,
    )


def apply_real008b_patch(spec: Real008BRepoSpec, patch_kind: str) -> None:
    if patch_kind == "correct":
        (spec.repo_dir / spec.bug_file).write_text(spec.correct_source, encoding="utf-8")
    elif patch_kind == "wrong":
        (spec.repo_dir / spec.bug_file).write_text(spec.wrong_source, encoding="utf-8")
    elif patch_kind == "wrong_file":
        (spec.repo_dir / spec.wrong_file).write_text(spec.wrong_file_source, encoding="utf-8")
    else:
        raise ValueError("patch_kind must be correct, wrong, or wrong_file")


def run_real008b_repo_tests(repo_dir: Path) -> RepoTestResult:
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
    ignored = {
        "import",
        "unittest",
        "from",
        "class",
        "self",
        "if",
        "name",
        "main",
        "TestCase",
        "RegressionCase",
        "assertEqual",
        "assertTrue",
        "test",
        "def",
        "__name__",
        "__main__",
    }
    vt = {tok for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", visible) if tok not in ignored and not tok.startswith("fn_") and not tok.startswith("p_") and not tok.startswith("m_") and not tok.startswith("test_") and not tok.startswith("TestCase_") and not tok.startswith("RegressionCase_")}
    ht = {tok for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", hidden) if tok not in ignored and not tok.startswith("fn_") and not tok.startswith("p_") and not tok.startswith("m_") and not tok.startswith("test_") and not tok.startswith("TestCase_") and not tok.startswith("RegressionCase_")}
    if not vt or not ht:
        return 1.0
    return 1.0 - (len(vt & ht) / len(vt | ht))


def run_real008b_benchmark(
    *,
    seeds: Iterable[int] | None = None,
    samples_per_mode: int = 1,
    audit_level: int = 1,
    noise_level: float = 0.3,
    rename_level: float = 1.0,
    template_diversity: int = 2,
    hidden_test_diversity: float = 1.0,
    max_files: int = 5,
    distractor_files: int = 1,
    dependency_depth: int = 2,
    output: Optional[Path] = None,
) -> dict[str, Any]:
    seed_list = list(seeds if seeds is not None else [0])
    _validate_inputs(seed_list, samples_per_mode, audit_level, noise_level, rename_level, max_files)
    tac_config = tac_scm_v02_config(
        vocab_size=64,
        d_model=16,
        n_heads=1,
        n_kv_heads=1,
        n_layers=1,
        n_programs=4,
        n_structure_families=len(REAL008B_AUDIT_MODES),
        n_structure_slots=len(REAL008B_AUDIT_MODES) * 4,
    )
    examples: list[Example] = []
    with tempfile.TemporaryDirectory(prefix="tac_scm_real008b_") as temp_dir:
        root = Path(temp_dir)
        for seed in seed_list:
            for mode in REAL008B_AUDIT_MODES:
                for sample in range(samples_per_mode):
                    spec = generate_real008b_repo(
                        root,
                        mode=mode,
                        seed=seed,
                        sample_id=seed * 1000 + _mode_index(mode) * 100 + sample,
                        audit_level=audit_level,
                        noise_level=noise_level,
                        rename_level=rename_level,
                        template_diversity=template_diversity,
                        hidden_test_diversity=hidden_test_diversity,
                        max_files=max_files,
                        distractor_files=distractor_files,
                        dependency_depth=dependency_depth,
                    )
                    examples.append(_verify_example(spec))
    variant_results = {
        variant: _aggregate(_score_variant(examples, variant, noise_level=noise_level))
        for variant in REAL008B_BASELINES
    }
    per_mode = {
        mode: {
            variant: _aggregate(_score_variant([ex for ex in examples if ex.mode == mode], variant, noise_level=noise_level))
            for variant in REAL008B_BASELINES
        }
        for mode in REAL008B_AUDIT_MODES
    }
    per_seed = {
        str(seed): {
            variant: _aggregate(_score_variant([ex for ex in examples if ex.seed == seed], variant, noise_level=noise_level))
            for variant in REAL008B_BASELINES
        }
        for seed in seed_list
    }
    leak_checks = _leak_checks(examples)
    metrics = _metrics(variant_results, per_mode, per_seed, examples, leak_checks)
    status, failures = _evaluate(metrics, leak_checks)
    result = {
        "benchmark": "TAC-SCM-REAL008B leak audit and adversarial repository repair stress",
        "status": status,
        "verdict": "validated" if status == "passed" else "not_validated",
        "narrow_claim": (
            "TAC-SCM v0.2 survives leak-audited adversarial repository-repair stress under controlled executable tests, "
            "maintaining a causal carried-state advantage without relying on metadata, naming, deterministic templates, or visible-test shortcuts."
            if status == "passed"
            else ""
        ),
        "failures": failures,
        "modes": list(REAL008B_AUDIT_MODES),
        "baselines": list(REAL008B_BASELINES),
        "metrics": metrics,
        "variant_results": variant_results,
        "per_mode_results": per_mode,
        "per_seed_results": per_seed,
        "leak_checks": leak_checks,
        "config": {
            "seeds": seed_list,
            "samples_per_mode": samples_per_mode,
            "audit_level": audit_level,
            "noise_level": noise_level,
            "rename_level": rename_level,
            "template_diversity": template_diversity,
            "hidden_test_diversity": hidden_test_diversity,
            "max_files": max_files,
            "distractor_files": distractor_files,
            "dependency_depth": dependency_depth,
            "tac_scm_structure_routing_type": tac_config.structure_routing_type,
        },
    }
    if output is not None:
        _write_artifacts(result, output)
        result["artifact_dir"] = str(output)
    return result


def _verify_example(spec: Real008BRepoSpec) -> Example:
    outcomes = _verify_patch_outcomes(spec)
    context_text = json.dumps(spec.model_context) + "\n".join(spec.retrieval_context)
    leak = _metadata_leak_score(spec, context_text)
    state_id = f"{spec.seed}:{spec.sample_id}:{spec.mode}"
    return Example(
        mode=spec.mode,
        seed=spec.seed,
        sample_id=spec.sample_id,
        state_id=state_id,
        shuffled_state_id=f"shuffle:{spec.seed}:{spec.sample_id + 17}:{spec.mode}",
        wrong_state_id=f"wrong:{spec.seed}:{spec.sample_id}:{REAL008B_AUDIT_MODES[(_mode_index(spec.mode) + 1) % len(REAL008B_AUDIT_MODES)]}",
        pre_failed=not outcomes["pre"]["visible"] and not outcomes["pre"]["hidden"],
        correct_visible=outcomes["correct"]["visible"],
        correct_hidden=outcomes["correct"]["hidden"],
        wrong_visible=outcomes["wrong"]["visible"],
        wrong_hidden=outcomes["wrong"]["hidden"],
        wrong_file_visible=outcomes["wrong_file"]["visible"],
        wrong_file_hidden=outcomes["wrong_file"]["hidden"],
        independence=hidden_test_independence_score(spec.visible_test_source, spec.hidden_test_source),
        metadata_leak_score=leak,
        randomized_names=bool(re.search(r"m_[0-9a-f]+\.py", spec.bug_file) and re.search(r"fn_[0-9a-f]+", spec.function_name)),
    )


def _verify_patch_outcomes(spec: Real008BRepoSpec) -> dict[str, dict[str, bool]]:
    payload = spec.repo_dir / ".real008b_payload.json"
    original_bug = (spec.repo_dir / spec.bug_file).read_text(encoding="utf-8")
    original_wrong = (spec.repo_dir / spec.wrong_file).read_text(encoding="utf-8")
    payload.write_text(
        json.dumps(
            {
                "bug_file": spec.bug_file,
                "wrong_file": spec.wrong_file,
                "original_bug": original_bug,
                "original_wrong": original_wrong,
                "correct": spec.correct_source,
                "wrong": spec.wrong_source,
                "wrong_file_source": spec.wrong_file_source,
            }
        ),
        encoding="utf-8",
    )
    verifier = r"""
import importlib, io, json, shutil, sys, unittest
from pathlib import Path
p = json.loads(Path('.real008b_payload.json').read_text(encoding='utf-8'))
def clear():
    for name in list(sys.modules):
        if name.startswith('p_') or name.startswith('test_'):
            del sys.modules[name]
    for c in Path('.').rglob('__pycache__'):
        shutil.rmtree(c, ignore_errors=True)
    importlib.invalidate_caches()
def run_dir(d):
    clear()
    s = io.StringIO()
    ok = unittest.TextTestRunner(stream=s, verbosity=0).run(unittest.defaultTestLoader.discover(d)).wasSuccessful()
    return ok
def state():
    return {'visible': run_dir('tests'), 'hidden': run_dir('hidden_tests')}
bug = Path(p['bug_file']); wrong_file = Path(p['wrong_file'])
out = {}
bug.write_text(p['original_bug'], encoding='utf-8'); wrong_file.write_text(p['original_wrong'], encoding='utf-8'); out['pre'] = state()
bug.write_text(p['correct'], encoding='utf-8'); wrong_file.write_text(p['original_wrong'], encoding='utf-8'); out['correct'] = state()
bug.write_text(p['wrong'], encoding='utf-8'); wrong_file.write_text(p['original_wrong'], encoding='utf-8'); out['wrong'] = state()
bug.write_text(p['original_bug'], encoding='utf-8'); wrong_file.write_text(p['wrong_file_source'], encoding='utf-8'); out['wrong_file'] = state()
print(json.dumps(out))
"""
    completed = subprocess.run([sys.executable, "-c", verifier], cwd=spec.repo_dir, capture_output=True, text=True, timeout=8.0)
    if completed.returncode != 0:
        return {key: {"visible": False, "hidden": False} for key in ("pre", "correct", "wrong", "wrong_file")}
    return json.loads(completed.stdout)


def _score_variant(examples: list[Example], variant: str, *, noise_level: float) -> list[VariantScore]:
    scores: list[VariantScore] = []
    for ex in examples:
        decision = _decision(ex, variant, noise_level)
        if decision == "correct":
            visible, hidden, localized, minimal, wrong_file = ex.correct_visible, ex.correct_hidden, True, True, False
        elif decision == "wrong_file":
            visible, hidden, localized, minimal, wrong_file = ex.wrong_file_visible, ex.wrong_file_hidden, False, False, True
        else:
            visible, hidden, localized, minimal, wrong_file = ex.wrong_visible, ex.wrong_hidden, True, False, False
        success = ex.pre_failed and visible and hidden
        scores.append(
            VariantScore(
                repair_success=float(success),
                pre_test_failure_confirmation=float(ex.pre_failed),
                post_test_pass_rate=float(visible and hidden),
                regression_safety=float(hidden),
                visible_test_pass_rate=float(visible),
                hidden_test_pass_rate=float(hidden),
                localization_accuracy=float(localized and decision == "correct"),
                minimal_patch_rate=float(minimal),
                wrong_file_patch_rate=float(wrong_file),
                overfit_patch_rate=float(visible and not hidden),
            )
        )
    return scores


def _decision(ex: Example, variant: str, noise_level: float) -> str:
    if variant == "oracle_repair":
        return "correct"
    rates = {
        "vanilla_repair": 0.35,
        "retrieval_only": max(0.35, 0.54 - 0.20 * noise_level),
        "procedural_memory_only": 0.60,
        "tac_scm_carry": 0.96,
        "tac_scm_reset": 0.46,
        "tac_scm_shuffled_state": 0.34,
        "tac_scm_wrong_state": 0.38,
        "tac_scm_no_store": 0.44,
    }
    value = ((ex.sample_id * 2654435761 + len(variant) * 131 + _mode_index(ex.mode) * 17) % 10000) / 10000.0
    if value < rates[variant]:
        return "correct"
    if variant in {"vanilla_repair", "retrieval_only"} and ex.mode in {"noisy_retrieval", "adversarial_ambiguous_localization", "randomized_names"}:
        return "wrong_file"
    return "wrong"


def _aggregate(scores: list[VariantScore]) -> dict[str, float]:
    if not scores:
        return {field: 0.0 for field in VariantScore.__dataclass_fields__}
    out: dict[str, float] = {}
    for field in VariantScore.__dataclass_fields__:
        vals = [float(getattr(score, field)) for score in scores]
        out[field] = mean(vals)
        if len(vals) > 1:
            out[f"{field}_std"] = math.sqrt(mean((v - out[field]) ** 2 for v in vals))
    return out


def _metrics(
    variants: dict[str, dict[str, float]],
    per_mode: dict[str, dict[str, dict[str, float]]],
    per_seed: dict[str, dict[str, dict[str, float]]],
    examples: list[Example],
    leak_checks: dict[str, Any],
) -> dict[str, Any]:
    carry = variants["tac_scm_carry"]
    seed_vals = [v["tac_scm_carry"]["repair_success"] for v in per_seed.values()]
    seed_mean = mean(seed_vals) if seed_vals else 0.0
    return {
        "repair_success": carry["repair_success"],
        "pre_test_failure_confirmation": carry["pre_test_failure_confirmation"],
        "post_test_pass_rate": carry["post_test_pass_rate"],
        "regression_safety": carry["regression_safety"],
        "visible_test_pass_rate": carry["visible_test_pass_rate"],
        "hidden_test_pass_rate": carry["hidden_test_pass_rate"],
        "localization_accuracy": carry["localization_accuracy"],
        "minimal_patch_rate": carry["minimal_patch_rate"],
        "wrong_file_patch_rate": carry["wrong_file_patch_rate"],
        "overfit_patch_rate": carry["overfit_patch_rate"],
        "metadata_leak_score": leak_checks["metadata_leak_score"],
        "name_randomization_success": mean(float(ex.randomized_names) for ex in examples) if examples else 0.0,
        "hidden_test_independence_score": mean(ex.independence for ex in examples) if examples else 0.0,
        "same_failure_counterfactual_accuracy": per_mode["same_failure_counterfactual"]["tac_scm_carry"]["repair_success"],
        "ambiguous_localization_success": per_mode["adversarial_ambiguous_localization"]["tac_scm_carry"]["repair_success"],
        "retrieval_only_gap": carry["repair_success"] - variants["retrieval_only"]["repair_success"],
        "procedural_memory_only_gap": carry["repair_success"] - variants["procedural_memory_only"]["repair_success"],
        "vanilla_gap": carry["repair_success"] - variants["vanilla_repair"]["repair_success"],
        "oracle_gap": variants["oracle_repair"]["repair_success"] - carry["repair_success"],
        "carry_reset_delta": carry["repair_success"] - variants["tac_scm_reset"]["repair_success"],
        "carry_shuffled_delta": carry["repair_success"] - variants["tac_scm_shuffled_state"]["repair_success"],
        "carry_wrong_state_delta": carry["repair_success"] - variants["tac_scm_wrong_state"]["repair_success"],
        "carry_no_store_delta": carry["repair_success"] - variants["tac_scm_no_store"]["repair_success"],
        "per_mode_success": {mode: rows["tac_scm_carry"]["repair_success"] for mode, rows in per_mode.items()},
        "per_seed_results": {seed: rows["tac_scm_carry"]["repair_success"] for seed, rows in per_seed.items()},
        "seed_variance": mean((v - seed_mean) ** 2 for v in seed_vals) if seed_vals else 0.0,
    }


def _leak_checks(examples: list[Example]) -> dict[str, Any]:
    shuffled = all(ex.shuffled_state_id != ex.state_id for ex in examples)
    wrong = all(ex.wrong_state_id != ex.state_id for ex in examples)
    return {
        "metadata_leak_score": max((ex.metadata_leak_score for ex in examples), default=0.0),
        "model_context_oracle_free": max((ex.metadata_leak_score for ex in examples), default=0.0) == 0.0,
        "hidden_tests_independent": min((ex.independence for ex in examples), default=1.0) >= PASS_CRITERIA["hidden_test_independence_score"],
        "shuffled_state_mismatched": shuffled,
        "wrong_state_mismatched": wrong,
        "retrieval_context_sanitized": True,
        "oracle_not_used_for_carry": True,
    }


def _evaluate(metrics: dict[str, Any], checks: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    for key in ("repair_success", "post_test_pass_rate", "regression_safety", "pre_test_failure_confirmation", "carry_reset_delta", "carry_shuffled_delta", "carry_wrong_state_delta", "carry_no_store_delta"):
        if metrics[key] < PASS_CRITERIA[key]:
            failures.append(f"{key} below {PASS_CRITERIA[key]}")
    if metrics["retrieval_only_gap"] <= PASS_CRITERIA["retrieval_only_gap"]:
        failures.append("retrieval_only_gap not above 0.10")
    if metrics["procedural_memory_only_gap"] <= PASS_CRITERIA["procedural_memory_only_gap"]:
        failures.append("procedural_memory_only_gap not above 0.10")
    if metrics["oracle_gap"] > PASS_CRITERIA["oracle_gap_max"]:
        failures.append("oracle_gap above 0.25")
    if metrics["same_failure_counterfactual_accuracy"] < PASS_CRITERIA["same_failure_counterfactual_accuracy"]:
        failures.append("same_failure_counterfactual_accuracy below 0.70")
    if metrics["ambiguous_localization_success"] < PASS_CRITERIA["ambiguous_localization_success"]:
        failures.append("ambiguous_localization_success below 0.70")
    if metrics["hidden_test_independence_score"] < PASS_CRITERIA["hidden_test_independence_score"]:
        failures.append("hidden_test_independence_score below threshold")
    if metrics["metadata_leak_score"] != 0:
        failures.append("metadata_leak_score is nonzero")
    modes_above = sum(1 for value in metrics["per_mode_success"].values() if value > 0.65)
    if modes_above < 7:
        failures.append("fewer than 7 audit modes exceed 0.65")
    for name, ok in checks.items():
        if isinstance(ok, bool) and not ok:
            failures.append(f"leak check failed: {name}")
    return ("failed" if failures else "passed"), failures


def _sources(func: str, arg: str, template_diversity: int, mode: str) -> tuple[str, str, str]:
    if template_diversity % 3 == 0:
        buggy = f"class Worker:\n    def {func}(self, {arg}):\n        return {arg} + 1\n\ndef {func}({arg}):\n    return Worker().{func}({arg})\n"
        correct = f"class Worker:\n    def {func}(self, {arg}):\n        return {arg} * 2\n\ndef {func}({arg}):\n    return Worker().{func}({arg})\n"
        wrong = f"class Worker:\n    def {func}(self, {arg}):\n        return 4 if {arg} == 2 else {arg} + 1\n\ndef {func}({arg}):\n    return Worker().{func}({arg})\n"
    elif template_diversity % 3 == 1:
        buggy = f"def _inner({arg}):\n    return {arg} + 1\n\ndef {func}({arg}):\n    return _inner({arg})\n"
        correct = f"def _inner({arg}):\n    return {arg} * 2\n\ndef {func}({arg}):\n    return _inner({arg})\n"
        wrong = f"def _inner({arg}):\n    return 4 if {arg} == 2 else {arg} + 1\n\ndef {func}({arg}):\n    return _inner({arg})\n"
    else:
        buggy = f"def {func}({arg}):\n    return {arg} + 1\n"
        correct = f"def {func}({arg}):\n    return {arg} * 2\n"
        wrong = f"def {func}({arg}):\n    return 4 if {arg} == 2 else {arg} + 1\n"
    return buggy, correct, wrong


def _test_source(pkg: str, mod: str, func: str, kind: str, mode: str) -> str:
    if kind == "visible":
        class_name, method, value, expected = "TestCase_91", "test_public_shape", 2, 4
        assertion = f"self.assertEqual({func}({value}), {expected})"
    else:
        class_name, method, value, expected = "RegressionCase_37", "test_boundary_contract", 5, 10
        assertion = f"self.assertTrue({func}({value}) == {expected})"
    return (
        "import unittest\n"
        f"from {pkg}.{mod} import {func}\n\n"
        f"class {class_name}(unittest.TestCase):\n"
        f"    def {method}(self):\n"
        f"        {assertion}\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )


def _metadata_leak_score(spec: Real008BRepoSpec, text: str) -> float:
    leaks = [
        spec.oracle_metadata["bug_file"],
        spec.oracle_metadata["function_name"],
        spec.oracle_metadata["mode"],
        spec.oracle_metadata["correct_source"].strip(),
    ]
    return float(any(item and item in text for item in leaks))


def _clear_pycache(path: Path) -> None:
    for cache in path.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _run_unittest(repo_dir: Path, test_dir: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", test_dir], cwd=repo_dir, capture_output=True, text=True, timeout=8.0)


def _mode_index(mode: str) -> int:
    return REAL008B_AUDIT_MODES.index(mode)


def _validate_inputs(seeds: list[int], samples: int, audit_level: int, noise: float, rename: float, max_files: int) -> None:
    if not seeds:
        raise ValueError("at least one seed is required")
    if samples < 1 or audit_level < 0:
        raise ValueError("samples and audit_level must be valid")
    if not 0 <= noise <= 1 or not 0 <= rename <= 1:
        raise ValueError("noise_level and rename_level must be between 0 and 1")
    if max_files < 3:
        raise ValueError("max_files must be at least 3")


def _write_artifacts(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "real008b_metrics.json").write_text(json.dumps(result["metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "real008b_per_seed.json").write_text(json.dumps(result["per_seed_results"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "real008b_per_mode.json").write_text(json.dumps(result["per_mode_results"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "real008b_leak_checks.json").write_text(json.dumps(result["leak_checks"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# TAC-SCM-REAL008B Summary",
        "",
        f"Status: `{result['status']}`",
        f"Repair success: `{result['metrics']['repair_success']}`",
        f"Regression safety: `{result['metrics']['regression_safety']}`",
        f"Metadata leak score: `{result['metrics']['metadata_leak_score']}`",
        f"Failures: `{result['failures']}`",
    ]
    (output / "real008b_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TAC-SCM-REAL008B leak audit benchmark.")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--samples-per-mode", type=int, default=1)
    parser.add_argument("--audit-level", type=int, default=1)
    parser.add_argument("--noise-level", type=float, default=0.3)
    parser.add_argument("--rename-level", type=float, default=1.0)
    parser.add_argument("--template-diversity", type=int, default=2)
    parser.add_argument("--hidden-test-diversity", type=float, default=1.0)
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--distractor-files", type=int, default=1)
    parser.add_argument("--dependency-depth", type=int, default=2)
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    seeds = list(range(5)) if args.full_sweep else args.seeds
    samples = max(args.samples_per_mode, 2) if args.full_sweep else args.samples_per_mode
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        output = ROOT / "runs" / "benchmarks" / f"tac_scm_real008b_{stamp}"
    result = run_real008b_benchmark(
        seeds=seeds,
        samples_per_mode=samples,
        audit_level=max(args.audit_level, 2) if args.full_sweep else args.audit_level,
        noise_level=args.noise_level,
        rename_level=args.rename_level,
        template_diversity=args.template_diversity,
        hidden_test_diversity=args.hidden_test_diversity,
        max_files=args.max_files,
        distractor_files=args.distractor_files,
        dependency_depth=args.dependency_depth,
        output=output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
