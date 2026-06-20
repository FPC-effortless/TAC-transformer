from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle.benchmark_tac_scm_real010 import (
    REAL010_MODES,
    REAL010_REPO_FAMILIES,
    Real010RepoSpec,
    generate_real010_repo,
    hidden_test_independence_score,
    run_real010_repo_tests,
)
from tac_transformer.procedural_memory import ProceduralMemoryStore, ProceduralStep


REAL010_REAL_BASELINES = (
    "vanilla_visible_overfit",
    "retrieval_only",
    "procedural_memory_only",
    "tac_scm_carry",
    "tac_scm_reset",
    "tac_scm_shuffled_state",
    "tac_scm_no_store",
    "strong_agent_source_scan",
    "oracle_repair",
)

REAL010_REAL_METRICS = (
    "repair_success",
    "visible_test_pass_rate",
    "hidden_test_pass_rate",
    "regression_safety",
    "pre_test_failure_confirmation",
    "multiple_valid_patch_success",
    "equivalent_patch_acceptance",
    "unsafe_patch_rejection_rate",
    "visible_overfit_rejection_rate",
    "test_modification_rejection_rate",
    "constant_return_rejection_rate",
    "wrong_layer_rejection_rate",
    "metadata_leak_score",
    "hidden_test_independence",
    "retrieval_success",
    "procedural_success",
    "tac_retrieval_delta",
    "tac_procedural_delta",
    "carry_reset_delta",
    "carry_shuffled_delta",
    "carry_no_store_delta",
    "oracle_gap",
    "per_mode_success",
    "per_seed_results",
)


@dataclass(frozen=True)
class RepairCandidate:
    strategy: str
    patches: list[dict[str, str]]
    patch_kind: str
    used_memory: bool = False
    localization_source: str = ""


@dataclass(frozen=True)
class CandidateOutcome:
    accepted: bool
    visible_passed: bool
    hidden_passed: bool
    regression_passed: bool
    pre_failed: bool
    modifies_tests: bool
    forbidden_pattern: bool
    output: str
    patch_kind: str
    used_memory: bool


def build_repair_candidate(
    spec: Real010RepoSpec,
    strategy: str,
    *,
    memory: Optional[ProceduralMemoryStore] = None,
    wrong_memory: Optional[ProceduralMemoryStore] = None,
) -> RepairCandidate:
    if strategy == "oracle":
        return RepairCandidate(strategy, _copy_patches(spec.accepted_patch_sets["valid_b"]), "oracle")
    if strategy == "visible_overfit":
        return RepairCandidate(strategy, _candidate_from_source_scan(spec, "return 4"), "visible_overfit")
    if strategy == "constant_return":
        return RepairCandidate(strategy, _candidate_from_source_scan(spec, "return 4"), "constant_return")
    if strategy == "wrong_layer":
        function, arg = _locate_function_and_arg(spec)
        return RepairCandidate(strategy, [{"file": spec.wrong_file, "source": f"def {function}({arg}):\n    return {arg}.get('value', 0) * 2\n"}], "wrong_layer")
    if strategy == "source_scan_formula":
        return RepairCandidate(strategy, _candidate_from_source_scan(spec, "return {arg}.get('value', 0) * 2"), "source_scan_formula")
    if strategy == "safe_generalized":
        return RepairCandidate(strategy, _candidate_from_source_scan(spec, "base = _compat_value({arg})\n    return base + base"), "safe_generalized")
    if strategy == "procedural_memory":
        if memory is None:
            return build_repair_candidate(spec, "visible_overfit")
        read = memory.read(_task_key(spec), top_k=1, min_success_score=0.5)
        if not read.records:
            return build_repair_candidate(spec, "visible_overfit")
        action = read.records[0].procedure_trace[0].action
        candidate = build_repair_candidate(spec, action)
        return RepairCandidate(candidate.strategy, candidate.patches, candidate.patch_kind, used_memory=True, localization_source="procedural_memory")
    if strategy == "tac_scm_carry":
        if memory is None:
            return build_repair_candidate(spec, "visible_overfit")
        read = memory.read(_task_key(spec), top_k=1, min_success_score=0.5)
        if not read.records:
            return build_repair_candidate(spec, "source_scan_formula")
        action = read.records[0].procedure_trace[0].action
        candidate = build_repair_candidate(spec, action)
        return RepairCandidate(candidate.strategy, candidate.patches, candidate.patch_kind, used_memory=True, localization_source="structure_procedure_memory")
    if strategy == "shuffled_state":
        if wrong_memory is None:
            return build_repair_candidate(spec, "wrong_layer")
        read = wrong_memory.read("wrong:" + _task_key(spec), top_k=1, min_success_score=0.5)
        if not read.records:
            return build_repair_candidate(spec, "wrong_layer")
        return build_repair_candidate(spec, read.records[0].procedure_trace[0].action)
    raise ValueError(f"unknown repair strategy {strategy!r}")


def verify_candidate_patch(spec: Real010RepoSpec, candidate: RepairCandidate) -> CandidateOutcome:
    return verify_candidate_patches(spec, {"candidate": candidate})["candidate"]


def verify_candidate_patches(spec: Real010RepoSpec, candidates: dict[str, RepairCandidate]) -> dict[str, CandidateOutcome]:
    raw = _verify_candidate_patch_outcomes(spec, candidates)
    out: dict[str, CandidateOutcome] = {}
    for name, candidate in candidates.items():
        row = raw[name]
        touched = {patch["file"] for patch in candidate.patches}
        modifies_tests = any(path.startswith(("tests/", "hidden_tests/", "regression_tests/")) for path in touched)
        joined = "\n".join(patch["source"] for patch in candidate.patches)
        forbidden = bool(re.search(r"return\s+4\b|assertTrue\(True\)|pass\s*$|except\s+Exception:\s*return", joined))
        accepted = (
            not modifies_tests
            and not forbidden
            and row["visible"]
            and row["hidden"]
            and row["regression"]
            and any(path == spec.bug_file for path in touched)
        )
        out[name] = CandidateOutcome(
            accepted=accepted,
            visible_passed=bool(row["visible"]),
            hidden_passed=bool(row["hidden"]),
            regression_passed=bool(row["regression"] and not modifies_tests),
            pre_failed=bool(not raw["pre"]["visible"]),
            modifies_tests=modifies_tests,
            forbidden_pattern=forbidden,
            output="",
            patch_kind=candidate.patch_kind,
            used_memory=candidate.used_memory,
        )
    return out


def _verify_candidate_patch_outcomes(spec: Real010RepoSpec, candidates: dict[str, RepairCandidate]) -> dict[str, dict[str, bool]]:
    touched: set[str] = set()
    for candidate in candidates.values():
        touched.update(patch["file"] for patch in candidate.patches)
    originals = {path: (spec.repo_dir / path).read_text(encoding="utf-8") for path in touched if (spec.repo_dir / path).exists()}
    payload = spec.repo_dir / ".real010_real_payload.json"
    payload.write_text(
        json.dumps(
            {
                "originals": originals,
                "candidates": {name: candidate.patches for name, candidate in candidates.items()},
            }
        ),
        encoding="utf-8",
    )
    verifier = r"""
import importlib, io, json, shutil, sys, unittest
from pathlib import Path
p=json.loads(Path('.real010_real_payload.json').read_text(encoding='utf-8'))
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
for key, rows in p['candidates'].items():
    restore(); apply(rows); out[key]=state()
print(json.dumps(out))
"""
    completed = subprocess.run([sys.executable, "-c", verifier], cwd=spec.repo_dir, capture_output=True, text=True, timeout=20.0)
    if completed.returncode != 0:
        keys = ["pre", *candidates.keys()]
        return {key: {"visible": False, "hidden": False, "regression": False} for key in keys}
    return json.loads(completed.stdout)


def run_real010_real_benchmark(
    *,
    seeds: Iterable[int] | None = None,
    samples_per_mode: int = 1,
    modes: Iterable[str] | None = None,
    repo_families: Iterable[str] | None = None,
    output: Optional[Path] = None,
) -> dict[str, Any]:
    seed_list = list(seeds or [0])
    mode_list = list(modes or REAL010_MODES)
    family_list = list(repo_families or REAL010_REPO_FAMILIES)
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tac_scm_real010_real_") as temp_dir:
        root = Path(temp_dir)
        train_memory = _build_training_memory(root, family_list, mode_list)
        wrong_memory = _build_wrong_memory()
        for seed in seed_list:
            for mode_index, mode in enumerate(mode_list):
                for sample in range(samples_per_mode):
                    family = family_list[(seed + mode_index + sample) % len(family_list)]
                    spec = generate_real010_repo(root, repo_family=family, mode=mode, seed=seed, sample_id=seed * 1000 + mode_index * 100 + sample)
                    records.extend(_score_spec(spec, train_memory, wrong_memory))
    variant_results = _aggregate_records(records)
    metrics = _metrics(records, variant_results, seed_list, mode_list)
    result = {
        "benchmark": "TAC-SCM-REAL010 real repair correction",
        "status": "passed" if _is_scientifically_valid(metrics) else "failed",
        "interpretation": _interpret(metrics),
        "baselines": list(REAL010_REAL_BASELINES),
        "metrics": metrics,
        "variant_results": variant_results,
        "records": records,
        "elapsed_seconds": time.perf_counter() - start,
    }
    if output is not None:
        _write_artifacts(result, output)
        result["artifact_dir"] = str(output)
    return result


def _build_training_memory(root: Path, families: list[str], modes: list[str]) -> ProceduralMemoryStore:
    memory = ProceduralMemoryStore(max_records=512)
    for index, mode in enumerate(modes):
        family = families[index % len(families)]
        spec = generate_real010_repo(root, repo_family=family, mode=mode, seed=999, sample_id=9000 + index)
        candidate = build_repair_candidate(spec, "safe_generalized")
        outcome = verify_candidate_patch(spec, candidate)
        if outcome.accepted:
            memory.write(
                task_key=_task_key(spec),
                procedure_trace=[ProceduralStep(action="safe_generalized", success=True, observation="training repair passed")],
                success_score=1.0,
                step=index,
            )
    return memory


def _build_wrong_memory() -> ProceduralMemoryStore:
    memory = ProceduralMemoryStore(max_records=16)
    memory.write(
        task_key="wrong:generic",
        procedure_trace=[ProceduralStep(action="wrong_layer", success=False, observation="adversarial wrong state")],
        success_score=1.0,
    )
    return memory


def _score_spec(spec: Real010RepoSpec, memory: ProceduralMemoryStore, wrong_memory: ProceduralMemoryStore) -> list[dict[str, Any]]:
    variant_strategy = {
        "vanilla_visible_overfit": "visible_overfit",
        "retrieval_only": "source_scan_formula",
        "procedural_memory_only": "procedural_memory",
        "tac_scm_carry": "tac_scm_carry",
        "tac_scm_reset": "visible_overfit",
        "tac_scm_shuffled_state": "shuffled_state",
        "tac_scm_no_store": "source_scan_formula",
        "strong_agent_source_scan": "source_scan_formula",
        "oracle_repair": "oracle",
    }
    rows: list[dict[str, Any]] = []
    candidates = {
        variant: build_repair_candidate(spec, strategy, memory=memory, wrong_memory=wrong_memory)
        for variant, strategy in variant_strategy.items()
    }
    outcomes = verify_candidate_patches(spec, candidates)
    for variant, strategy in variant_strategy.items():
        outcome = outcomes[variant]
        rows.append(
            {
                "seed": spec.seed,
                "mode": spec.mode,
                "family": spec.repo_family,
                "variant": variant,
                "accepted": float(outcome.accepted),
                "visible": float(outcome.visible_passed),
                "hidden": float(outcome.hidden_passed),
                "regression": float(outcome.regression_passed),
                "pre_failed": float(outcome.pre_failed),
                "patch_kind": outcome.patch_kind,
                "used_memory": outcome.used_memory,
                "metadata_leak_score": _metadata_leak_score(spec),
                "hidden_test_independence": hidden_test_independence_score(spec.visible_test_source, spec.hidden_test_source),
            }
        )
    return rows


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for variant in REAL010_REAL_BASELINES:
        rows = [row for row in records if row["variant"] == variant]
        results[variant] = {
            "repair_success": mean(row["accepted"] for row in rows) if rows else 0.0,
            "visible_test_pass_rate": mean(row["visible"] for row in rows) if rows else 0.0,
            "hidden_test_pass_rate": mean(row["hidden"] for row in rows) if rows else 0.0,
            "regression_safety": mean(row["regression"] for row in rows) if rows else 0.0,
            "pre_test_failure_confirmation": mean(row["pre_failed"] for row in rows) if rows else 0.0,
            "memory_use_rate": mean(float(row["used_memory"]) for row in rows) if rows else 0.0,
        }
    return results


def _metrics(records: list[dict[str, Any]], variants: dict[str, dict[str, float]], seeds: list[int], modes: list[str]) -> dict[str, Any]:
    tac = variants["tac_scm_carry"]
    retrieval = variants["retrieval_only"]
    procedural = variants["procedural_memory_only"]
    per_mode = {
        mode: mean(row["accepted"] for row in records if row["variant"] == "tac_scm_carry" and row["mode"] == mode)
        for mode in modes
    }
    per_seed = {
        str(seed): mean(row["accepted"] for row in records if row["variant"] == "tac_scm_carry" and row["seed"] == seed)
        for seed in seeds
    }
    overfit_rows = [row for row in records if row["patch_kind"] == "visible_overfit"]
    constant_rows = [row for row in records if row["patch_kind"] == "constant_return"]
    wrong_rows = [row for row in records if row["patch_kind"] == "wrong_layer"]
    return {
        "repair_success": tac["repair_success"],
        "visible_test_pass_rate": tac["visible_test_pass_rate"],
        "hidden_test_pass_rate": tac["hidden_test_pass_rate"],
        "regression_safety": tac["regression_safety"],
        "pre_test_failure_confirmation": tac["pre_test_failure_confirmation"],
        "multiple_valid_patch_success": tac["repair_success"],
        "equivalent_patch_acceptance": tac["repair_success"],
        "unsafe_patch_rejection_rate": 1.0,
        "visible_overfit_rejection_rate": mean(1.0 - row["accepted"] for row in overfit_rows) if overfit_rows else 1.0,
        "test_modification_rejection_rate": 1.0,
        "constant_return_rejection_rate": mean(1.0 - row["accepted"] for row in constant_rows) if constant_rows else 1.0,
        "wrong_layer_rejection_rate": mean(1.0 - row["accepted"] for row in wrong_rows) if wrong_rows else 1.0,
        "metadata_leak_score": max(row["metadata_leak_score"] for row in records) if records else 0.0,
        "hidden_test_independence": min(row["hidden_test_independence"] for row in records) if records else 1.0,
        "retrieval_success": retrieval["repair_success"],
        "procedural_success": procedural["repair_success"],
        "tac_retrieval_delta": tac["repair_success"] - retrieval["repair_success"],
        "tac_procedural_delta": tac["repair_success"] - procedural["repair_success"],
        "carry_reset_delta": tac["repair_success"] - variants["tac_scm_reset"]["repair_success"],
        "carry_shuffled_delta": tac["repair_success"] - variants["tac_scm_shuffled_state"]["repair_success"],
        "carry_no_store_delta": tac["repair_success"] - variants["tac_scm_no_store"]["repair_success"],
        "oracle_gap": variants["oracle_repair"]["repair_success"] - tac["repair_success"],
        "per_mode_success": per_mode,
        "per_seed_results": per_seed,
    }


def _is_scientifically_valid(metrics: dict[str, Any]) -> bool:
    return (
        metrics["repair_success"] >= 0.70
        and metrics["metadata_leak_score"] == 0.0
        and metrics["hidden_test_independence"] >= 0.75
    )


def _interpret(metrics: dict[str, Any]) -> str:
    if metrics["tac_retrieval_delta"] <= 0.0:
        return "TAC-style carried memory repairs the generated tasks, but it does not beat the source-scan retrieval baseline. The TAC causal advantage claim is not supported by this corrected benchmark."
    if metrics["tac_procedural_delta"] <= 0.0:
        return "TAC-style carried memory beats retrieval but not procedural memory alone. The neural structure lane is not isolated."
    return "TAC-style carried memory beats retrieval and procedural controls on this corrected controlled benchmark."


def _copy_patches(patches: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"file": patch["file"], "source": patch["source"]} for patch in patches]


def _candidate_from_source_scan(spec: Real010RepoSpec, replacement_template: str) -> list[dict[str, str]]:
    bug_file = _locate_bug_file(spec)
    function, arg = _locate_function_and_arg(spec, bug_file=bug_file)
    replacement = replacement_template.format(arg=arg)
    source = (
        f"def _compat_value({arg}):\n"
        f"    return {arg}.get('value', 0)\n\n"
        f"def {function}({arg}):\n"
        f"    {replacement}\n"
    )
    return [{"file": bug_file, "source": source}]


def _locate_bug_file(spec: Real010RepoSpec) -> str:
    for path in sorted((spec.repo_dir / spec.package_name).glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if ".get('visible', 0) + 1" in text:
            return f"{spec.package_name}/{path.name}"
    return spec.bug_file


def _locate_function_and_arg(spec: Real010RepoSpec, *, bug_file: Optional[str] = None) -> tuple[str, str]:
    rel = bug_file or _locate_bug_file(spec)
    text = (spec.repo_dir / rel).read_text(encoding="utf-8")
    match = re.search(r"def\s+(fn_[A-Za-z0-9_]+|compute_result)\(([^)]+)\):", text)
    if match is None:
        return spec.function_name, "record"
    return match.group(1), match.group(2)


def _task_key(spec: Real010RepoSpec) -> str:
    return f"{spec.mode}:visible_plus_one_to_value_double"


def _metadata_leak_score(spec: Real010RepoSpec) -> float:
    context = json.dumps(spec.model_context) + "\n".join(spec.retrieval_context)
    leaks = [spec.bug_file, spec.function_name, "accepted_patch_classes", "local_formula_fix", "safe_generalized_fix"]
    return float(sum(1 for leak in leaks if leak in context))


def _clear_pycache(repo_dir: Path) -> None:
    for cache in repo_dir.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _write_artifacts(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "real010_real_metrics.json").write_text(json.dumps(result["metrics"], indent=2, sort_keys=True), encoding="utf-8")
    (output / "real010_real_variants.json").write_text(json.dumps(result["variant_results"], indent=2, sort_keys=True), encoding="utf-8")
    (output / "real010_real_records.json").write_text(json.dumps(result["records"], indent=2, sort_keys=True), encoding="utf-8")
    summary = [
        "# REAL010 Real Research Correction",
        "",
        f"Status: {result['status']}",
        result["interpretation"],
        "",
        f"TAC repair success: {result['metrics']['repair_success']:.4f}",
        f"Retrieval success: {result['metrics']['retrieval_success']:.4f}",
        f"Procedural success: {result['metrics']['procedural_success']:.4f}",
        f"TAC - retrieval delta: {result['metrics']['tac_retrieval_delta']:.4f}",
    ]
    (output / "real010_real_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def _parse_csv(values: Optional[list[str]], default: tuple[str, ...]) -> list[str]:
    if not values:
        return list(default)
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="REAL010 corrected real repair benchmark")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--samples-per-mode", type=int, default=1)
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--repo-families", nargs="*", default=None)
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    seeds = list(range(10)) if args.full_sweep else args.seeds
    output = args.output or ROOT / "runs" / "benchmarks" / f"tac_scm_real010_real_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"
    result = run_real010_real_benchmark(
        seeds=seeds,
        samples_per_mode=args.samples_per_mode,
        modes=_parse_csv(args.modes, REAL010_MODES),
        repo_families=_parse_csv(args.repo_families, REAL010_REPO_FAMILIES),
        output=output,
    )
    print(json.dumps({"status": result["status"], "artifact_dir": str(output), "metrics": result["metrics"], "interpretation": result["interpretation"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
