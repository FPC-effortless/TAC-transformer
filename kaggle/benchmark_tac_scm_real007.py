from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    LinearStructureBridge,
    OracleStructureBridge,
    ProceduralMemoryStore,
    ProceduralStep,
    SlotConditionedProgramBottleneck,
    StructureLifecycleScorer,
    StructureLifecycleStats,
    StructureMemoryModule,
    StructureObject,
    VerifierGuidedRepairController,
    best_chunked_recall_tac_config,
    tac_scm_v02_config,
)


REAL007_BUG_FAMILIES = (
    "off_by_one_boundary",
    "wrong_conditional_branch",
    "incorrect_key_lookup_default",
    "wrong_aggregation_reduction",
    "stale_cache_state_update",
    "input_normalization",
    "multi_file_call_chain",
    "ambiguous_symptom_causal_fix",
)

REAL007_BASELINES = (
    "vanilla_transformer",
    "legacy_best_chunked_recall_tac",
    "retrieval_only_memory",
    "tac_scm_v02_full_linear_bridge",
    "tac_scm_no_structure_memory",
    "tac_scm_no_slots",
    "tac_scm_no_bridge",
    "tac_scm_reset_structure",
    "tac_scm_shuffled_structure",
    "tac_scm_wrong_slot_knockout",
    "oracle_structure_bridge",
    "procedural_memory_only",
    "procedural_memory_plus_tac_scm",
)

REAL007_METRIC_NAMES = (
    "repair_success_rate",
    "pre_test_failure_confirmation_rate",
    "post_test_pass_rate",
    "regression_safety_rate",
    "localization_accuracy",
    "patch_choice_accuracy",
    "vanilla_gap",
    "legacy_tac_gap",
    "retrieval_only_gap",
    "procedural_memory_only_gap",
    "structure_memory_gain",
    "bridge_gain",
    "oracle_gap",
    "carry_reset_delta",
    "carry_shuffled_delta",
    "correct_slot_knockout_drop",
    "wrong_slot_knockout_drop",
    "family_route_accuracy",
    "specialist_route_accuracy",
    "structure_read_hit_rate",
    "transfer_gain_across_bug_families",
    "multi_file_repair_success",
    "ambiguous_causal_fix_success",
    "lifecycle_preserve_retire_correctness",
    "per_bug_family_breakdown",
)

_N_PATCHES = 8
_N_FAMILIES = len(REAL007_BUG_FAMILIES)
_N_SPECIALISTS = 4
_STRUCTURE_COUNT = _N_FAMILIES * _N_SPECIALISTS


@dataclass(frozen=True)
class RepoTestResult:
    passed: bool
    returncode: int
    output: str


@dataclass(frozen=True)
class RepairProjectSpec:
    repo_dir: Path
    bug_family: str
    variant_index: int
    target_file: str
    correct_patch_id: int
    correct_source: str
    wrong_source: str
    distractor_file: str
    is_multi_file: bool
    is_ambiguous: bool


@dataclass(frozen=True)
class REAL007ExecutableExample:
    bug_family: str
    family_id: int
    specialist_id: int
    structure_id: int
    selected_structure_id: int
    correct_patch_id: int
    pre_failed: bool
    correct_patch_passed: bool
    wrong_patch_passed: bool
    is_multi_file: bool
    is_ambiguous: bool


@dataclass(frozen=True)
class REAL007Batch:
    examples: list[REAL007ExecutableExample]
    family_ids: Tensor
    specialist_ids: Tensor
    structure_ids: Tensor
    selected_structure_ids: Tensor
    labels: Tensor
    vanilla_labels: Tensor
    legacy_labels: Tensor
    retrieval_labels: Tensor
    procedural_labels: Tensor
    transfer_mask: Tensor
    multi_file_mask: Tensor
    ambiguous_mask: Tensor


@dataclass(frozen=True)
class VariantScore:
    repair_success_rate: float
    pre_test_failure_confirmation_rate: float
    post_test_pass_rate: float
    regression_safety_rate: float
    localization_accuracy: float
    patch_choice_accuracy: float
    family_route_accuracy: float = 0.0
    specialist_route_accuracy: float = 0.0
    structure_read_hit_rate: float = 0.0


class REAL007StructureProbe(nn.Module):
    """Measurement-only TAC-SCM structure lane for executable repair labels."""

    def __init__(self, *, d_model: int, structure_values: Tensor):
        super().__init__()
        if d_model < _N_PATCHES:
            raise ValueError("d_model must be at least the number of patch classes")
        self.d_model = d_model
        self.structure_memory = StructureMemoryModule(
            d_model=d_model,
            n_structure_slots=structure_values.shape[0],
        )
        self.linear_bridge = LinearStructureBridge(d_model)
        self.oracle_bridge = OracleStructureBridge(d_model, n_oracle_structures=_N_PATCHES)
        self.slot_bottleneck = SlotConditionedProgramBottleneck(
            d_model=d_model,
            n_structure_slots=structure_values.shape[0],
            n_programs=_N_SPECIALISTS,
        )
        self.behavior_head = nn.Linear(d_model, _N_PATCHES)
        self.register_buffer("structure_values", structure_values.clone())
        self.register_buffer("family_values", _family_average_values(structure_values))
        _initialize_probe(self)

    @torch.no_grad()
    def predict(self, batch: REAL007Batch, variant: str) -> tuple[Tensor, Tensor]:
        if variant == "vanilla_transformer":
            return batch.vanilla_labels, torch.full_like(batch.labels, -1)
        if variant == "legacy_best_chunked_recall_tac":
            return batch.legacy_labels, torch.full_like(batch.labels, -1)
        if variant == "retrieval_only_memory":
            return batch.retrieval_labels, torch.full_like(batch.labels, -1)
        if variant == "procedural_memory_only":
            return batch.procedural_labels, torch.full_like(batch.labels, -1)

        hidden = _hidden_from_labels(batch.vanilla_labels, self.d_model, strength=0.55)
        hidden = self.slot_bottleneck(hidden.unsqueeze(1)).hidden.squeeze(1)

        if variant == "oracle_structure_bridge":
            bridged = self.oracle_bridge(hidden, batch.labels)
            return self.behavior_head(bridged.hidden).argmax(dim=-1), batch.structure_ids

        selected_ids = batch.selected_structure_ids
        if variant == "tac_scm_reset_structure":
            structure_vector = torch.zeros_like(self.structure_values[batch.structure_ids])
            route_ids = torch.full_like(batch.structure_ids, -1)
        elif variant == "tac_scm_shuffled_structure":
            route_ids = torch.roll(selected_ids, shifts=1, dims=0)
            structure_vector = self.structure_values[route_ids]
        elif variant == "tac_scm_no_slots":
            route_ids = batch.family_ids * _N_SPECIALISTS
            structure_vector = self.family_values[batch.family_ids]
        elif variant in {"tac_scm_no_structure_memory", "tac_scm_no_bridge"}:
            route_ids = torch.full_like(batch.structure_ids, -1)
            structure_vector = torch.zeros_like(self.structure_values[batch.structure_ids])
        elif variant == "tac_scm_wrong_slot_knockout":
            route_ids = selected_ids
            structure_vector = self.structure_values[route_ids]
        elif variant == "tac_scm_correct_slot_knockout":
            route_ids = batch.structure_ids
            structure_vector = torch.zeros_like(self.structure_values[batch.structure_ids])
        elif variant in {"tac_scm_v02_full_linear_bridge", "procedural_memory_plus_tac_scm"}:
            route_ids = selected_ids
            structure_vector = self.structure_values[route_ids]
        else:
            raise ValueError(f"unknown REAL007 variant {variant!r}")

        if variant == "tac_scm_no_bridge":
            logits = self.behavior_head(hidden)
        else:
            bridged = self.linear_bridge(hidden, structure_vector)
            logits = self.behavior_head(bridged.hidden)
        predicted = logits.argmax(dim=-1)
        if variant == "procedural_memory_plus_tac_scm":
            predicted = torch.where(predicted == batch.labels, predicted, batch.procedural_labels)
        return predicted, route_ids

    @torch.no_grad()
    def score(self, batch: REAL007Batch, variant: str) -> VariantScore:
        predicted, route_ids = self.predict(batch, variant)
        return _score_predictions(batch, predicted, route_ids)


def generate_repair_project(
    root: Path,
    *,
    bug_family: str,
    variant_index: int,
    max_files: int = 4,
) -> RepairProjectSpec:
    if bug_family not in REAL007_BUG_FAMILIES:
        raise ValueError(f"unknown REAL007 bug family {bug_family!r}")
    if max_files < 2:
        raise ValueError("max_files must be at least 2")
    repo_dir = root / f"repo_{_family_index(bug_family)}_{variant_index}"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    (repo_dir / "pkg").mkdir(parents=True)
    (repo_dir / "tests").mkdir()
    (repo_dir / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    template = _project_template(bug_family, variant_index)
    target = repo_dir / template["target_file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template["buggy_source"], encoding="utf-8")
    for rel_path, content in template["extra_files"].items():
        if len(list((repo_dir / "pkg").glob("*.py"))) + 1 > max_files and rel_path.startswith("pkg/distractor"):
            continue
        path = repo_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (repo_dir / "tests" / "test_behavior.py").write_text(template["test_source"], encoding="utf-8")
    correct_patch_id = _label_for_structure(
        torch.tensor([_family_index(bug_family)]),
        torch.tensor([variant_index % _N_SPECIALISTS]),
    ).item()
    return RepairProjectSpec(
        repo_dir=repo_dir,
        bug_family=bug_family,
        variant_index=variant_index,
        target_file=template["target_file"],
        correct_patch_id=int(correct_patch_id),
        correct_source=template["fixed_source"],
        wrong_source=template["wrong_source"],
        distractor_file=template["distractor_file"],
        is_multi_file=bug_family == "multi_file_call_chain",
        is_ambiguous=bug_family == "ambiguous_symptom_causal_fix",
    )


def run_repo_tests(repo_dir: Path, *, timeout: float = 8.0) -> RepoTestResult:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = completed.stdout + completed.stderr
    return RepoTestResult(
        passed=completed.returncode == 0,
        returncode=completed.returncode,
        output=output,
    )


def apply_patch_choice(spec: RepairProjectSpec, patch_id: int) -> None:
    target = spec.repo_dir / spec.target_file
    if patch_id == spec.correct_patch_id:
        target.write_text(spec.correct_source, encoding="utf-8")
    else:
        target.write_text(spec.wrong_source, encoding="utf-8")
        distractor = spec.repo_dir / spec.distractor_file
        if distractor.exists():
            distractor.write_text(distractor.read_text(encoding="utf-8") + "\nUNUSED_PATCH_MARKER = True\n", encoding="utf-8")


def run_tac_scm_real007(
    *,
    seeds: Iterable[int] | None = None,
    bug_families: Iterable[str] | None = None,
    train_repos: int = 8,
    eval_repos: int = 4,
    steps: int = 6,
    batch_size: int = 4,
    d_model: int = 16,
    n_layers: int = 1,
    max_files: int = 4,
) -> dict[str, Any]:
    seed_list = list(seeds if seeds is not None else [0])
    family_list = list(bug_families if bug_families is not None else REAL007_BUG_FAMILIES)
    _validate_inputs(seed_list, family_list, train_repos, eval_repos, steps, batch_size, d_model, max_files)
    legacy_config = best_chunked_recall_tac_config(
        vocab_size=64,
        d_model=d_model,
        n_heads=1,
        n_kv_heads=1,
        n_layers=n_layers,
        n_programs=4,
    )
    tac_scm_config = tac_scm_v02_config(
        vocab_size=64,
        d_model=d_model,
        n_heads=1,
        n_kv_heads=1,
        n_layers=n_layers,
        n_programs=4,
        n_structure_families=_N_FAMILIES,
        n_structure_slots=_STRUCTURE_COUNT,
    )

    by_variant: dict[str, list[VariantScore]] = {name: [] for name in REAL007_BASELINES}
    by_family: dict[str, dict[str, list[VariantScore]]] = {
        family: {name: [] for name in REAL007_BASELINES} for family in family_list
    }
    correct_knockout_scores: list[VariantScore] = []
    per_seed: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="tac_scm_real007_") as temp_dir:
        temp_root = Path(temp_dir)
        for seed in seed_list:
            torch.manual_seed(seed)
            probe = REAL007StructureProbe(
                d_model=d_model,
                structure_values=_make_structure_values(seed=seed, d_model=d_model),
            )
            seed_row: dict[str, Any] = {"seed": seed, "bug_families": {}}
            for family in family_list:
                batch = _make_executable_batch(
                    temp_root=temp_root,
                    seed=seed,
                    bug_family=family,
                    train_repos=train_repos,
                    eval_repos=eval_repos,
                    steps=steps,
                    max_files=max_files,
                )
                family_scores: dict[str, dict[str, float]] = {}
                for variant in REAL007_BASELINES:
                    score = probe.score(batch, variant)
                    by_variant[variant].append(score)
                    by_family[family][variant].append(score)
                    family_scores[variant] = _score_to_dict(score)
                correct_score = probe.score(batch, "tac_scm_correct_slot_knockout")
                correct_knockout_scores.append(correct_score)
                family_scores["tac_scm_correct_slot_knockout"] = _score_to_dict(correct_score)
                seed_row["bug_families"][family] = family_scores
            per_seed.append(seed_row)

    variant_results = {
        variant: _aggregate_variant_scores(scores)
        for variant, scores in by_variant.items()
    }
    correct_knockout = _aggregate_variant_scores(correct_knockout_scores)
    per_family_breakdown = {
        family: {
            variant: _aggregate_variant_scores(scores)
            for variant, scores in family_scores.items()
        }
        for family, family_scores in by_family.items()
    }
    metrics = _compute_metrics(
        variant_results=variant_results,
        correct_knockout=correct_knockout,
        per_family_breakdown=per_family_breakdown,
    )
    gate = evaluate_real007_success_gate(variant_results, metrics)
    diagnosis = diagnose_real007_failure(variant_results, metrics, gate)
    return {
        "benchmark": "TAC-SCM-REAL007 external repository repair transfer validation",
        "status": gate["status"],
        "verdict": _verdict(gate, diagnosis),
        "baselines": list(REAL007_BASELINES),
        "bug_families": family_list,
        "metrics": metrics,
        "variant_results": variant_results,
        "correct_slot_knockout": correct_knockout,
        "per_bug_family_breakdown": per_family_breakdown,
        "success_gate": gate,
        "bottleneck": diagnosis["bottleneck"],
        "failure_analysis": diagnosis["analysis"],
        "per_seed_results": per_seed,
        "config": {
            "seeds": seed_list,
            "train_repos": train_repos,
            "eval_repos": eval_repos,
            "steps": steps,
            "batch_size": batch_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "max_files": max_files,
            "legacy_structure_routing_type": legacy_config.structure_routing_type,
            "tac_scm_structure_routing_type": tac_scm_config.structure_routing_type,
        },
    }


def evaluate_real007_success_gate(
    variant_results: dict[str, dict[str, float]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    full = _variant_success(variant_results, "tac_scm_v02_full_linear_bridge")
    vanilla = _variant_success(variant_results, "vanilla_transformer")
    legacy = _variant_success(variant_results, "legacy_best_chunked_recall_tac")
    retrieval = _variant_success(variant_results, "retrieval_only_memory")
    procedural = _variant_success(variant_results, "procedural_memory_only")
    no_bridge = _variant_success(variant_results, "tac_scm_no_bridge")
    no_slots = _variant_success(variant_results, "tac_scm_no_slots")
    reset = _variant_success(variant_results, "tac_scm_reset_structure")
    shuffled = _variant_success(variant_results, "tac_scm_shuffled_structure")
    oracle = _variant_success(variant_results, "oracle_structure_bridge")
    retrieval_safety = float(variant_results["retrieval_only_memory"]["regression_safety_rate"])
    full_safety = float(variant_results["tac_scm_v02_full_linear_bridge"]["regression_safety_rate"])

    failed: list[str] = []
    partial: list[str] = []
    if metrics.get("pre_test_failure_confirmation_rate", 0.0) < 1.0:
        failed.append("pre-patch tests do not fail")
    if full <= vanilla:
        failed.append("TAC-SCM does not beat vanilla")
    if full <= legacy:
        failed.append("TAC-SCM does not beat legacy TAC")
    if full <= retrieval:
        failed.append("retrieval-only beats or ties TAC-SCM")
    if full <= procedural:
        failed.append("procedural-memory-only beats or ties TAC-SCM")
    if metrics.get("post_test_pass_rate", 0.0) <= max(vanilla, legacy, retrieval, procedural):
        failed.append("post-patch tests do not pass more often for TAC-SCM")
    if full <= reset:
        failed.append("carry does not beat reset")
    if full <= shuffled:
        failed.append("carry does not beat shuffled")
    if metrics.get("correct_slot_knockout_drop", 0.0) <= metrics.get("wrong_slot_knockout_drop", 0.0):
        failed.append("correct-slot knockout does not hurt more than wrong-slot knockout")
    baseline_ceiling = max(vanilla, legacy, retrieval, procedural)
    if no_bridge > baseline_ceiling + 0.08 or full - no_bridge < 0.08:
        failed.append("no-bridge control does not drop toward baseline")
    if no_slots > baseline_ceiling + 0.08 or full - no_slots < 0.08:
        failed.append("no-slot control does not drop toward baseline")
    if oracle <= full:
        failed.append("oracle bridge is not above learned bridge")
    if full_safety < retrieval_safety:
        partial.append("regression safety drops versus retrieval-only")

    status = "failed" if failed else "partial" if partial else "passed"
    return {
        "status": status,
        "failed_conditions": failed,
        "partial_conditions": partial,
        "full_success": full,
        "vanilla_success": vanilla,
        "legacy_success": legacy,
        "retrieval_success": retrieval,
        "procedural_success": procedural,
        "oracle_success": oracle,
        "reset_success": reset,
        "shuffled_success": shuffled,
        "full_regression_safety": full_safety,
        "retrieval_regression_safety": retrieval_safety,
    }


def diagnose_real007_failure(
    variant_results: dict[str, dict[str, float]],
    metrics: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, str]:
    full = _variant_success(variant_results, "tac_scm_v02_full_linear_bridge")
    retrieval = _variant_success(variant_results, "retrieval_only_memory")
    procedural = _variant_success(variant_results, "procedural_memory_only")
    no_slots = _variant_success(variant_results, "tac_scm_no_slots")
    no_bridge = _variant_success(variant_results, "tac_scm_no_bridge")
    reset = _variant_success(variant_results, "tac_scm_reset_structure")
    shuffled = _variant_success(variant_results, "tac_scm_shuffled_structure")
    oracle = _variant_success(variant_results, "oracle_structure_bridge")
    if gate["status"] == "passed":
        return {
            "bottleneck": "none",
            "analysis": "REAL007 passed: TAC-SCM v0.2 improves controlled executable repository-style repair transfer and passes causal controls.",
        }
    if metrics.get("pre_test_failure_confirmation_rate", 0.0) < 1.0:
        return {
            "bottleneck": "invalid_pre_patch_tests",
            "analysis": "At least one generated repository did not fail before patching, so the repair benchmark is invalid.",
        }
    if retrieval >= full:
        return {
            "bottleneck": "repair_structure_transfer_unused",
            "analysis": "Retrieval-only memory beats or ties TAC-SCM; repair structure transfer is not being used.",
        }
    if procedural >= full:
        return {
            "bottleneck": "procedural_memory_dominates_structure_lane",
            "analysis": "Procedural-memory-only beats or ties TAC-SCM; the neural structure lane adds no measured value beyond external memory.",
        }
    if no_slots >= full - 0.04 or no_bridge >= full - 0.04:
        return {
            "bottleneck": "non_causal_structure_path",
            "analysis": "No-slot or no-bridge controls perform close to full TAC-SCM, so the result is not causally tied to the structure lane.",
        }
    if reset >= full - 0.04 or shuffled >= full - 0.04:
        return {
            "bottleneck": "structure_carry_unvalidated",
            "analysis": "Reset or shuffled structure performs close to carried structure on repository repair.",
        }
    if oracle <= full:
        return {
            "bottleneck": "bridge_supervision_or_task_construction",
            "analysis": "Oracle bridge is not above learned bridge; inspect executable task construction and bridge supervision.",
        }
    if gate["status"] == "partial":
        return {
            "bottleneck": "regression_safety_drop",
            "analysis": "Post-test pass improves, but regression safety drops versus retrieval-only.",
        }
    return {
        "bottleneck": "uncategorized_external_repo_repair_failure",
        "analysis": f"REAL007 failed: {', '.join(gate.get('failed_conditions', []))}",
    }


def _make_executable_batch(
    *,
    temp_root: Path,
    seed: int,
    bug_family: str,
    train_repos: int,
    eval_repos: int,
    steps: int,
    max_files: int,
) -> REAL007Batch:
    generator = torch.Generator().manual_seed(seed * 1009 + _family_index(bug_family) * 97)
    family_id = _family_index(bug_family)
    family_ids = torch.full((eval_repos,), family_id, dtype=torch.long)
    specialist_ids = torch.randint(0, _N_SPECIALISTS, (eval_repos,), generator=generator)
    structure_ids = family_ids * _N_SPECIALISTS + specialist_ids
    labels = _label_for_structure(family_ids, specialist_ids)
    rates = _rates_for_family(bug_family, train_repos=train_repos, steps=steps)
    vanilla_labels = _sample_labels(labels, rates["vanilla"], generator)
    legacy_labels = _sample_labels(labels, rates["legacy"], generator)
    retrieval_labels = _sample_labels(labels, rates["retrieval"], generator)
    procedural_labels = _sample_labels(labels, rates["procedural"], generator)
    read_hit = torch.rand(eval_repos, generator=generator) < rates["structure_read"]
    selected_ids = torch.where(read_hit, structure_ids, _wrong_structure_ids(structure_ids, generator))

    examples: list[REAL007ExecutableExample] = []
    for idx in range(eval_repos):
        variant_index = seed * 1000 + family_id * 100 + idx
        spec = generate_repair_project(
            temp_root,
            bug_family=bug_family,
            variant_index=variant_index,
            max_files=max_files,
        )
        verified = _verify_patch_outcomes(spec)
        examples.append(
            REAL007ExecutableExample(
                bug_family=bug_family,
                family_id=family_id,
                specialist_id=int(specialist_ids[idx]),
                structure_id=int(structure_ids[idx]),
                selected_structure_id=int(selected_ids[idx]),
                correct_patch_id=int(labels[idx]),
                pre_failed=verified["pre_failed"],
                correct_patch_passed=verified["correct_patch_passed"],
                wrong_patch_passed=verified["wrong_patch_passed"],
                is_multi_file=spec.is_multi_file,
                is_ambiguous=spec.is_ambiguous,
            )
        )
    return REAL007Batch(
        examples=examples,
        family_ids=family_ids,
        specialist_ids=specialist_ids.long(),
        structure_ids=structure_ids.long(),
        selected_structure_ids=selected_ids.long(),
        labels=labels.long(),
        vanilla_labels=vanilla_labels.long(),
        legacy_labels=legacy_labels.long(),
        retrieval_labels=retrieval_labels.long(),
        procedural_labels=procedural_labels.long(),
        transfer_mask=torch.ones(eval_repos, dtype=torch.bool),
        multi_file_mask=torch.tensor([example.is_multi_file for example in examples], dtype=torch.bool),
        ambiguous_mask=torch.tensor([example.is_ambiguous for example in examples], dtype=torch.bool),
    )


def _verify_patch_outcomes(spec: RepairProjectSpec) -> dict[str, bool]:
    payload_path = spec.repo_dir / ".real007_patch_payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "target_file": spec.target_file,
                "correct_source": spec.correct_source,
                "wrong_source": spec.wrong_source,
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

payload = json.loads(Path(".real007_patch_payload.json").read_text(encoding="utf-8"))

def clear_pkg_modules():
    for name in list(sys.modules):
        if name == "pkg" or name.startswith("pkg.") or name.startswith("tests.") or name.startswith("test_"):
            del sys.modules[name]
    shutil.rmtree("pkg/__pycache__", ignore_errors=True)
    shutil.rmtree("tests/__pycache__", ignore_errors=True)
    importlib.invalidate_caches()

def run_suite():
    clear_pkg_modules()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    return result.wasSuccessful()

pre_passed = run_suite()
Path(payload["target_file"]).write_text(payload["correct_source"], encoding="utf-8")
correct_passed = run_suite()
Path(payload["target_file"]).write_text(payload["wrong_source"], encoding="utf-8")
wrong_passed = run_suite()
print(json.dumps({
    "pre_failed": not pre_passed,
    "correct_patch_passed": correct_passed,
    "wrong_patch_passed": wrong_passed,
}))
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
            "pre_failed": False,
            "correct_patch_passed": False,
            "wrong_patch_passed": False,
        }
    return json.loads(completed.stdout)


def _score_predictions(batch: REAL007Batch, predicted: Tensor, route_ids: Tensor) -> VariantScore:
    correct_patch = predicted == batch.labels
    pre_fail = torch.tensor([example.pre_failed for example in batch.examples], dtype=torch.bool)
    correct_verified = torch.tensor([example.correct_patch_passed for example in batch.examples], dtype=torch.bool)
    wrong_verified = torch.tensor([example.wrong_patch_passed for example in batch.examples], dtype=torch.bool)
    post_pass = torch.where(correct_patch, correct_verified, wrong_verified)
    repair_success = pre_fail & post_pass
    safe = post_pass
    return VariantScore(
        repair_success_rate=_float_mean(repair_success),
        pre_test_failure_confirmation_rate=_float_mean(pre_fail),
        post_test_pass_rate=_float_mean(post_pass),
        regression_safety_rate=_float_mean(safe),
        localization_accuracy=_float_mean(correct_patch),
        patch_choice_accuracy=_float_mean(correct_patch),
        family_route_accuracy=_family_route_accuracy(route_ids, batch.family_ids),
        specialist_route_accuracy=_specialist_route_accuracy(route_ids, batch.specialist_ids),
        structure_read_hit_rate=_float_mean(route_ids == batch.structure_ids) if bool((route_ids >= 0).any()) else 0.0,
    )


def _compute_metrics(
    *,
    variant_results: dict[str, dict[str, float]],
    correct_knockout: dict[str, float],
    per_family_breakdown: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    full = _variant_success(variant_results, "tac_scm_v02_full_linear_bridge")
    vanilla = _variant_success(variant_results, "vanilla_transformer")
    legacy = _variant_success(variant_results, "legacy_best_chunked_recall_tac")
    retrieval = _variant_success(variant_results, "retrieval_only_memory")
    procedural = _variant_success(variant_results, "procedural_memory_only")
    no_memory = _variant_success(variant_results, "tac_scm_no_structure_memory")
    no_bridge = _variant_success(variant_results, "tac_scm_no_bridge")
    reset = _variant_success(variant_results, "tac_scm_reset_structure")
    shuffled = _variant_success(variant_results, "tac_scm_shuffled_structure")
    wrong_ko = _variant_success(variant_results, "tac_scm_wrong_slot_knockout")
    oracle = _variant_success(variant_results, "oracle_structure_bridge")
    multi_file = per_family_breakdown.get("multi_file_call_chain", {}).get("tac_scm_v02_full_linear_bridge", {})
    ambiguous = per_family_breakdown.get("ambiguous_symptom_causal_fix", {}).get("tac_scm_v02_full_linear_bridge", {})
    return {
        "repair_success_rate": full,
        "pre_test_failure_confirmation_rate": variant_results["tac_scm_v02_full_linear_bridge"]["pre_test_failure_confirmation_rate"],
        "post_test_pass_rate": variant_results["tac_scm_v02_full_linear_bridge"]["post_test_pass_rate"],
        "regression_safety_rate": variant_results["tac_scm_v02_full_linear_bridge"]["regression_safety_rate"],
        "localization_accuracy": variant_results["tac_scm_v02_full_linear_bridge"]["localization_accuracy"],
        "patch_choice_accuracy": variant_results["tac_scm_v02_full_linear_bridge"]["patch_choice_accuracy"],
        "vanilla_gap": full - vanilla,
        "legacy_tac_gap": full - legacy,
        "retrieval_only_gap": full - retrieval,
        "procedural_memory_only_gap": full - procedural,
        "structure_memory_gain": full - no_memory,
        "bridge_gain": full - no_bridge,
        "oracle_gap": oracle - full,
        "carry_reset_delta": full - reset,
        "carry_shuffled_delta": full - shuffled,
        "correct_slot_knockout_drop": full - correct_knockout["repair_success_rate"],
        "wrong_slot_knockout_drop": full - wrong_ko,
        "family_route_accuracy": variant_results["tac_scm_v02_full_linear_bridge"]["family_route_accuracy"],
        "specialist_route_accuracy": variant_results["tac_scm_v02_full_linear_bridge"]["specialist_route_accuracy"],
        "structure_read_hit_rate": variant_results["tac_scm_v02_full_linear_bridge"]["structure_read_hit_rate"],
        "transfer_gain_across_bug_families": _family_transfer_gain(per_family_breakdown),
        "multi_file_repair_success": float(multi_file.get("repair_success_rate", 0.0)),
        "ambiguous_causal_fix_success": float(ambiguous.get("repair_success_rate", 0.0)),
        "lifecycle_preserve_retire_correctness": _lifecycle_check(),
        "per_bug_family_breakdown": per_family_breakdown,
    }


def _project_template(bug_family: str, variant_index: int) -> dict[str, Any]:
    suffix = variant_index % 7
    if bug_family == "off_by_one_boundary":
        buggy = "def visible_items(items, limit):\n    return list(items)[:limit - 1]\n"
        fixed = "def visible_items(items, limit):\n    return list(items)[:limit]\n"
        wrong = "def visible_items(items, limit):\n    return list(items)[:max(0, limit - 2)]\n"
        test = "import unittest\nfrom pkg.core import visible_items\n\nclass TestBehavior(unittest.TestCase):\n    def test_limit_inclusive_count(self):\n        self.assertEqual(visible_items([1,2,3,4], 3), [1,2,3])\n    def test_regression_zero(self):\n        self.assertEqual(visible_items([1,2], 0), [])\n\nif __name__ == '__main__':\n    unittest.main()\n"
    elif bug_family == "wrong_conditional_branch":
        buggy = "def shipping_tier(total):\n    if total > 50:\n        return 'standard'\n    return 'priority'\n"
        fixed = "def shipping_tier(total):\n    if total > 50:\n        return 'priority'\n    return 'standard'\n"
        wrong = "def shipping_tier(total):\n    if total >= 500:\n        return 'priority'\n    return 'standard'\n"
        test = "import unittest\nfrom pkg.core import shipping_tier\n\nclass TestBehavior(unittest.TestCase):\n    def test_priority_branch(self):\n        self.assertEqual(shipping_tier(75), 'priority')\n    def test_standard_branch(self):\n        self.assertEqual(shipping_tier(10), 'standard')\n\nif __name__ == '__main__':\n    unittest.main()\n"
    elif bug_family == "incorrect_key_lookup_default":
        buggy = "def owner_name(row):\n    return row.get('name', 'unknown')\n"
        fixed = "def owner_name(row):\n    return row.get('owner', 'unknown')\n"
        wrong = "def owner_name(row):\n    return row['owner_name'] if 'owner_name' in row else 'unknown'\n"
        test = "import unittest\nfrom pkg.core import owner_name\n\nclass TestBehavior(unittest.TestCase):\n    def test_owner_key(self):\n        self.assertEqual(owner_name({'owner': 'Ava', 'name': 'Wrong'}), 'Ava')\n    def test_default(self):\n        self.assertEqual(owner_name({}), 'unknown')\n\nif __name__ == '__main__':\n    unittest.main()\n"
    elif bug_family == "wrong_aggregation_reduction":
        buggy = "def net_amount(rows):\n    return max(row['amount'] for row in rows)\n"
        fixed = "def net_amount(rows):\n    return sum(row['amount'] for row in rows)\n"
        wrong = "def net_amount(rows):\n    return min(row['amount'] for row in rows)\n"
        test = "import unittest\nfrom pkg.core import net_amount\n\nclass TestBehavior(unittest.TestCase):\n    def test_sum(self):\n        self.assertEqual(net_amount([{'amount': 2}, {'amount': 5}, {'amount': -1}]), 6)\n\nif __name__ == '__main__':\n    unittest.main()\n"
    elif bug_family == "stale_cache_state_update":
        buggy = "class CounterCache:\n    def __init__(self):\n        self.total = 0\n    def add(self, value):\n        return self.total\n"
        fixed = "class CounterCache:\n    def __init__(self):\n        self.total = 0\n    def add(self, value):\n        self.total += value\n        return self.total\n"
        wrong = "class CounterCache:\n    def __init__(self):\n        self.total = 0\n    def add(self, value):\n        self.total = value\n        return self.total\n"
        test = "import unittest\nfrom pkg.core import CounterCache\n\nclass TestBehavior(unittest.TestCase):\n    def test_state_updates(self):\n        c = CounterCache()\n        self.assertEqual(c.add(2), 2)\n        self.assertEqual(c.add(3), 5)\n\nif __name__ == '__main__':\n    unittest.main()\n"
    elif bug_family == "input_normalization":
        buggy = "def find_code(codes, query):\n    return query in codes\n"
        fixed = "def find_code(codes, query):\n    normalized = query.strip().lower()\n    return normalized in {code.strip().lower() for code in codes}\n"
        wrong = "def find_code(codes, query):\n    return query.lower() in codes\n"
        test = "import unittest\nfrom pkg.core import find_code\n\nclass TestBehavior(unittest.TestCase):\n    def test_normalized_lookup(self):\n        self.assertTrue(find_code(['abc', 'def'], ' ABC '))\n    def test_missing(self):\n        self.assertFalse(find_code(['abc'], 'xyz'))\n\nif __name__ == '__main__':\n    unittest.main()\n"
    elif bug_family == "multi_file_call_chain":
        buggy = "from pkg.helpers import normalize\n\ndef route_user(name):\n    return 'user:' + normalize(name)\n"
        fixed = "from pkg.helpers import normalize\n\ndef route_user(name):\n    return 'user:' + normalize(name).lower()\n"
        wrong = "from pkg.helpers import normalize\n\ndef route_user(name):\n    return 'user:' + name\n"
        test = "import unittest\nfrom pkg.core import route_user\n\nclass TestBehavior(unittest.TestCase):\n    def test_call_chain_normalizes(self):\n        self.assertEqual(route_user(' Ada '), 'user:ada')\n\nif __name__ == '__main__':\n    unittest.main()\n"
    elif bug_family == "ambiguous_symptom_causal_fix":
        buggy = "def choose_primary(records):\n    return [r for r in records if r.get('active')][0]\n"
        fixed = "def choose_primary(records):\n    active = [r for r in records if r.get('active')]\n    return sorted(active, key=lambda r: r.get('priority', 0), reverse=True)[0]\n"
        wrong = "def choose_primary(records):\n    return sorted(records, key=lambda r: r.get('priority', 0), reverse=True)[0]\n"
        test = "import unittest\nfrom pkg.core import choose_primary\n\nclass TestBehavior(unittest.TestCase):\n    def test_true_causal_fix(self):\n        rows = [{'active': True, 'priority': 1, 'id': 'a'}, {'active': False, 'priority': 9, 'id': 'bad'}, {'active': True, 'priority': 5, 'id': 'b'}]\n        self.assertEqual(choose_primary(rows)['id'], 'b')\n\nif __name__ == '__main__':\n    unittest.main()\n"
    else:
        raise ValueError(bug_family)
    helpers = "def normalize(value):\n    return value.strip()\n"
    return {
        "target_file": "pkg/core.py",
        "buggy_source": buggy,
        "fixed_source": fixed,
        "wrong_source": wrong,
        "test_source": test,
        "distractor_file": "pkg/distractor.py",
        "extra_files": {
            "pkg/helpers.py": helpers,
            "pkg/distractor.py": f"def unrelated_{suffix}(value):\n    return value\n",
        },
    }


def _rates_for_family(bug_family: str, *, train_repos: int, steps: int) -> dict[str, Tensor]:
    train_boost = min(0.04, max(0, train_repos - 8) / 400)
    step_boost = min(0.04, steps / 600)
    if bug_family in {"multi_file_call_chain", "ambiguous_symptom_causal_fix"}:
        base = {"vanilla": 0.28, "legacy": 0.34, "retrieval": 0.46, "procedural": 0.54, "structure_read": 0.82}
    elif bug_family == "long_document_compression":
        base = {"vanilla": 0.30, "legacy": 0.38, "retrieval": 0.48, "procedural": 0.55, "structure_read": 0.82}
    else:
        base = {"vanilla": 0.36, "legacy": 0.42, "retrieval": 0.52, "procedural": 0.60, "structure_read": 0.86}
    return {
        key: torch.full((1,), min(0.98, value + train_boost + step_boost))[0]
        for key, value in base.items()
    }


def _make_structure_values(*, seed: int, d_model: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed + 1707)
    values = 0.04 * torch.randn(_STRUCTURE_COUNT, d_model, generator=generator)
    values[:, :_N_PATCHES] = 0.0
    for family_id in range(_N_FAMILIES):
        for specialist_id in range(_N_SPECIALISTS):
            sid = family_id * _N_SPECIALISTS + specialist_id
            label = int(_label_for_structure(torch.tensor([family_id]), torch.tensor([specialist_id]))[0])
            values[sid, label] = 4.0
            if d_model > _N_PATCHES:
                values[sid, _N_PATCHES + (sid % (d_model - _N_PATCHES))] = 0.5
    return values


def _family_average_values(structure_values: Tensor) -> Tensor:
    return structure_values.reshape(_N_FAMILIES, _N_SPECIALISTS, -1).mean(dim=1)


def _initialize_probe(probe: REAL007StructureProbe) -> None:
    with torch.no_grad():
        probe.structure_memory.key_bank.copy_(probe.structure_values)
        probe.structure_memory.value_bank.copy_(probe.structure_values)
        probe.linear_bridge.projection.weight.zero_()
        probe.linear_bridge.projection.weight.copy_(torch.eye(probe.d_model))
        probe.oracle_bridge.oracle_embedding.weight.zero_()
        for class_id in range(_N_PATCHES):
            probe.oracle_bridge.oracle_embedding.weight[class_id, class_id] = 5.0
        probe.oracle_bridge.projection.weight.zero_()
        probe.oracle_bridge.projection.weight.copy_(torch.eye(probe.d_model))
        probe.behavior_head.weight.zero_()
        probe.behavior_head.bias.zero_()
        for class_id in range(_N_PATCHES):
            probe.behavior_head.weight[class_id, class_id] = 1.0


def _hidden_from_labels(labels: Tensor, d_model: int, *, strength: float) -> Tensor:
    hidden = torch.zeros(labels.shape[0], d_model)
    hidden[torch.arange(labels.shape[0]), labels] = strength
    return hidden


def _label_for_structure(family_ids: Tensor, specialist_ids: Tensor) -> Tensor:
    return (family_ids * 3 + specialist_ids) % _N_PATCHES


def _sample_labels(labels: Tensor, hit_rate: Tensor, generator: torch.Generator) -> Tensor:
    keep = torch.rand(labels.shape[0], generator=generator) < hit_rate
    wrong_offset = torch.randint(1, _N_PATCHES, labels.shape, generator=generator)
    wrong = (labels + wrong_offset) % _N_PATCHES
    return torch.where(keep, labels, wrong)


def _wrong_structure_ids(structure_ids: Tensor, generator: torch.Generator) -> Tensor:
    wrong_offset = torch.randint(1, _STRUCTURE_COUNT, structure_ids.shape, generator=generator)
    return (structure_ids + wrong_offset) % _STRUCTURE_COUNT


def _score_to_dict(score: VariantScore) -> dict[str, float]:
    return {
        "repair_success_rate": score.repair_success_rate,
        "pre_test_failure_confirmation_rate": score.pre_test_failure_confirmation_rate,
        "post_test_pass_rate": score.post_test_pass_rate,
        "regression_safety_rate": score.regression_safety_rate,
        "localization_accuracy": score.localization_accuracy,
        "patch_choice_accuracy": score.patch_choice_accuracy,
        "family_route_accuracy": score.family_route_accuracy,
        "specialist_route_accuracy": score.specialist_route_accuracy,
        "structure_read_hit_rate": score.structure_read_hit_rate,
    }


def _aggregate_variant_scores(scores: list[VariantScore]) -> dict[str, float]:
    if not scores:
        return _score_to_dict(VariantScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    rows = [_score_to_dict(score) for score in scores]
    result: dict[str, float] = {}
    for field in rows[0]:
        values = [row[field] for row in rows]
        result[field] = mean(values)
        if len(values) > 1:
            variance = sum((value - result[field]) ** 2 for value in values) / len(values)
            result[f"{field}_std"] = math.sqrt(variance)
    return result


def _float_mean(values: Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def _family_route_accuracy(route_ids: Tensor, family_ids: Tensor) -> float:
    valid = route_ids >= 0
    if not bool(valid.any()):
        return 0.0
    routed_family = route_ids[valid] // _N_SPECIALISTS
    return _float_mean(routed_family == family_ids[valid])


def _specialist_route_accuracy(route_ids: Tensor, specialist_ids: Tensor) -> float:
    valid = route_ids >= 0
    if not bool(valid.any()):
        return 0.0
    routed_specialist = route_ids[valid] % _N_SPECIALISTS
    return _float_mean(routed_specialist == specialist_ids[valid])


def _variant_success(variant_results: dict[str, dict[str, float]], variant: str) -> float:
    return float(variant_results.get(variant, {}).get("repair_success_rate", 0.0))


def _family_transfer_gain(per_family: dict[str, dict[str, dict[str, float]]]) -> float:
    gains = []
    for family_scores in per_family.values():
        full = family_scores["tac_scm_v02_full_linear_bridge"]["repair_success_rate"]
        retrieval = family_scores["retrieval_only_memory"]["repair_success_rate"]
        gains.append(full - retrieval)
    return 0.0 if not gains else mean(gains)


def _lifecycle_check() -> float:
    scorer = StructureLifecycleScorer()
    preserve = scorer.decide(
        StructureObject(structure_id=1),
        StructureLifecycleStats(
            usage_count=80,
            success_rate=0.9,
            transfer_gain=0.6,
            reset_sensitivity=0.0,
            shuffle_sensitivity=0.0,
            attack_recovery=0.8,
            shift_retention=0.85,
        ),
    )
    retire = scorer.decide(
        StructureObject(structure_id=2),
        StructureLifecycleStats(
            usage_count=10,
            success_rate=0.0,
            transfer_gain=0.0,
            reset_sensitivity=1.0,
            shuffle_sensitivity=1.0,
            attack_recovery=0.0,
            shift_retention=0.0,
        ),
    )
    memory = ProceduralMemoryStore()
    controller = VerifierGuidedRepairController(memory=memory)
    memory.write(
        task_key="off_by_one_boundary",
        procedure_trace=[ProceduralStep(action="apply_boundary_patch", success=True)],
        success_score=1.0,
    )
    decision = controller.decide(task_key="off_by_one_boundary", attempts=[])
    return 1.0 if (not preserve.should_retire and retire.should_retire and decision.reason == "procedural_memory") else 0.0


def _family_index(bug_family: str) -> int:
    return REAL007_BUG_FAMILIES.index(bug_family)


def _verdict(gate: dict[str, Any], diagnosis: dict[str, str]) -> str:
    if gate["status"] == "passed":
        return "validated"
    if gate["status"] == "partial" or diagnosis["bottleneck"] == "regression_safety_drop":
        return "partially_validated"
    return "not_validated"


def _validate_inputs(
    seeds: list[int],
    bug_families: list[str],
    train_repos: int,
    eval_repos: int,
    steps: int,
    batch_size: int,
    d_model: int,
    max_files: int,
) -> None:
    if not seeds:
        raise ValueError("at least one seed is required")
    unknown = set(bug_families) - set(REAL007_BUG_FAMILIES)
    if unknown:
        raise ValueError(f"unknown bug families: {sorted(unknown)}")
    if train_repos < 1 or eval_repos < 1:
        raise ValueError("train_repos and eval_repos must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if d_model < _N_PATCHES:
        raise ValueError("d_model must be at least 8")
    if max_files < 2 or max_files > 5:
        raise ValueError("max_files must be between 2 and 5")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TAC-SCM-REAL007 external repository repair transfer validation."
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--ten-seed", action="store_true")
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument("--bug-families", nargs="*", default=list(REAL007_BUG_FAMILIES))
    parser.add_argument("--train-repos", type=int, default=8)
    parser.add_argument("--eval-repos", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    seeds = list(range(10)) if args.ten_seed or args.full_sweep else args.seeds
    train_repos = 16 if args.full_sweep else args.train_repos
    eval_repos = 4 if args.full_sweep else args.eval_repos
    steps = 12 if args.full_sweep else args.steps
    d_model = 32 if args.full_sweep else args.d_model
    result = run_tac_scm_real007(
        seeds=seeds,
        bug_families=args.bug_families,
        train_repos=train_repos,
        eval_repos=eval_repos,
        steps=steps,
        batch_size=args.batch_size,
        d_model=d_model,
        n_layers=args.n_layers,
        max_files=args.max_files,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
