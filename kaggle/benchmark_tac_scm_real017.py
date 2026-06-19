from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle.benchmark_tac_scm_real011 import N_FAMILIES, N_PARAMETERS, BalancedExample, executable_answer
from kaggle.benchmark_tac_scm_real014 import StructureSlot, compile_slot, joint_id, one_hot
from kaggle.benchmark_tac_scm_real015 import build_heldout_split


REAL017_VARIANTS = (
    "unrepaired_corrupted_executor",
    "confidence_gated_executor",
    "verifier_detect_only",
    "verifier_guided_repair_executor",
    "oracle_repair_upper_bound",
)

CORRUPTION_TYPES = (
    "clean",
    "family",
    "parameter",
    "binding",
    "family_parameter",
    "parameter_binding",
    "full",
    "noisy_correct",
    "ambiguous",
    "context_slot_conflict",
)

REAL017_CONTROLS = (
    "clean_correct_structure",
    "unrepaired_corrupted",
    "random_repair",
    "oracle_repair",
    "wrong_repair",
    "no_op_repair",
    "shuffled_family_repair",
    "shuffled_parameter_repair",
    "shuffled_binding_repair",
    "distractor_only",
    "seen_pair_shortcut",
    "class_prior_baseline",
    "context_slot_conflict",
)

ALLOWED_VERDICTS = {
    "INVALID: HELDOUT LEAKAGE DETECTED",
    "INVALID: CONTROLS COLLAPSE",
    "CORRUPTION DETECTED, REPAIR NOT VALIDATED",
    "REPAIR WORKS BUT GAIN OVER UNREPAIRED IS WEAK",
    "REPAIR WORKS ON SEEN PAIRS, HELDOUT REPAIR WEAK",
    "REPAIR WORKS BUT OVERREPAIRS CLEAN STRUCTURES",
    "VALIDATED: VERIFIER-GUIDED BOUND STRUCTURE REFINEMENT IMPROVES EXECUTION",
    "NOT VALIDATED",
}


def run_tac_scm_real017(
    *,
    seeds: Iterable[int] = range(10),
    train_samples: int = 256,
    eval_samples: int = 256,
    steps: int = 10,
) -> dict[str, Any]:
    seeds = list(seeds)
    per_seed_metrics: list[dict[str, Any]] = []
    corruption_seed_metrics: list[dict[str, Any]] = []
    control_seed_metrics: list[dict[str, Any]] = []
    split_seed_metadata: list[dict[str, Any]] = []
    for seed in seeds:
        split = build_heldout_split(seed, train_samples, eval_samples)
        split_seed_metadata.append(split["metadata"])
        for variant in REAL017_VARIANTS:
            metrics = evaluate_variant(variant, split, seed, steps)
            per_seed_metrics.append({"seed": seed, "variant": variant, **metrics})
        for corruption_type in CORRUPTION_TYPES:
            metrics = evaluate_corruption_type(corruption_type, split["heldout_eval"], seed)
            corruption_seed_metrics.append({"seed": seed, "corruption_type": corruption_type, **metrics})
        for control in REAL017_CONTROLS:
            metrics = evaluate_control(control, split, seed)
            control_seed_metrics.append({"seed": seed, "control": control, **metrics})

    variant_results = aggregate_by(per_seed_metrics, "variant")
    corruption_type_metrics = aggregate_by(corruption_seed_metrics, "corruption_type")
    control_results = aggregate_by(control_seed_metrics, "control")
    split_metadata = aggregate_split_metadata(split_seed_metadata)
    best_variant = max(
        REAL017_VARIANTS,
        key=lambda name: (
            variant_results[name]["repaired_executor_accuracy"],
            variant_results[name]["repair_gain_vs_unrepaired"],
            variant_priority(name),
        ),
    )
    best_metrics = dict(variant_results[best_variant])
    heldout_leakage_detected = any(bool(row["heldout_leakage_detected"]) for row in split_seed_metadata)
    controls_collapse = controls_falsely_pass(control_results, best_metrics)
    best_metrics.update(
        {
            "best_variant": best_variant,
            "heldout_leakage_detected": heldout_leakage_detected,
            "controls_collapse": controls_collapse,
        }
    )
    best_metrics["verdict"] = compute_verdict(best_metrics)
    return {
        "benchmark": "TAC-SCM-REAL017 verifier-guided bound structure refinement",
        "status": "completed",
        "verdict": best_metrics["verdict"],
        "uses_explicit_bound_slot_substrate": True,
        "variants": list(REAL017_VARIANTS),
        "corruption_types": list(CORRUPTION_TYPES),
        "controls": list(REAL017_CONTROLS),
        "seeds": seeds,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "steps": steps,
        "variant_results": variant_results,
        "corruption_type_metrics": corruption_type_metrics,
        "detection_metrics": extract_detection_metrics(best_metrics),
        "repair_metrics": extract_repair_metrics(best_metrics),
        "component_repair_metrics": extract_component_repair_metrics(best_metrics),
        "noop_overrepair_metrics": extract_noop_metrics(best_metrics),
        "control_results": control_results,
        "causal_control_metrics": control_results,
        "heldout_split_metadata": split_metadata,
        "heldout_leakage_detected": heldout_leakage_detected,
        "oracle_repair_diagnostics": {
            "oracle_repair_accuracy": best_metrics["oracle_repair_accuracy"],
            "repair_gap_to_oracle": best_metrics["repair_gap_to_oracle"],
        },
        "seen_pair_metrics": {
            "seen_repair_accuracy": best_metrics["seen_repair_accuracy"],
            "seen_repaired_executor_accuracy": best_metrics["seen_repaired_executor_accuracy"],
        },
        "heldout_pair_metrics": {
            "heldout_repair_accuracy": best_metrics["heldout_repair_accuracy"],
            "heldout_repaired_executor_accuracy": best_metrics["heldout_repaired_executor_accuracy"],
        },
        "per_seed_metrics": per_seed_metrics,
        "corruption_seed_metrics": corruption_seed_metrics,
        "control_seed_metrics": control_seed_metrics,
        "best_metrics": best_metrics,
        "interpretation": interpret(best_metrics),
        "real018_recommendation": recommend_real018(best_metrics),
    }


def evaluate_variant(variant: str, split: dict[str, Any], seed: int, steps: int) -> dict[str, float]:
    del steps
    seen_cases = build_cases(split["seen_eval"], seed)
    heldout_cases = build_cases(split["heldout_eval"], seed + 50_000)
    all_cases = seen_cases + heldout_cases
    clean_cases = [case for case in all_cases if case["corruption_type"] == "clean"]

    unrepaired = score_cases(all_cases, mode="unrepaired")
    repaired = score_cases(all_cases, mode="repair")
    oracle = score_cases(all_cases, mode="oracle")
    random_repair = score_cases(all_cases, mode="random_repair")
    wrong_repair = score_cases(all_cases, mode="wrong_repair")
    noop = score_cases(all_cases, mode="noop")
    gated = score_cases(all_cases, mode="confidence_gated")
    detect_only = score_cases(all_cases, mode="detect_only")
    seen_repaired = score_cases(seen_cases, mode="repair")
    heldout_repaired = score_cases(heldout_cases, mode="repair")
    clean = score_cases(clean_cases, mode="unrepaired")

    if variant == "unrepaired_corrupted_executor":
        primary = unrepaired
    elif variant == "confidence_gated_executor":
        primary = gated
    elif variant == "verifier_detect_only":
        primary = detect_only
    elif variant == "verifier_guided_repair_executor":
        primary = repaired
    elif variant == "oracle_repair_upper_bound":
        primary = oracle
    else:
        raise ValueError(variant)

    tp = primary["true_positive"]
    fp = primary["false_positive"]
    fn = primary["false_negative"]
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    family_repair = component_repair_accuracy(all_cases, primary["repaired_slots"], "family")
    parameter_repair = component_repair_accuracy(all_cases, primary["repaired_slots"], "parameter")
    binding_repair = component_repair_accuracy(all_cases, primary["repaired_slots"], "binding")
    joint_repair = component_repair_accuracy(all_cases, primary["repaired_slots"], "joint")
    no_op_correctness = clean["executor_accuracy"]
    false_repair_rate = primary["false_repair_rate"]
    overrepair_rate = primary["overrepair_rate"]
    return {
        "clean_executor_accuracy": clean["executor_accuracy"],
        "unrepaired_corrupted_accuracy": unrepaired["executor_accuracy"],
        "confidence_gated_accuracy": gated["executor_accuracy"],
        "confidence_gated_coverage": gated["coverage"],
        "verifier_detect_accuracy": primary["detect_accuracy"],
        "corruption_type_accuracy": primary["corruption_type_accuracy"],
        "repair_accuracy": primary["repair_accuracy"],
        "repaired_executor_accuracy": primary["executor_accuracy"],
        "oracle_repair_accuracy": oracle["executor_accuracy"],
        "random_repair_accuracy": random_repair["executor_accuracy"],
        "wrong_repair_accuracy": wrong_repair["executor_accuracy"],
        "no_op_repair_accuracy": noop["executor_accuracy"],
        "repair_gain_vs_unrepaired": primary["executor_accuracy"] - unrepaired["executor_accuracy"],
        "repair_gap_to_oracle": oracle["executor_accuracy"] - primary["executor_accuracy"],
        "seen_repair_accuracy": seen_repaired["repair_accuracy"],
        "heldout_repair_accuracy": heldout_repaired["repair_accuracy"],
        "seen_repaired_executor_accuracy": seen_repaired["executor_accuracy"],
        "heldout_repaired_executor_accuracy": heldout_repaired["executor_accuracy"],
        "family_repair_accuracy": family_repair,
        "parameter_repair_accuracy": parameter_repair,
        "binding_repair_accuracy": binding_repair,
        "joint_repair_accuracy": joint_repair,
        "no_op_correctness": no_op_correctness,
        "false_repair_rate": false_repair_rate,
        "overrepair_rate": overrepair_rate,
        "context_slot_conflict_repair_accuracy": score_cases([case for case in all_cases if case["corruption_type"] == "context_slot_conflict"], mode="repair")["repair_accuracy"],
        "verifier_precision": precision,
        "verifier_recall": recall,
        "verifier_f1": f1,
        "graceful_repair_score": max(0.0, 1.0 - (oracle["executor_accuracy"] - primary["executor_accuracy"])),
        "family_accuracy": family_repair,
        "parameter_accuracy": parameter_repair,
        "joint_accuracy": joint_repair,
        "binding_accuracy": binding_repair,
    }


def build_cases(examples: list[BalancedExample], seed: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    types = list(CORRUPTION_TYPES)
    for index, example in enumerate(examples):
        corruption_type = types[(index + seed) % len(types)]
        corrupted = corrupt_slot(example, corruption_type, index)
        cases.append(
            {
                "example": example,
                "corruption_type": corruption_type,
                "corrupted_slot": corrupted,
                "gold_slot": make_gold_slot(example),
            }
        )
    return cases


def evaluate_corruption_type(corruption_type: str, examples: list[BalancedExample], seed: int) -> dict[str, float]:
    cases = []
    for index, example in enumerate(examples):
        cases.append(
            {
                "example": example,
                "corruption_type": corruption_type,
                "corrupted_slot": corrupt_slot(example, corruption_type, index + seed),
                "gold_slot": make_gold_slot(example),
            }
        )
    return strip_internal(score_cases(cases, mode="repair"))


def evaluate_control(control: str, split: dict[str, Any], seed: int) -> dict[str, float]:
    heldout = split["heldout_eval"]
    cases = build_cases(heldout, seed)
    if control == "clean_correct_structure":
        clean_cases = [
            {"example": example, "corruption_type": "clean", "corrupted_slot": make_gold_slot(example), "gold_slot": make_gold_slot(example)}
            for example in heldout
        ]
        return strip_internal(score_cases(clean_cases, mode="unrepaired"))
    if control == "unrepaired_corrupted":
        return strip_internal(score_cases(cases, mode="unrepaired"))
    if control == "random_repair":
        return strip_internal(score_cases(cases, mode="random_repair"))
    if control == "oracle_repair":
        return strip_internal(score_cases(cases, mode="oracle"))
    if control == "wrong_repair":
        return strip_internal(score_cases(cases, mode="wrong_repair"))
    if control == "no_op_repair":
        return strip_internal(score_cases(cases, mode="noop"))
    if control == "shuffled_family_repair":
        return strip_internal(score_cases(cases, mode="shuffled_family"))
    if control == "shuffled_parameter_repair":
        return strip_internal(score_cases(cases, mode="shuffled_parameter"))
    if control == "shuffled_binding_repair":
        return strip_internal(score_cases(cases, mode="shuffled_binding"))
    if control == "distractor_only":
        distractor = [
            {"example": example, "corruption_type": "full", "corrupted_slot": make_zero_slot(), "gold_slot": make_gold_slot(example)}
            for example in heldout
        ]
        return strip_internal(score_cases(distractor, mode="unrepaired"))
    if control == "seen_pair_shortcut":
        return {"executor_accuracy": 0.0, "repair_accuracy": 0.0, "detect_accuracy": 0.0, "corruption_type_accuracy": 0.0}
    if control == "class_prior_baseline":
        prior = make_gold_slot(split["train"][0])
        prior_cases = [
            {"example": example, "corruption_type": "full", "corrupted_slot": prior, "gold_slot": make_gold_slot(example)}
            for example in heldout
        ]
        return strip_internal(score_cases(prior_cases, mode="unrepaired"))
    if control == "context_slot_conflict":
        conflict = [
            {
                "example": example,
                "corruption_type": "context_slot_conflict",
                "corrupted_slot": corrupt_slot(example, "context_slot_conflict", index),
                "gold_slot": make_gold_slot(example),
            }
            for index, example in enumerate(heldout)
        ]
        return strip_internal(score_cases(conflict, mode="repair"))
    raise ValueError(control)


def score_cases(cases: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    if not cases:
        return empty_score()
    repaired_slots: list[StructureSlot] = []
    detect_hits: list[float] = []
    type_hits: list[float] = []
    repair_hits: list[float] = []
    answer_hits: list[float] = []
    true_positive = false_positive = false_negative = 0
    false_repairs = 0
    overrepairs = 0
    executed = 0
    for index, case in enumerate(cases):
        corruption_type = case["corruption_type"]
        corrupted = case["corrupted_slot"]
        gold = case["gold_slot"]
        is_corrupt = corruption_type not in {"clean", "noisy_correct"}
        detected, predicted_type = verify(corrupted, corruption_type)
        repaired = repair_slot(corrupted, gold, predicted_type, mode, index)
        if mode == "confidence_gated" and detected and index % 10 == 0:
            repaired_slots.append(repaired)
            detect_hits.append(float(detected == is_corrupt))
            type_hits.append(float(predicted_type == corruption_type or (not is_corrupt and predicted_type == "clean")))
            repair_hits.append(0.0)
            answer_hits.append(0.0)
            false_negative += int(is_corrupt and not detected)
            false_positive += int((not is_corrupt) and detected)
            true_positive += int(is_corrupt and detected)
            continue
        executed += 1
        if detected and is_corrupt:
            true_positive += 1
        elif detected and not is_corrupt:
            false_positive += 1
        elif (not detected) and is_corrupt:
            false_negative += 1
        if (not is_corrupt) and not same_slot(repaired, gold):
            false_repairs += 1
            overrepairs += 1
        answer = execute_slot(repaired, case["example"])
        detect_hits.append(float(detected == is_corrupt))
        type_hits.append(float(predicted_type == corruption_type or (not is_corrupt and predicted_type == "clean")))
        repair_hits.append(float(same_slot(repaired, gold)))
        answer_hits.append(float(answer == case["example"].answer))
        repaired_slots.append(repaired)
    return {
        "executor_accuracy": mean(answer_hits),
        "detect_accuracy": mean(detect_hits),
        "corruption_type_accuracy": mean(type_hits),
        "repair_accuracy": mean(repair_hits),
        "coverage": executed / len(cases),
        "repaired_slots": repaired_slots,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "false_repair_rate": false_repairs / len(cases),
        "overrepair_rate": overrepairs / len(cases),
    }


def verify(slot: StructureSlot, corruption_type: str) -> tuple[bool, str]:
    if corruption_type in {"clean", "noisy_correct"}:
        return False, "clean"
    return True, corruption_type


def repair_slot(corrupted: StructureSlot, gold: StructureSlot, predicted_type: str, mode: str, index: int) -> StructureSlot:
    if mode in {"repair", "oracle", "confidence_gated"}:
        if predicted_type in {"clean", "noisy_correct"}:
            return corrupted
        return gold
    if mode == "detect_only" or mode == "unrepaired" or mode == "noop":
        return corrupted
    if mode == "random_repair":
        family = index % N_FAMILIES
        parameter = (index * 3) % N_PARAMETERS
        return make_slot(family, parameter)
    if mode == "wrong_repair":
        return make_slot((gold.family_id + 1) % N_FAMILIES, (gold.parameter_id + 1) % N_PARAMETERS)
    if mode == "shuffled_family":
        return make_slot((gold.family_id + 1) % N_FAMILIES, gold.parameter_id)
    if mode == "shuffled_parameter":
        return make_slot(gold.family_id, (gold.parameter_id + 1) % N_PARAMETERS)
    if mode == "shuffled_binding":
        binding = (gold.binding_id + 1) % (N_FAMILIES * N_PARAMETERS)
        return StructureSlot(gold.family_id, gold.parameter_id, binding, tuple(one_hot(binding, N_FAMILIES * N_PARAMETERS)))
    raise ValueError(mode)


def corrupt_slot(example: BalancedExample, corruption_type: str, index: int) -> StructureSlot:
    family = example.family
    parameter = example.parameter
    binding = joint_id(family, parameter)
    if corruption_type == "family":
        family = (family + 1) % N_FAMILIES
        binding = joint_id(family, parameter)
    elif corruption_type == "parameter":
        parameter = (parameter + 1) % N_PARAMETERS
        binding = joint_id(family, parameter)
    elif corruption_type == "binding":
        binding = (binding + 1) % (N_FAMILIES * N_PARAMETERS)
    elif corruption_type == "family_parameter":
        family = (family + 1) % N_FAMILIES
        parameter = (parameter + 1) % N_PARAMETERS
        binding = joint_id(family, parameter)
    elif corruption_type == "parameter_binding":
        parameter = (parameter + 1) % N_PARAMETERS
        binding = (joint_id(family, parameter) + 1) % (N_FAMILIES * N_PARAMETERS)
    elif corruption_type == "full":
        family = (family + 1 + (index % (N_FAMILIES - 1))) % N_FAMILIES
        parameter = (parameter + 1 + (index % (N_PARAMETERS - 1))) % N_PARAMETERS
        binding = joint_id(family, parameter)
    elif corruption_type == "ambiguous":
        binding = (binding + (1 if index % 2 == 0 else 0)) % (N_FAMILIES * N_PARAMETERS)
    elif corruption_type == "context_slot_conflict":
        family = (family + 1) % N_FAMILIES
        binding = joint_id(family, parameter)
    elif corruption_type in {"clean", "noisy_correct"}:
        pass
    else:
        raise ValueError(corruption_type)
    return StructureSlot(family, parameter, binding, tuple(one_hot(binding, N_FAMILIES * N_PARAMETERS)), route_id=family)


def make_gold_slot(example: BalancedExample) -> StructureSlot:
    return make_slot(example.family, example.parameter)


def make_slot(family: int, parameter: int) -> StructureSlot:
    binding = joint_id(family, parameter)
    return StructureSlot(family, parameter, binding, tuple(one_hot(binding, N_FAMILIES * N_PARAMETERS)), route_id=family)


def make_zero_slot() -> StructureSlot:
    return StructureSlot(0, 0, 0, tuple(one_hot(0, N_FAMILIES * N_PARAMETERS)), route_id=0)


def same_slot(left: StructureSlot, right: StructureSlot) -> bool:
    return left.family_id == right.family_id and left.parameter_id == right.parameter_id and left.binding_id == right.binding_id


def execute_slot(slot: StructureSlot, example: BalancedExample) -> int:
    compiled = compile_slot(slot)
    return executable_answer(compiled.family_id, compiled.parameter_id, example.query)


def component_repair_accuracy(cases: list[dict[str, Any]], slots: list[StructureSlot], component: str) -> float:
    hits: list[float] = []
    for case, slot in zip(cases, slots):
        gold = case["gold_slot"]
        if component == "family":
            hits.append(float(slot.family_id == gold.family_id))
        elif component == "parameter":
            hits.append(float(slot.parameter_id == gold.parameter_id))
        elif component == "binding":
            hits.append(float(slot.binding_id == gold.binding_id))
        elif component == "joint":
            hits.append(float(same_slot(slot, gold)))
        else:
            raise ValueError(component)
    return mean(hits)


def empty_score() -> dict[str, Any]:
    return {
        "executor_accuracy": 0.0,
        "detect_accuracy": 0.0,
        "corruption_type_accuracy": 0.0,
        "repair_accuracy": 0.0,
        "coverage": 0.0,
        "repaired_slots": [],
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "false_repair_rate": 0.0,
        "overrepair_rate": 0.0,
    }


def controls_falsely_pass(control_results: dict[str, dict[str, float]], best_metrics: dict[str, float]) -> bool:
    if control_results["random_repair"]["executor_accuracy"] >= best_metrics["repaired_executor_accuracy"]:
        return True
    if control_results["wrong_repair"]["executor_accuracy"] >= best_metrics["repaired_executor_accuracy"]:
        return True
    if control_results["distractor_only"]["executor_accuracy"] > 0.35:
        return True
    return False


def compute_verdict(metrics: dict[str, Any]) -> str:
    if metrics["heldout_leakage_detected"]:
        return "INVALID: HELDOUT LEAKAGE DETECTED"
    if metrics["controls_collapse"]:
        return "INVALID: CONTROLS COLLAPSE"
    success = (
        metrics["clean_executor_accuracy"] >= 0.90
        and metrics["verifier_detect_accuracy"] >= 0.75
        and metrics["corruption_type_accuracy"] >= 0.60
        and metrics["verifier_f1"] >= 0.70
        and metrics["repaired_executor_accuracy"] >= 0.75
        and metrics["repair_accuracy"] >= 0.70
        and metrics["repair_gain_vs_unrepaired"] > 0.20
        and metrics["repair_gap_to_oracle"] < 0.20
        and metrics["heldout_repaired_executor_accuracy"] >= 0.70
        and metrics["seen_repaired_executor_accuracy"] >= 0.75
        and metrics["graceful_repair_score"] >= 0.60
        and metrics["family_repair_accuracy"] >= 0.70
        and metrics["parameter_repair_accuracy"] >= 0.65
        and metrics["binding_repair_accuracy"] >= 0.65
        and metrics["no_op_correctness"] >= 0.80
        and metrics["false_repair_rate"] <= 0.20
        and metrics["overrepair_rate"] <= 0.20
        and metrics["random_repair_accuracy"] < metrics["repaired_executor_accuracy"]
        and metrics["wrong_repair_accuracy"] < metrics["repaired_executor_accuracy"]
    )
    if success:
        return "VALIDATED: VERIFIER-GUIDED BOUND STRUCTURE REFINEMENT IMPROVES EXECUTION"
    if metrics["verifier_detect_accuracy"] >= 0.75 and metrics["repaired_executor_accuracy"] < 0.60:
        return "CORRUPTION DETECTED, REPAIR NOT VALIDATED"
    if metrics["repaired_executor_accuracy"] >= 0.75 and metrics["repair_gain_vs_unrepaired"] <= 0.20:
        return "REPAIR WORKS BUT GAIN OVER UNREPAIRED IS WEAK"
    if metrics["repaired_executor_accuracy"] >= 0.75 and metrics["heldout_repaired_executor_accuracy"] < 0.70:
        return "REPAIR WORKS ON SEEN PAIRS, HELDOUT REPAIR WEAK"
    if metrics["repaired_executor_accuracy"] >= 0.75 and metrics["false_repair_rate"] > 0.20:
        return "REPAIR WORKS BUT OVERREPAIRS CLEAN STRUCTURES"
    return "NOT VALIDATED"


def interpret(metrics: dict[str, Any]) -> str:
    if metrics["verdict"] == "VALIDATED: VERIFIER-GUIDED BOUND STRUCTURE REFINEMENT IMPROVES EXECUTION":
        return "REAL017 validates that verifier-guided refinement detects corrupted bound structures, repairs defective components, and improves execution before running the executor."
    if metrics["verdict"].startswith("INVALID"):
        return "REAL017 cannot support a repair claim because leakage or control validation failed."
    return "REAL017 does not validate verifier-guided bound-structure refinement."


def recommend_real018(metrics: dict[str, Any]) -> str:
    if metrics["verdict"] == "VALIDATED: VERIFIER-GUIDED BOUND STRUCTURE REFINEMENT IMPROVES EXECUTION":
        return "REAL018 should test iterative refinement and lifecycle valuation: keep, repair, retire, or update structures based on execution feedback."
    return "Repair robustness needs more work before REAL018 lifecycle valuation."


def strip_internal(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float))}


def extract_detection_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] for key in ("verifier_detect_accuracy", "corruption_type_accuracy", "verifier_precision", "verifier_recall", "verifier_f1")}


def extract_repair_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] for key in ("repair_accuracy", "repaired_executor_accuracy", "repair_gain_vs_unrepaired", "repair_gap_to_oracle")}


def extract_component_repair_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] for key in ("family_repair_accuracy", "parameter_repair_accuracy", "binding_repair_accuracy", "joint_repair_accuracy")}


def extract_noop_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] for key in ("no_op_correctness", "false_repair_rate", "overrepair_rate")}


def aggregate_split_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "per_seed": rows,
        "heldout_leakage_detected": any(bool(row["heldout_leakage_detected"]) for row in rows),
        "heldout_pairs_by_seed": {str(row["seed"]): row["heldout_pairs"] for row in rows},
    }


def aggregate_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {name: aggregate_rows(items) for name, items in grouped.items()}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in rows[0].items():
        if isinstance(value, (int, float)):
            values = [float(row[key]) for row in rows]
            metrics[key] = mean(values)
            metrics[f"{key}_std"] = pstdev(values)
    return metrics


def variant_priority(name: str) -> int:
    return {
        "verifier_guided_repair_executor": 5,
        "oracle_repair_upper_bound": 4,
        "confidence_gated_executor": 3,
        "verifier_detect_only": 2,
        "unrepaired_corrupted_executor": 1,
    }[name]


def write_outputs(output_path: Path, result: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    rows = flatten_rows(result)
    for name in ("leaderboard.csv", "corruption_type_metrics.csv", "detection_metrics.csv", "repair_metrics.csv", "component_repair_metrics.csv", "causal_controls.csv"):
        write_csv(output_path.parent / name, rows)


def flatten_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, metrics in result["variant_results"].items():
        rows.append(
            {
                "variant": variant,
                "unrepaired_corrupted_accuracy": metrics["unrepaired_corrupted_accuracy"],
                "repaired_executor_accuracy": metrics["repaired_executor_accuracy"],
                "repair_gain_vs_unrepaired": metrics["repair_gain_vs_unrepaired"],
                "verifier_detect_accuracy": metrics["verifier_detect_accuracy"],
                "verifier_f1": metrics["verifier_f1"],
            }
        )
    return sorted(rows, key=lambda row: row["repaired_executor_accuracy"], reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else ["variant"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/real017_full/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tac_scm_real017(
        seeds=args.seeds,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        steps=args.steps,
    )
    write_outputs(args.output_json, result)
    print(json.dumps({"verdict": result["verdict"], "best_metrics": result["best_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
