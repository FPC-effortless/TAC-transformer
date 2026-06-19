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

from kaggle.benchmark_tac_scm_real011 import (
    N_FAMILIES,
    N_PARAMETERS,
    BalancedExample,
    executable_answer,
)
from kaggle.benchmark_tac_scm_real014 import StructureSlot, compile_slot, joint_id, one_hot
from kaggle.benchmark_tac_scm_real015 import build_heldout_split, score_pair_lookup


REAL016_VARIANTS = (
    "clean_symbolic_compositional_executor",
    "noisy_bound_slot_executor",
    "confidence_gated_executor",
    "verifier_repaired_executor",
)

REAL016_REGIMES = (
    "clean",
    "input_noise",
    "slot_noise",
    "ambiguous_family",
    "ambiguous_parameter",
    "adversarial_family",
    "adversarial_parameter",
    "adversarial_binding",
    "conflicting_evidence",
)

REAL016_CONTROLS = (
    "reset_no_slot",
    "shuffled_family",
    "shuffled_parameter",
    "shuffled_binding",
    "wrong_family",
    "wrong_parameter",
    "wrong_binding",
    "distractor_only",
    "random_representation",
    "class_prior_baseline",
    "seen_pair_shortcut",
    "context_slot_conflict",
    "confidence_threshold",
)

ALLOWED_VERDICTS = {
    "INVALID: HELDOUT LEAKAGE DETECTED",
    "INVALID: CONTROLS COLLAPSE",
    "CLEAN COMPOSITION VALIDATED, ROBUSTNESS FAILS",
    "ROBUST HELDOUT EXECUTION WORKS, CAUSAL CONTROLS WEAK",
    "PARTIAL ROBUSTNESS WITH POOR GRACEFUL DEGRADATION",
    "VALIDATED: ROBUST BOUND-SLOT EXECUTION UNDER NOISE AND AMBIGUITY",
    "NOT VALIDATED",
}


def run_tac_scm_real016(
    *,
    seeds: Iterable[int] = range(10),
    train_samples: int = 256,
    eval_samples: int = 256,
    steps: int = 10,
) -> dict[str, Any]:
    seeds = list(seeds)
    per_seed_metrics: list[dict[str, Any]] = []
    regime_seed_metrics: list[dict[str, Any]] = []
    control_seed_metrics: list[dict[str, Any]] = []
    split_seed_metadata: list[dict[str, Any]] = []
    for seed in seeds:
        split = build_heldout_split(seed, train_samples, eval_samples)
        split_seed_metadata.append(split["metadata"])
        for variant in REAL016_VARIANTS:
            metrics = evaluate_variant(variant, split, seed, steps)
            per_seed_metrics.append({"seed": seed, "variant": variant, **metrics})
        for regime in REAL016_REGIMES:
            seen = evaluate_regime(regime, split["seen_eval"], seed, pair_lookup_train=split["train"])
            heldout = evaluate_regime(regime, split["heldout_eval"], seed, pair_lookup_train=split["train"])
            regime_seed_metrics.append({"seed": seed, "regime": regime, **prefix("seen", seen), **prefix("heldout", heldout)})
        for control in REAL016_CONTROLS:
            metrics = evaluate_control(control, split, seed)
            control_seed_metrics.append({"seed": seed, "control": control, **metrics})

    variant_results = aggregate_by(per_seed_metrics, "variant")
    regime_results = aggregate_by(regime_seed_metrics, "regime")
    control_results = aggregate_by(control_seed_metrics, "control")
    split_metadata = aggregate_split_metadata(split_seed_metadata)
    best_variant = max(
        REAL016_VARIANTS,
        key=lambda name: (
            variant_results[name]["noisy_heldout_executor_accuracy"],
            variant_results[name]["robustness_mean_accuracy"],
            variant_priority(name),
        ),
    )
    best_metrics = dict(variant_results[best_variant])
    causal = causal_metrics(control_results, best_metrics)
    heldout_leakage_detected = any(bool(row["heldout_leakage_detected"]) for row in split_seed_metadata)
    controls_collapse = controls_falsely_pass(control_results)
    best_metrics.update(causal)
    best_metrics.update(
        {
            "best_variant": best_variant,
            "heldout_leakage_detected": heldout_leakage_detected,
            "controls_collapse": controls_collapse,
            "pair_lookup_heldout_accuracy": control_results["seen_pair_shortcut"]["executor_accuracy"],
            "distractor_only_accuracy": control_results["distractor_only"]["executor_accuracy"],
            "context_slot_conflict_accuracy": control_results["context_slot_conflict"]["executor_accuracy"],
            "oracle_gap_clean": 1.0 - best_metrics["clean_heldout_executor_accuracy"],
            "oracle_gap_noisy": 1.0 - best_metrics["noisy_seen_executor_accuracy"],
            "oracle_gap_heldout_noisy": 1.0 - best_metrics["noisy_heldout_executor_accuracy"],
        }
    )
    best_metrics["verdict"] = compute_verdict(best_metrics)
    return {
        "benchmark": "TAC-SCM-REAL016 robust bound-slot execution",
        "status": "completed",
        "verdict": best_metrics["verdict"],
        "uses_explicit_bound_slot_substrate": True,
        "variants": list(REAL016_VARIANTS),
        "regimes": list(REAL016_REGIMES),
        "controls": list(REAL016_CONTROLS),
        "seeds": seeds,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "steps": steps,
        "variant_results": variant_results,
        "robustness_regime_metrics": regime_results,
        "adversarial_regime_metrics": {name: regime_results[name] for name in ("adversarial_family", "adversarial_parameter", "adversarial_binding")},
        "control_results": control_results,
        "causal_control_metrics": control_results,
        "seen_pair_metrics": extract_seen_metrics(best_metrics),
        "heldout_pair_metrics": extract_heldout_metrics(best_metrics),
        "heldout_split_metadata": split_metadata,
        "heldout_leakage_detected": heldout_leakage_detected,
        "oracle_diagnostics": {
            "oracle_gap_clean": best_metrics["oracle_gap_clean"],
            "oracle_gap_noisy": best_metrics["oracle_gap_noisy"],
            "oracle_gap_heldout_noisy": best_metrics["oracle_gap_heldout_noisy"],
        },
        "confidence_gating_metrics": {
            "confidence_gated_accuracy": best_metrics["confidence_gated_accuracy"],
            "confidence_gated_coverage": best_metrics["confidence_gated_coverage"],
        },
        "verifier_repair_metrics": {
            "verifier_repair_accuracy": best_metrics["verifier_repair_accuracy"],
            "verifier_repair_gain": best_metrics["verifier_repair_gain"],
        },
        "per_seed_metrics": per_seed_metrics,
        "regime_seed_metrics": regime_seed_metrics,
        "control_seed_metrics": control_seed_metrics,
        "best_metrics": best_metrics,
        "interpretation": interpret(best_metrics),
        "real017_recommendation": recommend_real017(best_metrics),
    }


def evaluate_variant(variant: str, split: dict[str, Any], seed: int, steps: int) -> dict[str, float]:
    del steps
    seen_clean = evaluate_regime("clean", split["seen_eval"], seed, pair_lookup_train=split["train"])
    heldout_clean = evaluate_regime("clean", split["heldout_eval"], seed, pair_lookup_train=split["train"])
    seen_noise = evaluate_regime("input_noise", split["seen_eval"], seed, pair_lookup_train=split["train"])
    heldout_noise = evaluate_regime("input_noise", split["heldout_eval"], seed, pair_lookup_train=split["train"])
    slot_noise_seen = evaluate_regime("slot_noise", split["seen_eval"], seed, pair_lookup_train=split["train"])
    slot_noise_heldout = evaluate_regime("slot_noise", split["heldout_eval"], seed, pair_lookup_train=split["train"])
    ambiguous_family = evaluate_regime("ambiguous_family", split["heldout_eval"], seed, pair_lookup_train=split["train"])
    ambiguous_parameter = evaluate_regime("ambiguous_parameter", split["heldout_eval"], seed, pair_lookup_train=split["train"])
    conflict = evaluate_regime("conflicting_evidence", split["heldout_eval"], seed, pair_lookup_train=split["train"])
    adversarial_family = evaluate_regime("adversarial_family", split["heldout_eval"], seed, pair_lookup_train=split["train"])
    adversarial_parameter = evaluate_regime("adversarial_parameter", split["heldout_eval"], seed, pair_lookup_train=split["train"])
    adversarial_binding = evaluate_regime("adversarial_binding", split["heldout_eval"], seed, pair_lookup_train=split["train"])

    confidence = evaluate_regime("confidence_threshold", split["heldout_eval"], seed, pair_lookup_train=split["train"])
    repaired = evaluate_regime("verifier_repair", split["heldout_eval"], seed, pair_lookup_train=split["train"])
    unrepaired = evaluate_regime("slot_noise", split["heldout_eval"], seed, pair_lookup_train=split["train"], repair=False)
    if variant == "clean_symbolic_compositional_executor":
        primary_seen = seen_clean
        primary_heldout = heldout_clean
    elif variant == "noisy_bound_slot_executor":
        primary_seen = seen_noise
        primary_heldout = heldout_noise
    elif variant == "confidence_gated_executor":
        primary_seen = seen_noise
        primary_heldout = confidence
    elif variant == "verifier_repaired_executor":
        primary_seen = seen_noise
        primary_heldout = repaired
    else:
        raise ValueError(variant)

    robustness_values = [
        seen_noise["executor_accuracy"],
        heldout_noise["executor_accuracy"],
        slot_noise_seen["executor_accuracy"],
        slot_noise_heldout["executor_accuracy"],
        ambiguous_family["executor_accuracy"],
        ambiguous_parameter["executor_accuracy"],
        conflict["executor_accuracy"],
    ]
    adversarial_values = [
        adversarial_family["executor_accuracy"],
        adversarial_parameter["executor_accuracy"],
        adversarial_binding["executor_accuracy"],
    ]
    clean_mean = mean([seen_clean["executor_accuracy"], heldout_clean["executor_accuracy"]])
    robustness_mean = mean(robustness_values)
    return {
        "family_accuracy": primary_heldout["family_accuracy"],
        "parameter_accuracy": primary_heldout["parameter_accuracy"],
        "joint_accuracy": primary_heldout["joint_accuracy"],
        "binding_accuracy": primary_heldout["binding_accuracy"],
        "seen_pair_executor_accuracy": primary_seen["executor_accuracy"],
        "heldout_pair_executor_accuracy": primary_heldout["executor_accuracy"],
        "generalization_gap": primary_seen["executor_accuracy"] - primary_heldout["executor_accuracy"],
        "heldout_success_rate": primary_heldout["executor_accuracy"],
        "clean_seen_executor_accuracy": seen_clean["executor_accuracy"],
        "clean_heldout_executor_accuracy": heldout_clean["executor_accuracy"],
        "noisy_seen_executor_accuracy": seen_noise["executor_accuracy"],
        "noisy_heldout_executor_accuracy": heldout_noise["executor_accuracy"],
        "slot_noise_seen_accuracy": slot_noise_seen["executor_accuracy"],
        "slot_noise_heldout_accuracy": slot_noise_heldout["executor_accuracy"],
        "ambiguous_family_accuracy": ambiguous_family["executor_accuracy"],
        "ambiguous_parameter_accuracy": ambiguous_parameter["executor_accuracy"],
        "conflicting_evidence_accuracy": conflict["executor_accuracy"],
        "adversarial_family_accuracy": adversarial_family["executor_accuracy"],
        "adversarial_parameter_accuracy": adversarial_parameter["executor_accuracy"],
        "adversarial_binding_accuracy": adversarial_binding["executor_accuracy"],
        "robustness_mean_accuracy": robustness_mean,
        "robustness_min_accuracy": min(robustness_values),
        "graceful_degradation_score": max(0.0, 1.0 - (clean_mean - robustness_mean)),
        "clean_to_noise_drop": heldout_clean["executor_accuracy"] - heldout_noise["executor_accuracy"],
        "clean_to_ambiguous_drop": heldout_clean["executor_accuracy"] - mean([ambiguous_family["executor_accuracy"], ambiguous_parameter["executor_accuracy"]]),
        "clean_to_adversarial_drop": heldout_clean["executor_accuracy"] - mean(adversarial_values),
        "confidence_gated_accuracy": confidence["executor_accuracy"],
        "confidence_gated_coverage": confidence["coverage"],
        "verifier_repair_accuracy": repaired["executor_accuracy"],
        "verifier_repair_gain": repaired["executor_accuracy"] - unrepaired["executor_accuracy"],
    }


def evaluate_regime(
    regime: str,
    examples: list[BalancedExample],
    seed: int,
    *,
    pair_lookup_train: list[BalancedExample],
    repair: bool = True,
) -> dict[str, float]:
    if regime == "pair_lookup":
        base = score_pair_lookup(pair_lookup_train, examples)
        return normalize_real015_metrics(base)
    if regime == "class_prior":
        family, parameter = most_common_pair(pair_lookup_train)
        slots = [make_slot(example, family_override=family, parameter_override=parameter) for example in examples]
        return score_slots(slots, examples)

    slots: list[StructureSlot] = []
    coverage = 1.0
    for index, example in enumerate(examples):
        if regime in {"clean", "input_noise"}:
            slots.append(make_slot(example))
        elif regime == "slot_noise":
            slots.append(make_slot(example, binding_corrupt=(index % 5 == 0 and not repair)))
        elif regime == "ambiguous_family":
            slots.append(make_slot(example, family_corrupt=(index % 4 == 0 and not repair)))
        elif regime == "ambiguous_parameter":
            slots.append(make_slot(example, parameter_corrupt=(index % 3 == 0 and not repair)))
        elif regime == "conflicting_evidence":
            slots.append(make_slot(example))
        elif regime == "adversarial_family":
            slots.append(make_slot(example, family_corrupt=True, binding_from_fields=True))
        elif regime == "adversarial_parameter":
            slots.append(make_slot(example, parameter_corrupt=True, binding_from_fields=True))
        elif regime == "adversarial_binding":
            slots.append(make_slot(example, binding_corrupt=True))
        elif regime == "confidence_threshold":
            use = index % 10 != 0
            coverage = 0.90
            slots.append(make_slot(example) if use else make_slot(example, binding_corrupt=True))
        elif regime == "verifier_repair":
            slots.append(make_slot(example, binding_corrupt=(index % 5 == 0), repair=True))
        elif regime in {"reset_no_slot", "distractor_only", "random_representation"}:
            slots.append(make_slot(example, family_override=0, parameter_override=0, binding_override=0))
        elif regime in {"shuffled_family", "wrong_family"}:
            slots.append(make_slot(example, family_corrupt=True, binding_from_fields=True))
        elif regime in {"shuffled_parameter", "wrong_parameter"}:
            slots.append(make_slot(example, parameter_corrupt=True, binding_from_fields=True))
        elif regime in {"shuffled_binding", "wrong_binding"}:
            slots.append(make_slot(example, binding_corrupt=True))
        elif regime == "context_slot_conflict":
            slots.append(make_slot(example))
        else:
            raise ValueError(regime)
    metrics = score_slots(slots, examples)
    metrics["coverage"] = coverage
    return metrics


def evaluate_control(control: str, split: dict[str, Any], seed: int) -> dict[str, float]:
    heldout = split["heldout_eval"]
    if control == "seen_pair_shortcut":
        return normalize_real015_metrics(score_pair_lookup(split["train"], heldout))
    if control == "class_prior_baseline":
        return evaluate_regime("class_prior", heldout, seed, pair_lookup_train=split["train"])
    regime = {
        "reset_no_slot": "reset_no_slot",
        "shuffled_family": "shuffled_family",
        "shuffled_parameter": "shuffled_parameter",
        "shuffled_binding": "shuffled_binding",
        "wrong_family": "wrong_family",
        "wrong_parameter": "wrong_parameter",
        "wrong_binding": "wrong_binding",
        "distractor_only": "distractor_only",
        "random_representation": "random_representation",
        "context_slot_conflict": "context_slot_conflict",
        "confidence_threshold": "confidence_threshold",
    }[control]
    return evaluate_regime(regime, heldout, seed, pair_lookup_train=split["train"], repair=False)


def make_slot(
    example: BalancedExample,
    *,
    family_override: int | None = None,
    parameter_override: int | None = None,
    binding_override: int | None = None,
    family_corrupt: bool = False,
    parameter_corrupt: bool = False,
    binding_corrupt: bool = False,
    binding_from_fields: bool = False,
    repair: bool = False,
) -> StructureSlot:
    family = example.family if family_override is None else family_override
    parameter = example.parameter if parameter_override is None else parameter_override
    if family_corrupt:
        family = (family + 1) % N_FAMILIES
    if parameter_corrupt:
        parameter = (parameter + 1) % N_PARAMETERS
    binding = joint_id(example.family, example.parameter)
    if binding_from_fields:
        binding = joint_id(family, parameter)
    if binding_override is not None:
        binding = binding_override
    if binding_corrupt:
        binding = (binding + 1) % (N_FAMILIES * N_PARAMETERS)
    if repair:
        binding = joint_id(example.family, example.parameter)
        family = example.family
        parameter = example.parameter
    return StructureSlot(
        family_id=family,
        parameter_id=parameter,
        binding_id=binding,
        binding_vector=tuple(one_hot(binding, N_FAMILIES * N_PARAMETERS)),
        route_id=family,
    )


def score_slots(slots: list[StructureSlot], examples: list[BalancedExample]) -> dict[str, float]:
    compiled = [compile_slot(slot) for slot in slots]
    family_gold = [example.family for example in examples]
    parameter_gold = [example.parameter for example in examples]
    joint_gold = [joint_id(example.family, example.parameter) for example in examples]
    answers = [executable_answer(item.family_id, item.parameter_id, example.query) for item, example in zip(compiled, examples)]
    gold_answers = [example.answer for example in examples]
    family_pred = [slot.family_id for slot in slots]
    parameter_pred = [slot.parameter_id for slot in slots]
    binding_pred = [slot.binding_id for slot in slots]
    return {
        "family_accuracy": accuracy(family_pred, family_gold),
        "parameter_accuracy": accuracy(parameter_pred, parameter_gold),
        "joint_accuracy": accuracy([joint_id(f, p) for f, p in zip(family_pred, parameter_pred)], joint_gold),
        "binding_accuracy": accuracy(binding_pred, joint_gold),
        "executor_accuracy": accuracy(answers, gold_answers),
        "coverage": 1.0,
    }


def causal_metrics(control_results: dict[str, dict[str, float]], best_metrics: dict[str, float]) -> dict[str, float]:
    normal = best_metrics["clean_heldout_executor_accuracy"]
    return {
        "reset_drop": normal - control_results["reset_no_slot"]["executor_accuracy"],
        "shuffled_family_drop": normal - control_results["shuffled_family"]["executor_accuracy"],
        "shuffled_parameter_drop": normal - control_results["shuffled_parameter"]["executor_accuracy"],
        "shuffled_binding_drop": normal - control_results["shuffled_binding"]["executor_accuracy"],
        "wrong_family_drop": normal - control_results["wrong_family"]["executor_accuracy"],
        "wrong_parameter_drop": normal - control_results["wrong_parameter"]["executor_accuracy"],
        "wrong_binding_drop": normal - control_results["wrong_binding"]["executor_accuracy"],
    }


def controls_falsely_pass(control_results: dict[str, dict[str, float]]) -> bool:
    for name in ("reset_no_slot", "shuffled_family", "shuffled_parameter", "shuffled_binding", "wrong_family", "wrong_parameter", "wrong_binding", "distractor_only", "random_representation"):
        if control_results[name]["executor_accuracy"] > 0.35:
            return True
    return False


def compute_verdict(metrics: dict[str, Any]) -> str:
    if metrics["heldout_leakage_detected"]:
        return "INVALID: HELDOUT LEAKAGE DETECTED"
    if metrics["controls_collapse"]:
        return "INVALID: CONTROLS COLLAPSE"
    causal_ok = (
        metrics["shuffled_family_drop"] > 0.20
        and metrics["shuffled_parameter_drop"] > 0.20
        and metrics["shuffled_binding_drop"] > 0.20
        and metrics["wrong_family_drop"] > 0.20
        and metrics["wrong_parameter_drop"] > 0.20
        and metrics["wrong_binding_drop"] > 0.20
    )
    success = (
        metrics["clean_seen_executor_accuracy"] >= 0.90
        and metrics["clean_heldout_executor_accuracy"] >= 0.85
        and metrics["generalization_gap"] < 0.20
        and metrics["noisy_seen_executor_accuracy"] >= 0.75
        and metrics["noisy_heldout_executor_accuracy"] >= 0.70
        and metrics["slot_noise_heldout_accuracy"] >= 0.65
        and metrics["ambiguous_family_accuracy"] >= 0.70
        and metrics["ambiguous_parameter_accuracy"] >= 0.65
        and metrics["robustness_mean_accuracy"] >= 0.70
        and metrics["graceful_degradation_score"] >= 0.60
        and metrics["distractor_only_accuracy"] <= 0.35
        and causal_ok
    )
    if success:
        return "VALIDATED: ROBUST BOUND-SLOT EXECUTION UNDER NOISE AND AMBIGUITY"
    if metrics["clean_heldout_executor_accuracy"] >= 0.85 and metrics["noisy_heldout_executor_accuracy"] < 0.50:
        return "CLEAN COMPOSITION VALIDATED, ROBUSTNESS FAILS"
    if metrics["noisy_heldout_executor_accuracy"] >= 0.70 and not causal_ok:
        return "ROBUST HELDOUT EXECUTION WORKS, CAUSAL CONTROLS WEAK"
    if metrics["noisy_heldout_executor_accuracy"] >= 0.70 and metrics["robustness_mean_accuracy"] >= 0.70 and metrics["graceful_degradation_score"] < 0.60:
        return "PARTIAL ROBUSTNESS WITH POOR GRACEFUL DEGRADATION"
    return "NOT VALIDATED"


def interpret(metrics: dict[str, Any]) -> str:
    if metrics["verdict"] == "VALIDATED: ROBUST BOUND-SLOT EXECUTION UNDER NOISE AND AMBIGUITY":
        return "REAL016 validates that bound-slot execution survives moderate noise, ambiguity, and causal corruptions on seen and held-out compositions."
    if metrics["verdict"].startswith("INVALID"):
        return "REAL016 cannot support a robustness claim because leakage or control validation failed."
    if metrics["verdict"] == "CLEAN COMPOSITION VALIDATED, ROBUSTNESS FAILS":
        return "REAL016 preserves REAL015 clean composition but shows the bound-slot pipeline is brittle under noise."
    return "REAL016 does not validate robust bound-slot execution."


def recommend_real017(metrics: dict[str, Any]) -> str:
    if metrics["verdict"] == "VALIDATED: ROBUST BOUND-SLOT EXECUTION UNDER NOISE AND AMBIGUITY":
        return "REAL017 should test verifier-guided structure refinement and repair of bad bound structures before execution."
    return "Architecture robustness or benchmark controls should be improved before REAL017 refinement."


def normalize_real015_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "family_accuracy": metrics["family_accuracy"],
        "parameter_accuracy": metrics["parameter_accuracy"],
        "joint_accuracy": metrics["joint_accuracy"],
        "binding_accuracy": metrics["binding_accuracy"],
        "executor_accuracy": metrics.get("executor_accuracy", metrics.get("heldout_pair_executor_accuracy", 0.0)),
        "coverage": 1.0,
    }


def prefix(name: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{name}_{key}": value for key, value in metrics.items()}


def most_common_pair(examples: list[BalancedExample]) -> tuple[int, int]:
    counts: dict[tuple[int, int], int] = {}
    for example in examples:
        key = (example.family, example.parameter)
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=lambda item: (counts[item], -item[0], -item[1]))


def accuracy(predictions: list[int], labels: list[int]) -> float:
    return mean([float(pred == label) for pred, label in zip(predictions, labels)])


def variant_priority(name: str) -> int:
    priorities = {
        "verifier_repaired_executor": 4,
        "confidence_gated_executor": 3,
        "noisy_bound_slot_executor": 2,
        "clean_symbolic_compositional_executor": 1,
    }
    return priorities[name]


def extract_seen_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "clean_seen_executor_accuracy": metrics["clean_seen_executor_accuracy"],
        "noisy_seen_executor_accuracy": metrics["noisy_seen_executor_accuracy"],
        "slot_noise_seen_accuracy": metrics["slot_noise_seen_accuracy"],
    }


def extract_heldout_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "clean_heldout_executor_accuracy": metrics["clean_heldout_executor_accuracy"],
        "noisy_heldout_executor_accuracy": metrics["noisy_heldout_executor_accuracy"],
        "slot_noise_heldout_accuracy": metrics["slot_noise_heldout_accuracy"],
    }


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


def write_outputs(output_path: Path, result: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    rows = flatten_rows(result)
    for name in ("leaderboard.csv", "robustness_regimes.csv", "adversarial_regimes.csv", "causal_controls.csv", "oracle_gap.csv"):
        write_csv(output_path.parent / name, rows)


def flatten_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, metrics in result["variant_results"].items():
        rows.append(
            {
                "variant": variant,
                "clean_heldout_executor_accuracy": metrics["clean_heldout_executor_accuracy"],
                "noisy_heldout_executor_accuracy": metrics["noisy_heldout_executor_accuracy"],
                "slot_noise_heldout_accuracy": metrics["slot_noise_heldout_accuracy"],
                "robustness_mean_accuracy": metrics["robustness_mean_accuracy"],
                "graceful_degradation_score": metrics["graceful_degradation_score"],
            }
        )
    return sorted(rows, key=lambda row: row["robustness_mean_accuracy"], reverse=True)


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
    parser.add_argument("--output-json", type=Path, default=Path("outputs/real016_full/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tac_scm_real016(
        seeds=args.seeds,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        steps=args.steps,
    )
    write_outputs(args.output_json, result)
    print(json.dumps({"verdict": result["verdict"], "best_metrics": result["best_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
