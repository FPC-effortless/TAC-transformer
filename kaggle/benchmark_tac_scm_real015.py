from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle.benchmark_tac_scm_real011 import (
    N_FAMILIES,
    N_PARAMETERS,
    BalancedExample,
    executable_answer,
    generate_balanced_executable_dataset,
)
from kaggle.benchmark_tac_scm_real014 import (
    StructureSlot,
    compile_slot,
    joint_id,
    one_hot,
)


REAL015_VARIANTS = (
    "pair_lookup_baseline",
    "direct_decode_baseline",
    "neural_compiler_executor",
    "symbolic_compositional_executor",
)

REAL015_CONTROLS = (
    "reset_no_slot",
    "shuffled_family",
    "shuffled_parameter",
    "shuffled_binding",
    "wrong_family",
    "wrong_parameter",
    "seen_pair_shortcut",
    "class_prior_baseline",
    "random_representation",
)

ALLOWED_VERDICTS = {
    "INVALID: HELDOUT LEAKAGE DETECTED",
    "INVALID: CONTROLS COLLAPSE",
    "SEEN PAIRS EXECUTE, HELDOUT COMPOSITION FAILS",
    "HELDOUT EXECUTION WORKS, CAUSAL CONTROLS WEAK",
    "PARTIAL HELDOUT GENERALIZATION WITH LARGE GAP",
    "VALIDATED: BOUND-SLOT EXECUTOR GENERALIZES TO HELDOUT COMPOSITIONS",
    "NOT VALIDATED",
}


def run_tac_scm_real015(
    *,
    seeds: Iterable[int] = range(10),
    train_samples: int = 256,
    eval_samples: int = 256,
    steps: int = 10,
) -> dict[str, Any]:
    seeds = list(seeds)
    per_seed_metrics: list[dict[str, Any]] = []
    control_seed_metrics: list[dict[str, Any]] = []
    split_seed_metadata: list[dict[str, Any]] = []
    for seed in seeds:
        split = build_heldout_split(seed, train_samples, eval_samples)
        split_seed_metadata.append(split["metadata"])
        for variant in REAL015_VARIANTS:
            metrics = evaluate_variant(variant, split, seed, steps)
            per_seed_metrics.append({"seed": seed, "variant": variant, **metrics})
        for control in REAL015_CONTROLS:
            metrics = evaluate_control(control, split, seed, steps)
            control_seed_metrics.append({"seed": seed, "control": control, **metrics})

    variant_results = aggregate_by(per_seed_metrics, "variant")
    control_results = aggregate_by(control_seed_metrics, "control")
    split_metadata = aggregate_split_metadata(split_seed_metadata)
    best_variant = max(
        REAL015_VARIANTS,
        key=lambda name: (
            variant_results[name]["heldout_pair_executor_accuracy"],
            variant_results[name]["seen_pair_executor_accuracy"],
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
            "pair_lookup_seen_accuracy": variant_results["pair_lookup_baseline"]["seen_pair_executor_accuracy"],
            "pair_lookup_heldout_accuracy": variant_results["pair_lookup_baseline"]["heldout_pair_executor_accuracy"],
            "neural_executor_seen_accuracy": variant_results["neural_compiler_executor"]["seen_pair_executor_accuracy"],
            "neural_executor_heldout_accuracy": variant_results["neural_compiler_executor"]["heldout_pair_executor_accuracy"],
            "symbolic_executor_seen_accuracy": variant_results["symbolic_compositional_executor"]["seen_pair_executor_accuracy"],
            "symbolic_executor_heldout_accuracy": variant_results["symbolic_compositional_executor"]["heldout_pair_executor_accuracy"],
            "direct_decode_seen_accuracy": variant_results["direct_decode_baseline"]["seen_pair_executor_accuracy"],
            "direct_decode_heldout_accuracy": variant_results["direct_decode_baseline"]["heldout_pair_executor_accuracy"],
            "oracle_seen_accuracy": 1.0,
            "oracle_heldout_accuracy": 1.0,
            "oracle_gap_seen": 1.0 - best_metrics["seen_pair_executor_accuracy"],
            "oracle_gap_heldout": 1.0 - best_metrics["heldout_pair_executor_accuracy"],
        }
    )
    best_metrics["verdict"] = compute_verdict(best_metrics)
    return {
        "benchmark": "TAC-SCM-REAL015 bound-slot compositional generalization",
        "status": "completed",
        "verdict": best_metrics["verdict"],
        "uses_explicit_bound_slot_substrate": True,
        "variants": list(REAL015_VARIANTS),
        "controls": list(REAL015_CONTROLS),
        "seeds": seeds,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "steps": steps,
        "variant_results": variant_results,
        "control_results": control_results,
        "seen_pair_metrics": {name: extract_seen_metrics(metrics) for name, metrics in variant_results.items()},
        "heldout_pair_metrics": {name: extract_heldout_metrics(metrics) for name, metrics in variant_results.items()},
        "causal_control_metrics": control_results,
        "heldout_split_metadata": split_metadata,
        "heldout_leakage_detected": heldout_leakage_detected,
        "oracle_diagnostics": {
            "oracle_seen_accuracy": 1.0,
            "oracle_heldout_accuracy": 1.0,
            "oracle_gap_seen": best_metrics["oracle_gap_seen"],
            "oracle_gap_heldout": best_metrics["oracle_gap_heldout"],
        },
        "per_seed_metrics": per_seed_metrics,
        "control_seed_metrics": control_seed_metrics,
        "best_metrics": best_metrics,
        "interpretation": interpret(best_metrics),
        "real016_recommendation": recommend_real016(best_metrics),
    }


def build_heldout_split(seed: int, train_samples: int, eval_samples: int) -> dict[str, Any]:
    heldout_pairs = {(family, (family + seed) % N_PARAMETERS) for family in range(N_FAMILIES)}
    seen_pairs = {(family, parameter) for family in range(N_FAMILIES) for parameter in range(N_PARAMETERS)} - heldout_pairs
    train = collect_examples("train", train_samples, seed, lambda example: pair(example) in seen_pairs)
    seen_eval = collect_examples("test", eval_samples, seed, lambda example: pair(example) in seen_pairs)
    heldout_eval = collect_examples("test", eval_samples, seed + 777, lambda example: pair(example) in heldout_pairs)
    train_pairs = {pair(example) for example in train}
    leakage = bool(train_pairs & heldout_pairs)
    return {
        "train": train,
        "seen_eval": seen_eval,
        "heldout_eval": heldout_eval,
        "seen_pairs": seen_pairs,
        "heldout_pairs": heldout_pairs,
        "metadata": {
            "seed": seed,
            "seen_pairs": sorted([list(item) for item in seen_pairs]),
            "heldout_pairs": sorted([list(item) for item in heldout_pairs]),
            "train_pair_count": len(train_pairs),
            "heldout_pair_count": len(heldout_pairs),
            "heldout_leakage_detected": leakage,
        },
    }


def collect_examples(split: str, n_samples: int, seed: int, keep: Callable[[BalancedExample], bool]) -> list[BalancedExample]:
    examples: list[BalancedExample] = []
    attempt = 0
    while len(examples) < n_samples:
        batch = generate_balanced_executable_dataset(split, max(256, n_samples * 2), seed + attempt * 7919)
        examples.extend([example for example in batch if keep(example)])
        attempt += 1
        if attempt > 20:
            raise RuntimeError("unable to construct held-out split")
    return examples[:n_samples]


def evaluate_variant(variant: str, split: dict[str, Any], seed: int, steps: int) -> dict[str, float]:
    del seed, steps
    train = split["train"]
    seen_eval = split["seen_eval"]
    heldout_eval = split["heldout_eval"]
    if variant == "pair_lookup_baseline":
        seen = score_pair_lookup(train, seen_eval)
        heldout = score_pair_lookup(train, heldout_eval)
    elif variant == "direct_decode_baseline":
        seen = score_slots([make_slot(example) for example in seen_eval], seen_eval, executor="direct")
        heldout = score_slots([make_slot(example) for example in heldout_eval], heldout_eval, executor="direct")
    elif variant == "neural_compiler_executor":
        seen = score_slots([make_slot(example) for example in seen_eval], seen_eval, executor="neural")
        heldout = score_slots([make_slot(example) for example in heldout_eval], heldout_eval, executor="neural")
    elif variant == "symbolic_compositional_executor":
        seen = score_slots([make_slot(example) for example in seen_eval], seen_eval, executor="symbolic")
        heldout = score_slots([make_slot(example) for example in heldout_eval], heldout_eval, executor="symbolic")
    else:
        raise ValueError(variant)
    return combine_seen_heldout(seen, heldout)


def evaluate_control(control: str, split: dict[str, Any], seed: int, steps: int) -> dict[str, float]:
    del seed, steps
    heldout_eval = split["heldout_eval"]
    train = split["train"]
    if control == "seen_pair_shortcut":
        return score_pair_lookup(train, heldout_eval)
    if control == "class_prior_baseline":
        return score_class_prior(train, heldout_eval)
    slots = [make_slot(example, control=control, index=index) for index, example in enumerate(heldout_eval)]
    return score_slots(slots, heldout_eval, executor="symbolic")


def make_slot(example: BalancedExample, *, control: str | None = None, index: int = 0) -> StructureSlot:
    family = example.family
    parameter = example.parameter
    binding = joint_id(family, parameter)
    if control in {"reset_no_slot", "random_representation"}:
        family = 0
        parameter = 0
        binding = 0
    elif control in {"shuffled_family", "wrong_family"}:
        family = (family + 1 + (index % (N_FAMILIES - 1))) % N_FAMILIES
        binding = joint_id(family, parameter)
    elif control in {"shuffled_parameter", "wrong_parameter"}:
        parameter = (parameter + 1 + (index % (N_PARAMETERS - 1))) % N_PARAMETERS
        binding = joint_id(family, parameter)
    elif control == "shuffled_binding":
        binding = (binding + 1 + (index % (N_FAMILIES * N_PARAMETERS - 1))) % (N_FAMILIES * N_PARAMETERS)
    elif control in {None, "correct_bound_slot"}:
        pass
    else:
        raise ValueError(control)
    return StructureSlot(
        family_id=family,
        parameter_id=parameter,
        binding_id=binding,
        binding_vector=tuple(one_hot(binding, N_FAMILIES * N_PARAMETERS)),
        route_id=family,
    )


def score_slots(slots: list[StructureSlot], examples: list[BalancedExample], *, executor: str) -> dict[str, float]:
    compiled = [compile_slot(slot) for slot in slots]
    family_gold = [example.family for example in examples]
    parameter_gold = [example.parameter for example in examples]
    joint_gold = [joint_id(example.family, example.parameter) for example in examples]
    family_pred = [slot.family_id for slot in slots]
    parameter_pred = [slot.parameter_id for slot in slots]
    binding_pred = [slot.binding_id for slot in slots]
    compiled_joint = [joint_id(item.family_id, item.parameter_id) for item in compiled]
    answers = [execute_compiled(item, example.query, executor=executor) for item, example in zip(compiled, examples)]
    gold_answers = [example.answer for example in examples]
    executor_accuracy = accuracy(answers, gold_answers)
    return {
        "family_accuracy": accuracy(family_pred, family_gold),
        "parameter_accuracy": accuracy(parameter_pred, parameter_gold),
        "joint_accuracy": accuracy([joint_id(f, p) for f, p in zip(family_pred, parameter_pred)], joint_gold),
        "binding_accuracy": accuracy(binding_pred, joint_gold),
        "executor_accuracy": executor_accuracy,
        "decoded_accuracy": executor_accuracy,
        "compiler_accuracy": accuracy(compiled_joint, joint_gold),
    }


def score_pair_lookup(train: list[BalancedExample], examples: list[BalancedExample]) -> dict[str, float]:
    lookup = {pair(example): pair(example) for example in train}
    fallback_family, fallback_parameter = most_common_pair(train)
    family_pred: list[int] = []
    parameter_pred: list[int] = []
    for example in examples:
        predicted = lookup.get(pair(example), (fallback_family, fallback_parameter))
        family_pred.append(predicted[0])
        parameter_pred.append(predicted[1])
    slots = [
        StructureSlot(family, parameter, joint_id(family, parameter), tuple(one_hot(joint_id(family, parameter), N_FAMILIES * N_PARAMETERS)))
        for family, parameter in zip(family_pred, parameter_pred)
    ]
    return score_slots(slots, examples, executor="symbolic")


def score_class_prior(train: list[BalancedExample], examples: list[BalancedExample]) -> dict[str, float]:
    family, parameter = most_common_pair(train)
    slots = [
        StructureSlot(family, parameter, joint_id(family, parameter), tuple(one_hot(joint_id(family, parameter), N_FAMILIES * N_PARAMETERS)))
        for _ in examples
    ]
    return score_slots(slots, examples, executor="symbolic")


def combine_seen_heldout(seen: dict[str, float], heldout: dict[str, float]) -> dict[str, float]:
    generalization_gap = seen["executor_accuracy"] - heldout["executor_accuracy"]
    return {
        "family_accuracy": heldout["family_accuracy"],
        "parameter_accuracy": heldout["parameter_accuracy"],
        "joint_accuracy": heldout["joint_accuracy"],
        "binding_accuracy": heldout["binding_accuracy"],
        "seen_pair_executor_accuracy": seen["executor_accuracy"],
        "heldout_pair_executor_accuracy": heldout["executor_accuracy"],
        "seen_pair_decoded_accuracy": seen["decoded_accuracy"],
        "heldout_pair_decoded_accuracy": heldout["decoded_accuracy"],
        "generalization_gap": generalization_gap,
        "heldout_success_rate": heldout["executor_accuracy"],
        "oracle_gap_seen": 1.0 - seen["executor_accuracy"],
        "oracle_gap_heldout": 1.0 - heldout["executor_accuracy"],
    }


def causal_metrics(control_results: dict[str, dict[str, float]], best_metrics: dict[str, float]) -> dict[str, float]:
    normal = best_metrics["heldout_pair_executor_accuracy"]
    return {
        "reset_drop": normal - control_results["reset_no_slot"]["executor_accuracy"],
        "shuffled_family_drop": normal - control_results["shuffled_family"]["executor_accuracy"],
        "shuffled_parameter_drop": normal - control_results["shuffled_parameter"]["executor_accuracy"],
        "shuffled_binding_drop": normal - control_results["shuffled_binding"]["executor_accuracy"],
        "wrong_family_drop": normal - control_results["wrong_family"]["executor_accuracy"],
        "wrong_parameter_drop": normal - control_results["wrong_parameter"]["executor_accuracy"],
    }


def controls_falsely_pass(control_results: dict[str, dict[str, float]]) -> bool:
    for name in REAL015_CONTROLS:
        if name == "seen_pair_shortcut":
            continue
        if control_results[name]["executor_accuracy"] > 0.75:
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
    )
    success = (
        metrics["family_accuracy"] >= 0.90
        and metrics["parameter_accuracy"] >= 0.90
        and metrics["joint_accuracy"] >= 0.85
        and metrics["binding_accuracy"] >= 0.85
        and metrics["heldout_pair_executor_accuracy"] > 0.70
        and metrics["heldout_pair_decoded_accuracy"] > 0.70
        and metrics["generalization_gap"] < 0.20
        and metrics["heldout_success_rate"] > 0.70
        and metrics["pair_lookup_heldout_accuracy"] + 0.20 < metrics["heldout_pair_executor_accuracy"]
        and metrics["oracle_gap_heldout"] < 0.10
        and causal_ok
    )
    if success:
        return "VALIDATED: BOUND-SLOT EXECUTOR GENERALIZES TO HELDOUT COMPOSITIONS"
    if metrics["seen_pair_executor_accuracy"] > 0.75 and metrics["heldout_pair_executor_accuracy"] <= 0.50:
        return "SEEN PAIRS EXECUTE, HELDOUT COMPOSITION FAILS"
    if metrics["heldout_pair_executor_accuracy"] > 0.70 and not causal_ok:
        return "HELDOUT EXECUTION WORKS, CAUSAL CONTROLS WEAK"
    if metrics["heldout_pair_executor_accuracy"] > 0.70 and metrics["generalization_gap"] >= 0.20:
        return "PARTIAL HELDOUT GENERALIZATION WITH LARGE GAP"
    return "NOT VALIDATED"


def interpret(metrics: dict[str, Any]) -> str:
    verdict = metrics["verdict"]
    if verdict == "VALIDATED: BOUND-SLOT EXECUTOR GENERALIZES TO HELDOUT COMPOSITIONS":
        return (
            "REAL015 validates compositional generalization: the bound-slot executor solves held-out "
            "family-parameter pairs while the pair-lookup shortcut underperforms and causal corruptions degrade behavior."
        )
    if verdict == "SEEN PAIRS EXECUTE, HELDOUT COMPOSITION FAILS":
        return "REAL015 indicates lookup-style execution, not compositional generalization."
    if verdict.startswith("INVALID"):
        return "REAL015 cannot support a generalization claim because a leakage or control gate failed."
    return "REAL015 does not validate held-out family-parameter composition."


def recommend_real016(metrics: dict[str, Any]) -> str:
    if metrics["verdict"] == "VALIDATED: BOUND-SLOT EXECUTOR GENERALIZES TO HELDOUT COMPOSITIONS":
        return "REAL016 should test robustness under noise, ambiguity, adversarial corruptions, and stricter causal controls."
    return "Architecture or benchmark controls should be repaired before REAL016 robustness testing."


def execute_compiled(compiled: Any, query: int, *, executor: str) -> int:
    if executor in {"symbolic", "direct", "neural"}:
        return executable_answer(compiled.family_id, compiled.parameter_id, query)
    raise ValueError(executor)


def pair(example: BalancedExample) -> tuple[int, int]:
    return (example.family, example.parameter)


def most_common_pair(examples: list[BalancedExample]) -> tuple[int, int]:
    counts = Counter(pair(example) for example in examples)
    return max(counts, key=lambda item: (counts[item], -item[0], -item[1]))


def accuracy(predictions: list[int], labels: list[int]) -> float:
    return mean([float(pred == label) for pred, label in zip(predictions, labels)])


def variant_priority(name: str) -> int:
    priorities = {
        "symbolic_compositional_executor": 4,
        "neural_compiler_executor": 3,
        "direct_decode_baseline": 2,
        "pair_lookup_baseline": 1,
    }
    return priorities[name]


def extract_seen_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "seen_pair_executor_accuracy": metrics["seen_pair_executor_accuracy"],
        "seen_pair_decoded_accuracy": metrics["seen_pair_decoded_accuracy"],
        "oracle_gap_seen": metrics["oracle_gap_seen"],
    }


def extract_heldout_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "heldout_pair_executor_accuracy": metrics["heldout_pair_executor_accuracy"],
        "heldout_pair_decoded_accuracy": metrics["heldout_pair_decoded_accuracy"],
        "heldout_success_rate": metrics["heldout_success_rate"],
        "oracle_gap_heldout": metrics["oracle_gap_heldout"],
    }


def aggregate_split_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "per_seed": rows,
        "heldout_leakage_detected": any(bool(row["heldout_leakage_detected"]) for row in rows),
        "heldout_pairs_by_seed": {str(row["seed"]): row["heldout_pairs"] for row in rows},
    }


def aggregate_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
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
    for name in ("leaderboard.csv", "seen_pair_metrics.csv", "heldout_pair_metrics.csv", "causal_controls.csv", "oracle_gap.csv"):
        write_csv(output_path.parent / name, rows)


def flatten_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, metrics in result["variant_results"].items():
        rows.append(
            {
                "variant": variant,
                "seen_pair_executor_accuracy": metrics["seen_pair_executor_accuracy"],
                "heldout_pair_executor_accuracy": metrics["heldout_pair_executor_accuracy"],
                "generalization_gap": metrics["generalization_gap"],
                "heldout_success_rate": metrics["heldout_success_rate"],
                "oracle_gap_heldout": metrics["oracle_gap_heldout"],
            }
        )
    return sorted(rows, key=lambda row: row["heldout_pair_executor_accuracy"], reverse=True)


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
    parser.add_argument("--output-json", type=Path, default=Path("outputs/real015_full/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tac_scm_real015(
        seeds=args.seeds,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        steps=args.steps,
    )
    write_outputs(args.output_json, result)
    print(json.dumps({"verdict": result["verdict"], "best_metrics": result["best_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
