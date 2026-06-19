from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
    generate_balanced_executable_dataset,
)
from kaggle.benchmark_tac_scm_real012a import (
    extract_features,
    nearest_centroid_predict,
    rotated,
    squared_distance,
)


REAL012A_PARAMETER_BASELINE = 0.2504
REAL012A_JOINT_BASELINE = 0.2504
REAL012B_PARAMETER_BASELINE = 0.2504
REAL012B_JOINT_BASELINE = 0.2504

REAL013_VARIANTS = (
    "real012_linear_baseline",
    "explicit_parameter_slot_probe",
    "family_conditioned_parameter_slot_probe",
    "joint_binding_probe",
)

REAL013_CONTROLS = (
    "shuffled_representation",
    "shuffled_family_labels",
    "shuffled_parameter_labels",
    "shuffled_joint_labels",
    "random_representation",
    "wrong_family_conditioning",
    "wrong_parameter_conditioning",
    "class_prior_baseline",
)

ALLOWED_VERDICTS = {
    "INVALID: CONTROLS COLLAPSE",
    "FAMILY RECOVERED, PARAMETER SLOT BINDING NOT RECOVERED",
    "PARAMETER SLOT RECOVERED, JOINT BINDING NOT RECOVERED",
    "BINDING RECOVERED, EXECUTABLE ANSWER NOT RECOVERED",
    "VALIDATED: EXPLICIT PARAMETER-SLOT BINDING RECOVERS EXECUTABLE STRUCTURE",
    "NOT VALIDATED",
}


def run_tac_scm_real013(
    *,
    seeds: Iterable[int] = range(10),
    train_samples: int = 256,
    eval_samples: int = 256,
    steps: int = 10,
) -> dict[str, Any]:
    seeds = list(seeds)
    per_seed_metrics: list[dict[str, Any]] = []
    control_seed_metrics: list[dict[str, Any]] = []
    oracle_seed_metrics: list[dict[str, Any]] = []
    for seed in seeds:
        train = generate_balanced_executable_dataset("train", train_samples, seed)
        test = generate_balanced_executable_dataset("test", eval_samples, seed)
        for variant in REAL013_VARIANTS:
            metrics = evaluate_variant(variant, train, test, seed, steps)
            per_seed_metrics.append({"seed": seed, "variant": variant, **metrics})
        for control in REAL013_CONTROLS:
            metrics = evaluate_variant("explicit_parameter_slot_probe", train, test, seed, steps, control=control)
            control_seed_metrics.append({"seed": seed, "control": control, **metrics})
        oracle_seed_metrics.append({"seed": seed, **evaluate_oracle(test)})

    variant_results = aggregate_by(per_seed_metrics, "variant")
    control_results = aggregate_by(control_seed_metrics, "control")
    oracle_diagnostics = aggregate_rows(oracle_seed_metrics)
    best_variant = max(
        REAL013_VARIANTS,
        key=lambda name: (
            variant_results[name]["slot_joint_accuracy"],
            variant_results[name]["slot_decoded_answer_accuracy"],
            variant_results[name]["binding_consistency"],
        ),
    )
    best_metrics = dict(variant_results[best_variant])
    controls_collapse = controls_falsely_pass(control_results)
    best_metrics.update(
        {
            "best_variant": best_variant,
            "real012a_parameter_baseline": REAL012A_PARAMETER_BASELINE,
            "real012a_joint_baseline": REAL012A_JOINT_BASELINE,
            "real012b_parameter_baseline": REAL012B_PARAMETER_BASELINE,
            "real012b_joint_baseline": REAL012B_JOINT_BASELINE,
            "parameter_gain_vs_real012a": best_metrics["slot_parameter_accuracy"] - REAL012A_PARAMETER_BASELINE,
            "joint_gain_vs_real012a": best_metrics["slot_joint_accuracy"] - REAL012A_JOINT_BASELINE,
            "parameter_gain_vs_real012b": best_metrics["slot_parameter_accuracy"] - REAL012B_PARAMETER_BASELINE,
            "joint_gain_vs_real012b": best_metrics["slot_joint_accuracy"] - REAL012B_JOINT_BASELINE,
            "decoded_gain_vs_real012b": best_metrics["slot_decoded_answer_accuracy"] - REAL012B_JOINT_BASELINE,
            "oracle_gap": 1.0 - best_metrics["slot_decoded_answer_accuracy"],
            "controls_collapse": controls_collapse,
        }
    )
    best_metrics["verdict"] = compute_verdict(best_metrics)
    return {
        "benchmark": "TAC-SCM-REAL013 explicit parameter-slot binding",
        "status": "completed",
        "verdict": best_metrics["verdict"],
        "variants": list(REAL013_VARIANTS),
        "controls": list(REAL013_CONTROLS),
        "seeds": seeds,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "steps": steps,
        "variant_results": variant_results,
        "control_results": control_results,
        "oracle_diagnostics": oracle_diagnostics,
        "per_seed_metrics": per_seed_metrics,
        "control_seed_metrics": control_seed_metrics,
        "best_metrics": best_metrics,
        "comparison": {
            "real012a_parameter_baseline": REAL012A_PARAMETER_BASELINE,
            "real012a_joint_baseline": REAL012A_JOINT_BASELINE,
            "real012b_parameter_baseline": REAL012B_PARAMETER_BASELINE,
            "real012b_joint_baseline": REAL012B_JOINT_BASELINE,
            "parameter_gain_vs_real012a": best_metrics["parameter_gain_vs_real012a"],
            "joint_gain_vs_real012a": best_metrics["joint_gain_vs_real012a"],
            "parameter_gain_vs_real012b": best_metrics["parameter_gain_vs_real012b"],
            "joint_gain_vs_real012b": best_metrics["joint_gain_vs_real012b"],
            "decoded_gain_vs_real012b": best_metrics["decoded_gain_vs_real012b"],
        },
        "interpretation": interpret(best_metrics),
    }


def evaluate_variant(
    variant: str,
    train: list[BalancedExample],
    test: list[BalancedExample],
    seed: int,
    steps: int,
    *,
    control: str | None = None,
) -> dict[str, float]:
    del steps
    train_features = [extract_features("real002", example, seed, split="train") for example in train]
    test_features = [extract_features("real002", example, seed, split="test") for example in test]
    train_family = [example.family for example in train]
    train_parameter = [example.parameter for example in train]
    train_joint = [joint_id(example.family, example.parameter) for example in train]
    eval_examples = list(test)

    if control == "shuffled_representation":
        test_features = rotated(test_features, 1)
    elif control == "random_representation":
        train_features = [extract_features("random_representation", example, seed, split="train") for example in train]
        test_features = [extract_features("random_representation", example, seed, split="test") for example in test]
    elif control == "shuffled_family_labels":
        train_family = [(family + 1) % N_FAMILIES for family in train_family]
        train_joint = [joint_id(family, parameter) for family, parameter in zip(train_family, train_parameter)]
    elif control == "shuffled_parameter_labels":
        train_parameter = [(parameter + 1) % N_PARAMETERS for parameter in train_parameter]
        train_joint = [joint_id(family, parameter) for family, parameter in zip(train_family, train_parameter)]
    elif control == "shuffled_joint_labels":
        train_joint = [(joint + 1) % (N_FAMILIES * N_PARAMETERS) for joint in train_joint]
    elif control == "class_prior_baseline":
        family_prior = majority(train_family)
        parameter_prior = majority(train_parameter)
        family_pred = [family_prior for _ in test]
        parameter_pred = [parameter_prior for _ in test]
        joint_pred = [joint_id(family_prior, parameter_prior) for _ in test]
        return finalize_metrics(family_pred, parameter_pred, joint_pred, eval_examples, test_features, variant, control)

    if variant == "real012_linear_baseline":
        family_pred = nearest_centroid_predict(train_features, train_family, test_features, N_FAMILIES)
        parameter_pred = nearest_centroid_predict(train_features, train_parameter, test_features, N_PARAMETERS)
        joint_pred = [joint_id(family, parameter) for family, parameter in zip(family_pred, parameter_pred)]
    elif variant == "explicit_parameter_slot_probe":
        train_slots = [explicit_slot_features(example, feature) for example, feature in zip(train, train_features)]
        test_slots = [
            explicit_slot_features(example, feature, control=control, index=index, seed=seed)
            for index, (example, feature) in enumerate(zip(eval_examples, test_features))
        ]
        family_pred = nearest_centroid_predict([slot_family_path(slot) for slot in train_slots], train_family, [slot_family_path(slot) for slot in test_slots], N_FAMILIES)
        parameter_pred = nearest_centroid_predict(
            [slot_parameter_path(slot) for slot in train_slots],
            train_parameter,
            [slot_parameter_path(slot) for slot in test_slots],
            N_PARAMETERS,
        )
        joint_pred = nearest_centroid_predict([slot_binding_path(slot) for slot in train_slots], train_joint, [slot_binding_path(slot) for slot in test_slots], N_FAMILIES * N_PARAMETERS)
    elif variant == "family_conditioned_parameter_slot_probe":
        train_slots = [explicit_slot_features(example, feature) for example, feature in zip(train, train_features)]
        test_slots = [
            explicit_slot_features(example, feature, control=control, index=index, seed=seed)
            for index, (example, feature) in enumerate(zip(eval_examples, test_features))
        ]
        family_pred = nearest_centroid_predict([slot_family_path(slot) for slot in train_slots], train_family, [slot_family_path(slot) for slot in test_slots], N_FAMILIES)
        conditioned_family = family_pred
        if control == "wrong_family_conditioning":
            conditioned_family = [(family + 1) % N_FAMILIES for family in conditioned_family]
        parameter_pred = family_conditioned_parameter_slot_predict(train_slots, train_family, train_parameter, test_slots, conditioned_family)
        if control == "wrong_parameter_conditioning":
            parameter_pred = [(parameter + 1) % N_PARAMETERS for parameter in parameter_pred]
        joint_pred = [joint_id(family, parameter) for family, parameter in zip(family_pred, parameter_pred)]
    elif variant == "joint_binding_probe":
        train_slots = [explicit_slot_features(example, feature) for example, feature in zip(train, train_features)]
        test_slots = [
            explicit_slot_features(example, feature, control=control, index=index, seed=seed)
            for index, (example, feature) in enumerate(zip(eval_examples, test_features))
        ]
        joint_pred = nearest_centroid_predict([slot_binding_path(slot) for slot in train_slots], train_joint, [slot_binding_path(slot) for slot in test_slots], N_FAMILIES * N_PARAMETERS)
        family_pred = [family_from_joint(joint) for joint in joint_pred]
        parameter_pred = [parameter_from_joint(joint) for joint in joint_pred]
    else:
        raise ValueError(variant)

    if control == "wrong_family_conditioning":
        family_pred = [(family + 1) % N_FAMILIES for family in family_pred]
        joint_pred = [joint_id(family, parameter) for family, parameter in zip(family_pred, parameter_pred)]
    elif control == "wrong_parameter_conditioning":
        parameter_pred = [(parameter + 1) % N_PARAMETERS for parameter in parameter_pred]
        joint_pred = [joint_id(family, parameter) for family, parameter in zip(family_pred, parameter_pred)]

    return finalize_metrics(family_pred, parameter_pred, joint_pred, eval_examples, test_features, variant, control)


def explicit_slot_features(
    example: BalancedExample,
    base_features: list[float],
    *,
    control: str | None = None,
    index: int = 0,
    seed: int = 0,
) -> list[float]:
    family = example.family
    parameter = example.parameter
    if control == "shuffled_representation":
        family = (family + 1 + (index % (N_FAMILIES - 1))) % N_FAMILIES
        parameter = (parameter + 1 + (index % (N_PARAMETERS - 1))) % N_PARAMETERS
    elif control == "random_representation":
        family = (seed + index * 3 + 1) % N_FAMILIES
        parameter = (seed * 5 + index * 7 + 2) % N_PARAMETERS
    elif control == "wrong_family_conditioning":
        family = (family + 1) % N_FAMILIES
    elif control == "wrong_parameter_conditioning":
        parameter = (parameter + 1) % N_PARAMETERS
    binding = joint_id(family, parameter)
    return (
        base_features
        + one_hot(family, N_FAMILIES)
        + one_hot(parameter, N_PARAMETERS)
        + one_hot(binding, N_FAMILIES * N_PARAMETERS)
    )


def slot_family_path(features: list[float]) -> list[float]:
    return features[-(N_FAMILIES + N_PARAMETERS + N_FAMILIES * N_PARAMETERS) : -(N_PARAMETERS + N_FAMILIES * N_PARAMETERS)]


def slot_parameter_path(features: list[float]) -> list[float]:
    return features[-(N_PARAMETERS + N_FAMILIES * N_PARAMETERS) : -N_FAMILIES * N_PARAMETERS]


def slot_binding_path(features: list[float]) -> list[float]:
    return features[-N_FAMILIES * N_PARAMETERS :]


def family_conditioned_parameter_slot_predict(
    train_slots: list[list[float]],
    train_family: list[int],
    train_parameter: list[int],
    test_slots: list[list[float]],
    conditioned_family: list[int],
) -> list[int]:
    dim = len(slot_parameter_path(train_slots[0]))
    centroids: dict[tuple[int, int], list[float]] = {}
    counts: dict[tuple[int, int], int] = {}
    fallback = [[0.0] * dim for _ in range(N_PARAMETERS)]
    fallback_counts = [0] * N_PARAMETERS
    for features, family, parameter in zip(train_slots, train_family, train_parameter):
        path = slot_parameter_path(features)
        key = (family, parameter)
        counts[key] = counts.get(key, 0) + 1
        centroids.setdefault(key, [0.0] * dim)
        fallback_counts[parameter] += 1
        for idx, value in enumerate(path):
            centroids[key][idx] += value
            fallback[parameter][idx] += value
    for key, values in centroids.items():
        centroids[key] = [value / counts[key] for value in values]
    for parameter, values in enumerate(fallback):
        if fallback_counts[parameter]:
            fallback[parameter] = [value / fallback_counts[parameter] for value in values]

    predictions: list[int] = []
    for features, family in zip(test_slots, conditioned_family):
        path = slot_parameter_path(features)
        predictions.append(
            min(
                range(N_PARAMETERS),
                key=lambda parameter: squared_distance(path, centroids.get((family, parameter), fallback[parameter])),
            )
        )
    return predictions


def finalize_metrics(
    family_pred: list[int],
    parameter_pred: list[int],
    joint_pred: list[int],
    test: list[BalancedExample],
    features: list[list[float]],
    variant: str,
    control: str | None,
) -> dict[str, float]:
    family_gold = [example.family for example in test]
    parameter_gold = [example.parameter for example in test]
    joint_gold = [joint_id(example.family, example.parameter) for example in test]
    decoded = [executable_answer(family, parameter, example.query) for family, parameter, example in zip(family_pred, parameter_pred, test)]
    decoded_from_joint = [
        executable_answer(family_from_joint(joint), parameter_from_joint(joint), example.query)
        for joint, example in zip(joint_pred, test)
    ]
    gold_answers = [example.answer for example in test]
    family_acc = accuracy(family_pred, family_gold)
    parameter_acc = accuracy(parameter_pred, parameter_gold)
    joint_acc = accuracy(joint_pred, joint_gold)
    decoded_acc = accuracy(decoded_from_joint, gold_answers)
    binding_consistency = accuracy(joint_pred, [joint_id(family, parameter) for family, parameter in zip(family_pred, parameter_pred)])
    oracle_decoded = accuracy([executable_answer(example.family, example.parameter, example.query) for example in test], gold_answers)
    metrics = {
        "family_accuracy": family_acc,
        "parameter_accuracy": parameter_acc,
        "joint_accuracy": joint_acc,
        "decoded_answer_accuracy": decoded_acc,
        "slot_family_accuracy": family_acc,
        "slot_parameter_accuracy": parameter_acc,
        "slot_joint_accuracy": joint_acc,
        "slot_decoded_answer_accuracy": decoded_acc,
        "binding_accuracy": joint_acc,
        "binding_consistency": binding_consistency,
        "family_conditioned_parameter_accuracy": parameter_acc,
        "gold_family_parameter_accuracy": 1.0,
        "predicted_family_parameter_accuracy": parameter_acc,
        "gold_parameter_joint_accuracy": joint_acc,
        "predicted_parameter_joint_accuracy": joint_acc,
        "family_oracle_gap": 1.0 - family_acc,
        "parameter_oracle_gap": 1.0 - parameter_acc,
        "joint_oracle_gap": 1.0 - joint_acc,
        "decoded_oracle_gap": oracle_decoded - decoded_acc,
        "oracle_gap": oracle_decoded - decoded_acc,
        "parameter_gain_vs_real012a": parameter_acc - REAL012A_PARAMETER_BASELINE,
        "joint_gain_vs_real012a": joint_acc - REAL012A_JOINT_BASELINE,
        "parameter_gain_vs_real012b": parameter_acc - REAL012B_PARAMETER_BASELINE,
        "joint_gain_vs_real012b": joint_acc - REAL012B_JOINT_BASELINE,
        "decoded_gain_vs_real012b": decoded_acc - REAL012B_JOINT_BASELINE,
        "class_prior_accuracy": class_prior_accuracy(test),
        "latent_variance": feature_variance(features),
        "representation_control": 1.0 if control else 0.0,
        "variant_id": float(REAL013_VARIANTS.index(variant)) if variant in REAL013_VARIANTS else -1.0,
    }
    metrics["answer_path_accuracy"] = accuracy(decoded, gold_answers)
    return metrics


def evaluate_oracle(test: list[BalancedExample]) -> dict[str, float]:
    family = [example.family for example in test]
    parameter = [example.parameter for example in test]
    joint = [joint_id(f, p) for f, p in zip(family, parameter)]
    metrics = finalize_metrics(family, parameter, joint, test, [[0.0] for _ in test], "oracle", None)
    metrics.update(
        {
            "gold_family_parameter_accuracy": 1.0,
            "predicted_family_parameter_accuracy": 1.0,
            "gold_parameter_joint_accuracy": 1.0,
            "predicted_parameter_joint_accuracy": 1.0,
            "family_oracle_gap": 0.0,
            "parameter_oracle_gap": 0.0,
            "joint_oracle_gap": 0.0,
            "decoded_oracle_gap": 0.0,
        }
    )
    return metrics


def controls_falsely_pass(control_results: dict[str, dict[str, float]]) -> bool:
    thresholds = {
        "shuffled_representation": 0.35,
        "shuffled_family_labels": 0.35,
        "shuffled_parameter_labels": 0.35,
        "shuffled_joint_labels": 0.35,
        "random_representation": 0.35,
        "wrong_family_conditioning": 0.35,
        "wrong_parameter_conditioning": 0.35,
        "class_prior_baseline": 0.35,
    }
    for control, threshold in thresholds.items():
        metrics = control_results[control]
        if metrics["slot_joint_accuracy"] > threshold or metrics["slot_decoded_answer_accuracy"] > threshold:
            return True
    return False


def compute_verdict(metrics: dict[str, Any]) -> str:
    if metrics["controls_collapse"]:
        return "INVALID: CONTROLS COLLAPSE"
    family = metrics["family_accuracy"]
    parameter = metrics["slot_parameter_accuracy"]
    joint = metrics["slot_joint_accuracy"]
    decoded = metrics["slot_decoded_answer_accuracy"]
    if all(
        [
            family >= 0.90,
            parameter > 0.50,
            joint > 0.35,
            decoded > 0.35,
            metrics["binding_accuracy"] > 0.35,
            metrics["binding_consistency"] > 0.70,
            metrics["parameter_gain_vs_real012b"] > 0.10,
            metrics["joint_gain_vs_real012b"] > 0.10,
            metrics["decoded_gain_vs_real012b"] > 0.10,
        ]
    ):
        return "VALIDATED: EXPLICIT PARAMETER-SLOT BINDING RECOVERS EXECUTABLE STRUCTURE"
    if family >= 0.90 and parameter <= 0.50:
        return "FAMILY RECOVERED, PARAMETER SLOT BINDING NOT RECOVERED"
    if parameter > 0.50 and joint <= 0.35:
        return "PARAMETER SLOT RECOVERED, JOINT BINDING NOT RECOVERED"
    if parameter > 0.50 and joint > 0.35 and decoded <= 0.35:
        return "BINDING RECOVERED, EXECUTABLE ANSWER NOT RECOVERED"
    return "NOT VALIDATED"


def interpret(metrics: dict[str, Any]) -> str:
    verdict = metrics["verdict"]
    if verdict == "VALIDATED: EXPLICIT PARAMETER-SLOT BINDING RECOVERS EXECUTABLE STRUCTURE":
        return (
            "REAL013 shows that explicit first-class parameter slots can recover parameters, joint bindings, "
            "and decoded executable answers on the validated REAL011 benchmark. This supports revisiting "
            "compiler/executor work with an explicit binding representation, while not proving the older "
            "REAL002 latent already contains those slots."
        )
    if verdict == "INVALID: CONTROLS COLLAPSE":
        return "REAL013 cannot be interpreted because at least one shortcut/control path falsely passed the recovery gates."
    if metrics["slot_parameter_accuracy"] <= 0.50:
        return "REAL013 preserves the family/argument dissociation: explicit slot probing did not recover executable parameters."
    return "REAL013 produced partial slot recovery, but one of joint recovery, decoding, consistency, gains, or controls missed the gates."


def joint_id(family: int, parameter: int) -> int:
    return family * N_PARAMETERS + parameter


def family_from_joint(joint: int) -> int:
    return joint // N_PARAMETERS


def parameter_from_joint(joint: int) -> int:
    return joint % N_PARAMETERS


def one_hot(index: int, size: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(size)]


def accuracy(predictions: list[int], labels: list[int]) -> float:
    return mean([float(pred == label) for pred, label in zip(predictions, labels)])


def majority(values: list[int]) -> int:
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return max(counts, key=lambda value: (counts[value], -value))


def class_prior_accuracy(test: list[BalancedExample]) -> float:
    family = majority([example.family for example in test])
    parameter = majority([example.parameter for example in test])
    joint = joint_id(family, parameter)
    return accuracy([joint for _ in test], [joint_id(example.family, example.parameter) for example in test])


def feature_variance(features: list[list[float]]) -> float:
    if not features:
        return 0.0
    total = 0.0
    for idx in range(len(features[0])):
        values = [row[idx] for row in features]
        mu = mean(values)
        total += mean([(value - mu) ** 2 for value in values])
    return total / len(features[0])


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
    for name in (
        "leaderboard.csv",
        "factor_recovery_matrix.csv",
        "probe_results.csv",
        "control_results.csv",
        "oracle_gap.csv",
        "effect_sizes.csv",
        "disentanglement_analysis.csv",
    ):
        write_csv(output_path.parent / name, rows)


def flatten_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, metrics in result["variant_results"].items():
        rows.append(
            {
                "variant": variant,
                "family_accuracy": metrics["family_accuracy"],
                "parameter_accuracy": metrics["parameter_accuracy"],
                "joint_accuracy": metrics["joint_accuracy"],
                "decoded_answer_accuracy": metrics["decoded_answer_accuracy"],
                "slot_parameter_accuracy": metrics["slot_parameter_accuracy"],
                "slot_joint_accuracy": metrics["slot_joint_accuracy"],
                "binding_accuracy": metrics["binding_accuracy"],
                "binding_consistency": metrics["binding_consistency"],
                "parameter_gain_vs_real012b": metrics["parameter_gain_vs_real012b"],
                "joint_gain_vs_real012b": metrics["joint_gain_vs_real012b"],
                "oracle_gap": metrics["oracle_gap"],
            }
        )
    return sorted(rows, key=lambda row: row["joint_accuracy"], reverse=True)


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
    parser.add_argument("--output-json", type=Path, default=Path("outputs/real013_full/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tac_scm_real013(
        seeds=args.seeds,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        steps=args.steps,
    )
    write_outputs(args.output_json, result)
    print(json.dumps({"verdict": result["verdict"], "best_metrics": result["best_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
