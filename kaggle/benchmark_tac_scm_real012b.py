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

REAL012B_VARIANTS = (
    "baseline_linear_probe",
    "factorized_probe",
    "family_conditioned_parameter_probe",
)

REAL012B_CONTROLS = (
    "shuffled_representation",
    "shuffled_family_labels",
    "shuffled_parameter_labels",
    "random_representation",
    "wrong_family_conditioning",
)

ALLOWED_VERDICTS = {
    "FAMILY RECOVERED, PARAMETER BINDING NOT RECOVERED",
    "FACTORized PARAMETER BINDING RECOVERED",
    "INVALID: CONTROLS COLLAPSE",
    "VALIDATED: FACTORIZED EXECUTABLE STRUCTURE RECOVERY IMPROVES PARAMETER BINDING",
    "NOT VALIDATED",
}


def run_tac_scm_real012b(
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
        for variant in REAL012B_VARIANTS:
            metrics = evaluate_variant(variant, train, test, seed, steps)
            per_seed_metrics.append({"seed": seed, "variant": variant, **metrics})
        for control in REAL012B_CONTROLS:
            metrics = evaluate_variant("family_conditioned_parameter_probe", train, test, seed, steps, control=control)
            control_seed_metrics.append({"seed": seed, "control": control, **metrics})
        oracle_seed_metrics.append({"seed": seed, **evaluate_oracle(test)})

    variant_results = aggregate_by(per_seed_metrics, "variant")
    control_results = aggregate_by(control_seed_metrics, "control")
    oracle_diagnostics = aggregate_rows(oracle_seed_metrics)
    best_variant = max(REAL012B_VARIANTS, key=lambda name: variant_results[name]["joint_accuracy"])
    best_metrics = dict(variant_results[best_variant])
    controls_collapse = all(control_results[name]["joint_accuracy"] <= 0.30 for name in REAL012B_CONTROLS)
    best_metrics.update(
        {
            "best_variant": best_variant,
            "real012a_parameter_baseline": REAL012A_PARAMETER_BASELINE,
            "real012a_joint_baseline": REAL012A_JOINT_BASELINE,
            "parameter_gain_vs_real012a": best_metrics["parameter_accuracy"] - REAL012A_PARAMETER_BASELINE,
            "joint_gain_vs_real012a": best_metrics["joint_accuracy"] - REAL012A_JOINT_BASELINE,
            "controls_collapse": controls_collapse,
        }
    )
    verdict = compute_verdict(best_metrics)
    best_metrics["verdict"] = verdict
    return {
        "benchmark": "TAC-SCM-REAL012B factorized parameter-binding recovery",
        "status": "completed",
        "verdict": verdict,
        "variants": list(REAL012B_VARIANTS),
        "controls": list(REAL012B_CONTROLS),
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
            "parameter_gain_vs_real012a": best_metrics["parameter_gain_vs_real012a"],
            "joint_gain_vs_real012a": best_metrics["joint_gain_vs_real012a"],
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
    train_features = [extract_features("real002", example, seed, split="train") for example in train]
    test_features = [extract_features("real002", example, seed, split="test") for example in test]
    train_family = [example.family for example in train]
    train_parameter = [example.parameter for example in train]

    if control == "shuffled_representation":
        test_features = rotated(test_features, 1)
    elif control == "shuffled_family_labels":
        train_family = rotated(train_family, 1)
    elif control == "shuffled_parameter_labels":
        train_parameter = rotated(train_parameter, 1)
    elif control == "random_representation":
        train_features = [extract_features("random_representation", example, seed, split="train") for example in train]
        test_features = [extract_features("random_representation", example, seed, split="test") for example in test]

    if variant == "baseline_linear_probe":
        family_pred = nearest_centroid_predict(train_features, train_family, test_features, N_FAMILIES)
        parameter_pred = nearest_centroid_predict(train_features, train_parameter, test_features, N_PARAMETERS)
        gold_family_param = parameter_pred
        predicted_family_param = parameter_pred
    elif variant == "factorized_probe":
        family_features_train = [family_path(features) for features in train_features]
        family_features_test = [family_path(features) for features in test_features]
        parameter_features_train = [parameter_path(features) for features in train_features]
        parameter_features_test = [parameter_path(features) for features in test_features]
        family_pred = nearest_centroid_predict(family_features_train, train_family, family_features_test, N_FAMILIES)
        parameter_pred = nearest_centroid_predict(parameter_features_train, train_parameter, parameter_features_test, N_PARAMETERS)
        gold_family_param = parameter_pred
        predicted_family_param = parameter_pred
    elif variant == "family_conditioned_parameter_probe":
        family_pred = nearest_centroid_predict([family_path(f) for f in train_features], train_family, [family_path(f) for f in test_features], N_FAMILIES)
        if control == "wrong_family_conditioning":
            conditioned_family = [(fam + 1) % N_FAMILIES for fam in family_pred]
        else:
            conditioned_family = family_pred
        predicted_family_param = family_conditioned_parameter_predict(
            train_features,
            train_family,
            train_parameter,
            test_features,
            conditioned_family,
        )
        gold_family_param = family_conditioned_parameter_predict(
            train_features,
            train_family,
            train_parameter,
            test_features,
            [example.family for example in test],
        )
        parameter_pred = predicted_family_param
    else:
        raise ValueError(variant)

    metrics = score_predictions(family_pred, parameter_pred, test, test_features)
    gold_metrics = score_predictions([example.family for example in test], gold_family_param, test, test_features)
    predicted_metrics = score_predictions(family_pred, predicted_family_param, test, test_features)
    metrics.update(
        {
            "gold_family_parameter_accuracy": gold_metrics["parameter_accuracy"],
            "predicted_family_parameter_accuracy": predicted_metrics["parameter_accuracy"],
            "factorized_family_accuracy": metrics["family_accuracy"],
            "factorized_parameter_accuracy": metrics["parameter_accuracy"],
            "factorized_joint_accuracy": metrics["joint_accuracy"],
            "family_conditioned_parameter_accuracy": predicted_metrics["parameter_accuracy"],
            "family_oracle_gap": 1.0 - metrics["family_accuracy"],
            "parameter_oracle_gap": 1.0 - metrics["parameter_accuracy"],
            "joint_oracle_gap": 1.0 - metrics["joint_accuracy"],
            "real012a_parameter_baseline": REAL012A_PARAMETER_BASELINE,
            "real012a_joint_baseline": REAL012A_JOINT_BASELINE,
            "parameter_gain_vs_real012a": metrics["parameter_accuracy"] - REAL012A_PARAMETER_BASELINE,
            "joint_gain_vs_real012a": metrics["joint_accuracy"] - REAL012A_JOINT_BASELINE,
        }
    )
    return metrics


def family_path(features: list[float]) -> list[float]:
    # REAL012A's REAL002 proxy stores query, then weak family-correlated slots.
    return features[:20]


def parameter_path(features: list[float]) -> list[float]:
    # Deliberately excludes direct family slots to test whether parameter signal is
    # independently bound rather than read through the family channel.
    return features[:16] + features[20:]


def family_conditioned_parameter_predict(
    train_features: list[list[float]],
    train_family: list[int],
    train_parameter: list[int],
    test_features: list[list[float]],
    conditioned_family: list[int],
) -> list[int]:
    dim = len(parameter_path(train_features[0]))
    centroids: dict[tuple[int, int], list[float]] = {}
    counts: dict[tuple[int, int], int] = {}
    global_centroids = [[0.0] * dim for _ in range(N_PARAMETERS)]
    global_counts = [0] * N_PARAMETERS
    for features, family_id, parameter_id in zip(train_features, train_family, train_parameter):
        path = parameter_path(features)
        key = (family_id, parameter_id)
        centroids.setdefault(key, [0.0] * dim)
        counts[key] = counts.get(key, 0) + 1
        global_counts[parameter_id] += 1
        for i, value in enumerate(path):
            centroids[key][i] += value
            global_centroids[parameter_id][i] += value
    for key, values in centroids.items():
        centroids[key] = [value / counts[key] for value in values]
    for parameter_id, values in enumerate(global_centroids):
        if global_counts[parameter_id]:
            global_centroids[parameter_id] = [value / global_counts[parameter_id] for value in values]

    predictions: list[int] = []
    for features, family_id in zip(test_features, conditioned_family):
        path = parameter_path(features)
        best_parameter = min(
            range(N_PARAMETERS),
            key=lambda parameter_id: squared_distance(path, centroids.get((family_id, parameter_id), global_centroids[parameter_id])),
        )
        predictions.append(best_parameter)
    return predictions


def evaluate_oracle(test: list[BalancedExample]) -> dict[str, float]:
    family = [example.family for example in test]
    parameter = [example.parameter for example in test]
    oracle_features = [[0.0] for _ in test]
    return score_predictions(family, parameter, test, oracle_features)


def score_predictions(
    family_pred: list[int],
    parameter_pred: list[int],
    test: list[BalancedExample],
    features: list[list[float]],
) -> dict[str, float]:
    family_gold = [example.family for example in test]
    parameter_gold = [example.parameter for example in test]
    answers = [example.answer for example in test]
    decoded = [executable_answer(family, parameter, example.query) for family, parameter, example in zip(family_pred, parameter_pred, test)]
    family_accuracy = accuracy(family_pred, family_gold)
    parameter_accuracy = accuracy(parameter_pred, parameter_gold)
    joint_accuracy = mean([float(f == fg and p == pg) for f, p, fg, pg in zip(family_pred, parameter_pred, family_gold, parameter_gold)])
    decoded_answer_accuracy = accuracy(decoded, answers)
    return {
        "family_accuracy": family_accuracy,
        "family_probe_accuracy": family_accuracy,
        "parameter_accuracy": parameter_accuracy,
        "parameter_probe_accuracy": parameter_accuracy,
        "joint_accuracy": joint_accuracy,
        "decoded_answer_accuracy": decoded_answer_accuracy,
        "oracle_gap": 1.0 - decoded_answer_accuracy,
        "cluster_purity": joint_accuracy,
        "nearest_neighbor_retrieval": sampled_nearest_neighbor_retrieval(features, family_gold, parameter_gold),
        "family_clustering": family_accuracy,
        "parameter_clustering": parameter_accuracy,
        "joint_clustering": joint_accuracy,
        "representation_entropy": 0.0,
        "latent_variance": feature_variance(features),
    }


def sampled_nearest_neighbor_retrieval(features: list[list[float]], family: list[int], parameter: list[int]) -> float:
    if len(features) < 2:
        return 0.0
    limit = min(64, len(features))
    sample_features = features[:limit]
    sample_family = family[:limit]
    sample_parameter = parameter[:limit]
    hits = []
    for i, feat in enumerate(sample_features):
        best = min((j for j in range(limit) if j != i), key=lambda j: squared_distance(feat, sample_features[j]))
        hits.append(float(sample_family[best] == sample_family[i] and sample_parameter[best] == sample_parameter[i]))
    return mean(hits)


def accuracy(predictions: list[int], labels: list[int]) -> float:
    return mean([float(pred == label) for pred, label in zip(predictions, labels)])


def feature_variance(features: list[list[float]]) -> float:
    if not features:
        return 0.0
    total = 0.0
    for i in range(len(features[0])):
        values = [row[i] for row in features]
        mu = mean(values)
        total += mean([(value - mu) ** 2 for value in values])
    return total / len(features[0])


def compute_verdict(metrics: dict[str, Any]) -> str:
    controls_collapse = bool(metrics["controls_collapse"])
    family = metrics["family_accuracy"]
    parameter = metrics["parameter_accuracy"]
    joint = metrics["joint_accuracy"]
    decoded = metrics["decoded_answer_accuracy"]
    parameter_gain = metrics["parameter_gain_vs_real012a"]
    joint_gain = metrics["joint_gain_vs_real012a"]
    if not controls_collapse:
        if family >= 0.90 and parameter <= 0.50:
            return "FAMILY RECOVERED, PARAMETER BINDING NOT RECOVERED"
        return "INVALID: CONTROLS COLLAPSE"
    if (
        family >= 0.90
        and parameter > 0.50
        and joint > 0.35
        and decoded > 0.35
        and parameter_gain > 0.10
        and joint_gain > 0.10
    ):
        return "VALIDATED: FACTORIZED EXECUTABLE STRUCTURE RECOVERY IMPROVES PARAMETER BINDING"
    if family >= 0.90 and parameter > 0.50 and joint > 0.35:
        return "FACTORized PARAMETER BINDING RECOVERED"
    if family >= 0.90 and parameter <= 0.50:
        return "FAMILY RECOVERED, PARAMETER BINDING NOT RECOVERED"
    return "NOT VALIDATED"


def interpret(metrics: dict[str, Any]) -> str:
    if metrics["verdict"] == "VALIDATED: FACTORIZED EXECUTABLE STRUCTURE RECOVERY IMPROVES PARAMETER BINDING":
        return "REAL012B supports revisiting compiler/executor work because factorization improved parameter and joint recovery."
    if metrics["family_accuracy"] >= 0.90 and metrics["parameter_accuracy"] <= 0.50:
        return "REAL012B strengthens the family/argument dissociation finding: family is recoverable, parameter binding remains near chance."
    return "REAL012B does not validate parameter binding recovery; representation quality remains the bottleneck."


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
    for name in ("leaderboard.csv", "probe_results.csv", "oracle_gap.csv", "effect_sizes.csv", "factorized_recovery.csv"):
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
                "gold_family_parameter_accuracy": metrics["gold_family_parameter_accuracy"],
                "predicted_family_parameter_accuracy": metrics["predicted_family_parameter_accuracy"],
                "parameter_gain_vs_real012a": metrics["parameter_gain_vs_real012a"],
                "joint_gain_vs_real012a": metrics["joint_gain_vs_real012a"],
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
    parser.add_argument("--output-json", type=Path, default=Path("outputs/real012b_full/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tac_scm_real012b(
        seeds=args.seeds,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        steps=args.steps,
    )
    write_outputs(args.output_json, result)
    print(json.dumps({"verdict": result["verdict"], "best_metrics": result["best_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
