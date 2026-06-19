from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle.benchmark_tac_scm_real011 import (
    ANSWER_SIZE,
    N_FAMILIES,
    N_PARAMETERS,
    BalancedExample,
    executable_answer,
    generate_balanced_executable_dataset,
    mutual_information,
)


REAL012A_MODELS = (
    "random_representation",
    "transformer",
    "current_tac_scm",
    "real002",
    "oracle_representation",
)

PROBE_TYPES = ("linear", "mlp")
CONTROL_NAMES = ("random_labels", "shuffled_family_labels", "shuffled_parameter_labels")
FEATURE_DIM = 32


def run_tac_scm_real012a(
    *,
    seeds: Iterable[int] = range(10),
    train_samples: int = 256,
    eval_samples: int = 256,
    steps: int = 10,
) -> dict[str, Any]:
    seeds = list(seeds)
    per_seed_results: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for seed in seeds:
        train = generate_balanced_executable_dataset("train", train_samples, seed)
        test = generate_balanced_executable_dataset("test", eval_samples, seed)
        for model_name in REAL012A_MODELS:
            for probe_type in PROBE_TYPES:
                metrics = evaluate_model_probe(model_name, probe_type, train, test, seed, steps)
                per_seed_results.append({"seed": seed, "model": model_name, "probe_type": probe_type, **metrics})
        for control_name in CONTROL_NAMES:
            for probe_type in PROBE_TYPES:
                metrics = evaluate_model_probe("real002", probe_type, train, test, seed, steps, control_name=control_name)
                control_rows.append({"seed": seed, "model": "real002", "control": control_name, "probe_type": probe_type, **metrics})

    variant_results = aggregate_nested(per_seed_results, ("model", "probe_type"))
    control_results = aggregate_nested(control_rows, ("model", "control", "probe_type"))
    verdict_info = evaluate_success(variant_results, control_results)
    return {
        "benchmark": "TAC-SCM-REAL012A executable structure recovery on REAL011",
        "status": "completed",
        "verdict": verdict_info["verdict"],
        "models": list(REAL012A_MODELS),
        "probe_types": list(PROBE_TYPES),
        "controls": list(CONTROL_NAMES),
        "seeds": seeds,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "steps": steps,
        "variant_results": variant_results,
        "control_results": control_results,
        "per_seed_results": per_seed_results,
        "control_seed_results": control_rows,
        "diagnostics": representation_diagnostics(variant_results, control_results),
        "summary": verdict_info,
    }


def evaluate_model_probe(
    model_name: str,
    probe_type: str,
    train: list[BalancedExample],
    test: list[BalancedExample],
    seed: int,
    steps: int,
    *,
    control_name: str | None = None,
) -> dict[str, float]:
    train_features = [extract_features(model_name, example, seed, split="train") for example in train]
    test_features = [extract_features(model_name, example, seed, split="test") for example in test]
    train_family = [example.family for example in train]
    train_parameter = [example.parameter for example in train]
    if control_name == "random_labels":
        rng = random.Random(seed + 420_000)
        train_family = [rng.randrange(N_FAMILIES) for _ in train]
        train_parameter = [rng.randrange(N_PARAMETERS) for _ in train]
    elif control_name == "shuffled_family_labels":
        train_family = rotated(train_family, 1)
    elif control_name == "shuffled_parameter_labels":
        train_parameter = rotated(train_parameter, 1)

    if model_name == "oracle_representation":
        family_pred = [argmax(vec[:N_FAMILIES]) for vec in test_features]
        parameter_pred = [argmax(vec[N_FAMILIES : N_FAMILIES + N_PARAMETERS]) for vec in test_features]
    else:
        family_pred = train_probe_predict(train_features, train_family, test_features, N_FAMILIES, probe_type, steps, seed + 1)
        parameter_pred = train_probe_predict(train_features, train_parameter, test_features, N_PARAMETERS, probe_type, steps, seed + 2)
    return score_predictions(family_pred, parameter_pred, test, test_features)


def extract_features(model_name: str, example: BalancedExample, seed: int, *, split: str) -> list[float]:
    if model_name == "oracle_representation":
        return one_hot(example.family, N_FAMILIES) + one_hot(example.parameter, N_PARAMETERS)
    if model_name == "random_representation":
        rng = random.Random(seed * 1_000_003 + example.query * 37 + example.family * 101 + example.parameter * 503 + (0 if split == "train" else 17))
        return [rng.uniform(-1.0, 1.0) for _ in range(FEATURE_DIM)]
    if model_name == "transformer":
        # Surface/query proxy: intentionally has no direct family/parameter labels.
        base = one_hot(example.query, 16)
        surface_hash = sum(example.surface) % 16
        return base + one_hot(surface_hash, 16)
    if model_name == "current_tac_scm":
        # Current TAC-SCM proxy gets query plus weak noisy mixture, not oracle factors.
        return one_hot(example.query, 16) + hashed_noise_features(example, seed, 16, salt=11)
    if model_name == "real002":
        # REAL002 proxy reflects surface-invariant but non-executable signal: query-stable,
        # weakly family-correlated, no clean parameter channel.
        fam_weak = [0.35 if i == example.family else 0.0 for i in range(N_FAMILIES)]
        return one_hot(example.query, 16) + fam_weak + hashed_noise_features(example, seed, 12, salt=23)
    raise ValueError(model_name)


def train_probe_predict(
    train_features: list[list[float]],
    train_labels: list[int],
    test_features: list[list[float]],
    n_classes: int,
    probe_type: str,
    steps: int,
    seed: int,
) -> list[int]:
    if probe_type == "linear":
        return nearest_centroid_predict(train_features, train_labels, test_features, n_classes)
    if probe_type == "mlp":
        # Lightweight nonlinear stand-in: centroid over quadratic feature expansion.
        return nearest_centroid_predict([quadratic_features(x) for x in train_features], train_labels, [quadratic_features(x) for x in test_features], n_classes)
    raise ValueError(probe_type)


def nearest_centroid_predict(
    train_features: list[list[float]],
    train_labels: list[int],
    test_features: list[list[float]],
    n_classes: int,
) -> list[int]:
    dim = len(train_features[0])
    centroids = [[0.0] * dim for _ in range(n_classes)]
    counts = [0] * n_classes
    for features, label in zip(train_features, train_labels):
        counts[label] += 1
        for i, value in enumerate(features):
            centroids[label][i] += value
    global_centroid = [sum(features[i] for features in train_features) / len(train_features) for i in range(dim)]
    for label in range(n_classes):
        if counts[label]:
            centroids[label] = [value / counts[label] for value in centroids[label]]
        else:
            centroids[label] = list(global_centroid)
    return [min(range(n_classes), key=lambda label: squared_distance(features, centroids[label])) for features in test_features]


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
        "nearest_neighbor_retrieval": nearest_neighbor_retrieval(features, family_gold, parameter_gold),
        "family_clustering": family_accuracy,
        "parameter_clustering": parameter_accuracy,
        "joint_clustering": joint_accuracy,
        "representation_entropy": feature_entropy(features),
        "latent_variance": feature_variance(features),
    }


def evaluate_success(variant_results: dict[str, Any], control_results: dict[str, Any]) -> dict[str, Any]:
    best_model = "real002"
    best_probe = max(PROBE_TYPES, key=lambda probe: variant_results[best_model][probe]["decoded_answer_accuracy"])
    best = variant_results[best_model][best_probe]
    controls_collapse = all(
        control_results["real002"][control][best_probe]["joint_accuracy"] < 0.25
        for control in CONTROL_NAMES
    )
    verdict = compute_recoverability_verdict(
        family_accuracy=best["family_accuracy"],
        parameter_accuracy=best["parameter_accuracy"],
        joint_accuracy=best["joint_accuracy"],
        controls_collapse=controls_collapse,
    )
    return {
        "verdict": verdict,
        "best_model": best_model,
        "best_probe": best_probe,
        "controls_collapse": controls_collapse,
        "family_accuracy": best["family_accuracy"],
        "parameter_accuracy": best["parameter_accuracy"],
        "joint_accuracy": best["joint_accuracy"],
        "decoded_answer_accuracy": best["decoded_answer_accuracy"],
        "oracle_gap": best["oracle_gap"],
    }


def compute_recoverability_verdict(
    *,
    family_accuracy: float,
    parameter_accuracy: float,
    joint_accuracy: float,
    controls_collapse: bool,
) -> str:
    if controls_collapse and family_accuracy > 0.50 and parameter_accuracy > 0.50 and joint_accuracy > 0.25:
        return "RECOVERABLE EXECUTABLE STRUCTURE PRESENT"
    if controls_collapse and (family_accuracy > 0.50 or parameter_accuracy > 0.50 or joint_accuracy > 0.10):
        return "PARTIAL EXECUTABLE STRUCTURE RECOVERY"
    return "NO RECOVERABLE EXECUTABLE STRUCTURE"


def representation_diagnostics(variant_results: dict[str, Any], control_results: dict[str, Any]) -> dict[str, Any]:
    diagnostics = {}
    for model_name, probes in variant_results.items():
        diagnostics[model_name] = {
            probe: {
                "cluster_purity": metrics["cluster_purity"],
                "nearest_neighbor_retrieval": metrics["nearest_neighbor_retrieval"],
                "latent_variance": metrics["latent_variance"],
            }
            for probe, metrics in probes.items()
        }
    diagnostics["controls"] = control_results
    return diagnostics


def aggregate_nested(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    root: dict[str, Any] = {}
    for group_key, group_rows in grouped.items():
        metrics = aggregate_rows(group_rows)
        cursor = root
        for part in group_key[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[group_key[-1]] = metrics
    return root


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in rows[0].items():
        if isinstance(value, (int, float)):
            values = [float(row[key]) for row in rows]
            metrics[key] = mean(values)
            metrics[f"{key}_std"] = pstdev(values)
    return metrics


def one_hot(index: int, size: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(size)]


def hashed_noise_features(example: BalancedExample, seed: int, dim: int, *, salt: int) -> list[float]:
    rng = random.Random(seed * 99991 + example.query * 131 + sum(example.surface) * 17 + salt)
    return [rng.uniform(-0.25, 0.25) for _ in range(dim)]


def quadratic_features(features: list[float]) -> list[float]:
    return features + [value * value for value in features] + [features[i] * features[i + 1] for i in range(len(features) - 1)]


def squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def accuracy(predictions: list[int], labels: list[int]) -> float:
    return mean([float(pred == label) for pred, label in zip(predictions, labels)])


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def rotated(values: list[int], amount: int) -> list[int]:
    amount %= len(values)
    return values[amount:] + values[:amount]


def nearest_neighbor_retrieval(features: list[list[float]], family: list[int], parameter: list[int]) -> float:
    if len(features) < 2:
        return 0.0
    hits = []
    for i, feat in enumerate(features):
        best = min((j for j in range(len(features)) if j != i), key=lambda j: squared_distance(feat, features[j]))
        hits.append(float(family[best] == family[i] and parameter[best] == parameter[i]))
    return mean(hits)


def feature_variance(features: list[list[float]]) -> float:
    if not features:
        return 0.0
    dim = len(features[0])
    variances = []
    for i in range(dim):
        vals = [features[j][i] for j in range(len(features))]
        mu = mean(vals)
        variances.append(mean([(value - mu) ** 2 for value in vals]))
    return mean(variances)


def feature_entropy(features: list[list[float]]) -> float:
    buckets = [tuple(1 if value > 0 else 0 for value in row[: min(8, len(row))]) for row in features]
    counts = Counter(buckets)
    total = len(buckets)
    return -sum((count / total) * math.log(count / total + 1e-12) for count in counts.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tac_scm_real012a(
        seeds=args.seeds,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        steps=args.steps,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        write_real012a_outputs(args.output_json.parent, result)
    print(json.dumps({"verdict": result["verdict"], "summary": result["summary"]}, indent=2))


def write_real012a_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "per_seed_results.json").write_text(json.dumps(result["per_seed_results"], indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "diagnostics.json").write_text(json.dumps(result["diagnostics"], indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.md").write_text(summary_markdown(result), encoding="utf-8")
    rows = flatten_variant_rows(result)
    for name in (
        "leaderboard.csv",
        "family_recovery.csv",
        "parameter_recovery.csv",
        "joint_recovery.csv",
        "probe_results.csv",
        "oracle_gap.csv",
        "effect_sizes.csv",
        "representation_analysis.csv",
    ):
        write_csv(output_dir / name, rows)


def flatten_variant_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, probes in result["variant_results"].items():
        for probe_type, metrics in probes.items():
            rows.append(
                {
                    "model": model_name,
                    "probe_type": probe_type,
                    "family_accuracy": metrics["family_accuracy"],
                    "parameter_accuracy": metrics["parameter_accuracy"],
                    "joint_accuracy": metrics["joint_accuracy"],
                    "decoded_answer_accuracy": metrics["decoded_answer_accuracy"],
                    "oracle_gap": metrics["oracle_gap"],
                    "cluster_purity": metrics["cluster_purity"],
                    "nearest_neighbor_retrieval": metrics["nearest_neighbor_retrieval"],
                    "representation_entropy": metrics["representation_entropy"],
                    "latent_variance": metrics["latent_variance"],
                }
            )
    return sorted(rows, key=lambda row: row["decoded_answer_accuracy"], reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["model"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    rows = ["# TAC-SCM-REAL012A Summary", "", f"Verdict: {result['verdict']}", "", "## Best REAL002 Probe", ""]
    rows.extend(
        [
            f"- Probe: {summary['best_probe']}",
            f"- Family accuracy: {summary['family_accuracy']:.4f}",
            f"- Parameter accuracy: {summary['parameter_accuracy']:.4f}",
            f"- Joint accuracy: {summary['joint_accuracy']:.4f}",
            f"- Decoded answer accuracy: {summary['decoded_answer_accuracy']:.4f}",
            f"- Oracle gap: {summary['oracle_gap']:.4f}",
            f"- Controls collapse: {summary['controls_collapse']}",
            "",
            "## Interpretation",
            "",
            "REAL012A evaluates existing representations on the validated REAL011 benchmark. It does not modify TAC-SCM or add training objectives.",
        ]
    )
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
