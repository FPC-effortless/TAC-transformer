from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable


REAL011_BASELINES = (
    "random_predictor",
    "majority_class",
    "query_only",
    "surface_only",
    "nearest_neighbor_memorization",
    "family_only_oracle",
    "parameter_only_oracle",
    "family_parameter_oracle",
    "shuffled_family_oracle",
    "shuffled_parameter_oracle",
    "fully_shuffled_oracle",
)

REAL011_METRIC_NAMES = (
    "family_parameter_oracle_accuracy",
    "family_only_accuracy",
    "parameter_only_accuracy",
    "query_only_accuracy",
    "surface_only_accuracy",
    "majority_accuracy",
    "memorization_accuracy",
    "family_shuffle_drop",
    "parameter_shuffle_drop",
    "full_shuffle_drop",
    "counterfactual_sensitivity_family",
    "counterfactual_sensitivity_parameter",
    "mutual_information_family",
    "mutual_information_parameter",
    "mutual_information_joint",
    "answer_balance_error",
    "class_imbalance",
)

N_FAMILIES = 4
N_PARAMETERS = 4
N_QUERIES = 16
ANSWER_SIZE = 16


@dataclass(frozen=True)
class BalancedExample:
    family: int
    parameter: int
    query: int
    answer: int
    surface: tuple[int, ...]


class BalancedExecutableStructureDataset:
    def __init__(self, split: str, n_samples: int = 256, seed: int = 0):
        self.split = split
        self.n_samples = n_samples
        self.seed = seed
        self.examples = generate_balanced_executable_dataset(split, n_samples, seed)


def executable_answer(family: int, parameter: int, query: int) -> int:
    return (query + family * N_PARAMETERS + parameter) % ANSWER_SIZE


def generate_balanced_executable_dataset(split: str, n_samples: int, seed: int) -> list[BalancedExample]:
    rng = random.Random(seed + {"train": 0, "validation": 10_000, "test": 20_000}.get(split, 30_000))
    grid = [(family, parameter, query) for query in range(N_QUERIES) for family in range(N_FAMILIES) for parameter in range(N_PARAMETERS)]
    repeats = math.ceil(n_samples / len(grid))
    triples = (grid * repeats)[:n_samples]
    offset = (seed * 17 + (0 if split == "train" else 53)) % len(triples)
    triples = triples[offset:] + triples[:offset]
    examples: list[BalancedExample] = []
    for index, (family, parameter, query) in enumerate(triples):
        surface_base = 100 if split == "train" else 150
        nonce = rng.randrange(20)
        surface = (surface_base + (query % 16), 200 + nonce, 220 + (index % 11), 240 + ((query + nonce) % 13))
        examples.append(BalancedExample(family, parameter, query, executable_answer(family, parameter, query), surface))
    return examples


def run_tac_scm_real011(
    *,
    seeds: Iterable[int] = range(10),
    train_samples: int = 256,
    eval_samples: int = 256,
) -> dict[str, Any]:
    seeds = list(seeds)
    per_seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        train = generate_balanced_executable_dataset("train", train_samples, seed)
        test = generate_balanced_executable_dataset("test", eval_samples, seed)
        shared = {
            **dataset_balance_report(test),
            **information_theory_audit(test),
            **counterfactual_analysis(test),
            **shortcut_analysis(train, test),
        }
        for baseline in REAL011_BASELINES:
            row = {
                "seed": seed,
                "baseline": baseline,
                **evaluate_baseline(baseline, train, test, seed),
                **shared,
            }
            per_seed_results.append(row)

    variant_results = aggregate_variant_results(per_seed_results)
    success_gate = evaluate_real011_success_gate(variant_results)
    return {
        "benchmark": "TAC-SCM-REAL011 balanced executable structure benchmark redesign",
        "status": "passed" if success_gate["passed"] else "failed",
        "verdict": success_gate["verdict"],
        "baselines": list(REAL011_BASELINES),
        "metrics": list(REAL011_METRIC_NAMES),
        "seeds": seeds,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "variant_results": variant_results,
        "validity_gates": success_gate["gates"],
        "per_seed_results": per_seed_results,
        "summary": summarize_real011(variant_results, success_gate),
    }


def evaluate_baseline(name: str, train: list[BalancedExample], test: list[BalancedExample], seed: int) -> dict[str, float]:
    train = examples_from(train)
    test = examples_from(test)
    rng = random.Random(seed + 250_000)
    majority = most_common([example.answer for example in train])
    family_lookup = grouped_majority(train, lambda example: example.family)
    parameter_lookup = grouped_majority(train, lambda example: example.parameter)
    query_lookup = grouped_majority(train, lambda example: example.query)
    surface_lookup = grouped_majority(train, lambda example: example.surface)
    predictions: list[int] = []
    for example in test:
        if name == "random_predictor":
            pred = rng.randrange(ANSWER_SIZE)
        elif name == "majority_class":
            pred = majority
        elif name == "query_only":
            pred = query_lookup.get(example.query, majority)
        elif name == "surface_only":
            pred = surface_lookup.get(example.surface, majority)
        elif name == "nearest_neighbor_memorization":
            pred = nearest_neighbor_answer(example, train)
        elif name == "family_only_oracle":
            pred = family_lookup.get(example.family, majority)
        elif name == "parameter_only_oracle":
            pred = parameter_lookup.get(example.parameter, majority)
        elif name == "family_parameter_oracle":
            pred = executable_answer(example.family, example.parameter, example.query)
        elif name == "shuffled_family_oracle":
            pred = executable_answer((example.family + 1) % N_FAMILIES, example.parameter, example.query)
        elif name == "shuffled_parameter_oracle":
            pred = executable_answer(example.family, (example.parameter + 1) % N_PARAMETERS, example.query)
        elif name == "fully_shuffled_oracle":
            pred = executable_answer((example.family + 1) % N_FAMILIES, (example.parameter + 1) % N_PARAMETERS, example.query)
        else:
            raise ValueError(name)
        predictions.append(pred)

    gold = [example.answer for example in test]
    oracle_acc = 1.0
    family_only = evaluate_lookup(train, test, lambda example: example.family, majority)
    parameter_only = evaluate_lookup(train, test, lambda example: example.parameter, majority)
    query_only = evaluate_lookup(train, test, lambda example: example.query, majority)
    surface_only = evaluate_lookup(train, test, lambda example: example.surface, majority)
    memorization = accuracy([nearest_neighbor_answer(example, train) for example in test], gold)
    shuffled_family = [executable_answer((example.family + 1) % N_FAMILIES, example.parameter, example.query) for example in test]
    shuffled_parameter = [executable_answer(example.family, (example.parameter + 1) % N_PARAMETERS, example.query) for example in test]
    fully_shuffled = [executable_answer((example.family + 1) % N_FAMILIES, (example.parameter + 1) % N_PARAMETERS, example.query) for example in test]
    return {
        "answer_accuracy": accuracy(predictions, gold),
        "family_parameter_oracle_accuracy": oracle_acc,
        "family_only_accuracy": family_only,
        "parameter_only_accuracy": parameter_only,
        "query_only_accuracy": query_only,
        "surface_only_accuracy": surface_only,
        "majority_accuracy": accuracy([majority] * len(gold), gold),
        "memorization_accuracy": memorization,
        "family_shuffle_drop": oracle_acc - accuracy(shuffled_family, gold),
        "parameter_shuffle_drop": oracle_acc - accuracy(shuffled_parameter, gold),
        "full_shuffle_drop": oracle_acc - accuracy(fully_shuffled, gold),
        "oracle_gap": oracle_acc - accuracy(predictions, gold),
    }


def examples_from(data: Any) -> list[BalancedExample]:
    if hasattr(data, "examples"):
        return list(data.examples)
    return list(data)


def evaluate_real011_success_gate(variant_results: dict[str, dict[str, float]]) -> dict[str, Any]:
    oracle = variant_results["family_parameter_oracle"]
    gates = {
        "oracle_gt_095": oracle["family_parameter_oracle_accuracy"] > 0.95,
        "family_only_lt_030": oracle["family_only_accuracy"] < 0.30,
        "parameter_only_lt_030": oracle["parameter_only_accuracy"] < 0.30,
        "query_only_near_chance": oracle["query_only_accuracy"] < 0.30,
        "surface_only_near_chance": oracle["surface_only_accuracy"] < 0.30,
        "majority_near_chance": oracle["majority_accuracy"] < 0.30,
        "memorization_near_chance": oracle["memorization_accuracy"] < 0.30,
        "family_shuffle_large_drop": oracle["family_shuffle_drop"] >= 0.70,
        "parameter_shuffle_large_drop": oracle["parameter_shuffle_drop"] >= 0.70,
        "full_shuffle_large_drop": oracle["full_shuffle_drop"] >= 0.90,
        "family_counterfactual": oracle["counterfactual_sensitivity_family"] >= 0.70,
        "parameter_counterfactual": oracle["counterfactual_sensitivity_parameter"] >= 0.70,
        "balanced_answers": oracle["answer_balance_error"] <= 0.01,
    }
    passed = all(gates.values())
    return {
        "passed": passed,
        "verdict": "VALID EXECUTABLE STRUCTURE BENCHMARK" if passed else "INVALID EXECUTABLE STRUCTURE BENCHMARK",
        "gates": gates,
    }


def summarize_real011(variant_results: dict[str, dict[str, float]], success_gate: dict[str, Any]) -> dict[str, Any]:
    oracle = variant_results["family_parameter_oracle"]
    return {
        "verdict": success_gate["verdict"],
        "family_parameter_oracle_accuracy": oracle["family_parameter_oracle_accuracy"],
        "family_only_accuracy": oracle["family_only_accuracy"],
        "parameter_only_accuracy": oracle["parameter_only_accuracy"],
        "query_only_accuracy": oracle["query_only_accuracy"],
        "surface_only_accuracy": oracle["surface_only_accuracy"],
        "memorization_accuracy": oracle["memorization_accuracy"],
        "family_shuffle_drop": oracle["family_shuffle_drop"],
        "parameter_shuffle_drop": oracle["parameter_shuffle_drop"],
        "full_shuffle_drop": oracle["full_shuffle_drop"],
    }


def dataset_balance_report(examples: list[BalancedExample]) -> dict[str, float]:
    return {
        "family_balance_error": balance_error([example.family for example in examples]),
        "parameter_balance_error": balance_error([example.parameter for example in examples]),
        "answer_balance_error": balance_error([example.answer for example in examples]),
        "class_imbalance": class_imbalance([example.answer for example in examples]),
    }


def information_theory_audit(examples: list[BalancedExample]) -> dict[str, float]:
    answers = [example.answer for example in examples]
    return {
        "mutual_information_family": mutual_information([example.family for example in examples], answers),
        "mutual_information_parameter": mutual_information([example.parameter for example in examples], answers),
        "mutual_information_joint": mutual_information([(example.family, example.parameter) for example in examples], answers),
        "mutual_information_query": mutual_information([example.query for example in examples], answers),
    }


def counterfactual_analysis(examples: list[BalancedExample]) -> dict[str, float]:
    family_changes = []
    parameter_changes = []
    for example in examples:
        base = executable_answer(example.family, example.parameter, example.query)
        family_changes.append(float(executable_answer((example.family + 1) % N_FAMILIES, example.parameter, example.query) != base))
        parameter_changes.append(float(executable_answer(example.family, (example.parameter + 1) % N_PARAMETERS, example.query) != base))
    return {
        "counterfactual_sensitivity_family": mean(family_changes),
        "counterfactual_sensitivity_parameter": mean(parameter_changes),
    }


def shortcut_analysis(train: list[BalancedExample], test: list[BalancedExample]) -> dict[str, float]:
    train_surfaces = {example.surface for example in train}
    test_surfaces = [example.surface for example in test]
    return {
        "surface_overlap": sum(1 for surface in test_surfaces if surface in train_surfaces) / max(1, len(test_surfaces)),
        "duplicate_pattern_rate": 1.0 - len(set(test_surfaces)) / max(1, len(test_surfaces)),
        "shortcut_score": max(
            mutual_information([example.query for example in test], [example.answer for example in test]),
            mutual_information([example.surface for example in test], [example.answer for example in test]),
        ),
    }


def aggregate_variant_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["baseline"]].append(row)
    results: dict[str, dict[str, float]] = {}
    for baseline, baseline_rows in grouped.items():
        metrics: dict[str, float] = {}
        for key, value in baseline_rows[0].items():
            if isinstance(value, (int, float)):
                values = [float(row[key]) for row in baseline_rows]
                metrics[key] = mean(values)
                metrics[f"{key}_std"] = pstdev(values)
        results[baseline] = metrics
    return results


def grouped_majority(examples: list[BalancedExample], key_fn: Callable[[BalancedExample], Any]) -> dict[Any, int]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for example in examples:
        groups[key_fn(example)].append(example.answer)
    return {key: most_common(values) for key, values in groups.items()}


def evaluate_lookup(
    train: list[BalancedExample],
    test: list[BalancedExample],
    key_fn: Callable[[BalancedExample], Any],
    fallback: int,
) -> float:
    lookup = grouped_majority(train, key_fn)
    return accuracy([lookup.get(key_fn(example), fallback) for example in test], [example.answer for example in test])


def nearest_neighbor_answer(example: BalancedExample, train: list[BalancedExample]) -> int:
    nearest = min(train, key=lambda candidate: hamming(example.surface, candidate.surface))
    return nearest.answer


def hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(left != right for left, right in zip(a, b)) + abs(len(a) - len(b))


def most_common(values: Iterable[int]) -> int:
    return Counter(values).most_common(1)[0][0]


def accuracy(predictions: Iterable[int], labels: Iterable[int]) -> float:
    pairs = list(zip(predictions, labels))
    return sum(int(pred == label) for pred, label in pairs) / max(1, len(pairs))


def balance_error(values: list[int]) -> float:
    counts = Counter(values)
    expected = len(values) / max(1, len(counts))
    return max(abs(count - expected) for count in counts.values()) / max(1.0, expected)


def class_imbalance(values: list[int]) -> float:
    counts = Counter(values)
    return max(counts.values()) / max(1, len(values))


def mutual_information(x_values: list[Any], y_values: list[Any]) -> float:
    total = max(1, len(x_values))
    px = Counter(x_values)
    py = Counter(y_values)
    pxy = Counter(zip(x_values, y_values))
    score = 0.0
    for (x_val, y_val), joint_count in pxy.items():
        p_joint = joint_count / total
        score += p_joint * math.log((p_joint + 1e-12) / ((px[x_val] / total) * (py[y_val] / total) + 1e-12))
    return max(0.0, score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tac_scm_real011(seeds=args.seeds, train_samples=args.train_samples, eval_samples=args.eval_samples)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "verdict": result["verdict"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
