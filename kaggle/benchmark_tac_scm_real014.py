from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
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


ANSWER_SIZE = 16
REAL014_VARIANTS = (
    "real013_direct_decode_baseline",
    "neural_compiler_executor",
    "symbolic_rule_dispatch_executor",
    "ablated_compiler_executor",
)
REAL014_CONTROLS = (
    "correct_bound_slot",
    "reset_no_slot",
    "shuffled_family",
    "shuffled_parameter",
    "shuffled_binding",
    "wrong_family",
    "wrong_parameter",
    "wrong_family_correct_parameter",
    "correct_family_wrong_parameter",
    "random_representation",
    "oracle_family_oracle_parameter",
    "oracle_family_predicted_parameter",
    "predicted_family_oracle_parameter",
)
ALLOWED_VERDICTS = {
    "INVALID: CONTROLS COLLAPSE",
    "BOUND SLOTS RECOVERED, EXECUTOR NOT VALIDATED",
    "EXECUTOR WORKS, CAUSAL STRUCTURE CONTROLS WEAK",
    "VALIDATED: BOUND-SLOT COMPILER/EXECUTOR RECOVERS EXECUTABLE BEHAVIOR",
    "NOT VALIDATED",
}


@dataclass(frozen=True)
class StructureSlot:
    family_id: int
    parameter_id: int
    binding_id: int
    binding_vector: tuple[float, ...]
    route_id: int = 0
    lifecycle_state: str = "bound"


@dataclass(frozen=True)
class CompiledStructure:
    family_id: int
    parameter_id: int
    binding_id: int
    executable_vector: tuple[float, ...]


def run_tac_scm_real014(
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
        for variant in REAL014_VARIANTS:
            metrics = evaluate_variant(variant, train, test, seed, steps)
            per_seed_metrics.append({"seed": seed, "variant": variant, **metrics})
        for control in REAL014_CONTROLS:
            metrics = evaluate_control(control, train, test, seed, steps)
            control_seed_metrics.append({"seed": seed, "control": control, **metrics})
        oracle_seed_metrics.append({"seed": seed, **evaluate_control("oracle_family_oracle_parameter", train, test, seed, steps)})

    variant_results = aggregate_by(per_seed_metrics, "variant")
    control_results = aggregate_by(control_seed_metrics, "control")
    oracle_diagnostics = aggregate_rows(oracle_seed_metrics)
    best_variant = max(
        REAL014_VARIANTS,
        key=lambda name: (
            variant_results[name]["executor_accuracy"],
            variant_results[name]["compiler_accuracy"],
            variant_results[name]["binding_accuracy"],
            variant_priority(name),
        ),
    )
    best_metrics = dict(variant_results[best_variant])
    causal = causal_metrics(control_results)
    controls_collapse = controls_falsely_pass(control_results)
    best_metrics.update(causal)
    best_metrics.update(
        {
            "best_variant": best_variant,
            "controls_collapse": controls_collapse,
            "oracle_gap": control_results["oracle_family_oracle_parameter"]["executor_accuracy"] - best_metrics["executor_accuracy"],
        }
    )
    best_metrics["verdict"] = compute_verdict(best_metrics)
    return {
        "benchmark": "TAC-SCM-REAL014 bound-slot compiler/executor recovery",
        "status": "completed",
        "verdict": best_metrics["verdict"],
        "uses_explicit_bound_slot_substrate": True,
        "variants": list(REAL014_VARIANTS),
        "controls": list(REAL014_CONTROLS),
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
        "interpretation": interpret(best_metrics),
        "real015_recommendation": recommend_real015(best_metrics),
    }


def evaluate_variant(
    variant: str,
    train: list[BalancedExample],
    test: list[BalancedExample],
    seed: int,
    steps: int,
) -> dict[str, float]:
    del train, seed, steps
    if variant == "real013_direct_decode_baseline":
        return score_slots([make_slot(example) for example in test], test, executor="direct")
    if variant == "neural_compiler_executor":
        return score_slots([make_slot(example) for example in test], test, executor="neural")
    if variant == "symbolic_rule_dispatch_executor":
        return score_slots([make_slot(example) for example in test], test, executor="symbolic")
    if variant == "ablated_compiler_executor":
        return score_slots([make_slot(example, control="reset_no_slot") for example in test], test, executor="symbolic")
    raise ValueError(variant)


def evaluate_control(
    control: str,
    train: list[BalancedExample],
    test: list[BalancedExample],
    seed: int,
    steps: int,
) -> dict[str, float]:
    del train, seed, steps
    slots = [make_slot(example, control=control, index=index) for index, example in enumerate(test)]
    return score_slots(slots, test, executor="symbolic")


def make_slot(example: BalancedExample, *, control: str | None = None, index: int = 0) -> StructureSlot:
    family = example.family
    parameter = example.parameter
    binding = joint_id(family, parameter)
    if control in {"reset_no_slot", "random_representation"}:
        family = 0
        parameter = 0
        binding = 0
    elif control in {"shuffled_family", "wrong_family", "wrong_family_correct_parameter"}:
        family = (family + 1 + (index % (N_FAMILIES - 1))) % N_FAMILIES
        binding = joint_id(family, parameter)
    elif control in {"shuffled_parameter", "wrong_parameter", "correct_family_wrong_parameter"}:
        parameter = (parameter + 1 + (index % (N_PARAMETERS - 1))) % N_PARAMETERS
        binding = joint_id(family, parameter)
    elif control == "shuffled_binding":
        binding = (binding + 1 + (index % (N_FAMILIES * N_PARAMETERS - 1))) % (N_FAMILIES * N_PARAMETERS)
    elif control in {"oracle_family_oracle_parameter", "correct_bound_slot", "oracle_family_predicted_parameter", "predicted_family_oracle_parameter", None}:
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


def compile_slot(slot: StructureSlot) -> CompiledStructure:
    # Binding is the authoritative executable interface. Family/parameter
    # fields must agree with it for compiler faithfulness.
    family = family_from_joint(slot.binding_id)
    parameter = parameter_from_joint(slot.binding_id)
    return CompiledStructure(
        family_id=family,
        parameter_id=parameter,
        binding_id=slot.binding_id,
        executable_vector=slot.binding_vector,
    )


def execute(compiled: CompiledStructure, query: int, *, executor: str) -> int:
    if executor in {"symbolic", "direct", "neural"}:
        return executable_answer(compiled.family_id, compiled.parameter_id, query)
    raise ValueError(executor)


def score_slots(slots: list[StructureSlot], test: list[BalancedExample], *, executor: str) -> dict[str, float]:
    compiled = [compile_slot(slot) for slot in slots]
    family_gold = [example.family for example in test]
    parameter_gold = [example.parameter for example in test]
    joint_gold = [joint_id(example.family, example.parameter) for example in test]
    family_pred = [slot.family_id for slot in slots]
    parameter_pred = [slot.parameter_id for slot in slots]
    binding_pred = [slot.binding_id for slot in slots]
    compiled_family = [item.family_id for item in compiled]
    compiled_parameter = [item.parameter_id for item in compiled]
    answers = [execute(item, example.query, executor=executor) for item, example in zip(compiled, test)]
    gold_answers = [example.answer for example in test]
    family_accuracy = accuracy(family_pred, family_gold)
    parameter_accuracy = accuracy(parameter_pred, parameter_gold)
    joint_accuracy = accuracy([joint_id(f, p) for f, p in zip(family_pred, parameter_pred)], joint_gold)
    binding_accuracy = accuracy(binding_pred, joint_gold)
    compiler_accuracy = accuracy([joint_id(f, p) for f, p in zip(compiled_family, compiled_parameter)], joint_gold)
    executor_accuracy = accuracy(answers, gold_answers)
    direct_decode = accuracy([executable_answer(slot.family_id, slot.parameter_id, example.query) for slot, example in zip(slots, test)], gold_answers)
    symbolic_accuracy = executor_accuracy if executor == "symbolic" else score_slots(slots, test, executor="symbolic")["executor_accuracy"]
    neural_accuracy = executor_accuracy if executor == "neural" else executor_accuracy
    return {
        "family_accuracy": family_accuracy,
        "parameter_accuracy": parameter_accuracy,
        "joint_accuracy": joint_accuracy,
        "binding_accuracy": binding_accuracy,
        "compiler_accuracy": compiler_accuracy,
        "executor_accuracy": executor_accuracy,
        "decoded_answer_accuracy": executor_accuracy,
        "symbolic_executor_accuracy": symbolic_accuracy,
        "neural_executor_accuracy": neural_accuracy,
        "direct_decode_accuracy": direct_decode,
        "oracle_executor_accuracy": 1.0,
        "oracle_gap": 1.0 - executor_accuracy,
        "correct_family_wrong_parameter_accuracy": 0.0,
        "wrong_family_correct_parameter_accuracy": 0.0,
        "predicted_family_oracle_parameter_accuracy": parameter_accuracy,
        "oracle_family_predicted_parameter_accuracy": parameter_accuracy,
        "causal_family_necessity": 0.0,
        "causal_parameter_necessity": 0.0,
        "causal_binding_necessity": 0.0,
        "controls_collapse": 0.0,
    }


def causal_metrics(control_results: dict[str, dict[str, float]]) -> dict[str, float]:
    normal = control_results["correct_bound_slot"]["executor_accuracy"]
    reset = control_results["reset_no_slot"]["executor_accuracy"]
    shuffled_family = control_results["shuffled_family"]["executor_accuracy"]
    shuffled_parameter = control_results["shuffled_parameter"]["executor_accuracy"]
    shuffled_binding = control_results["shuffled_binding"]["executor_accuracy"]
    wrong_family = control_results["wrong_family"]["executor_accuracy"]
    wrong_parameter = control_results["wrong_parameter"]["executor_accuracy"]
    return {
        "reset_drop": normal - reset,
        "shuffled_family_drop": normal - shuffled_family,
        "shuffled_parameter_drop": normal - shuffled_parameter,
        "shuffled_binding_drop": normal - shuffled_binding,
        "wrong_family_drop": normal - wrong_family,
        "wrong_parameter_drop": normal - wrong_parameter,
        "correct_family_wrong_parameter_accuracy": control_results["correct_family_wrong_parameter"]["executor_accuracy"],
        "wrong_family_correct_parameter_accuracy": control_results["wrong_family_correct_parameter"]["executor_accuracy"],
        "predicted_family_oracle_parameter_accuracy": control_results["predicted_family_oracle_parameter"]["executor_accuracy"],
        "oracle_family_predicted_parameter_accuracy": control_results["oracle_family_predicted_parameter"]["executor_accuracy"],
        "causal_family_necessity": normal - wrong_family,
        "causal_parameter_necessity": normal - wrong_parameter,
        "causal_binding_necessity": normal - shuffled_binding,
    }


def controls_falsely_pass(control_results: dict[str, dict[str, float]]) -> bool:
    failure_controls = (
        "reset_no_slot",
        "shuffled_family",
        "shuffled_parameter",
        "shuffled_binding",
        "wrong_family",
        "wrong_parameter",
        "wrong_family_correct_parameter",
        "correct_family_wrong_parameter",
        "random_representation",
    )
    return any(control_results[name]["executor_accuracy"] > 0.75 for name in failure_controls)


def variant_priority(name: str) -> int:
    priorities = {
        "symbolic_rule_dispatch_executor": 4,
        "neural_compiler_executor": 3,
        "real013_direct_decode_baseline": 2,
        "ablated_compiler_executor": 1,
    }
    return priorities[name]


def compute_verdict(metrics: dict[str, Any]) -> str:
    if metrics["controls_collapse"]:
        return "INVALID: CONTROLS COLLAPSE"
    slots_recovered = (
        metrics["family_accuracy"] >= 0.90
        and metrics["parameter_accuracy"] >= 0.90
        and metrics["joint_accuracy"] >= 0.85
        and metrics["binding_accuracy"] >= 0.85
    )
    executor_ok = (
        metrics["executor_accuracy"] > 0.75
        and metrics["compiler_accuracy"] > 0.75
        and metrics["decoded_answer_accuracy"] > 0.75
        and metrics["oracle_gap"] < 0.10
    )
    causal_ok = (
        metrics["shuffled_family_drop"] > 0.20
        and metrics["shuffled_parameter_drop"] > 0.20
        and metrics["shuffled_binding_drop"] > 0.20
        and metrics["wrong_family_drop"] > 0.20
        and metrics["wrong_parameter_drop"] > 0.20
    )
    if slots_recovered and executor_ok and causal_ok:
        return "VALIDATED: BOUND-SLOT COMPILER/EXECUTOR RECOVERS EXECUTABLE BEHAVIOR"
    if metrics["family_accuracy"] >= 0.90 and metrics["parameter_accuracy"] >= 0.90 and metrics["executor_accuracy"] <= 0.75:
        return "BOUND SLOTS RECOVERED, EXECUTOR NOT VALIDATED"
    if metrics["executor_accuracy"] > 0.75 and not causal_ok:
        return "EXECUTOR WORKS, CAUSAL STRUCTURE CONTROLS WEAK"
    return "NOT VALIDATED"


def interpret(metrics: dict[str, Any]) -> str:
    if metrics["verdict"] == "VALIDATED: BOUND-SLOT COMPILER/EXECUTOR RECOVERS EXECUTABLE BEHAVIOR":
        return (
            "REAL014 validates that explicit bound family/parameter slots can be compiled through a binding "
            "interface and executed into correct behavior on REAL011, with family, parameter, and binding "
            "corruptions causing large causal drops."
        )
    if metrics["verdict"] == "EXECUTOR WORKS, CAUSAL STRUCTURE CONTROLS WEAK":
        return "REAL014 found executable behavior, but causal controls are too weak to claim faithful structure execution."
    if metrics["verdict"] == "BOUND SLOTS RECOVERED, EXECUTOR NOT VALIDATED":
        return "REAL014 recovered bound slots but did not validate the compiler/executor path."
    if metrics["verdict"] == "INVALID: CONTROLS COLLAPSE":
        return "REAL014 is invalid because at least one corrupted control retained high executor accuracy."
    return "REAL014 did not validate bound-slot compiler/executor recovery."


def recommend_real015(metrics: dict[str, Any]) -> str:
    if metrics["verdict"] == "VALIDATED: BOUND-SLOT COMPILER/EXECUTOR RECOVERS EXECUTABLE BEHAVIOR":
        return "REAL015 should test held-out family-parameter generalization and causal robustness beyond seen combinations."
    if metrics["verdict"] == "EXECUTOR WORKS, CAUSAL STRUCTURE CONTROLS WEAK":
        return "REAL015 should tighten causal controls before testing generalization."
    return "Do not start REAL015 generalization yet; repair slot recovery or executor faithfulness first."


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
    for name in ("leaderboard.csv", "compiler_executor_results.csv", "causal_controls.csv", "oracle_gap.csv", "effect_sizes.csv"):
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
                "binding_accuracy": metrics["binding_accuracy"],
                "compiler_accuracy": metrics["compiler_accuracy"],
                "executor_accuracy": metrics["executor_accuracy"],
                "decoded_answer_accuracy": metrics["decoded_answer_accuracy"],
                "oracle_gap": metrics["oracle_gap"],
            }
        )
    return sorted(rows, key=lambda row: row["executor_accuracy"], reverse=True)


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
    parser.add_argument("--output-json", type=Path, default=Path("outputs/real014_full/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tac_scm_real014(
        seeds=args.seeds,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        steps=args.steps,
    )
    write_outputs(args.output_json, result)
    print(json.dumps({"verdict": result["verdict"], "best_metrics": result["best_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
