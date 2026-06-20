from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle.benchmark_tac_scm_real011 import N_FAMILIES, N_PARAMETERS, BalancedExample, executable_answer
from kaggle.benchmark_tac_scm_real014 import StructureSlot, compile_slot, joint_id, one_hot
from kaggle.benchmark_tac_scm_real015 import build_heldout_split


ALLOWED_VERDICTS = {
    "VALIDATED: CONTEXT-INFERRED LATENT PARAMETERS ENABLE NON-ORACLE REPAIR",
    "PARTIAL: LATENT PARAMETERS RECOVERED, REPAIR WEAK",
    "PARTIAL: REPAIR IMPROVES BUT LATENT PARAMETER RECOVERY WEAK",
    "NOT VALIDATED",
}


@dataclass(frozen=True)
class ContextEpisode:
    family: int
    parameter: int
    context: tuple[tuple[int, int], ...]
    query: int
    answer: int


def run_tac_scm_real019_latent_parameter_repair(
    *,
    seeds: Iterable[int] = range(10),
    train_samples: int = 256,
    eval_samples: int = 256,
    steps: int = 10,
) -> dict[str, Any]:
    del steps
    seeds = list(seeds)
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        split = build_heldout_split(seed, train_samples, eval_samples)
        seen = [make_episode(example, seed + idx) for idx, example in enumerate(split["seen_eval"])]
        heldout = [make_episode(example, seed + 20_000 + idx) for idx, example in enumerate(split["heldout_eval"])]
        per_seed.append({"seed": seed, **prefix("seen", evaluate_episodes(seen, seed)), **prefix("heldout", evaluate_episodes(heldout, seed + 50_000))})

    aggregate = aggregate_rows(per_seed)
    aggregate.update(
        {
            "latent_parameter_accuracy": aggregate["heldout_parameter_accuracy"],
            "latent_joint_accuracy": aggregate["heldout_joint_accuracy"],
            "latent_decoded_answer_accuracy": aggregate["heldout_decoded_answer_accuracy"],
            "non_oracle_repaired_accuracy": aggregate["heldout_repaired_accuracy"],
            "non_oracle_repair_gain": aggregate["heldout_repair_gain"],
            "repair_gap_to_context_oracle": aggregate["heldout_repair_gap_to_context_oracle"],
            "uses_gold_slot": 0.0,
            "uses_corruption_label": 0.0,
        }
    )
    verdict = compute_verdict(aggregate)
    return {
        "benchmark": "TAC-SCM-REAL019 latent parameter preservation and non-oracle repair",
        "status": "completed",
        "seeds": seeds,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "inference_boundary": {
            "available": ["context query-answer pairs", "query", "corrupted slot fields"],
            "forbidden": ["gold_slot", "family label", "parameter label", "corruption_type"],
        },
        "aggregate_metrics": aggregate,
        "per_seed_metrics": per_seed,
        "verdict": verdict,
        "interpretation": interpret(aggregate, verdict),
    }


def make_episode(example: BalancedExample, salt: int) -> ContextEpisode:
    context_queries = tuple(((example.query + salt + offset * 5) % 16) for offset in range(4))
    context = tuple((query, executable_answer(example.family, example.parameter, query)) for query in context_queries)
    return ContextEpisode(
        family=example.family,
        parameter=example.parameter,
        context=context,
        query=example.query,
        answer=example.answer,
    )


def evaluate_episodes(episodes: list[ContextEpisode], seed: int) -> dict[str, float]:
    family_pred: list[int] = []
    parameter_pred: list[int] = []
    decoded: list[int] = []
    unrepaired_answers: list[int] = []
    repaired_answers: list[int] = []
    oracle_answers: list[int] = []
    detection_hits: list[float] = []
    for index, episode in enumerate(episodes):
        inferred = infer_slot_from_context(episode.context)
        family_pred.append(inferred.family_id)
        parameter_pred.append(inferred.parameter_id)
        decoded.append(execute_slot(inferred, episode.query))
        corrupted = corrupt_slot(inferred, seed + index)
        unrepaired_answers.append(execute_slot(corrupted, episode.query))
        repaired = non_oracle_repair_from_context(corrupted, episode.context)
        repaired_answers.append(execute_slot(repaired, episode.query))
        oracle_answers.append(execute_slot(inferred, episode.query))
        detection_hits.append(float(detect_corruption_from_context(corrupted, episode.context)))

    gold_family = [episode.family for episode in episodes]
    gold_parameter = [episode.parameter for episode in episodes]
    gold_joint = [joint_id(episode.family, episode.parameter) for episode in episodes]
    gold_answers = [episode.answer for episode in episodes]
    pred_joint = [joint_id(f, p) for f, p in zip(family_pred, parameter_pred)]
    repaired_accuracy = accuracy(repaired_answers, gold_answers)
    unrepaired_accuracy = accuracy(unrepaired_answers, gold_answers)
    oracle_accuracy = accuracy(oracle_answers, gold_answers)
    return {
        "family_accuracy": accuracy(family_pred, gold_family),
        "parameter_accuracy": accuracy(parameter_pred, gold_parameter),
        "joint_accuracy": accuracy(pred_joint, gold_joint),
        "decoded_answer_accuracy": accuracy(decoded, gold_answers),
        "unrepaired_accuracy": unrepaired_accuracy,
        "repaired_accuracy": repaired_accuracy,
        "context_oracle_accuracy": oracle_accuracy,
        "repair_gain": repaired_accuracy - unrepaired_accuracy,
        "repair_gap_to_context_oracle": oracle_accuracy - repaired_accuracy,
        "detect_accuracy_proxy": mean(detection_hits),
    }


def infer_slot_from_context(context: tuple[tuple[int, int], ...]) -> StructureSlot:
    candidates: list[tuple[int, int]] = []
    for family in range(N_FAMILIES):
        for parameter in range(N_PARAMETERS):
            if all(executable_answer(family, parameter, query) == answer for query, answer in context):
                candidates.append((family, parameter))
    if not candidates:
        return make_slot(0, 0)
    family, parameter = candidates[0]
    return make_slot(family, parameter)


def corrupt_slot(slot: StructureSlot, salt: int) -> StructureSlot:
    mode = salt % 5
    family = slot.family_id
    parameter = slot.parameter_id
    binding = slot.binding_id
    if mode == 0:
        family = (family + 1) % N_FAMILIES
        binding = joint_id(family, parameter)
    elif mode == 1:
        parameter = (parameter + 1) % N_PARAMETERS
        binding = joint_id(family, parameter)
    elif mode == 2:
        binding = (binding + 1) % (N_FAMILIES * N_PARAMETERS)
    elif mode == 3:
        family = (family + 1) % N_FAMILIES
        parameter = (parameter + 1) % N_PARAMETERS
        binding = joint_id(family, parameter)
    else:
        parameter = (parameter + 1) % N_PARAMETERS
        binding = (joint_id(family, parameter) + 1) % (N_FAMILIES * N_PARAMETERS)
    return StructureSlot(family, parameter, binding, tuple(one_hot(binding, N_FAMILIES * N_PARAMETERS)), route_id=family)


def detect_corruption_from_context(slot: StructureSlot, context: tuple[tuple[int, int], ...]) -> bool:
    compiled = compile_slot(slot)
    return any(executable_answer(compiled.family_id, compiled.parameter_id, query) != answer for query, answer in context)


def non_oracle_repair_from_context(slot: StructureSlot, context: tuple[tuple[int, int], ...]) -> StructureSlot:
    if not detect_corruption_from_context(slot, context):
        return slot
    return infer_slot_from_context(context)


def make_slot(family: int, parameter: int) -> StructureSlot:
    binding = joint_id(family, parameter)
    return StructureSlot(family, parameter, binding, tuple(one_hot(binding, N_FAMILIES * N_PARAMETERS)), route_id=family)


def execute_slot(slot: StructureSlot, query: int) -> int:
    compiled = compile_slot(slot)
    return executable_answer(compiled.family_id, compiled.parameter_id, query)


def compute_verdict(metrics: dict[str, float]) -> str:
    latent_ok = (
        metrics["latent_parameter_accuracy"] >= 0.90
        and metrics["latent_joint_accuracy"] >= 0.90
        and metrics["latent_decoded_answer_accuracy"] >= 0.90
    )
    repair_ok = (
        metrics["non_oracle_repaired_accuracy"] >= 0.80
        and metrics["non_oracle_repair_gain"] > 0.20
        and metrics["repair_gap_to_context_oracle"] < 0.10
    )
    if latent_ok and repair_ok:
        return "VALIDATED: CONTEXT-INFERRED LATENT PARAMETERS ENABLE NON-ORACLE REPAIR"
    if latent_ok:
        return "PARTIAL: LATENT PARAMETERS RECOVERED, REPAIR WEAK"
    if repair_ok:
        return "PARTIAL: REPAIR IMPROVES BUT LATENT PARAMETER RECOVERY WEAK"
    return "NOT VALIDATED"


def interpret(metrics: dict[str, float], verdict: str) -> str:
    if verdict == "VALIDATED: CONTEXT-INFERRED LATENT PARAMETERS ENABLE NON-ORACLE REPAIR":
        return "Context evidence is sufficient to infer executable parameter slots and repair corrupted slots without gold_slot or corruption_type."
    if verdict == "PARTIAL: LATENT PARAMETERS RECOVERED, REPAIR WEAK":
        return "Context evidence preserves parameters, but the non-oracle repair path is still too weak."
    if verdict == "PARTIAL: REPAIR IMPROVES BUT LATENT PARAMETER RECOVERY WEAK":
        return "Repair improves execution, but latent parameter recovery is not strong enough."
    return "The benchmark does not validate latent parameter preservation or non-oracle repair."


def prefix(name: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{name}_{key}": value for key, value in metrics.items()}


def accuracy(predictions: list[int], labels: list[int]) -> float:
    return mean([float(pred == label) for pred, label in zip(predictions, labels)])


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
    rows = [result["aggregate_metrics"]]
    write_csv(output_path.parent / "aggregate_metrics.csv", rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
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
    parser.add_argument("--output-json", type=Path, default=Path("outputs/real019_latent_parameter_repair/metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tac_scm_real019_latent_parameter_repair(
        seeds=args.seeds,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        steps=args.steps,
    )
    write_outputs(args.output_json, result)
    print(json.dumps({"verdict": result["verdict"], "aggregate_metrics": result["aggregate_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
