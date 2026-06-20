from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

N_FAMILIES = 4
N_PARAMETERS = 4
N_QUERIES = 16
ANSWER_SIZE = 16

CORRUPTION_TYPES = (
    "clean",
    "family_only",
    "parameter_only",
    "binding_only",
    "route_only",
    "family_binding",
    "parameter_binding",
    "full",
)

AUDIT_VARIANTS = (
    "unrepaired",
    "blind_consistency_repair",
    "oracle_repair",
    "random_repair",
    "wrong_repair",
)


@dataclass(frozen=True)
class AuditExample:
    family: int
    parameter: int
    query: int
    answer: int


@dataclass(frozen=True)
class AuditSlot:
    family_id: int
    parameter_id: int
    binding_id: int
    route_id: int


def joint_id(family: int, parameter: int) -> int:
    return family * N_PARAMETERS + parameter


def split_binding(binding_id: int) -> tuple[int, int]:
    return binding_id // N_PARAMETERS, binding_id % N_PARAMETERS


def executable_answer(family: int, parameter: int, query: int) -> int:
    return (query + family * N_PARAMETERS + parameter) % ANSWER_SIZE


def make_gold_slot(example: AuditExample) -> AuditSlot:
    return AuditSlot(
        family_id=example.family,
        parameter_id=example.parameter,
        binding_id=joint_id(example.family, example.parameter),
        route_id=example.family,
    )


def make_slot(family: int, parameter: int, *, route_id: int | None = None, binding_id: int | None = None) -> AuditSlot:
    return AuditSlot(
        family_id=family,
        parameter_id=parameter,
        binding_id=joint_id(family, parameter) if binding_id is None else binding_id,
        route_id=family if route_id is None else route_id,
    )


def generate_examples(n_samples: int, seed: int, *, split_offset: int = 0) -> list[AuditExample]:
    rng = random.Random(seed + split_offset)
    grid = [
        (family, parameter, query)
        for query in range(N_QUERIES)
        for family in range(N_FAMILIES)
        for parameter in range(N_PARAMETERS)
    ]
    rng.shuffle(grid)
    rows: list[AuditExample] = []
    for family, parameter, query in (grid * ((n_samples // len(grid)) + 1))[:n_samples]:
        rows.append(AuditExample(family, parameter, query, executable_answer(family, parameter, query)))
    return rows


def corrupt_slot(slot: AuditSlot, corruption_type: str, rng: random.Random) -> AuditSlot:
    family = slot.family_id
    parameter = slot.parameter_id
    binding = slot.binding_id
    route = slot.route_id

    if corruption_type == "clean":
        pass
    elif corruption_type == "family_only":
        family = _different_int(family, N_FAMILIES, rng)
    elif corruption_type == "parameter_only":
        parameter = _different_int(parameter, N_PARAMETERS, rng)
    elif corruption_type == "binding_only":
        binding = _different_int(binding, N_FAMILIES * N_PARAMETERS, rng)
    elif corruption_type == "route_only":
        route = _different_int(route, N_FAMILIES, rng)
    elif corruption_type == "family_binding":
        family = _different_int(family, N_FAMILIES, rng)
        binding = _different_int(binding, N_FAMILIES * N_PARAMETERS, rng)
    elif corruption_type == "parameter_binding":
        parameter = _different_int(parameter, N_PARAMETERS, rng)
        binding = _different_int(binding, N_FAMILIES * N_PARAMETERS, rng)
    elif corruption_type == "full":
        family = _different_int(family, N_FAMILIES, rng)
        parameter = _different_int(parameter, N_PARAMETERS, rng)
        binding = _different_int(binding, N_FAMILIES * N_PARAMETERS, rng)
        route = _different_int(route, N_FAMILIES, rng)
    else:
        raise ValueError(corruption_type)

    return AuditSlot(family, parameter, binding, route)


def _different_int(value: int, modulo: int, rng: random.Random) -> int:
    if modulo <= 1:
        return value
    candidate = rng.randrange(modulo - 1)
    if candidate >= value:
        candidate += 1
    return candidate


def verify_slot(slot: AuditSlot) -> tuple[bool, str]:
    """Blind consistency verifier.

    This function deliberately receives only the candidate slot. It must not
    receive corruption labels, gold slots, clean copies, or examples.
    """
    binding_family, binding_parameter = split_binding(slot.binding_id)
    route_ok = slot.route_id == slot.family_id
    family_ok = binding_family == slot.family_id
    parameter_ok = binding_parameter == slot.parameter_id

    failed = []
    if not family_ok:
        failed.append("family")
    if not parameter_ok:
        failed.append("parameter")
    if not route_ok:
        failed.append("route")

    if not failed:
        return False, "clean"
    return True, "+".join(failed)


def repair_slot_blind(slot: AuditSlot) -> AuditSlot:
    """Repair using only internal consistency, never the gold slot.

    The repair is intentionally conservative. If family/parameter agree with the
    route but binding disagrees, it recomputes binding. If binding and route agree
    against one field, it repairs that field. If all signals conflict, it returns
    the original slot rather than hallucinating a clean object.
    """
    binding_family, binding_parameter = split_binding(slot.binding_id)

    # Binding and route agree on family, so repair the family field if needed.
    if binding_family == slot.route_id and slot.family_id != binding_family:
        return make_slot(binding_family, slot.parameter_id, route_id=binding_family, binding_id=slot.binding_id)

    # Family/parameter fields and route agree; binding alone is inconsistent.
    if slot.route_id == slot.family_id and joint_id(slot.family_id, slot.parameter_id) != slot.binding_id:
        return make_slot(slot.family_id, slot.parameter_id, route_id=slot.route_id)

    # Family and binding agree, but parameter field is inconsistent.
    if binding_family == slot.family_id and binding_parameter != slot.parameter_id:
        return make_slot(slot.family_id, binding_parameter, route_id=slot.family_id, binding_id=slot.binding_id)

    # Route is the only inconsistent field.
    if slot.family_id == binding_family and slot.parameter_id == binding_parameter and slot.route_id != slot.family_id:
        return make_slot(slot.family_id, slot.parameter_id, route_id=slot.family_id, binding_id=slot.binding_id)

    return slot


def oracle_repair(_slot: AuditSlot, gold_slot: AuditSlot) -> AuditSlot:
    return gold_slot


def random_repair(index: int) -> AuditSlot:
    family = index % N_FAMILIES
    parameter = (index * 3 + 1) % N_PARAMETERS
    return make_slot(family, parameter)


def wrong_repair(gold_slot: AuditSlot) -> AuditSlot:
    return make_slot((gold_slot.family_id + 1) % N_FAMILIES, (gold_slot.parameter_id + 1) % N_PARAMETERS)


def execute_slot(slot: AuditSlot, example: AuditExample) -> int:
    return executable_answer(slot.family_id, slot.parameter_id, example.query)


def run_tac_scm_real017_audit(
    *,
    seeds: Iterable[int] = range(10),
    train_samples: int = 256,
    eval_samples: int = 256,
) -> dict[str, Any]:
    del train_samples  # Kept for CLI compatibility; this audit has no training path.
    per_seed: list[dict[str, Any]] = []
    per_corruption: list[dict[str, Any]] = []

    for seed in seeds:
        examples = generate_examples(eval_samples, seed, split_offset=20_000)
        cases = build_cases(examples, seed)
        for variant in AUDIT_VARIANTS:
            score = score_cases(cases, variant=variant)
            per_seed.append({"seed": seed, "variant": variant, **score})
        for corruption_type in CORRUPTION_TYPES:
            subset = [case for case in cases if case["corruption_type"] == corruption_type]
            score = score_cases(subset, variant="blind_consistency_repair")
            per_corruption.append({"seed": seed, "corruption_type": corruption_type, **score})

    variant_results = aggregate(per_seed, "variant")
    corruption_results = aggregate(per_corruption, "corruption_type")
    verdict = compute_verdict(variant_results)
    return {
        "benchmark": "TAC-SCM-REAL017-AUDIT blind verifier-guided structure refinement",
        "status": "passed" if verdict.startswith("VALID") else "failed",
        "verdict": verdict,
        "variants": list(AUDIT_VARIANTS),
        "corruption_types": list(CORRUPTION_TYPES),
        "leakage_guardrails": {
            "verifier_receives_corruption_type": False,
            "blind_repair_receives_gold_slot": False,
            "oracle_repair_separate_variant": True,
            "no_training_labels_used": True,
        },
        "variant_results": variant_results,
        "corruption_results": corruption_results,
        "per_seed": per_seed,
    }


def build_cases(examples: list[AuditExample], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 100_000)
    cases: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        corruption_type = CORRUPTION_TYPES[(index + seed) % len(CORRUPTION_TYPES)]
        gold_slot = make_gold_slot(example)
        corrupted_slot = corrupt_slot(gold_slot, corruption_type, rng)
        # gold_slot is stored for scoring and the explicit oracle variant only.
        # It must never be passed into verify_slot or repair_slot_blind.
        cases.append({
            "index": index,
            "example": example,
            "corruption_type": corruption_type,
            "corrupted_slot": corrupted_slot,
            "gold_slot": gold_slot,
        })
    return cases


def score_cases(cases: list[dict[str, Any]], *, variant: str) -> dict[str, float]:
    if not cases:
        return empty_score()
    detect_hits: list[float] = []
    repair_hits: list[float] = []
    answer_hits: list[float] = []
    false_repairs = 0

    for case in cases:
        index = case["index"]
        example = case["example"]
        corruption_type = case["corruption_type"]
        corrupted = case["corrupted_slot"]
        gold = case["gold_slot"]
        is_corrupt = corruption_type != "clean"

        detected, _diagnosis = verify_slot(corrupted)
        if variant == "unrepaired":
            repaired = corrupted
        elif variant == "blind_consistency_repair":
            repaired = repair_slot_blind(corrupted)
        elif variant == "oracle_repair":
            repaired = oracle_repair(corrupted, gold)
        elif variant == "random_repair":
            repaired = random_repair(index)
        elif variant == "wrong_repair":
            repaired = wrong_repair(gold)
        else:
            raise ValueError(variant)

        if not is_corrupt and repaired != corrupted:
            false_repairs += 1
        detect_hits.append(float(detected == is_corrupt))
        repair_hits.append(float(repaired == gold))
        answer_hits.append(float(execute_slot(repaired, example) == example.answer))

    return {
        "detect_accuracy": mean(detect_hits),
        "repair_accuracy": mean(repair_hits),
        "executor_accuracy": mean(answer_hits),
        "false_repair_rate": false_repairs / len(cases),
    }


def empty_score() -> dict[str, float]:
    return {
        "detect_accuracy": 0.0,
        "repair_accuracy": 0.0,
        "executor_accuracy": 0.0,
        "false_repair_rate": 0.0,
    }


def aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    result: dict[str, dict[str, float]] = {}
    for group_key, group_rows in grouped.items():
        metrics: dict[str, float] = {}
        for metric, value in group_rows[0].items():
            if isinstance(value, (int, float)) and metric != "seed":
                values = [float(row[metric]) for row in group_rows]
                metrics[metric] = mean(values)
                metrics[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        result[group_key] = metrics
    return result


def compute_verdict(variant_results: dict[str, dict[str, float]]) -> str:
    blind = variant_results["blind_consistency_repair"]
    unrepaired = variant_results["unrepaired"]
    oracle = variant_results["oracle_repair"]
    random = variant_results["random_repair"]
    wrong = variant_results["wrong_repair"]

    if oracle["executor_accuracy"] < 0.99:
        return "INVALID: ORACLE CEILING BROKEN"
    if blind["executor_accuracy"] <= unrepaired["executor_accuracy"] + 0.05:
        return "NOT VALIDATED: BLIND REPAIR DOES NOT IMPROVE UNREPAIRED"
    if blind["executor_accuracy"] <= random["executor_accuracy"] or blind["executor_accuracy"] <= wrong["executor_accuracy"]:
        return "NOT VALIDATED: CONTROLS MATCH OR BEAT BLIND REPAIR"
    if blind["repair_accuracy"] >= 0.99 and blind["detect_accuracy"] >= 0.99:
        return "SUSPICIOUS: BLIND AUDIT IS PERFECT; ADD HARDER CORRUPTIONS"
    return "VALID AUDIT SCAFFOLD: BLIND REPAIR IMPROVES BUT IS NOT PERFECT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tac_scm_real017_audit(
        seeds=args.seeds,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": result["status"], "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
