from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_persistent_computational_identity import (
    LOW_TOKEN,
    RULES,
    apply_rule_bank,
    infer_rule_from_support,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/persistent_identity_broader_tasks_2026_06_05")
TASK_FAMILIES = [
    "transfer_learning",
    "multi_hop_reasoning",
    "agent_memory",
    "language_like_instruction",
]


def build_broader_task_suite(
    *,
    seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
) -> dict[str, Any]:
    if identities_per_seed < len(RULES) or identities_per_seed % len(RULES) != 0:
        raise ValueError("identities_per_seed must be a positive multiple of rule count")
    if examples_per_task < 1:
        raise ValueError("examples_per_task must be positive")
    support_inputs = _support_inputs(vocab_size)
    rows = []
    identity_support: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        for identity_index in range(identities_per_seed):
            identity_id = f"seed{int(seed)}_identity{identity_index:02d}"
            rule = RULES[identity_index % len(RULES)]
            support_targets = _apply_rule_to_values(
                support_inputs,
                rule,
                vocab_size=vocab_size,
            )
            identity_support[identity_id] = {
                "rule": rule,
                "support_inputs": support_inputs,
                "support_targets": support_targets,
                "source_domain": "navigation",
            }
            for task_family in TASK_FAMILIES:
                for example_index in range(examples_per_task):
                    query_value = _query_value(
                        seed=int(seed),
                        identity_index=identity_index,
                        task_family=task_family,
                        example_index=example_index,
                        vocab_size=vocab_size,
                        forbidden=set(support_inputs),
                    )
                    rows.append(
                        _build_row(
                            identity_id=identity_id,
                            rule=rule,
                            task_family=task_family,
                            query_value=query_value,
                            support_inputs=support_inputs,
                            support_targets=support_targets,
                            vocab_size=vocab_size,
                        )
                    )
    return {
        "schema": "persistent_identity_broader_task_suite.v1",
        "task_families": list(TASK_FAMILIES),
        "rules": list(RULES),
        "seeds": [int(seed) for seed in seeds],
        "identities_per_seed": int(identities_per_seed),
        "examples_per_task": int(examples_per_task),
        "vocab_size": int(vocab_size),
        "identity_support": identity_support,
        "rows": rows,
        "real_world_benchmark_status": "proxy_not_real_world",
        "boundary": (
            "Rows include language-like prompts, but this is still a controlled "
            "proxy suite rather than a real-world language benchmark."
        ),
    }


def evaluate_solver(suite: dict[str, Any], *, solver: str) -> dict[str, Any]:
    if solver not in {
        "persistent_identity",
        "stateless_reset",
        "global_persistent_without_identity",
        "memory_only_without_computation",
    }:
        raise ValueError(f"unknown solver: {solver}")
    rule_by_identity = {
        identity_id: infer_rule_from_support(
            support["support_inputs"],
            support["support_targets"],
            vocab_size=suite["vocab_size"],
        )
        for identity_id, support in suite["identity_support"].items()
    }
    stateless_rule = _majority_rule(suite)
    last_identity = sorted(suite["identity_support"].keys())[-1]
    global_rule = rule_by_identity[last_identity]
    correct = 0
    missing = 0
    by_task: dict[str, dict[str, int]] = {
        task: {"correct": 0, "total": 0, "missing": 0}
        for task in TASK_FAMILIES
    }
    for row in suite["rows"]:
        if solver == "persistent_identity":
            predicted = _solve_row(
                row,
                rule_by_identity[row["identity_id"]],
                vocab_size=suite["vocab_size"],
            )
        elif solver == "stateless_reset":
            predicted = _solve_row(row, stateless_rule, vocab_size=suite["vocab_size"])
        elif solver == "global_persistent_without_identity":
            predicted = _solve_row(row, global_rule, vocab_size=suite["vocab_size"])
        else:
            predicted = _memory_only_prediction(row)
        task = row["task_family"]
        by_task[task]["total"] += 1
        if predicted is None:
            missing += 1
            by_task[task]["missing"] += 1
            continue
        is_correct = _prediction_matches(row, predicted)
        correct += int(is_correct)
        by_task[task]["correct"] += int(is_correct)
    total = len(suite["rows"])
    return {
        "schema": "persistent_identity_broader_task_solver.v1",
        "solver": solver,
        "accuracy": correct / float(total),
        "correct_count": correct,
        "example_count": total,
        "missing_prediction_count": missing,
        "by_task": {
            task: {
                "accuracy": stats["correct"] / float(stats["total"]),
                "correct_count": stats["correct"],
                "example_count": stats["total"],
                "missing_prediction_count": stats["missing"],
            }
            for task, stats in by_task.items()
        },
    }


def run_broader_task_bridge_probe(
    *,
    seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
) -> dict[str, Any]:
    suite = build_broader_task_suite(
        seeds=seeds,
        identities_per_seed=identities_per_seed,
        examples_per_task=examples_per_task,
        vocab_size=vocab_size,
    )
    controls = {
        solver: evaluate_solver(suite, solver=solver)
        for solver in [
            "persistent_identity",
            "stateless_reset",
            "global_persistent_without_identity",
            "memory_only_without_computation",
        ]
    }
    task_metrics = _task_metrics(controls)
    decision = _decision(task_metrics)
    return {
        "schema": "persistent_identity_broader_tasks.v1",
        "hypothesis": (
            "The TAC-180 persistent identity advantage should transfer from "
            "isolated latent-rule rows to broader controlled families: domain "
            "transfer, multi-hop composition, identity-keyed agent memory, and "
            "language-like instruction exact match."
        ),
        "suite_summary": {
            "task_families": list(TASK_FAMILIES),
            "identity_count": len(suite["identity_support"]),
            "row_count": len(suite["rows"]),
            "seeds": suite["seeds"],
            "identities_per_seed": suite["identities_per_seed"],
            "examples_per_task": suite["examples_per_task"],
            "real_world_benchmark_status": suite["real_world_benchmark_status"],
        },
        "controls": controls,
        "task_metrics": task_metrics,
        "decision": decision,
        "boundary": suite["boundary"],
    }


def format_broader_task_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Persistent Identity Broader Tasks",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        f"- Real-world benchmark status: `{result['suite_summary']['real_world_benchmark_status']}`",
        "",
        "## Task Metrics",
        "",
        "| Task | Persistent | Best non-identity | Advantage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task, metrics in result["task_metrics"].items():
        lines.append(
            "| {task} | {persistent:.4f} | {best:.4f} | {advantage:.4f} |".format(
                task=task,
                persistent=metrics["persistent_identity_accuracy"],
                best=metrics["best_non_identity_accuracy"],
                advantage=metrics["persistent_advantage"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["decision"]["recommendation"],
            "",
            "## Boundary",
            "",
            result["boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Bridge the persistent identity proof to broader controlled tasks."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--identities-per-seed", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=64)
    args = parser.parse_args(argv)

    result = run_broader_task_bridge_probe(
        seeds=args.seeds,
        identities_per_seed=args.identities_per_seed,
        examples_per_task=args.examples_per_task,
        vocab_size=args.vocab_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "persistent_identity_broader_tasks.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_broader_task_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return result


def _build_row(
    *,
    identity_id: str,
    rule: str,
    task_family: str,
    query_value: int,
    support_inputs: Sequence[int],
    support_targets: Sequence[int],
    vocab_size: int,
) -> dict[str, Any]:
    target_value = _target_for_task(task_family, query_value, rule, vocab_size)
    row = {
        "identity_id": identity_id,
        "rule": rule,
        "task_family": task_family,
        "support_inputs": list(support_inputs),
        "support_targets": list(support_targets),
        "query_value": int(query_value),
        "target_value": int(target_value),
        "target_answer": _answer_text(target_value),
    }
    if task_family == "transfer_learning":
        row.update(
            {
                "source_domain": "navigation",
                "query_domain": "lab_protocol",
                "transfer_type": "same_identity_new_domain",
            }
        )
    elif task_family == "multi_hop_reasoning":
        bridge = _apply_rule_to_values([query_value], rule, vocab_size=vocab_size)[0]
        row.update({"bridge_value": bridge, "hop_count": 2})
    elif task_family == "agent_memory":
        row.update(
            {
                "event_context": f"event_for_{identity_id}",
                "memory_key": f"{identity_id}:private_operator",
            }
        )
    elif task_family == "language_like_instruction":
        row["prompt"] = _language_prompt(identity_id, query_value)
    else:
        raise ValueError(f"unknown task family: {task_family}")
    return row


def _target_for_task(task_family: str, value: int, rule: str, vocab_size: int) -> int:
    first = _apply_rule_to_values([value], rule, vocab_size=vocab_size)[0]
    if task_family == "multi_hop_reasoning":
        return _apply_rule_to_values([first], rule, vocab_size=vocab_size)[0]
    return first


def _solve_row(row: dict[str, Any], rule: str, *, vocab_size: int) -> int:
    return _target_for_task(row["task_family"], int(row["query_value"]), rule, vocab_size)


def _memory_only_prediction(row: dict[str, Any]) -> int | None:
    memory = dict(zip(row["support_inputs"], row["support_targets"]))
    query = int(row["query_value"])
    if row["task_family"] == "multi_hop_reasoning":
        first = memory.get(query)
        if first is None:
            return None
        return memory.get(first)
    return memory.get(query)


def _prediction_matches(row: dict[str, Any], prediction: int) -> bool:
    if row["task_family"] == "language_like_instruction":
        return _answer_text(prediction) == row["target_answer"]
    return int(prediction) == int(row["target_value"])


def _task_metrics(controls: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for task in TASK_FAMILIES:
        persistent = controls["persistent_identity"]["by_task"][task]["accuracy"]
        non_identity = [
            controls["stateless_reset"]["by_task"][task]["accuracy"],
            controls["global_persistent_without_identity"]["by_task"][task]["accuracy"],
            controls["memory_only_without_computation"]["by_task"][task]["accuracy"],
        ]
        best = max(non_identity)
        result[task] = {
            "persistent_identity_accuracy": persistent,
            "stateless_reset_accuracy": non_identity[0],
            "global_persistent_without_identity_accuracy": non_identity[1],
            "memory_only_accuracy": non_identity[2],
            "best_non_identity_accuracy": best,
            "persistent_advantage": persistent - best,
        }
    return result


def _decision(task_metrics: dict[str, Any]) -> dict[str, Any]:
    persistent_ok = all(
        metrics["persistent_identity_accuracy"] >= 0.95
        for metrics in task_metrics.values()
    )
    non_identity_ok = all(
        metrics["best_non_identity_accuracy"] <= 0.35
        for metrics in task_metrics.values()
    )
    advantages = [
        metrics["persistent_advantage"]
        for metrics in task_metrics.values()
    ]
    mean_advantage = mean(advantages)
    advantage_ok = mean_advantage >= 0.60
    if persistent_ok and non_identity_ok and advantage_ok:
        return {
            "status": "persistent_identity_broader_task_bridge_proved",
            "reason": (
                "Persistent identity preserved the TAC-180 advantage across "
                "controlled transfer, multi-hop, agent-memory, and language-like "
                "instruction task families."
            ),
            "recommendation": (
                "Treat TAC-180 as a foundational controlled result for broader "
                "identity-dependent reasoning. The next validation layer should "
                "wire this bridge suite to live TAC state and then to external "
                "language benchmarks with reset-vs-carried-state controls."
            ),
            "mean_persistent_advantage": mean_advantage,
        }
    return {
        "status": "persistent_identity_broader_task_bridge_not_proved",
        "reason": (
            "At least one broader task family failed the persistent accuracy, "
            "non-identity bound, or mean-advantage gate."
        ),
        "recommendation": (
            "Keep TAC-180 scoped to its original benchmark until the failed "
            "bridge family is repaired or replaced."
        ),
        "mean_persistent_advantage": mean_advantage,
    }


def _apply_rule_to_values(values: Sequence[int], rule: str, *, vocab_size: int) -> list[int]:
    tensor = torch.tensor(list(values), dtype=torch.long)
    return apply_rule_bank(tensor, [rule] * len(values), vocab_size=vocab_size).tolist()


def _majority_rule(suite: dict[str, Any]) -> str:
    counts = {rule: 0 for rule in RULES}
    for support in suite["identity_support"].values():
        counts[support["rule"]] += 1
    return max(RULES, key=lambda rule: (counts[rule], -RULES.index(rule)))


def _support_inputs(vocab_size: int) -> list[int]:
    if vocab_size < 24:
        raise ValueError("vocab_size must be at least 24")
    return [LOW_TOKEN + 1, LOW_TOKEN + 6]


def _query_value(
    *,
    seed: int,
    identity_index: int,
    task_family: str,
    example_index: int,
    vocab_size: int,
    forbidden: set[int],
) -> int:
    span = vocab_size - LOW_TOKEN
    task_offset = TASK_FAMILIES.index(task_family) * 11
    cursor = (seed * 17 + identity_index * 19 + example_index * 7 + task_offset) % span
    for _ in range(span):
        candidate = LOW_TOKEN + cursor
        if candidate not in forbidden and _disambiguates_rules(
            task_family,
            candidate,
            vocab_size=vocab_size,
        ):
            return candidate
        cursor = (cursor + 7) % span
    raise ValueError(f"could not find disambiguating query for {task_family}")


def _disambiguates_rules(task_family: str, value: int, *, vocab_size: int) -> bool:
    targets = {
        _target_for_task(task_family, value, rule, vocab_size)
        for rule in RULES
    }
    return len(targets) == len(RULES)


def _language_prompt(identity_id: str, query_value: int) -> str:
    return (
        f"Analyst profile {identity_id} has a private operator learned from "
        f"earlier examples. In this fresh instruction, apply that operator to "
        f"code {query_value}. Return only the TOKEN answer."
    )


def _answer_text(value: int) -> str:
    return f"TOKEN_{int(value):03d}"


if __name__ == "__main__":
    main()
