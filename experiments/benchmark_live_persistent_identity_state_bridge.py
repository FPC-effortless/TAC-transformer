from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_persistent_computational_identity import (
    RULES,
    infer_rule_from_support,
)
from experiments.benchmark_persistent_identity_broader_tasks import (
    TASK_FAMILIES,
    _memory_only_prediction,
    _prediction_matches,
    _solve_row,
    build_broader_task_suite,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/live_persistent_identity_state_bridge_2026_06_05")
CONTROLS = [
    "carried_identity_state",
    "reset_per_query_state",
    "global_persistent_without_identity",
    "memory_only_without_computation",
]


@dataclass
class LiveIdentityStateAdapter:
    """Small live-state contract adapter for identity-keyed computation."""

    vocab_size: int
    state_by_identity: dict[str, str] = field(default_factory=dict)
    global_rule: str | None = None
    state_update_count: int = 0
    hidden_rule_labels_used: bool = False

    def update_identity(self, identity_id: str, support: dict[str, Any]) -> None:
        rule = infer_rule_from_support(
            support["support_inputs"],
            support["support_targets"],
            vocab_size=self.vocab_size,
        )
        self.state_by_identity[str(identity_id)] = rule
        self.global_rule = rule
        self.state_update_count += 1

    def predict_identity(self, row: dict[str, Any]) -> int | None:
        rule = self.state_by_identity.get(str(row["identity_id"]))
        if rule is None:
            return None
        return _solve_row(row, rule, vocab_size=self.vocab_size)

    def predict_global(self, row: dict[str, Any]) -> int | None:
        if self.global_rule is None:
            return None
        return _solve_row(row, self.global_rule, vocab_size=self.vocab_size)


def build_live_state_suite(
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
    return {
        **suite,
        "schema": "live_persistent_identity_state_suite.v1",
        "state_contract": (
            "Support observations are processed into identity-keyed live state. "
            "Held-out query rows do not include the hidden rule label."
        ),
    }


def evaluate_live_state_adapter(suite: dict[str, Any], *, control: str) -> dict[str, Any]:
    if control not in CONTROLS:
        raise ValueError(f"unknown control: {control}")

    adapter = LiveIdentityStateAdapter(vocab_size=int(suite["vocab_size"]))
    state_update_count = 0
    if control in {"carried_identity_state", "global_persistent_without_identity"}:
        for identity_id in sorted(suite["identity_support"].keys()):
            adapter.update_identity(identity_id, suite["identity_support"][identity_id])
        state_update_count = adapter.state_update_count

    prior_rule = _majority_rule(suite)
    correct = 0
    missing = 0
    by_task = {
        task: {"correct": 0, "total": 0, "missing": 0}
        for task in TASK_FAMILIES
    }
    for row in suite["rows"]:
        if control == "carried_identity_state":
            predicted = adapter.predict_identity(row)
        elif control == "reset_per_query_state":
            predicted = _solve_row(row, prior_rule, vocab_size=suite["vocab_size"])
        elif control == "global_persistent_without_identity":
            predicted = adapter.predict_global(row)
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
        "schema": "live_persistent_identity_state_control.v1",
        "control": control,
        "accuracy": correct / float(total),
        "correct_count": correct,
        "example_count": total,
        "missing_prediction_count": missing,
        "state_update_count": state_update_count,
        "hidden_rule_labels_used": adapter.hidden_rule_labels_used,
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


def run_live_state_bridge_probe(
    *,
    seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
) -> dict[str, Any]:
    suite = build_live_state_suite(
        seeds=seeds,
        identities_per_seed=identities_per_seed,
        examples_per_task=examples_per_task,
        vocab_size=vocab_size,
    )
    controls = {
        control: evaluate_live_state_adapter(suite, control=control)
        for control in CONTROLS
    }
    task_metrics = _task_metrics(controls)
    decision = _decision(task_metrics)
    return {
        "schema": "live_persistent_identity_state_bridge.v1",
        "hypothesis": (
            "The TAC-181 controlled broader-task advantage should survive when "
            "implemented as a live identity-state contract: support observations "
            "update carried identity-keyed computational state, while reset, "
            "global, and memory-only controls stay bounded."
        ),
        "suite_summary": {
            "task_families": list(TASK_FAMILIES),
            "identity_count": len(suite["identity_support"]),
            "row_count": len(suite["rows"]),
            "seeds": suite["seeds"],
            "identities_per_seed": suite["identities_per_seed"],
            "examples_per_task": suite["examples_per_task"],
            "real_world_benchmark_status": "not_real_world_benchmark",
        },
        "state_adapter": {
            "contract": suite["state_contract"],
            "uses_identity_keyed_state": True,
            "hidden_rule_labels_used": False,
            "support_update_source": "diagnostic support observations",
            "query_rule_label_available": False,
        },
        "controls": controls,
        "task_metrics": task_metrics,
        "decision": decision,
        "boundary": (
            "This is a live-state contract proof over controlled TAC-181 rows, "
            "not a trained checkpoint result and not a real-world language "
            "benchmark. It validates the state-carry interface before external "
            "language evaluation."
        ),
    }


def format_live_state_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Live Persistent Identity State Bridge",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        "- Boundary: live-state contract proof, not a trained checkpoint or real-world language benchmark.",
        "",
        "## Task Metrics",
        "",
        "| Task | Carried state | Best non-identity | Advantage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task, metrics in result["task_metrics"].items():
        lines.append(
            "| {task} | {carried:.4f} | {best:.4f} | {advantage:.4f} |".format(
                task=task,
                carried=metrics["carried_identity_state_accuracy"],
                best=metrics["best_non_identity_accuracy"],
                advantage=metrics["carried_state_advantage"],
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
        description="Bridge persistent identity proof to a live state-carry contract."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--identities-per-seed", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=64)
    args = parser.parse_args(argv)

    result = run_live_state_bridge_probe(
        seeds=args.seeds,
        identities_per_seed=args.identities_per_seed,
        examples_per_task=args.examples_per_task,
        vocab_size=args.vocab_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "live_persistent_identity_state_bridge.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_live_state_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return result


def _task_metrics(controls: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for task in TASK_FAMILIES:
        carried = controls["carried_identity_state"]["by_task"][task]["accuracy"]
        non_identity = [
            controls["reset_per_query_state"]["by_task"][task]["accuracy"],
            controls["global_persistent_without_identity"]["by_task"][task]["accuracy"],
            controls["memory_only_without_computation"]["by_task"][task]["accuracy"],
        ]
        best = max(non_identity)
        result[task] = {
            "carried_identity_state_accuracy": carried,
            "reset_per_query_state_accuracy": non_identity[0],
            "global_persistent_without_identity_accuracy": non_identity[1],
            "memory_only_accuracy": non_identity[2],
            "best_non_identity_accuracy": best,
            "carried_state_advantage": carried - best,
        }
    return result


def _decision(task_metrics: dict[str, Any]) -> dict[str, Any]:
    carried_ok = all(
        metrics["carried_identity_state_accuracy"] >= 0.95
        for metrics in task_metrics.values()
    )
    non_identity_ok = all(
        metrics["best_non_identity_accuracy"] <= 0.35
        for metrics in task_metrics.values()
    )
    advantages = [
        metrics["carried_state_advantage"]
        for metrics in task_metrics.values()
    ]
    mean_advantage = mean(advantages)
    advantage_ok = mean_advantage >= 0.60
    if carried_ok and non_identity_ok and advantage_ok:
        return {
            "status": "live_persistent_identity_state_bridge_proved",
            "reason": (
                "Carried identity-keyed live state preserved the broader-task "
                "advantage while reset, global, and memory-only controls stayed "
                "bounded."
            ),
            "recommendation": (
                "Use this as the live-state acceptance contract for the next "
                "TAC checkpoint experiment. The next layer should replace the "
                "adapter with trained TAC state updates and then run external "
                "language benchmarks with reset-vs-carried-state controls."
            ),
            "mean_carried_state_advantage": mean_advantage,
        }
    return {
        "status": "live_persistent_identity_state_bridge_not_proved",
        "reason": (
            "At least one task family failed the carried-state accuracy, "
            "non-identity bound, or mean-advantage gate."
        ),
        "recommendation": (
            "Keep TAC-181 as a controlled solver proof until the live-state "
            "interface is repaired."
        ),
        "mean_carried_state_advantage": mean_advantage,
    }


def _majority_rule(suite: dict[str, Any]) -> str:
    counts = {rule: 0 for rule in RULES}
    for support in suite["identity_support"].values():
        counts[support["rule"]] += 1
    return max(RULES, key=lambda rule: (counts[rule], -RULES.index(rule)))


if __name__ == "__main__":
    main()
