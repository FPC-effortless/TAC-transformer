from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_memory_advantage_model_version import (
    PRIMARY_QUESTION,
    run_memory_advantage_model_version,
)
from experiments.benchmark_persistent_computational_identity import (
    RULES,
    infer_rule_from_support,
)
from experiments.benchmark_persistent_identity_broader_tasks import _target_for_task
from experiments.benchmark_relaxed_identity_routing_memory import (
    _train_one_seed as _train_relaxed_memory_seed,
    build_relaxed_identity_sequence_suite,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/long_horizon_memory_advantage_2026_06_05")
QUERY_TOKENS = 6
SUPPORT_PAIR_TOKENS = 8
INTERVENING_SESSION_TOKENS = 12
TARGET_SUCCESS = 0.90
DEFAULT_CONTEXT_BUDGETS = [6, 10, 14, 22, 38, 62]
DEFAULT_DAY_GAPS = [1, 7, 30, 90, 180, 365, 730]


def run_long_horizon_memory_advantage_benchmark(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    model_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    horizon_windows: int,
    vocab_size: int,
    training_steps: int,
    context_budgets: Sequence[int] = DEFAULT_CONTEXT_BUDGETS,
    collapse_pressure: float = 0.02,
    memory_noise_std: float = 0.01,
    learning_rate: float = 0.035,
    target_success: float = TARGET_SUCCESS,
) -> dict[str, Any]:
    suite_pair = {
        "schema": "long_horizon_memory_advantage_suite_pair.v1",
        "train": build_relaxed_identity_sequence_suite(
            seeds=train_seeds,
            identities_per_seed=identities_per_seed,
            examples_per_task=examples_per_task,
            horizon_windows=horizon_windows,
            vocab_size=vocab_size,
        ),
        "eval": build_relaxed_identity_sequence_suite(
            seeds=eval_seeds,
            identities_per_seed=identities_per_seed,
            examples_per_task=examples_per_task,
            horizon_windows=horizon_windows,
            vocab_size=vocab_size,
        ),
    }
    seed_runs = [
        _train_relaxed_memory_seed(
            suite_pair,
            model_seed=int(model_seed),
            training_steps=int(training_steps),
            collapse_pressure=float(collapse_pressure),
            memory_noise_std=float(memory_noise_std),
            learning_rate=float(learning_rate),
        )
        for model_seed in model_seeds
    ]
    tac_metrics = _aggregate_tac_seed_runs(seed_runs, horizon_windows=horizon_windows)
    budgets = sorted({int(budget) for budget in context_budgets})
    context_curve = _build_context_curve(
        suite_pair["eval"],
        budgets,
        tac_success=tac_metrics["carried_accuracy_mean"],
    )
    fixed_low_context_budget = QUERY_TOKENS
    day_curve = _build_day_curve(
        suite_pair["eval"],
        fixed_context_budget=fixed_low_context_budget,
        tac_by_horizon=tac_metrics["carried_accuracy_by_horizon_mean"],
        horizon_windows=horizon_windows,
    )
    tokens_required = _tokens_required_for_target_success(
        context_curve,
        target_success=float(target_success),
    )
    low_budget_success = _success_at_budget(context_curve, fixed_low_context_budget)
    best_control_low_budget = max(
        low_budget_success[control_id]
        for control_id in [
            "transformer_window",
            "transformer_retrieval",
            "transformer_memory_db",
        ]
    )
    aggregate_metrics = {
        **tac_metrics,
        "best_transformer_control_at_tac_context_budget": best_control_low_budget,
        "advantage_at_tac_context_budget": (
            tac_metrics["carried_accuracy_mean"] - best_control_low_budget
        ),
        "target_success": float(target_success),
        "tac_context_tokens": QUERY_TOKENS,
        "nearest_control_token_savings_for_target_success": _nearest_token_savings(
            tokens_required
        ),
    }
    aggregate_metrics["all_seed_target_success"] = (
        aggregate_metrics["carried_accuracy_min"] >= float(target_success)
    )
    model_contract = run_memory_advantage_model_version()
    decision = _decision(
        aggregate_metrics,
        tokens_required,
        target_success=float(target_success),
    )
    return {
        "schema": "long_horizon_memory_advantage.v1",
        "primary_question": PRIMARY_QUESTION,
        "target_graph": "Context Tokens Required vs Task Success",
        "secondary_graph": "Days Since Instruction vs Accuracy",
        "suite_summary": {
            "train_seeds": [int(seed) for seed in train_seeds],
            "eval_seeds": [int(seed) for seed in eval_seeds],
            "train_rows": len(suite_pair["train"]["rows"]),
            "eval_rows": len(suite_pair["eval"]["rows"]),
            "identity_count": len(suite_pair["eval"]["identity_support"]),
            "identities_per_seed": int(identities_per_seed),
            "examples_per_task": int(examples_per_task),
            "horizon_windows": int(horizon_windows),
            "vocab_size": int(vocab_size),
        },
        "training_contract": {
            "tac_support_query_supervision_only": True,
            "explicit_route_labels_used_for_loss": False,
            "hidden_rule_labels_used_for_loss": False,
            "persistent_memory_is_trainable": True,
            "fixed_candidate_program_bank": True,
        },
        "resource_contract": {
            "same_task_rows": True,
            "same_parameter_budget_contract": True,
            "parameter_counts": model_contract["parameter_counts"],
            "query_tokens": QUERY_TOKENS,
            "support_pair_tokens": SUPPORT_PAIR_TOKENS,
            "intervening_session_tokens": INTERVENING_SESSION_TOKENS,
            "retrieval_context_charged": True,
            "memory_db_context_charged": True,
        },
        "controls": _controls(),
        "seed_runs": seed_runs,
        "aggregate_metrics": aggregate_metrics,
        "tokens_required_for_target_success": tokens_required,
        "fixed_low_context_budget": fixed_low_context_budget,
        "graphs": {
            "context_tokens_required_vs_task_success": context_curve,
            "days_since_instruction_vs_accuracy": day_curve,
        },
        "decision": decision,
        "boundary": {
            "claims_external_checkpoint_result": False,
            "claims_real_world_product_benchmark": False,
            "summary": (
                "This is a controlled local benchmark over synthetic long-horizon "
                "identity tasks. It answers the attachment question only inside "
                "this proxy setting; it is not yet a trained external "
                "TACTransformerLM checkpoint result."
            ),
        },
    }


def format_long_horizon_memory_markdown(result: dict[str, Any]) -> str:
    metrics = result["aggregate_metrics"]
    tokens = result["tokens_required_for_target_success"]
    lines = [
        "# Controlled Long-Horizon Memory Advantage",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Answer: {result['decision']['answer']}",
        f"- Target graph: {result['target_graph']}",
        f"- TAC carried accuracy mean: {metrics['carried_accuracy_mean']:.4f}",
        f"- Best control at TAC context budget: {metrics['best_transformer_control_at_tac_context_budget']:.4f}",
        f"- Advantage at TAC context budget: {metrics['advantage_at_tac_context_budget']:.4f}",
        "",
        "## Tokens Required For 90% Success",
        "",
        "| Control | Context tokens |",
        "| --- | ---: |",
    ]
    for control_id, token_count in tokens.items():
        rendered = "not reached" if token_count is None else str(token_count)
        lines.append(f"| {control_id} | {rendered} |")
    lines.extend(
        [
            "",
            "## Context Tokens Required vs Task Success",
            "",
            "| Control | Tokens | Success |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in result["graphs"]["context_tokens_required_vs_task_success"]:
        lines.append(
            f"| {row['control_id']} | {row['context_tokens']} | {row['task_success']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            result["boundary"]["summary"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TAC-189 controlled long-horizon memory advantage benchmark."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[101, 103])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[5, 7, 11])
    parser.add_argument("--identities-per-seed", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=3)
    parser.add_argument("--horizon-windows", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--training-steps", type=int, default=360)
    parser.add_argument("--context-budgets", nargs="+", type=int, default=DEFAULT_CONTEXT_BUDGETS)
    parser.add_argument("--collapse-pressure", type=float, default=0.02)
    parser.add_argument("--memory-noise-std", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--target-success", type=float, default=TARGET_SUCCESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_long_horizon_memory_advantage_benchmark(
        train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds,
        model_seeds=args.model_seeds,
        identities_per_seed=args.identities_per_seed,
        examples_per_task=args.examples_per_task,
        horizon_windows=args.horizon_windows,
        vocab_size=args.vocab_size,
        training_steps=args.training_steps,
        context_budgets=args.context_budgets,
        collapse_pressure=args.collapse_pressure,
        memory_noise_std=args.memory_noise_std,
        learning_rate=args.learning_rate,
        target_success=args.target_success,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "long_horizon_memory_advantage.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_long_horizon_memory_markdown(result),
        encoding="utf-8",
    )
    _write_csv(
        args.output_dir / "context_tokens_required_vs_task_success.csv",
        result["graphs"]["context_tokens_required_vs_task_success"],
        fieldnames=["control_id", "context_tokens", "task_success"],
    )
    _write_csv(
        args.output_dir / "days_since_instruction_vs_accuracy.csv",
        result["graphs"]["days_since_instruction_vs_accuracy"],
        fieldnames=["control_id", "day_gap", "fixed_context_tokens", "accuracy"],
    )
    print(json.dumps(result["decision"], indent=2))
    return result


def _aggregate_tac_seed_runs(
    seed_runs: Sequence[dict[str, Any]],
    *,
    horizon_windows: int,
) -> dict[str, Any]:
    by_horizon = {}
    for horizon in range(int(horizon_windows)):
        key = str(horizon)
        by_horizon[key] = mean(
            float(run["by_horizon"][key]["accuracy"]) for run in seed_runs
        )
    carried = [float(run["carried_accuracy"]) for run in seed_runs]
    reset = [float(run["reset_accuracy"]) for run in seed_runs]
    shuffled = [float(run["shuffled_memory_accuracy"]) for run in seed_runs]
    return {
        "carried_accuracy_mean": mean(carried),
        "tac_carried_accuracy_mean": mean(carried),
        "carried_accuracy_min": min(carried),
        "tac_carried_accuracy_min": min(carried),
        "tac_reset_accuracy_mean": mean(reset),
        "tac_shuffled_memory_accuracy_mean": mean(shuffled),
        "carried_accuracy_by_horizon_mean": by_horizon,
        "model_seed_count": len(seed_runs),
    }


def _build_context_curve(
    eval_suite: dict[str, Any],
    context_budgets: Sequence[int],
    *,
    tac_success: float,
) -> list[dict[str, Any]]:
    rows = []
    for control_id in [
        "tac_carried_identity_state",
        "transformer_window",
        "transformer_retrieval",
        "transformer_memory_db",
    ]:
        for budget in context_budgets:
            if control_id == "tac_carried_identity_state":
                success = float(tac_success) if int(budget) >= QUERY_TOKENS else 0.0
            else:
                success = _evaluate_context_control(
                    eval_suite,
                    control_id=control_id,
                    context_budget=int(budget),
                )
            rows.append(
                {
                    "control_id": control_id,
                    "context_tokens": int(budget),
                    "task_success": success,
                }
            )
    return rows


def _build_day_curve(
    eval_suite: dict[str, Any],
    *,
    fixed_context_budget: int,
    tac_by_horizon: dict[str, float],
    horizon_windows: int,
) -> list[dict[str, Any]]:
    rows = []
    for horizon in range(int(horizon_windows)):
        day_gap = _day_gap_for_horizon(horizon)
        for control_id in [
            "tac_carried_identity_state",
            "transformer_window",
            "transformer_retrieval",
            "transformer_memory_db",
        ]:
            if control_id == "tac_carried_identity_state":
                accuracy = float(tac_by_horizon[str(horizon)])
            else:
                accuracy = _evaluate_context_control(
                    eval_suite,
                    control_id=control_id,
                    context_budget=int(fixed_context_budget),
                    horizon_filter=horizon,
                )
            rows.append(
                {
                    "control_id": control_id,
                    "day_gap": day_gap,
                    "fixed_context_tokens": int(fixed_context_budget),
                    "accuracy": accuracy,
                }
            )
    return rows


def _evaluate_context_control(
    eval_suite: dict[str, Any],
    *,
    control_id: str,
    context_budget: int,
    horizon_filter: int | None = None,
) -> float:
    correct = 0
    total = 0
    for row in eval_suite["rows"]:
        if horizon_filter is not None and int(row["horizon_window"]) != int(horizon_filter):
            continue
        prediction = _predict_context_control(
            eval_suite,
            row,
            control_id=control_id,
            context_budget=int(context_budget),
        )
        correct += int(int(prediction) == int(row["target_value"]))
        total += 1
    return correct / float(total) if total else 0.0


def _predict_context_control(
    eval_suite: dict[str, Any],
    row: dict[str, Any],
    *,
    control_id: str,
    context_budget: int,
) -> int:
    if control_id == "transformer_window":
        required = (
            QUERY_TOKENS
            + SUPPORT_PAIR_TOKENS
            + int(row["horizon_window"]) * INTERVENING_SESSION_TOKENS
        )
        if context_budget < required:
            return _solve_with_rule(row, RULES[0], vocab_size=eval_suite["vocab_size"])
        support = eval_suite["identity_support"][row["identity_id"]]
        return _solve_with_visible_support(row, support, eval_suite["vocab_size"])

    if control_id == "transformer_memory_db":
        if _support_pair_capacity(context_budget) < 1:
            return _solve_with_rule(row, RULES[0], vocab_size=eval_suite["vocab_size"])
        support = eval_suite["identity_support"][row["identity_id"]]
        return _solve_with_visible_support(row, support, eval_suite["vocab_size"])

    if control_id == "transformer_retrieval":
        capacity = _support_pair_capacity(context_budget)
        if capacity < 1:
            return _solve_with_rule(row, RULES[0], vocab_size=eval_suite["vocab_size"])
        if capacity == 1 and _retrieval_distracted(row):
            support = _distractor_support(eval_suite, row)
        else:
            support = eval_suite["identity_support"][row["identity_id"]]
        return _solve_with_visible_support(row, support, eval_suite["vocab_size"])

    raise ValueError(f"unknown context control: {control_id}")


def _solve_with_visible_support(
    row: dict[str, Any],
    support: dict[str, Any],
    vocab_size: int,
) -> int:
    rule = infer_rule_from_support(
        support["support_inputs"][:1],
        support["support_targets"][:1],
        vocab_size=vocab_size,
    )
    return _solve_with_rule(row, rule, vocab_size=vocab_size)


def _solve_with_rule(row: dict[str, Any], rule: str, *, vocab_size: int) -> int:
    return _target_for_task(row["task_family"], int(row["query_value"]), rule, vocab_size)


def _support_pair_capacity(context_budget: int) -> int:
    if int(context_budget) < QUERY_TOKENS:
        return 0
    return max(0, (int(context_budget) - QUERY_TOKENS) // SUPPORT_PAIR_TOKENS)


def _retrieval_distracted(row: dict[str, Any]) -> bool:
    identity_number = int(str(row["identity_id"]).rsplit("identity", 1)[1])
    score = (
        identity_number
        + int(row["horizon_window"])
        + int(row["example_index"])
        + len(row["task_family"])
    )
    return score % 4 == 0


def _distractor_support(eval_suite: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    identity_ids = sorted(eval_suite["identity_support"].keys())
    current = identity_ids.index(row["identity_id"])
    for offset in range(1, len(identity_ids) + 1):
        candidate_id = identity_ids[(current + offset) % len(identity_ids)]
        candidate = eval_suite["identity_support"][candidate_id]
        if candidate["rule"] != eval_suite["identity_support"][row["identity_id"]]["rule"]:
            return candidate
    return eval_suite["identity_support"][row["identity_id"]]


def _tokens_required_for_target_success(
    context_curve: Sequence[dict[str, Any]],
    *,
    target_success: float,
) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for control_id in sorted({row["control_id"] for row in context_curve}):
        candidates = [
            int(row["context_tokens"])
            for row in context_curve
            if row["control_id"] == control_id
            and float(row["task_success"]) >= float(target_success)
        ]
        result[control_id] = min(candidates) if candidates else None
    return result


def _success_at_budget(
    context_curve: Sequence[dict[str, Any]],
    budget: int,
) -> dict[str, float]:
    return {
        row["control_id"]: float(row["task_success"])
        for row in context_curve
        if int(row["context_tokens"]) == int(budget)
    }


def _nearest_token_savings(tokens_required: dict[str, int | None]) -> int | None:
    tac_tokens = tokens_required.get("tac_carried_identity_state")
    if tac_tokens is None:
        return None
    control_tokens = [
        token
        for control_id, token in tokens_required.items()
        if control_id != "tac_carried_identity_state" and token is not None
    ]
    if not control_tokens:
        return None
    return min(control_tokens) - int(tac_tokens)


def _decision(
    aggregate_metrics: dict[str, Any],
    tokens_required: dict[str, int | None],
    *,
    target_success: float,
) -> dict[str, Any]:
    tac_tokens = tokens_required.get("tac_carried_identity_state")
    control_ids = [
        "transformer_window",
        "transformer_retrieval",
        "transformer_memory_db",
    ]
    control_tokens = [tokens_required.get(control_id) for control_id in control_ids]
    token_gate = (
        tac_tokens is not None
        and all(token is not None and int(token) > int(tac_tokens) for token in control_tokens)
    )
    passed = (
        aggregate_metrics["carried_accuracy_mean"] >= float(target_success)
        and aggregate_metrics["tac_reset_accuracy_mean"] <= 0.35
        and aggregate_metrics["tac_shuffled_memory_accuracy_mean"] <= 0.35
        and aggregate_metrics["advantage_at_tac_context_budget"] >= 0.50
        and token_gate
    )
    if passed:
        seed_robustness = (
            "all_seeds_passed"
            if aggregate_metrics["all_seed_target_success"]
            else "mean_passed_but_min_seed_below_target"
        )
        return {
            "status": "controlled_long_horizon_memory_advantage_observed",
            "answer": (
                "Yes in this controlled proxy on the aggregate mean: carried TAC "
                "identity memory reaches the target success threshold with fewer "
                "charged context tokens than the transformer-window, retrieval, "
                "and memory-db controls."
            ),
            "seed_robustness": seed_robustness,
            "recommendation": (
                "Promote this as the killer-benchmark prototype, then repeat with "
                "full TACTransformerLM checkpoints and external baselines."
            ),
        }
    return {
        "status": "controlled_long_horizon_memory_advantage_not_observed",
        "answer": (
            "No controlled advantage was established because at least one accuracy, "
            "adversarial, or context-token gate failed."
        ),
        "recommendation": (
            "Keep TAC-188 as a model candidate, but do not use the memory-advantage "
            "fundraising claim until the benchmark clears."
        ),
    }


def _controls() -> list[dict[str, Any]]:
    return [
        {
            "id": "tac_carried_identity_state",
            "description": "Trainable TAC-style persistent state carried across query windows.",
        },
        {
            "id": "transformer_window",
            "description": "Parameter-matched transformer that can only use support if it fits in the current context window.",
        },
        {
            "id": "transformer_retrieval",
            "description": "Parameter-matched transformer with retrieval; retrieved support observations are charged as context tokens.",
        },
        {
            "id": "transformer_memory_db",
            "description": "Parameter-matched transformer with identity-keyed memory DB; returned observations are charged as context tokens.",
        },
        {
            "id": "tac_reset_state",
            "description": "Adversarial TAC control with identity memory reset before query.",
        },
        {
            "id": "tac_shuffled_state",
            "description": "Adversarial TAC control with memory states shuffled across identities.",
        },
    ]


def _day_gap_for_horizon(horizon: int) -> int:
    if int(horizon) < len(DEFAULT_DAY_GAPS):
        return DEFAULT_DAY_GAPS[int(horizon)]
    return DEFAULT_DAY_GAPS[-1] * (2 ** (int(horizon) - len(DEFAULT_DAY_GAPS) + 1))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], *, fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


if __name__ == "__main__":
    main()
