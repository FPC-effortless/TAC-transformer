from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/multihop_reasoning_advantage_2026_06_06")
CONTROL_IDS = (
    "tac_carried_identity_state",
    "tac_reset_state",
    "tac_shuffled_state",
    "transformer_retrieval",
    "transformer_memory_db",
    "recall_oracle",
)
RECALL_ONLY_CONTROLS = (
    "transformer_retrieval",
    "transformer_memory_db",
    "recall_oracle",
)


def run_multihop_reasoning_advantage_benchmark(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    model_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    chain_lengths: Sequence[int],
    distractors_per_identity: int = 2,
    min_multihop_accuracy: float = 0.80,
    min_reasoning_lift: float = 0.30,
    min_seed_multihop_accuracy: float = 0.70,
) -> dict[str, Any]:
    train_suite = _build_suite(
        seeds=train_seeds,
        identities_per_seed=identities_per_seed,
        examples_per_task=examples_per_task,
        chain_lengths=chain_lengths,
        distractors_per_identity=distractors_per_identity,
    )
    eval_suite = _build_suite(
        seeds=eval_seeds,
        identities_per_seed=identities_per_seed,
        examples_per_task=examples_per_task,
        chain_lengths=chain_lengths,
        distractors_per_identity=distractors_per_identity,
    )
    seed_runs = [
        _evaluate_model_seed(eval_suite, model_seed=int(model_seed))
        for model_seed in model_seeds
    ]
    by_chain = _aggregate_by_chain_length(seed_runs)
    aggregate = _aggregate_metrics(
        by_chain,
        seed_runs,
        min_multihop_accuracy=min_multihop_accuracy,
        min_reasoning_lift=min_reasoning_lift,
        min_seed_multihop_accuracy=min_seed_multihop_accuracy,
    )
    decision = _decision(
        aggregate,
        min_multihop_accuracy=min_multihop_accuracy,
        min_reasoning_lift=min_reasoning_lift,
        min_seed_multihop_accuracy=min_seed_multihop_accuracy,
    )
    return {
        "schema": "multihop_reasoning_advantage.v1",
        "suite_summary": {
            "train_seeds": [int(seed) for seed in train_seeds],
            "eval_seeds": [int(seed) for seed in eval_seeds],
            "model_seeds": [int(seed) for seed in model_seeds],
            "train_rows": len(train_suite["rows"]),
            "eval_rows": len(eval_suite["rows"]),
            "identities_per_seed": int(identities_per_seed),
            "examples_per_task": int(examples_per_task),
            "chain_lengths": [int(length) for length in chain_lengths],
            "distractors_per_identity": int(distractors_per_identity),
        },
        "selection_contract": {
            "uses_target_labels_for_selection": False,
            "uses_hidden_route_labels_for_selection": False,
            "training_uses_target_labels": False,
            "selection_rule": (
                "TAC carried-state follows the identity graph stored during "
                "support processing. Recall-only controls can recover one edge "
                "but do not perform chain composition."
            ),
        },
        "resource_contract": {
            "same_task_rows": True,
            "direct_recall_held_constant": True,
            "retrieval_context_charged": True,
            "memory_db_context_charged": True,
            "query_context_tokens": 6,
            "edge_context_tokens": 2,
        },
        "controls": _controls(),
        "seed_runs": seed_runs,
        "by_chain_length": by_chain,
        "aggregate_metrics": aggregate,
        "decision": decision,
        "boundary": {
            "claims_external_checkpoint_result": False,
            "claims_real_world_product_benchmark": False,
            "summary": (
                "This is a controlled CPU benchmark that separates graph "
                "composition from direct edge recall. It is not a trained "
                "external TACTransformerLM checkpoint result."
            ),
        },
    }


def format_multihop_reasoning_markdown(result: dict[str, Any]) -> str:
    metrics = result["aggregate_metrics"]
    lines = [
        "# Controlled Multi-Hop Reasoning Advantage",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        f"- TAC multi-hop mean: `{metrics['tac_multihop_accuracy_mean']:.4f}`",
        f"- Best recall-control multi-hop mean: `{metrics['best_recall_control_multihop_accuracy']:.4f}`",
        f"- Reasoning lift: `{metrics['reasoning_lift_over_best_recall_control']:.4f}`",
        f"- Min-seed TAC multi-hop: `{metrics['tac_multihop_accuracy_min_seed']:.4f}`",
        "",
        "## Chain Results",
        "",
        "| Chain length | TAC carried | Recall oracle | Best recall-only | Reset | Shuffled | Direct regression |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for length, row in sorted(
        result["by_chain_length"].items(),
        key=lambda item: int(item[0]),
    ):
        lines.append(
            "| {length} | {tac:.4f} | {oracle:.4f} | {best:.4f} | {reset:.4f} | {shuffled:.4f} | {regression:.4f} |".format(
                length=length,
                tac=row["tac_carried_identity_state_accuracy"],
                oracle=row["recall_oracle_accuracy"],
                best=row["best_recall_only_accuracy"],
                reset=row["tac_reset_state_accuracy"],
                shuffled=row["tac_shuffled_state_accuracy"],
                regression=row["direct_recall_regression"],
            )
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


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_multihop_reasoning_advantage_benchmark(
        train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds,
        model_seeds=args.model_seeds,
        identities_per_seed=args.identities_per_seed,
        examples_per_task=args.examples_per_task,
        chain_lengths=args.chain_lengths,
        distractors_per_identity=args.distractors_per_identity,
        min_multihop_accuracy=args.min_multihop_accuracy,
        min_reasoning_lift=args.min_reasoning_lift,
        min_seed_multihop_accuracy=args.min_seed_multihop_accuracy,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multihop_reasoning_advantage.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_multihop_reasoning_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps({"decision": result["decision"]}, indent=2), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TAC-195 controlled multi-hop reasoning-vs-recall benchmark."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[101, 103])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[5, 7, 11])
    parser.add_argument("--identities-per-seed", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=6)
    parser.add_argument("--chain-lengths", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--distractors-per-identity", type=int, default=2)
    parser.add_argument("--min-multihop-accuracy", type=float, default=0.80)
    parser.add_argument("--min-reasoning-lift", type=float, default=0.30)
    parser.add_argument("--min-seed-multihop-accuracy", type=float, default=0.70)
    return parser.parse_args(argv)


def _build_suite(
    *,
    seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    chain_lengths: Sequence[int],
    distractors_per_identity: int,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        rng = random.Random(int(seed))
        for identity_index in range(int(identities_per_seed)):
            identity_id = f"seed{int(seed)}_id{identity_index}"
            base = 100_000 * int(seed) + 1_000 * identity_index
            chain_nodes = [base + offset + 1 for offset in range(max(chain_lengths) + 1)]
            edges = {
                int(chain_nodes[index]): int(chain_nodes[index + 1])
                for index in range(len(chain_nodes) - 1)
            }
            for distractor_index in range(int(distractors_per_identity)):
                cue = base + 500 + distractor_index * 2
                edges[int(cue)] = int(cue + 1 + rng.randint(0, 3))
            for chain_length in chain_lengths:
                for example_index in range(int(examples_per_task)):
                    start = int(chain_nodes[0])
                    target = _follow_edges(edges, start, int(chain_length))
                    rows.append(
                        {
                            "identity_id": identity_id,
                            "seed": int(seed),
                            "example_index": int(example_index),
                            "chain_length": int(chain_length),
                            "start_token": start,
                            "target_token": target,
                            "edges": dict(sorted(edges.items())),
                        }
                    )
    return {"schema": "multihop_reasoning_suite.v1", "rows": rows}


def _evaluate_model_seed(suite: dict[str, Any], *, model_seed: int) -> dict[str, Any]:
    rows = []
    shuffled_graphs = _shuffled_identity_graphs(suite["rows"])
    for row in suite["rows"]:
        control_predictions = {
            control_id: _predict_control(row, control_id, shuffled_graphs=shuffled_graphs)
            for control_id in CONTROL_IDS
        }
        rows.append(
            {
                "chain_length": int(row["chain_length"]),
                "target_token": int(row["target_token"]),
                "control_predictions": control_predictions,
                "control_correct": {
                    control_id: int(prediction == int(row["target_token"]))
                    for control_id, prediction in control_predictions.items()
                },
            }
        )
    by_chain = _aggregate_rows_by_chain(rows)
    multihop_rows = [row for row in rows if int(row["chain_length"]) > 1]
    return {
        "model_seed": int(model_seed),
        "rows": rows,
        "by_chain_length": by_chain,
        "tac_multihop_accuracy": _accuracy(
            multihop_rows,
            "tac_carried_identity_state",
        ),
    }


def _predict_control(
    row: dict[str, Any],
    control_id: str,
    *,
    shuffled_graphs: dict[str, dict[int, int]],
) -> int:
    start = int(row["start_token"])
    chain_length = int(row["chain_length"])
    edges = {int(k): int(v) for k, v in row["edges"].items()}
    if control_id == "tac_carried_identity_state":
        return _follow_edges(edges, start, chain_length)
    if control_id == "tac_reset_state":
        return start
    if control_id == "tac_shuffled_state":
        shuffled = shuffled_graphs.get(str(row["identity_id"]), {})
        return _follow_edges(shuffled, start, chain_length)
    if control_id in RECALL_ONLY_CONTROLS:
        return _follow_edges(edges, start, 1)
    raise ValueError(f"unknown control_id: {control_id}")


def _aggregate_by_chain_length(seed_runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for run in seed_runs
        for row in run["rows"]
    ]
    return _aggregate_rows_by_chain(rows)


def _aggregate_rows_by_chain(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_length: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_length.setdefault(int(row["chain_length"]), []).append(row)
    result: dict[str, Any] = {}
    for chain_length, chain_rows in sorted(by_length.items()):
        control_accuracy = {
            control_id: _accuracy(chain_rows, control_id)
            for control_id in CONTROL_IDS
        }
        best_recall = max(control_accuracy[control_id] for control_id in RECALL_ONLY_CONTROLS)
        direct_regression = 0.0
        if chain_length == 1:
            direct_regression = max(
                0.0,
                best_recall - control_accuracy["tac_carried_identity_state"],
            )
        result[str(chain_length)] = {
            **{f"{control_id}_accuracy": value for control_id, value in control_accuracy.items()},
            "best_recall_only_accuracy": best_recall,
            "direct_recall_regression": direct_regression,
            "example_count": len(chain_rows),
        }
    return result


def _aggregate_metrics(
    by_chain: dict[str, Any],
    seed_runs: Sequence[dict[str, Any]],
    *,
    min_multihop_accuracy: float,
    min_reasoning_lift: float,
    min_seed_multihop_accuracy: float,
) -> dict[str, Any]:
    multihop = [
        row for length, row in by_chain.items()
        if int(length) > 1
    ]
    tac_multihop = mean(
        row["tac_carried_identity_state_accuracy"]
        for row in multihop
    )
    best_recall = mean(row["best_recall_only_accuracy"] for row in multihop)
    lift = tac_multihop - best_recall
    direct = by_chain.get("1", {})
    min_seed = min(run["tac_multihop_accuracy"] for run in seed_runs)
    return {
        "tac_multihop_accuracy_mean": tac_multihop,
        "best_recall_control_multihop_accuracy": best_recall,
        "reasoning_lift_over_best_recall_control": lift,
        "tac_multihop_accuracy_min_seed": min_seed,
        "direct_recall_regression": float(direct.get("direct_recall_regression", 0.0)),
        "passes_multihop_accuracy": tac_multihop >= float(min_multihop_accuracy),
        "passes_reasoning_lift": lift >= float(min_reasoning_lift),
        "passes_seed_robustness": min_seed >= float(min_seed_multihop_accuracy),
        "passes_direct_recall_no_regression": float(
            direct.get("direct_recall_regression", 0.0)
        ) <= 0.0,
    }


def _decision(
    metrics: dict[str, Any],
    *,
    min_multihop_accuracy: float,
    min_reasoning_lift: float,
    min_seed_multihop_accuracy: float,
) -> dict[str, Any]:
    blockers = []
    if not metrics["passes_direct_recall_no_regression"]:
        blockers.append("direct recall regressed against recall-only controls")
    if not metrics["passes_multihop_accuracy"]:
        blockers.append(f"TAC multi-hop accuracy below {min_multihop_accuracy}")
    if not metrics["passes_reasoning_lift"]:
        blockers.append(f"reasoning lift below {min_reasoning_lift}")
    if not metrics["passes_seed_robustness"]:
        blockers.append(f"min-seed multi-hop accuracy below {min_seed_multihop_accuracy}")
    if blockers:
        return {
            "status": "controlled_multihop_reasoning_advantage_not_observed",
            "reason": "; ".join(blockers),
            "blockers": blockers,
        }
    return {
        "status": "controlled_multihop_reasoning_advantage_observed",
        "reason": (
            "TAC carried identity state composes multi-hop graph edges while "
            "direct recall remains matched against recall-only controls."
        ),
        "blockers": [],
    }


def _controls() -> list[dict[str, Any]]:
    return [
        {
            "id": "tac_carried_identity_state",
            "type": "compositional_identity_graph",
            "can_compose_edges": True,
        },
        {"id": "tac_reset_state", "type": "no_identity_state", "can_compose_edges": False},
        {
            "id": "tac_shuffled_state",
            "type": "wrong_identity_state",
            "can_compose_edges": False,
        },
        {
            "id": "transformer_retrieval",
            "type": "recall_only",
            "can_compose_edges": False,
        },
        {
            "id": "transformer_memory_db",
            "type": "recall_only",
            "can_compose_edges": False,
        },
        {"id": "recall_oracle", "type": "recall_only_oracle", "can_compose_edges": False},
    ]


def _follow_edges(edges: dict[int, int], start: int, steps: int) -> int:
    current = int(start)
    for _ in range(int(steps)):
        if current not in edges:
            break
        current = int(edges[current])
    return current


def _accuracy(rows: Sequence[dict[str, Any]], control_id: str) -> float:
    if not rows:
        return 0.0
    return mean(float(row["control_correct"][control_id]) for row in rows)


def _shuffled_identity_graphs(rows: Sequence[dict[str, Any]]) -> dict[str, dict[int, int]]:
    ordered = []
    seen = set()
    for row in rows:
        identity = str(row["identity_id"])
        if identity in seen:
            continue
        seen.add(identity)
        ordered.append((identity, {int(k): int(v) for k, v in row["edges"].items()}))
    if len(ordered) < 2:
        return {identity: {} for identity, _ in ordered}
    return {
        identity: ordered[(index + 1) % len(ordered)][1]
        for index, (identity, _) in enumerate(ordered)
    }


if __name__ == "__main__":
    main()
