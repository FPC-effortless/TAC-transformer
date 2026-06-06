from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_persistent_computational_identity import (
    LOW_TOKEN,
    RULES,
    apply_rule_bank,
)
from experiments.benchmark_persistent_identity_broader_tasks import (
    TASK_FAMILIES,
    _answer_text,
    _disambiguates_rules,
    _target_for_task,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/relaxed_identity_routing_memory_2026_06_05")


class RelaxedRoutingMemoryModel(torch.nn.Module):
    """Trainable support-to-memory adapter with soft latent program routing."""

    def __init__(
        self,
        *,
        vocab_size: int,
        state_dim: int = 40,
        route_count: int = len(RULES),
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.state_dim = int(state_dim)
        self.route_count = int(route_count)
        feature_dim = 6
        self.support_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, state_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(state_dim, state_dim),
            torch.nn.Tanh(),
        )
        self.memory_update = torch.nn.GRUCell(state_dim, state_dim)
        self.memory_norm = torch.nn.LayerNorm(state_dim)
        self.carry_input = torch.nn.Parameter(torch.zeros(state_dim))
        self.carry_update = torch.nn.GRUCell(state_dim, state_dim)
        self.carry_norm = torch.nn.LayerNorm(state_dim)
        self.router = torch.nn.Linear(state_dim, route_count)

    def initialize_memory(
        self,
        support_inputs: torch.Tensor,
        support_targets: torch.Tensor,
    ) -> torch.Tensor:
        span = float(max(self.vocab_size - LOW_TOKEN, 1))
        x = (support_inputs.float() - float(LOW_TOKEN)) / span
        y = (support_targets.float() - float(LOW_TOKEN)) / span
        delta = torch.remainder(support_targets - support_inputs, self.vocab_size).float()
        delta = delta / float(max(self.vocab_size, 1))
        product = x * y
        distance = torch.abs(y - x)
        bias = torch.ones_like(x)
        features = torch.stack([x, y, delta, product, distance, bias], dim=-1)
        encoded = self.support_encoder(features).mean(dim=1)
        previous = torch.zeros(
            encoded.shape[0],
            self.state_dim,
            dtype=encoded.dtype,
            device=encoded.device,
        )
        memory = self.memory_update(encoded, previous)
        return self.memory_norm(memory)

    def advance_memory(self, memory: torch.Tensor, steps: int) -> torch.Tensor:
        carried = memory
        carry_input = self.carry_input.unsqueeze(0).expand(memory.shape[0], -1)
        for _ in range(int(steps)):
            carried = self.carry_update(carry_input, carried)
            carried = self.carry_norm(carried)
        return carried

    def route_logits(self, memory: torch.Tensor) -> torch.Tensor:
        return self.router(memory)


def build_relaxed_identity_sequence_suite(
    *,
    seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    horizon_windows: int,
    vocab_size: int,
) -> dict[str, Any]:
    if identities_per_seed < len(RULES) or identities_per_seed % len(RULES) != 0:
        raise ValueError("identities_per_seed must be a positive multiple of rule count")
    if examples_per_task < 1:
        raise ValueError("examples_per_task must be positive")
    if horizon_windows < 1:
        raise ValueError("horizon_windows must be positive")
    support_inputs = _support_inputs(vocab_size)
    identity_support: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for identity_index in range(int(identities_per_seed)):
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
                "hidden_rule_label_for_evaluation_only": True,
            }
            for task_family in TASK_FAMILIES:
                for horizon_window in range(int(horizon_windows)):
                    for example_index in range(int(examples_per_task)):
                        query_value = _query_value(
                            seed=int(seed),
                            identity_index=identity_index,
                            task_family=task_family,
                            horizon_window=horizon_window,
                            example_index=example_index,
                            vocab_size=vocab_size,
                            forbidden=set(support_inputs),
                        )
                        target_value = _target_for_task(
                            task_family,
                            query_value,
                            rule,
                            vocab_size,
                        )
                        rows.append(
                            {
                                "identity_id": identity_id,
                                "task_family": task_family,
                                "query_value": int(query_value),
                                "target_value": int(target_value),
                                "target_answer": _answer_text(target_value),
                                "horizon_window": int(horizon_window),
                                "example_index": int(example_index),
                            }
                        )
    return {
        "schema": "relaxed_identity_sequence_suite.v1",
        "rules": list(RULES),
        "task_families": list(TASK_FAMILIES),
        "seeds": [int(seed) for seed in seeds],
        "identities_per_seed": int(identities_per_seed),
        "examples_per_task": int(examples_per_task),
        "horizon_windows": int(horizon_windows),
        "vocab_size": int(vocab_size),
        "identity_support": identity_support,
        "rows": rows,
        "training_contract": {
            "query_rows_include_rule_label": False,
            "route_labels_available_to_model": False,
            "support_targets_available": True,
            "hidden_rule_labels_used_for_evaluation_only": True,
        },
    }


def run_relaxed_identity_routing_memory_probe(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    model_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    horizon_windows: int,
    vocab_size: int,
    training_steps: int,
    collapse_pressure: float,
    memory_noise_std: float,
    learning_rate: float = 0.035,
) -> dict[str, Any]:
    suite = {
        "schema": "relaxed_identity_routing_memory_suite_pair.v1",
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
        _train_one_seed(
            suite,
            model_seed=int(model_seed),
            training_steps=int(training_steps),
            collapse_pressure=float(collapse_pressure),
            memory_noise_std=float(memory_noise_std),
            learning_rate=float(learning_rate),
        )
        for model_seed in model_seeds
    ]
    aggregate = _aggregate(seed_runs)
    decision = _decision(aggregate)
    return {
        "schema": "relaxed_identity_routing_memory.v1",
        "hypothesis": (
            "A TAC-style persistent identity mechanism can relax explicit route "
            "rules into a trained soft router and recurrent memory update while "
            "preserving long-horizon consistency under reset and shuffled-memory "
            "controls."
        ),
        "suite_summary": {
            "train_seeds": [int(seed) for seed in train_seeds],
            "eval_seeds": [int(seed) for seed in eval_seeds],
            "train_rows": len(suite["train"]["rows"]),
            "eval_rows": len(suite["eval"]["rows"]),
            "identity_count": len(suite["eval"]["identity_support"]),
            "identities_per_seed": int(identities_per_seed),
            "examples_per_task": int(examples_per_task),
            "horizon_windows": int(horizon_windows),
            "task_families": list(TASK_FAMILIES),
            "vocab_size": int(vocab_size),
        },
        "training_contract": {
            "support_query_supervision_only": True,
            "explicit_route_labels_used_for_loss": False,
            "hidden_rule_labels_used_for_loss": False,
            "hidden_rule_labels_used_for_posthoc_metrics": True,
            "hand_defined_routing_logic": False,
            "soft_routing": True,
            "trainable_memory_subsystem": True,
            "fixed_candidate_program_bank": True,
        },
        "controls": {
            "reset_state": "router receives zero memory instead of support-derived identity memory",
            "shuffled_memory": "support-derived memory states are rolled across identity rows before routing",
            "horizon_tail": "last query window after repeated learned carry updates",
        },
        "seed_runs": seed_runs,
        "aggregate_metrics": aggregate,
        "decision": decision,
        "boundary": (
            "This relaxes TAC-183 by removing route-label supervision and "
            "hand-defined route selection while making memory update/carry "
            "trainable. It still uses a fixed candidate program bank to score "
            "controlled latent-rule rows, so it is not yet a full "
            "TACTransformerLM checkpoint or real-world language benchmark."
        ),
    }


def format_relaxed_identity_markdown(result: dict[str, Any]) -> str:
    metrics = result["aggregate_metrics"]
    lines = [
        "# Relaxed Identity Routing Memory",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        "- Training condition: support/query supervision only; no explicit route-label loss.",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| carried accuracy mean | {metrics['carried_accuracy_mean']:.4f} |",
        f"| horizon tail accuracy mean | {metrics['horizon_tail_accuracy_mean']:.4f} |",
        f"| reset accuracy mean | {metrics['reset_accuracy_mean']:.4f} |",
        f"| shuffled memory accuracy mean | {metrics['shuffled_memory_accuracy_mean']:.4f} |",
        f"| carried advantage over best control | {metrics['carried_advantage_over_best_control']:.4f} |",
        f"| route-rule NMI min | {metrics['route_rule_nmi_min']:.4f} |",
        f"| route consistency min | {metrics['route_consistency_min']:.4f} |",
        f"| memory drift mean | {metrics['memory_drift_mean']:.4f} |",
        "",
        "## Seed Runs",
        "",
        "| Seed | Carried | Tail | Reset | Shuffled memory | Route NMI | Route consistency |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in result["seed_runs"]:
        lines.append(
            "| {seed} | {carried:.4f} | {tail:.4f} | {reset:.4f} | {shuffled:.4f} | {nmi:.4f} | {consistency:.4f} |".format(
                seed=run["model_seed"],
                carried=run["carried_accuracy"],
                tail=run["horizon_tail_accuracy"],
                reset=run["reset_accuracy"],
                shuffled=run["shuffled_memory_accuracy"],
                nmi=run["route_rule_nmi"],
                consistency=run["route_consistency"],
            )
        )
    lines.extend(["", "## Boundary", "", result["boundary"], ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Train a relaxed soft-router and persistent memory TAC probe."
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
    parser.add_argument("--collapse-pressure", type=float, default=0.02)
    parser.add_argument("--memory-noise-std", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    args = parser.parse_args(argv)

    result = run_relaxed_identity_routing_memory_probe(
        train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds,
        model_seeds=args.model_seeds,
        identities_per_seed=args.identities_per_seed,
        examples_per_task=args.examples_per_task,
        horizon_windows=args.horizon_windows,
        vocab_size=args.vocab_size,
        training_steps=args.training_steps,
        collapse_pressure=args.collapse_pressure,
        memory_noise_std=args.memory_noise_std,
        learning_rate=args.learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "relaxed_identity_routing_memory.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_relaxed_identity_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return result


def _train_one_seed(
    suite: dict[str, Any],
    *,
    model_seed: int,
    training_steps: int,
    collapse_pressure: float,
    memory_noise_std: float,
    learning_rate: float,
) -> dict[str, Any]:
    torch.manual_seed(int(model_seed))
    model = RelaxedRoutingMemoryModel(vocab_size=suite["train"]["vocab_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    train_batch = _tensorize_suite(suite["train"])
    final_task_loss = 0.0
    final_collapse_loss = 0.0
    for step_index in range(int(training_steps)):
        optimizer.zero_grad(set_to_none=True)
        memories_by_horizon = _memories_by_horizon(
            model,
            train_batch,
            horizon_windows=suite["train"]["horizon_windows"],
            memory_noise_std=memory_noise_std,
            model_seed=model_seed,
            step_index=step_index,
        )
        route_logits = _row_route_logits(model, memories_by_horizon, train_batch)
        program_targets, target_values = _row_program_targets_by_horizon(train_batch)
        task_loss = _mixture_program_loss(
            route_logits,
            program_targets,
            target_values,
        )
        route_probs = torch.softmax(model.route_logits(memories_by_horizon[0]), dim=-1)
        collapse_loss = torch.var(route_probs, dim=0, unbiased=False).mean()
        loss = task_loss + float(collapse_pressure) * collapse_loss
        loss.backward()
        optimizer.step()
        final_task_loss = float(task_loss.detach())
        final_collapse_loss = float(collapse_loss.detach())

    eval_batch = _tensorize_suite(suite["eval"])
    metrics = _evaluate_model(
        model,
        eval_batch,
        horizon_windows=suite["eval"]["horizon_windows"],
    )
    metrics.update(
        {
            "model_seed": int(model_seed),
            "final_task_loss": final_task_loss,
            "final_collapse_loss": final_collapse_loss,
            "collapse_pressure": float(collapse_pressure),
            "memory_noise_std": float(memory_noise_std),
        }
    )
    return metrics


def _tensorize_suite(suite: dict[str, Any]) -> dict[str, Any]:
    identity_ids = sorted(suite["identity_support"].keys())
    identity_index = {identity_id: index for index, identity_id in enumerate(identity_ids)}
    support_inputs = torch.tensor(
        [
            suite["identity_support"][identity_id]["support_inputs"]
            for identity_id in identity_ids
        ],
        dtype=torch.long,
    )
    support_targets = torch.tensor(
        [
            suite["identity_support"][identity_id]["support_targets"]
            for identity_id in identity_ids
        ],
        dtype=torch.long,
    )
    true_rule_index = torch.tensor(
        [
            RULES.index(suite["identity_support"][identity_id]["rule"])
            for identity_id in identity_ids
        ],
        dtype=torch.long,
    )
    row_identity_index = []
    row_targets = []
    row_program_targets = []
    row_horizon = []
    row_task_families = []
    for row in suite["rows"]:
        row_identity_index.append(identity_index[row["identity_id"]])
        row_targets.append(int(row["target_value"]))
        row_horizon.append(int(row["horizon_window"]))
        row_task_families.append(row["task_family"])
        row_program_targets.append(
            [
                _target_for_task(
                    row["task_family"],
                    int(row["query_value"]),
                    rule,
                    suite["vocab_size"],
                )
                for rule in RULES
            ]
        )
    return {
        "support_inputs": support_inputs,
        "support_targets": support_targets,
        "true_rule_index": true_rule_index,
        "identity_index": torch.tensor(row_identity_index, dtype=torch.long),
        "target_values": torch.tensor(row_targets, dtype=torch.long),
        "program_targets": torch.tensor(row_program_targets, dtype=torch.long),
        "horizon_window": torch.tensor(row_horizon, dtype=torch.long),
        "task_families": row_task_families,
    }


def _memories_by_horizon(
    model: RelaxedRoutingMemoryModel,
    batch: dict[str, Any],
    *,
    horizon_windows: int,
    memory_noise_std: float = 0.0,
    model_seed: int = 0,
    step_index: int = 0,
) -> list[torch.Tensor]:
    memory = model.initialize_memory(batch["support_inputs"], batch["support_targets"])
    if memory_noise_std > 0.0:
        generator = torch.Generator(device=memory.device).manual_seed(
            int(model_seed) * 1009 + int(step_index)
        )
        noise = torch.randn(
            memory.shape,
            generator=generator,
            device=memory.device,
            dtype=memory.dtype,
        )
        memory = memory + noise * float(memory_noise_std)
    memories = [memory]
    carried = memory
    for _ in range(1, int(horizon_windows)):
        carried = model.advance_memory(carried, 1)
        memories.append(carried)
    return memories


def _row_route_logits(
    model: RelaxedRoutingMemoryModel,
    memories_by_horizon: Sequence[torch.Tensor],
    batch: dict[str, Any],
) -> torch.Tensor:
    logits_by_horizon = [model.route_logits(memory) for memory in memories_by_horizon]
    row_logits = []
    for horizon_index, logits in enumerate(logits_by_horizon):
        mask = batch["horizon_window"].eq(int(horizon_index))
        if not bool(mask.any()):
            continue
        row_logits.append(logits[batch["identity_index"][mask]])
    return torch.cat(row_logits, dim=0)


def _row_program_targets_by_horizon(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    program_targets = []
    target_values = []
    max_horizon = int(batch["horizon_window"].max().item()) if batch["horizon_window"].numel() else -1
    for horizon_index in range(max_horizon + 1):
        mask = batch["horizon_window"].eq(int(horizon_index))
        if not bool(mask.any()):
            continue
        program_targets.append(batch["program_targets"][mask])
        target_values.append(batch["target_values"][mask])
    return torch.cat(program_targets, dim=0), torch.cat(target_values, dim=0)


def _mixture_program_loss(
    row_logits: torch.Tensor,
    program_targets: torch.Tensor,
    target_values: torch.Tensor,
) -> torch.Tensor:
    route_probs = torch.softmax(row_logits, dim=-1)
    target_mask = (program_targets == target_values.unsqueeze(-1)).float()
    target_prob = (route_probs * target_mask).sum(dim=-1).clamp_min(1e-8)
    return -torch.log(target_prob).mean()


def _evaluate_model(
    model: RelaxedRoutingMemoryModel,
    batch: dict[str, Any],
    *,
    horizon_windows: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        memories = _memories_by_horizon(model, batch, horizon_windows=horizon_windows)
        row_logits = _row_route_logits(model, memories, batch)
        program_targets, target_values = _row_program_targets_by_horizon(batch)
        carried_predictions = _predict_from_logits(row_logits, program_targets)
        carried_correct = carried_predictions.eq(target_values)

        reset_memory = torch.zeros_like(memories[0])
        reset_logits = model.route_logits(reset_memory)
        reset_row_logits = _row_logits_from_single_memory(reset_logits, batch)
        reset_predictions = _predict_from_logits(reset_row_logits, batch["program_targets"])
        reset_correct = reset_predictions.eq(batch["target_values"])

        shuffled_memories = [memory.roll(shifts=1, dims=0) for memory in memories]
        shuffled_logits = _row_route_logits(model, shuffled_memories, batch)
        shuffled_predictions = _predict_from_logits(shuffled_logits, program_targets)
        shuffled_correct = shuffled_predictions.eq(target_values)

        route_by_horizon = [torch.argmax(model.route_logits(memory), dim=-1) for memory in memories]
        route_rule_nmi = _normalized_mutual_information(
            batch["true_rule_index"].tolist(),
            route_by_horizon[0].tolist(),
        )
        route_consistency = _route_consistency(route_by_horizon)
        tail_mask = batch["horizon_window"].eq(int(horizon_windows) - 1)
        ordered_tail_mask = _ordered_mask_for_horizon(batch, int(horizon_windows) - 1)
        memory_drift = _memory_drift(memories[0], memories[-1])
        by_horizon = {}
        offset = 0
        for horizon_index in range(int(horizon_windows)):
            count = int(batch["horizon_window"].eq(horizon_index).sum().item())
            if count == 0:
                by_horizon[str(horizon_index)] = {"accuracy": 0.0, "example_count": 0}
                continue
            slice_correct = carried_correct[offset : offset + count]
            by_horizon[str(horizon_index)] = {
                "accuracy": float(slice_correct.float().mean().item()),
                "example_count": count,
            }
            offset += count
        return {
            "carried_accuracy": float(carried_correct.float().mean().item()),
            "horizon_tail_accuracy": float(
                carried_correct[ordered_tail_mask].float().mean().item()
                if bool(tail_mask.any())
                else 0.0
            ),
            "reset_accuracy": float(reset_correct.float().mean().item()),
            "shuffled_memory_accuracy": float(shuffled_correct.float().mean().item()),
            "best_control_accuracy": float(
                max(reset_correct.float().mean().item(), shuffled_correct.float().mean().item())
            ),
            "route_rule_nmi": route_rule_nmi,
            "route_consistency": route_consistency,
            "memory_drift": memory_drift,
            "by_horizon": by_horizon,
        }


def _predict_from_logits(row_logits: torch.Tensor, program_targets: torch.Tensor) -> torch.Tensor:
    routes = torch.argmax(row_logits, dim=-1)
    return program_targets[torch.arange(program_targets.shape[0]), routes]


def _row_logits_from_single_memory(route_logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    return route_logits[batch["identity_index"]]


def _ordered_mask_for_horizon(batch: dict[str, Any], horizon: int) -> torch.Tensor:
    ordered = []
    max_horizon = int(batch["horizon_window"].max().item()) if batch["horizon_window"].numel() else -1
    for horizon_index in range(max_horizon + 1):
        mask = batch["horizon_window"].eq(int(horizon_index))
        if not bool(mask.any()):
            continue
        ordered.append(mask[mask].new_full((int(mask.sum().item()),), horizon_index == horizon))
    return torch.cat(ordered)


def _route_consistency(route_by_horizon: Sequence[torch.Tensor]) -> float:
    if not route_by_horizon:
        return 0.0
    base = route_by_horizon[0]
    agreements = [
        routes.eq(base).float().mean().item()
        for routes in route_by_horizon[1:]
    ]
    return float(mean(agreements)) if agreements else 1.0


def _memory_drift(start: torch.Tensor, end: torch.Tensor) -> float:
    cosine = F.cosine_similarity(start.float(), end.float(), dim=-1)
    return float((1.0 - cosine).mean().item())


def _aggregate(seed_runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    carried = [run["carried_accuracy"] for run in seed_runs]
    tail = [run["horizon_tail_accuracy"] for run in seed_runs]
    reset = [run["reset_accuracy"] for run in seed_runs]
    shuffled = [run["shuffled_memory_accuracy"] for run in seed_runs]
    best_controls = [run["best_control_accuracy"] for run in seed_runs]
    route_nmi = [run["route_rule_nmi"] for run in seed_runs]
    route_consistency = [run["route_consistency"] for run in seed_runs]
    drift = [run["memory_drift"] for run in seed_runs]
    carried_mean = mean(carried)
    best_control_mean = mean(best_controls)
    return {
        "carried_accuracy_mean": carried_mean,
        "carried_accuracy_min": min(carried),
        "horizon_tail_accuracy_mean": mean(tail),
        "horizon_tail_accuracy_min": min(tail),
        "reset_accuracy_mean": mean(reset),
        "shuffled_memory_accuracy_mean": mean(shuffled),
        "best_control_accuracy_mean": best_control_mean,
        "carried_advantage_over_best_control": carried_mean - best_control_mean,
        "route_rule_nmi_mean": mean(route_nmi),
        "route_rule_nmi_min": min(route_nmi),
        "route_consistency_mean": mean(route_consistency),
        "route_consistency_min": min(route_consistency),
        "memory_drift_mean": mean(drift),
        "memory_drift_max": max(drift),
        "model_seed_count": len(seed_runs),
    }


def _decision(metrics: dict[str, Any]) -> dict[str, Any]:
    passed = (
        metrics["carried_accuracy_mean"] >= 0.90
        and metrics["horizon_tail_accuracy_mean"] >= 0.90
        and metrics["reset_accuracy_mean"] <= 0.35
        and metrics["shuffled_memory_accuracy_mean"] <= 0.35
        and metrics["carried_advantage_over_best_control"] >= 0.55
        and metrics["route_rule_nmi_min"] >= 0.75
        and metrics["route_consistency_min"] >= 0.90
    )
    if passed:
        return {
            "status": "relaxed_identity_routing_memory_promote_candidate",
            "reason": (
                "The trained soft router and recurrent memory update preserved "
                "long-horizon carried accuracy while reset and shuffled-memory "
                "controls stayed bounded, and posthoc route structure aligned "
                "with the latent programs."
            ),
            "recommendation": (
                "Promote as an opt-in architecture probe. The next layer should "
                "replace the fixed candidate program bank with learned experts "
                "inside TACTransformerLM and keep the same reset/shuffle/horizon "
                "gate."
            ),
        }
    return {
        "status": "relaxed_identity_routing_memory_not_proved",
        "reason": (
            "At least one carried-accuracy, control, route-structure, or "
            "long-horizon consistency gate failed."
        ),
        "recommendation": (
            "Keep TAC-183/TAC-184 as the trained local boundary until the "
            "relaxed router-memory probe clears its gate."
        ),
    }


def _normalized_mutual_information(labels: Sequence[int], routes: Sequence[int]) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    label_values = sorted(set(labels))
    route_values = sorted(set(routes))
    label_counts = {value: labels.count(value) for value in label_values}
    route_counts = {value: routes.count(value) for value in route_values}
    mi = 0.0
    for label in label_values:
        for route in route_values:
            joint = sum(1 for l, r in zip(labels, routes) if l == label and r == route)
            if joint == 0:
                continue
            pxy = joint / total
            px = label_counts[label] / total
            py = route_counts[route] / total
            mi += pxy * math.log2(pxy / (px * py))
    entropy = 0.0
    for count in label_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    if entropy <= 0.0:
        return 0.0
    return mi / entropy


def _support_inputs(vocab_size: int) -> list[int]:
    if vocab_size < 24:
        raise ValueError("vocab_size must be at least 24")
    return [LOW_TOKEN + 1, LOW_TOKEN + 6]


def _apply_rule_to_values(values: Sequence[int], rule: str, *, vocab_size: int) -> list[int]:
    tensor = torch.tensor(list(values), dtype=torch.long)
    return apply_rule_bank(tensor, [rule] * len(values), vocab_size=vocab_size).tolist()


def _query_value(
    *,
    seed: int,
    identity_index: int,
    task_family: str,
    horizon_window: int,
    example_index: int,
    vocab_size: int,
    forbidden: set[int],
) -> int:
    span = int(vocab_size) - LOW_TOKEN
    task_offset = TASK_FAMILIES.index(task_family) * 11
    cursor = (
        int(seed) * 17
        + int(identity_index) * 19
        + int(horizon_window) * 23
        + int(example_index) * 7
        + task_offset
    ) % span
    for _ in range(span):
        candidate = LOW_TOKEN + cursor
        if candidate not in forbidden and _disambiguates_rules(
            task_family,
            candidate,
            vocab_size=vocab_size,
        ):
            return int(candidate)
        cursor = (cursor + 7) % span
    raise ValueError(f"could not find disambiguating query for {task_family}")


if __name__ == "__main__":
    main()
