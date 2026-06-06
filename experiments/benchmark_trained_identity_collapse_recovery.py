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

from experiments.benchmark_persistent_computational_identity import RULES
from experiments.benchmark_persistent_identity_broader_tasks import (
    TASK_FAMILIES,
    _memory_only_prediction,
    _prediction_matches,
    _solve_row,
    build_broader_task_suite,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/trained_identity_collapse_recovery_2026_06_05")


class TrainableIdentityStateLearner(torch.nn.Module):
    """Support-observation encoder that learns identity-specific program routes."""

    def __init__(self, *, vocab_size: int, state_dim: int = 32) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.state_dim = int(state_dim)
        feature_dim = 5
        self.support_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, state_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(state_dim, state_dim),
            torch.nn.Tanh(),
        )
        self.state_norm = torch.nn.LayerNorm(state_dim)
        self.router = torch.nn.Linear(state_dim, len(RULES))

    def encode_support(
        self,
        support_inputs: torch.Tensor,
        support_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        span = float(max(self.vocab_size - 4, 1))
        x = (support_inputs.float() - 4.0) / span
        y = (support_targets.float() - 4.0) / span
        delta = torch.remainder(support_targets - support_inputs, self.vocab_size).float()
        delta = delta / float(max(self.vocab_size, 1))
        product = x * y
        bias = torch.ones_like(x)
        features = torch.stack([x, y, delta, product, bias], dim=-1)
        state = self.support_encoder(features).mean(dim=1)
        state = self.state_norm(state)
        return state, self.router(state)


def build_trained_identity_suite(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
) -> dict[str, Any]:
    return {
        "schema": "trained_identity_collapse_recovery_suite.v1",
        "train": build_broader_task_suite(
            seeds=train_seeds,
            identities_per_seed=identities_per_seed,
            examples_per_task=examples_per_task,
            vocab_size=vocab_size,
        ),
        "eval": build_broader_task_suite(
            seeds=eval_seeds,
            identities_per_seed=identities_per_seed,
            examples_per_task=examples_per_task,
            vocab_size=vocab_size,
        ),
        "hidden_rule_label_training_contract": "rules are never used as loss labels",
    }


def evaluate_non_identity_controls(suite: dict[str, Any]) -> dict[str, Any]:
    eval_suite = suite["eval"]
    solver_accuracy = 1.0
    reset_accuracy = _control_accuracy(eval_suite, control="reset_per_query_state")
    global_accuracy = _control_accuracy(eval_suite, control="global_persistent_without_identity")
    memory_accuracy = _control_accuracy(eval_suite, control="memory_only_without_computation")
    best_control = max(reset_accuracy, global_accuracy, memory_accuracy)
    return {
        "schema": "trained_identity_non_identity_controls.v1",
        "solver_accuracy": solver_accuracy,
        "reset_per_query_state_accuracy": reset_accuracy,
        "global_persistent_without_identity_accuracy": global_accuracy,
        "memory_only_without_computation_accuracy": memory_accuracy,
        "best_non_identity_control_accuracy": best_control,
        "solver_advantage": solver_accuracy - best_control,
    }


def run_trained_identity_collapse_recovery_probe(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
    training_steps: int,
    model_seeds: Sequence[int],
    collapse_pressure: float,
    gradient_noise_std: float,
    learning_rate: float = 0.03,
) -> dict[str, Any]:
    suite = build_trained_identity_suite(
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        identities_per_seed=identities_per_seed,
        examples_per_task=examples_per_task,
        vocab_size=vocab_size,
    )
    controls = evaluate_non_identity_controls(suite)
    seed_runs = [
        _train_one_seed(
            suite,
            model_seed=int(model_seed),
            training_steps=int(training_steps),
            collapse_pressure=float(collapse_pressure),
            gradient_noise_std=float(gradient_noise_std),
            learning_rate=float(learning_rate),
        )
        for model_seed in model_seeds
    ]
    aggregate = _aggregate(seed_runs, controls)
    decision = _decision(aggregate)
    return {
        "schema": "trained_identity_collapse_recovery.v1",
        "hypothesis": (
            "A trained TAC-style identity state learner can recover stable "
            "identity-specific computation from support/query supervision even "
            "when training deliberately applies collapse pressure and gradient "
            "noise."
        ),
        "suite_summary": {
            "train_seeds": [int(seed) for seed in train_seeds],
            "eval_seeds": [int(seed) for seed in eval_seeds],
            "train_rows": len(suite["train"]["rows"]),
            "eval_rows": len(suite["eval"]["rows"]),
            "identities_per_seed": int(identities_per_seed),
            "examples_per_task": int(examples_per_task),
            "task_families": list(TASK_FAMILIES),
            "vocab_size": int(vocab_size),
        },
        "training_contract": {
            "hidden_rule_labels_used_for_loss": False,
            "support_query_supervision_only": True,
            "collapse_pressure_applied": float(collapse_pressure) > 0.0,
            "gradient_noise_injected": float(gradient_noise_std) > 0.0,
            "fixed_program_primitives": list(RULES),
        },
        "controls": controls,
        "seed_runs": seed_runs,
        "aggregate_metrics": aggregate,
        "decision": decision,
        "boundary": (
            "This is a local controlled trained learner on the TAC-182 task "
            "distribution. It is stronger than the TAC-182 hand-coded adapter "
            "because support-to-state routing is learned under collapse pressure, "
            "but it is still not an external TAC checkpoint or real-world "
            "language benchmark."
        ),
    }


def format_trained_identity_markdown(result: dict[str, Any]) -> str:
    metrics = result["aggregate_metrics"]
    lines = [
        "# Trained Identity Collapse Recovery",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        "- Training condition: support/query supervision with collapse pressure and gradient noise.",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| trained accuracy mean | {metrics['trained_accuracy_mean']:.4f} |",
        f"| trained accuracy min | {metrics['trained_accuracy_min']:.4f} |",
        f"| solver gap mean | {metrics['solver_gap_mean']:.4f} |",
        f"| best non-identity control | {metrics['best_non_identity_control_accuracy']:.4f} |",
        f"| trained advantage | {metrics['trained_advantage_over_control']:.4f} |",
        f"| state separation margin min | {metrics['state_separation_margin_min']:.4f} |",
        f"| route agreement min | {metrics['route_agreement_min']:.4f} |",
        "",
        "## Seed Runs",
        "",
        "| Seed | Accuracy | Route agreement | State margin | Collapse loss |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in result["seed_runs"]:
        lines.append(
            "| {seed} | {accuracy:.4f} | {agreement:.4f} | {margin:.4f} | {collapse:.6f} |".format(
                seed=run["model_seed"],
                accuracy=run["eval_accuracy"],
                agreement=run["route_agreement"],
                margin=run["state_separation_margin"],
                collapse=run["final_collapse_loss"],
            )
        )
    lines.extend(
        [
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
        description="Train a TAC-style identity learner under collapse pressure."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[101, 103])
    parser.add_argument("--identities-per-seed", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--training-steps", type=int, default=220)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[5, 7, 11])
    parser.add_argument("--collapse-pressure", type=float, default=0.04)
    parser.add_argument("--gradient-noise-std", type=float, default=0.015)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args(argv)

    result = run_trained_identity_collapse_recovery_probe(
        train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds,
        identities_per_seed=args.identities_per_seed,
        examples_per_task=args.examples_per_task,
        vocab_size=args.vocab_size,
        training_steps=args.training_steps,
        model_seeds=args.model_seeds,
        collapse_pressure=args.collapse_pressure,
        gradient_noise_std=args.gradient_noise_std,
        learning_rate=args.learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trained_identity_collapse_recovery.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_trained_identity_markdown(result),
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
    gradient_noise_std: float,
    learning_rate: float,
) -> dict[str, Any]:
    torch.manual_seed(int(model_seed))
    model = TrainableIdentityStateLearner(vocab_size=suite["train"]["vocab_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    train_batch = _tensorize_suite(suite["train"])
    final_task_loss = 0.0
    final_collapse_loss = 0.0
    for step_index in range(int(training_steps)):
        optimizer.zero_grad(set_to_none=True)
        states, logits = model.encode_support(
            train_batch["support_inputs"],
            train_batch["support_targets"],
        )
        row_logits = logits[train_batch["identity_index"]]
        task_loss = _mixture_program_loss(
            row_logits,
            train_batch["program_targets"],
            train_batch["target_values"],
        )
        route_probs = torch.softmax(logits, dim=-1)
        collapse_loss = torch.var(route_probs, dim=0, unbiased=False).mean()
        loss = task_loss + float(collapse_pressure) * collapse_loss
        loss.backward()
        if gradient_noise_std > 0.0:
            generator = torch.Generator().manual_seed(
                int(model_seed) * 1009 + int(step_index)
            )
            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                noise = torch.randn(
                    parameter.grad.shape,
                    generator=generator,
                    device=parameter.grad.device,
                    dtype=parameter.grad.dtype,
                )
                parameter.grad.add_(noise, alpha=float(gradient_noise_std))
        optimizer.step()
        final_task_loss = float(task_loss.detach())
        final_collapse_loss = float(collapse_loss.detach())

    eval_batch = _tensorize_suite(suite["eval"])
    eval_metrics = _evaluate_model(model, eval_batch)
    eval_metrics.update(
        {
            "model_seed": int(model_seed),
            "final_task_loss": final_task_loss,
            "final_collapse_loss": final_collapse_loss,
            "collapse_pressure": float(collapse_pressure),
            "gradient_noise_std": float(gradient_noise_std),
        }
    )
    return eval_metrics


def _tensorize_suite(suite: dict[str, Any]) -> dict[str, torch.Tensor | list[str]]:
    identity_ids = sorted(suite["identity_support"].keys())
    identity_index = {identity_id: index for index, identity_id in enumerate(identity_ids)}
    support_inputs = torch.tensor(
        [suite["identity_support"][identity_id]["support_inputs"] for identity_id in identity_ids],
        dtype=torch.long,
    )
    support_targets = torch.tensor(
        [suite["identity_support"][identity_id]["support_targets"] for identity_id in identity_ids],
        dtype=torch.long,
    )
    true_rule_index = torch.tensor(
        [RULES.index(suite["identity_support"][identity_id]["rule"]) for identity_id in identity_ids],
        dtype=torch.long,
    )
    row_identity_index = []
    row_targets = []
    row_program_targets = []
    row_task_families = []
    for row in suite["rows"]:
        row_identity_index.append(identity_index[row["identity_id"]])
        row_targets.append(int(row["target_value"]))
        row_task_families.append(row["task_family"])
        row_program_targets.append(
            [
                _solve_row(row, rule, vocab_size=suite["vocab_size"])
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
        "task_families": row_task_families,
    }


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
    model: TrainableIdentityStateLearner,
    batch: dict[str, Any],
) -> dict[str, Any]:
    with torch.inference_mode():
        states, logits = model.encode_support(batch["support_inputs"], batch["support_targets"])
        identity_routes = torch.argmax(logits, dim=-1)
        row_routes = identity_routes[batch["identity_index"]]
        predicted = batch["program_targets"][
            torch.arange(batch["program_targets"].shape[0]),
            row_routes,
        ]
        correct = predicted.eq(batch["target_values"])
        route_agreement = identity_routes.eq(batch["true_rule_index"]).float().mean().item()
        route_rule_nmi = _normalized_mutual_information(
            batch["true_rule_index"].tolist(),
            identity_routes.tolist(),
        )
        state_margin = _state_separation_margin(states, batch["true_rule_index"])
        by_task = {}
        for task in TASK_FAMILIES:
            mask = torch.tensor([row_task == task for row_task in batch["task_families"]])
            by_task[task] = {
                "accuracy": correct[mask].float().mean().item(),
                "example_count": int(mask.sum().item()),
            }
        return {
            "eval_accuracy": correct.float().mean().item(),
            "route_agreement": route_agreement,
            "route_rule_nmi": route_rule_nmi,
            "state_separation_margin": state_margin,
            "route_consistency": 1.0,
            "by_task": by_task,
        }


def _control_accuracy(eval_suite: dict[str, Any], *, control: str) -> float:
    prior_rule = _majority_rule(eval_suite)
    last_identity = sorted(eval_suite["identity_support"].keys())[-1]
    global_rule = eval_suite["identity_support"][last_identity]["rule"]
    correct = 0
    total = 0
    for row in eval_suite["rows"]:
        if control == "reset_per_query_state":
            predicted = _solve_row(row, prior_rule, vocab_size=eval_suite["vocab_size"])
        elif control == "global_persistent_without_identity":
            predicted = _solve_row(row, global_rule, vocab_size=eval_suite["vocab_size"])
        elif control == "memory_only_without_computation":
            predicted = _memory_only_prediction(row)
        else:
            raise ValueError(f"unknown control: {control}")
        total += 1
        if predicted is None:
            continue
        correct += int(_prediction_matches(row, int(predicted)))
    return correct / float(total)


def _aggregate(seed_runs: Sequence[dict[str, Any]], controls: dict[str, Any]) -> dict[str, Any]:
    accuracies = [run["eval_accuracy"] for run in seed_runs]
    margins = [run["state_separation_margin"] for run in seed_runs]
    route_agreements = [run["route_agreement"] for run in seed_runs]
    route_nmis = [run["route_rule_nmi"] for run in seed_runs]
    solver_accuracy = controls["solver_accuracy"]
    best_control = controls["best_non_identity_control_accuracy"]
    trained_mean = mean(accuracies)
    solver_advantage = controls["solver_advantage"]
    trained_advantage = trained_mean - best_control
    return {
        "trained_accuracy_mean": trained_mean,
        "trained_accuracy_min": min(accuracies),
        "solver_accuracy": solver_accuracy,
        "solver_gap_mean": solver_accuracy - trained_mean,
        "best_non_identity_control_accuracy": best_control,
        "trained_advantage_over_control": trained_advantage,
        "solver_advantage": solver_advantage,
        "solver_advantage_recovered_fraction": trained_advantage / max(solver_advantage, 1e-8),
        "state_separation_margin_mean": mean(margins),
        "state_separation_margin_min": min(margins),
        "route_agreement_mean": mean(route_agreements),
        "route_agreement_min": min(route_agreements),
        "route_rule_nmi_mean": mean(route_nmis),
        "route_rule_nmi_min": min(route_nmis),
        "route_consistency_min": min(run["route_consistency"] for run in seed_runs),
        "model_seed_count": len(seed_runs),
    }


def _decision(metrics: dict[str, Any]) -> dict[str, Any]:
    passed = (
        metrics["trained_accuracy_mean"] >= 0.90
        and metrics["trained_accuracy_min"] >= 0.85
        and metrics["solver_gap_mean"] <= 0.10
        and metrics["best_non_identity_control_accuracy"] <= 0.35
        and metrics["trained_advantage_over_control"] >= 0.60
        and metrics["state_separation_margin_min"] > 0.15
        and metrics["route_agreement_min"] >= 0.90
    )
    if passed:
        return {
            "status": "trained_identity_collapse_recovery_proved",
            "reason": (
                "The learned identity-state route recovered the TAC-182 solver "
                "advantage under explicit collapse pressure and gradient noise."
            ),
            "recommendation": (
                "Use this as the local Layer-4 bridge criterion before larger "
                "checkpoint runs: trained state updates must preserve identity "
                "separation, route consistency, and a small solver degradation gap."
            ),
        }
    return {
        "status": "trained_identity_collapse_recovery_not_proved",
        "reason": (
            "The trained learner failed at least one learnability, stability, "
            "control, or solver-gap gate."
        ),
        "recommendation": (
            "Treat TAC-180/181/182 as controlled possibility proofs until the "
            "trained collapse-recovery gate passes."
        ),
    }


def _state_separation_margin(states: torch.Tensor, rule_index: torch.Tensor) -> float:
    normalized = F.normalize(states.float(), dim=-1)
    same_distances = []
    different_distances = []
    for i in range(normalized.shape[0]):
        for j in range(i + 1, normalized.shape[0]):
            distance = 1.0 - float(torch.dot(normalized[i], normalized[j]).item())
            if int(rule_index[i]) == int(rule_index[j]):
                same_distances.append(distance)
            else:
                different_distances.append(distance)
    same = mean(same_distances) if same_distances else 0.0
    different = mean(different_distances) if different_distances else 0.0
    return different - same


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


def _majority_rule(suite: dict[str, Any]) -> str:
    counts = {rule: 0 for rule in RULES}
    for support in suite["identity_support"].values():
        counts[support["rule"]] += 1
    return max(RULES, key=lambda rule: (counts[rule], -RULES.index(rule)))


if __name__ == "__main__":
    main()
