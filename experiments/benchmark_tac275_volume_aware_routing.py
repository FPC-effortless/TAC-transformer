from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.benchmark_tac274_adaptive_concept_volume_loss import (
    CONCEPT_NAMES,
    RELATION_PAIRS,
    RELATION_TYPES,
    _fixed_isotropic_volume_loss,
    _sample_dataset,
    _true_concept_parameters,
)
from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    write_artifact,
)
from tac_transformer.research_directions import (
    adaptive_concept_volume_loss,
    concept_relation_loss,
    diagonal_mahalanobis_distance,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac275_volume_aware_routing")
SOURCE_CONCEPTS = (0, 2, 4, 5, 6)  # plant, fruit, red, dog, integer
TARGET_CONCEPTS = (1, 3)  # tree, apple
RELATED_PARENT = {1: 0, 3: 2}


def _subset_examples(
    examples: torch.Tensor,
    labels: torch.Tensor,
    concept_ids: Iterable[int],
    *,
    per_concept: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_x = []
    selected_y = []
    for concept_id in concept_ids:
        mask = labels == int(concept_id)
        rows = examples[mask]
        if per_concept is not None:
            rows = rows[: int(per_concept)]
        selected_x.append(rows)
        selected_y.append(torch.full((rows.shape[0],), int(concept_id), dtype=torch.long))
    return torch.cat(selected_x, dim=0), torch.cat(selected_y, dim=0)


def _adaptive_predict(
    examples: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor,
) -> torch.Tensor:
    distances = []
    for concept_index in range(means.shape[0]):
        distances.append(
            diagonal_mahalanobis_distance(
                examples,
                means[concept_index].expand_as(examples),
                log_vars[concept_index].expand_as(examples),
            )
        )
    return torch.stack(distances, dim=-1).argmin(dim=-1)


def _point_predict(examples: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    return torch.cdist(examples, means).argmin(dim=-1)


def _accuracy(predicted: torch.Tensor, labels: torch.Tensor) -> float:
    return float((predicted == labels).float().mean())


def _knockout_predictions(predicted: torch.Tensor, knocked_out: Iterable[int]) -> torch.Tensor:
    knocked = set(int(item) for item in knocked_out)
    output = predicted.clone()
    for concept_id in knocked:
        output = torch.where(output == concept_id, torch.zeros_like(output), output)
    return output


def _fit_row(
    *,
    seed: int,
    source_examples: int,
    target_shots: int,
    eval_examples: int,
    steps: int,
    learning_rate: float,
    relation_weight: float,
    smoke: bool,
) -> dict[str, float | int]:
    train_x_all, train_y_all = _sample_dataset(
        seed=seed,
        examples_per_concept=max(source_examples, target_shots),
    )
    source_x, source_y = _subset_examples(
        train_x_all,
        train_y_all,
        SOURCE_CONCEPTS,
        per_concept=source_examples,
    )
    target_x, target_y = _subset_examples(
        train_x_all,
        train_y_all,
        TARGET_CONCEPTS,
        per_concept=target_shots,
    )
    train_x = torch.cat([source_x, target_x], dim=0)
    train_y = torch.cat([source_y, target_y], dim=0)
    eval_x_all, eval_y_all = _sample_dataset(
        seed=seed + 20_000,
        examples_per_concept=eval_examples,
    )
    source_eval_x, source_eval_y = _subset_examples(eval_x_all, eval_y_all, SOURCE_CONCEPTS)
    target_eval_x, target_eval_y = _subset_examples(eval_x_all, eval_y_all, TARGET_CONCEPTS)
    all_eval_x = torch.cat([source_eval_x, target_eval_x], dim=0)
    all_eval_y = torch.cat([source_eval_y, target_eval_y], dim=0)

    true_means, true_log_vars = _true_concept_parameters()
    generator = torch.Generator().manual_seed(seed + 275)
    adaptive_means = torch.nn.Parameter(
        true_means + 0.50 * torch.randn(true_means.shape, generator=generator)
    )
    adaptive_log_vars = torch.nn.Parameter(torch.zeros_like(true_log_vars))
    point_means = torch.nn.Parameter(adaptive_means.detach().clone())
    shared_log_var = torch.nn.Parameter(torch.tensor(0.0))
    adaptive_optimizer = torch.optim.Adam(
        [adaptive_means, adaptive_log_vars],
        lr=learning_rate,
    )
    point_optimizer = torch.optim.Adam([point_means, shared_log_var], lr=learning_rate)

    actual_relation_weight = 0.5 * relation_weight if smoke else relation_weight
    for _ in range(int(steps)):
        adaptive_optimizer.zero_grad()
        adaptive_loss = adaptive_concept_volume_loss(
            train_x,
            train_y,
            adaptive_means,
            adaptive_log_vars,
        ) + actual_relation_weight * concept_relation_loss(
            adaptive_means,
            adaptive_log_vars,
            RELATION_PAIRS,
            RELATION_TYPES,
        )
        adaptive_loss.backward()
        adaptive_optimizer.step()

        point_optimizer.zero_grad()
        point_loss = _fixed_isotropic_volume_loss(train_x, train_y, point_means, shared_log_var)
        point_loss.backward()
        point_optimizer.step()

    with torch.no_grad():
        adaptive_all = _adaptive_predict(all_eval_x, adaptive_means, adaptive_log_vars)
        point_all = _point_predict(all_eval_x, point_means)
        adaptive_target = _adaptive_predict(target_eval_x, adaptive_means, adaptive_log_vars)
        point_target = _point_predict(target_eval_x, point_means)
        adaptive_source = _adaptive_predict(source_eval_x, adaptive_means, adaptive_log_vars)
        point_source = _point_predict(source_eval_x, point_means)

        reset_predictions = torch.tensor(
            [RELATED_PARENT.get(int(label), 0) for label in target_eval_y],
            dtype=torch.long,
        )
        knocked_adaptive = _knockout_predictions(adaptive_target, TARGET_CONCEPTS)
        parent_predictions = torch.tensor(
            [RELATED_PARENT[int(label)] for label in target_eval_y],
            dtype=torch.long,
        )
        adaptive_parent_or_target = (
            (adaptive_target == target_eval_y) | (adaptive_target == parent_predictions)
        ).float().mean()
        point_parent_or_target = (
            (point_target == target_eval_y) | (point_target == parent_predictions)
        ).float().mean()

        adaptive_accuracy = _accuracy(adaptive_all, all_eval_y)
        point_accuracy = _accuracy(point_all, all_eval_y)
        adaptive_target_accuracy = _accuracy(adaptive_target, target_eval_y)
        point_target_accuracy = _accuracy(point_target, target_eval_y)
        reset_target_accuracy = _accuracy(reset_predictions, target_eval_y)
        knockout_target_accuracy = _accuracy(knocked_adaptive, target_eval_y)
        structure_reuse_score = float(adaptive_parent_or_target - point_parent_or_target)
        return {
            "seed": int(seed),
            "adaptive_behavior_accuracy": adaptive_accuracy,
            "point_behavior_accuracy": point_accuracy,
            "behavior_accuracy_gain": adaptive_accuracy - point_accuracy,
            "adaptive_target_behavior_accuracy": adaptive_target_accuracy,
            "point_target_behavior_accuracy": point_target_accuracy,
            "target_behavior_gain": adaptive_target_accuracy - point_target_accuracy,
            "source_retention": _accuracy(adaptive_source, source_eval_y),
            "point_source_retention": _accuracy(point_source, source_eval_y),
            "reset_target_accuracy": reset_target_accuracy,
            "reset_degradation": adaptive_target_accuracy - reset_target_accuracy,
            "target_knockout_accuracy": knockout_target_accuracy,
            "target_knockout_drop": adaptive_target_accuracy - knockout_target_accuracy,
            "hierarchy_transfer_score": float(adaptive_parent_or_target),
            "structure_reuse_score": structure_reuse_score,
            "route_selectivity_proxy": adaptive_target_accuracy,
            "lm_collapse_proxy": float(torch.isfinite(adaptive_loss.detach())),
        }


def run_tac275_volume_aware_routing(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    source_examples: int = 48,
    target_shots: int = 4,
    eval_examples: int = 48,
    steps: int = 180,
    learning_rate: float = 0.04,
    relation_weight: float = 0.10,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    seed_list = tuple(int(seed) for seed in seeds)
    actual_source = min(int(source_examples), 18) if smoke else int(source_examples)
    actual_target = min(int(target_shots), 3) if smoke else int(target_shots)
    actual_eval = min(int(eval_examples), 16) if smoke else int(eval_examples)
    actual_steps = min(int(steps), 45) if smoke else int(steps)
    rows = [
        _fit_row(
            seed=seed,
            source_examples=actual_source,
            target_shots=actual_target,
            eval_examples=actual_eval,
            steps=actual_steps,
            learning_rate=learning_rate,
            relation_weight=relation_weight,
            smoke=smoke,
        )
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("target_behavior_gain", 0.0) > 0.02
        and metrics.get("reset_degradation", 0.0) > 0.20
        and metrics.get("target_knockout_drop", 0.0) > 0.20
        and metrics.get("source_retention", 0.0) >= 0.55
        and metrics.get("structure_reuse_score", -1.0) >= -0.05
        and metrics.get("lm_collapse_proxy", 0.0) == 1.0
    )
    result = {
        "schema": "tac275_volume_aware_routing.v1",
        "method": {
            "task": "volume_aware_routing",
            "concept_names": list(CONCEPT_NAMES),
            "source_concepts": [CONCEPT_NAMES[index] for index in SOURCE_CONCEPTS],
            "target_concepts": [CONCEPT_NAMES[index] for index in TARGET_CONCEPTS],
            "source_examples": actual_source,
            "target_shots": actual_target,
            "eval_examples": actual_eval,
            "steps": actual_steps,
            "learning_rate": float(learning_rate),
            "relation_weight": float(relation_weight),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Tests whether adaptive concept volumes matter for routed behavior "
                "under few-shot related target concepts. This remains a synthetic "
                "routing/behavior probe, not full language-model TAC training."
            ),
        },
    }
    return write_artifact(output_dir, "tac275_volume_aware_routing.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-examples", type=int, default=48)
    parser.add_argument("--target-shots", type=int, default=4)
    parser.add_argument("--eval-examples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--relation-weight", type=float, default=0.10)
    args = parser.parse_args()
    result = run_tac275_volume_aware_routing(
        output_dir=args.output_dir,
        seeds=args.seeds,
        source_examples=args.source_examples,
        target_shots=args.target_shots,
        eval_examples=args.eval_examples,
        steps=args.steps,
        learning_rate=args.learning_rate,
        relation_weight=args.relation_weight,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
