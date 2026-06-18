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
import torch.nn.functional as F

from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    write_artifact,
)
from tac_transformer.research_directions import (
    CONCEPT_RELATION_TYPES,
    adaptive_concept_volume_loss,
    concept_relation_loss,
    concept_subsumption_loss,
    diagonal_mahalanobis_distance,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac274_adaptive_concept_volume_loss")
CONCEPT_NAMES = ("plant", "tree", "fruit", "apple", "red", "dog", "integer")
RELATION_PAIRS = torch.tensor(
    [
        [1, 0],  # tree child_of plant
        [3, 2],  # apple child_of fruit
        [3, 4],  # apple overlaps red
        [5, 6],  # dog disjoint integer
    ],
    dtype=torch.long,
)
RELATION_TYPES = torch.tensor(
    [
        CONCEPT_RELATION_TYPES["child_of"],
        CONCEPT_RELATION_TYPES["child_of"],
        CONCEPT_RELATION_TYPES["overlaps"],
        CONCEPT_RELATION_TYPES["disjoint"],
    ],
    dtype=torch.long,
)


def _true_concept_parameters() -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.05, 0.0],
            [3.0, 0.0, 0.0],
            [3.15, 0.15, 0.0],
            [3.35, 0.20, 0.0],
            [0.0, 4.0, 0.0],
            [4.5, 4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    vars_ = torch.tensor(
        [
            [1.80, 0.75, 0.35],
            [0.30, 0.12, 0.08],
            [1.30, 0.50, 0.25],
            [0.22, 0.12, 0.08],
            [0.70, 0.28, 0.18],
            [0.35, 0.20, 0.10],
            [0.28, 0.18, 0.10],
        ],
        dtype=torch.float32,
    )
    return means, torch.log(vars_)


def _sample_dataset(
    *,
    seed: int,
    examples_per_concept: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    true_means, true_log_vars = _true_concept_parameters()
    examples = []
    labels = []
    for concept_index in range(true_means.shape[0]):
        std = torch.exp(0.5 * true_log_vars[concept_index])
        noise = torch.randn(
            examples_per_concept,
            true_means.shape[-1],
            generator=generator,
        )
        examples.append(true_means[concept_index] + noise * std)
        labels.extend([concept_index] * examples_per_concept)
    return torch.cat(examples, dim=0), torch.tensor(labels, dtype=torch.long)


def _fixed_isotropic_volume_loss(
    embeddings: torch.Tensor,
    concept_ids: torch.Tensor,
    concept_means: torch.Tensor,
    shared_log_var: torch.Tensor,
) -> torch.Tensor:
    selected = concept_means.index_select(0, concept_ids.long())
    log_var = shared_log_var.clamp(-8.0, 8.0)
    inv_var = torch.exp(-log_var)
    squared = (embeddings - selected).pow(2).sum(dim=-1)
    return 0.5 * (squared * inv_var + embeddings.shape[-1] * log_var).mean()


def _nearest_adaptive_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    means: torch.Tensor,
    log_vars: torch.Tensor,
) -> float:
    distances = []
    for concept_index in range(means.shape[0]):
        concept_mean = means[concept_index].expand_as(embeddings)
        concept_log_var = log_vars[concept_index].expand_as(embeddings)
        distances.append(
            diagonal_mahalanobis_distance(embeddings, concept_mean, concept_log_var)
        )
    predicted = torch.stack(distances, dim=-1).argmin(dim=-1)
    return float((predicted == labels).float().mean())


def _nearest_euclidean_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    means: torch.Tensor,
) -> float:
    distances = torch.cdist(embeddings, means)
    predicted = distances.argmin(dim=-1)
    return float((predicted == labels).float().mean())


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().reshape(-1).float()
    right = right.detach().reshape(-1).float()
    left = left - left.mean()
    right = right - right.mean()
    denom = left.norm() * right.norm()
    if float(denom) <= 1e-8:
        return 0.0
    return float(torch.dot(left, right) / denom)


def _fit_row(
    *,
    seed: int,
    examples_per_concept: int,
    steps: int,
    learning_rate: float,
    smoke: bool,
) -> dict[str, float | int]:
    train_x, train_y = _sample_dataset(
        seed=seed,
        examples_per_concept=examples_per_concept,
    )
    eval_x, eval_y = _sample_dataset(
        seed=seed + 10_000,
        examples_per_concept=max(8, examples_per_concept // 2),
    )
    true_means, true_log_vars = _true_concept_parameters()
    generator = torch.Generator().manual_seed(seed + 17)
    adaptive_means = torch.nn.Parameter(
        true_means + 0.35 * torch.randn(true_means.shape, generator=generator)
    )
    adaptive_log_vars = torch.nn.Parameter(torch.zeros_like(true_log_vars))
    fixed_means = torch.nn.Parameter(adaptive_means.detach().clone())
    fixed_log_var = torch.nn.Parameter(torch.tensor(0.0))
    adaptive_optimizer = torch.optim.Adam(
        [adaptive_means, adaptive_log_vars],
        lr=learning_rate,
    )
    fixed_optimizer = torch.optim.Adam([fixed_means, fixed_log_var], lr=learning_rate)

    relation_weight = 0.08 if not smoke else 0.04
    for _ in range(int(steps)):
        adaptive_optimizer.zero_grad()
        adaptive_loss = adaptive_concept_volume_loss(
            train_x,
            train_y,
            adaptive_means,
            adaptive_log_vars,
        ) + relation_weight * concept_relation_loss(
            adaptive_means,
            adaptive_log_vars,
            RELATION_PAIRS,
            RELATION_TYPES,
        )
        adaptive_loss.backward()
        adaptive_optimizer.step()

        fixed_optimizer.zero_grad()
        fixed_loss = _fixed_isotropic_volume_loss(
            train_x,
            train_y,
            fixed_means,
            fixed_log_var,
        )
        fixed_loss.backward()
        fixed_optimizer.step()

    with torch.no_grad():
        adaptive_eval_loss = adaptive_concept_volume_loss(
            eval_x,
            eval_y,
            adaptive_means,
            adaptive_log_vars,
        )
        fixed_eval_loss = _fixed_isotropic_volume_loss(
            eval_x,
            eval_y,
            fixed_means,
            fixed_log_var,
        )
        hierarchy_loss = concept_subsumption_loss(
            adaptive_means,
            adaptive_log_vars,
            RELATION_PAIRS[:2, 0],
            RELATION_PAIRS[:2, 1],
        )
        relation_loss = concept_relation_loss(
            adaptive_means,
            adaptive_log_vars,
            RELATION_PAIRS,
            RELATION_TYPES,
        )
        adaptive_accuracy = _nearest_adaptive_accuracy(
            eval_x,
            eval_y,
            adaptive_means,
            adaptive_log_vars,
        )
        fixed_accuracy = _nearest_euclidean_accuracy(eval_x, eval_y, fixed_means)
        shape_correlation = _correlation(true_log_vars, adaptive_log_vars)
        dog_integer_distance = diagonal_mahalanobis_distance(
            adaptive_means[5][None, :],
            adaptive_means[6][None, :],
            torch.logaddexp(adaptive_log_vars[5], adaptive_log_vars[6])[None, :],
        )
        apple_red_distance = diagonal_mahalanobis_distance(
            adaptive_means[3][None, :],
            adaptive_means[4][None, :],
            torch.logaddexp(adaptive_log_vars[3], adaptive_log_vars[4])[None, :],
        )
        chance = 1.0 / len(CONCEPT_NAMES)
        return {
            "seed": int(seed),
            "adaptive_eval_loss": float(adaptive_eval_loss),
            "fixed_isotropic_eval_loss": float(fixed_eval_loss),
            "adaptive_loss_advantage": float(fixed_eval_loss - adaptive_eval_loss),
            "adaptive_assignment_accuracy": adaptive_accuracy,
            "fixed_assignment_accuracy": fixed_accuracy,
            "assignment_accuracy_advantage": adaptive_accuracy - fixed_accuracy,
            "shape_logvar_correlation": shape_correlation,
            "hierarchy_subsumption_loss": float(hierarchy_loss),
            "relation_loss": float(relation_loss),
            "disjoint_distance": float(dog_integer_distance),
            "overlap_distance": float(apple_red_distance),
            "program_knockout_drop_proxy": max(0.0, adaptive_accuracy - chance),
            "reset_accuracy_proxy": chance,
            "lm_collapse_proxy": float(torch.isfinite(adaptive_eval_loss)),
        }


def run_tac274_adaptive_concept_volume_loss(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    examples_per_concept: int = 48,
    steps: int = 180,
    learning_rate: float = 0.04,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    seed_list = tuple(int(seed) for seed in seeds)
    actual_steps = min(int(steps), 45) if smoke else int(steps)
    actual_examples = min(int(examples_per_concept), 18) if smoke else int(examples_per_concept)
    rows = [
        _fit_row(
            seed=seed,
            examples_per_concept=actual_examples,
            steps=actual_steps,
            learning_rate=learning_rate,
            smoke=smoke,
        )
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("adaptive_loss_advantage", 0.0) > 0.05
        and metrics.get("shape_logvar_correlation", 0.0) > 0.35
        and metrics.get("adaptive_assignment_accuracy", 0.0)
        >= metrics.get("reset_accuracy_proxy", 1.0) + 0.25
        and metrics.get("hierarchy_subsumption_loss", 99.0) <= 1.50
        and metrics.get("lm_collapse_proxy", 0.0) == 1.0
    )
    result = {
        "schema": "tac274_adaptive_concept_volume_loss.v1",
        "method": {
            "task": "adaptive_concept_volume_loss",
            "requested_label": "TAC-273 Adaptive Concept Volume Loss",
            "assigned_label": "TAC-274 because TAC-273 already exists in prd.json",
            "concept_names": list(CONCEPT_NAMES),
            "relation_types": dict(CONCEPT_RELATION_TYPES),
            "examples_per_concept": actual_examples,
            "steps": actual_steps,
            "learning_rate": float(learning_rate),
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Validates the geometry-level loss on synthetic anisotropic "
                "concept/program regions. It does not yet prove carry retention, "
                "identity-probe MI, route selectivity, reset degradation, program "
                "knockout drops, or no-collapse behavior in full LM training."
            ),
        },
    }
    return write_artifact(output_dir, "tac274_adaptive_concept_volume_loss.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--examples-per-concept", type=int, default=48)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    args = parser.parse_args()
    result = run_tac274_adaptive_concept_volume_loss(
        output_dir=args.output_dir,
        seeds=args.seeds,
        examples_per_concept=args.examples_per_concept,
        steps=args.steps,
        learning_rate=args.learning_rate,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
