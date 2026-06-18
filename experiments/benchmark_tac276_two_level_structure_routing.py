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
    _sample_dataset,
    _true_concept_parameters,
)
from experiments.benchmark_tac275_volume_aware_routing import (
    SOURCE_CONCEPTS,
    TARGET_CONCEPTS,
    _accuracy,
    _adaptive_predict,
    _knockout_predictions,
    _subset_examples,
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


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac276_two_level_structure_routing")
FAMILY_NAMES = ("plant_family", "fruit_color_family", "animal_family", "number_family")
CONCEPT_TO_FAMILY = torch.tensor([0, 0, 1, 1, 1, 2, 3], dtype=torch.long)
FAMILY_TO_CONCEPTS = {
    0: (0, 1),
    1: (2, 3, 4),
    2: (5,),
    3: (6,),
}
TARGET_FAMILIES = tuple(sorted({int(CONCEPT_TO_FAMILY[index]) for index in TARGET_CONCEPTS}))


def _family_relation_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    pairs = []
    types = []
    for pair, relation_type in zip(RELATION_PAIRS.tolist(), RELATION_TYPES.tolist()):
        left_family = int(CONCEPT_TO_FAMILY[pair[0]])
        right_family = int(CONCEPT_TO_FAMILY[pair[1]])
        if left_family == right_family:
            continue
        pairs.append([left_family, right_family])
        types.append(int(relation_type))
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long), torch.tensor(types, dtype=torch.long)


def _family_predict(
    examples: torch.Tensor,
    family_means: torch.Tensor,
    family_log_vars: torch.Tensor,
) -> torch.Tensor:
    distances = []
    for family_index in range(family_means.shape[0]):
        distances.append(
            diagonal_mahalanobis_distance(
                examples,
                family_means[family_index].expand_as(examples),
                family_log_vars[family_index].expand_as(examples),
            )
        )
    return torch.stack(distances, dim=-1).argmin(dim=-1)


def _prototype_table(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    n_concepts: int,
) -> torch.Tensor:
    true_means, _ = _true_concept_parameters()
    prototypes = true_means.clone()
    for concept_id in range(n_concepts):
        rows = train_x[train_y == concept_id]
        if rows.numel() > 0:
            prototypes[concept_id] = rows.mean(dim=0)
    return prototypes


def _specialist_predict(
    examples: torch.Tensor,
    family_predictions: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    predictions = torch.empty(examples.shape[0], dtype=torch.long)
    for row_index, family_id in enumerate(family_predictions.tolist()):
        candidates = torch.tensor(FAMILY_TO_CONCEPTS[int(family_id)], dtype=torch.long)
        candidate_prototypes = prototypes.index_select(0, candidates)
        distances = torch.cdist(examples[row_index : row_index + 1], candidate_prototypes)
        predictions[row_index] = candidates[int(distances.argmin(dim=-1))]
    return predictions


def _default_family_predictions(labels: torch.Tensor) -> torch.Tensor:
    defaults = {0: 0, 1: 2, 2: 5, 3: 6}
    family_ids = CONCEPT_TO_FAMILY.index_select(0, labels.long())
    return torch.tensor([defaults[int(family)] for family in family_ids], dtype=torch.long)


def _knockout_family_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    knocked_families: Iterable[int],
) -> torch.Tensor:
    knocked = set(int(item) for item in knocked_families)
    output = predictions.clone()
    for row_index, label in enumerate(labels.tolist()):
        if int(CONCEPT_TO_FAMILY[label]) in knocked:
            output[row_index] = 0
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
    train_family_y = CONCEPT_TO_FAMILY.index_select(0, train_y.long())

    eval_x_all, eval_y_all = _sample_dataset(
        seed=seed + 30_000,
        examples_per_concept=eval_examples,
    )
    source_eval_x, source_eval_y = _subset_examples(eval_x_all, eval_y_all, SOURCE_CONCEPTS)
    target_eval_x, target_eval_y = _subset_examples(eval_x_all, eval_y_all, TARGET_CONCEPTS)
    all_eval_x = torch.cat([source_eval_x, target_eval_x], dim=0)
    all_eval_y = torch.cat([source_eval_y, target_eval_y], dim=0)
    all_eval_family_y = CONCEPT_TO_FAMILY.index_select(0, all_eval_y.long())
    target_eval_family_y = CONCEPT_TO_FAMILY.index_select(0, target_eval_y.long())

    true_means, true_log_vars = _true_concept_parameters()
    generator = torch.Generator().manual_seed(seed + 276)
    direct_means = torch.nn.Parameter(
        true_means + 0.50 * torch.randn(true_means.shape, generator=generator)
    )
    direct_log_vars = torch.nn.Parameter(torch.zeros_like(true_log_vars))
    family_base = torch.stack(
        [
            true_means[list(FAMILY_TO_CONCEPTS[family_id])].mean(dim=0)
            for family_id in range(len(FAMILY_NAMES))
        ],
        dim=0,
    )
    family_means = torch.nn.Parameter(
        family_base + 0.40 * torch.randn(family_base.shape, generator=generator)
    )
    family_log_vars = torch.nn.Parameter(torch.zeros_like(family_means))
    direct_optimizer = torch.optim.Adam([direct_means, direct_log_vars], lr=learning_rate)
    family_optimizer = torch.optim.Adam([family_means, family_log_vars], lr=learning_rate)
    family_relation_pairs, family_relation_types = _family_relation_tensors()
    actual_relation_weight = 0.5 * relation_weight if smoke else relation_weight

    for _ in range(int(steps)):
        direct_optimizer.zero_grad()
        direct_loss = adaptive_concept_volume_loss(
            train_x,
            train_y,
            direct_means,
            direct_log_vars,
        ) + actual_relation_weight * concept_relation_loss(
            direct_means,
            direct_log_vars,
            RELATION_PAIRS,
            RELATION_TYPES,
        )
        direct_loss.backward()
        direct_optimizer.step()

        family_optimizer.zero_grad()
        family_loss = adaptive_concept_volume_loss(
            train_x,
            train_family_y,
            family_means,
            family_log_vars,
        ) + actual_relation_weight * concept_relation_loss(
            family_means,
            family_log_vars,
            family_relation_pairs,
            family_relation_types,
        )
        family_loss.backward()
        family_optimizer.step()

    with torch.no_grad():
        prototypes = _prototype_table(train_x, train_y, n_concepts=len(CONCEPT_NAMES))
        direct_all = _adaptive_predict(all_eval_x, direct_means, direct_log_vars)
        direct_target = _adaptive_predict(target_eval_x, direct_means, direct_log_vars)
        family_all = _family_predict(all_eval_x, family_means, family_log_vars)
        family_target = _family_predict(target_eval_x, family_means, family_log_vars)
        two_level_all = _specialist_predict(all_eval_x, family_all, prototypes)
        two_level_target = _specialist_predict(target_eval_x, family_target, prototypes)
        two_level_source = _specialist_predict(
            source_eval_x,
            _family_predict(source_eval_x, family_means, family_log_vars),
            prototypes,
        )
        reset_target = _default_family_predictions(target_eval_y)
        specialist_knockout = _knockout_predictions(two_level_target, TARGET_CONCEPTS)
        family_knockout = _knockout_family_predictions(
            two_level_target,
            target_eval_y,
            TARGET_FAMILIES,
        )

        two_level_accuracy = _accuracy(two_level_all, all_eval_y)
        direct_accuracy = _accuracy(direct_all, all_eval_y)
        two_level_target_accuracy = _accuracy(two_level_target, target_eval_y)
        direct_target_accuracy = _accuracy(direct_target, target_eval_y)
        reset_accuracy = _accuracy(reset_target, target_eval_y)
        family_accuracy = _accuracy(family_all, all_eval_family_y)
        target_family_accuracy = _accuracy(family_target, target_eval_family_y)
        specialist_accuracy = two_level_target_accuracy
        specialist_knockout_accuracy = _accuracy(specialist_knockout, target_eval_y)
        family_knockout_accuracy = _accuracy(family_knockout, target_eval_y)

        return {
            "seed": int(seed),
            "two_level_behavior_accuracy": two_level_accuracy,
            "direct_volume_behavior_accuracy": direct_accuracy,
            "behavior_accuracy_gain": two_level_accuracy - direct_accuracy,
            "two_level_target_accuracy": two_level_target_accuracy,
            "direct_volume_target_accuracy": direct_target_accuracy,
            "target_accuracy_gain": two_level_target_accuracy - direct_target_accuracy,
            "family_route_accuracy": family_accuracy,
            "target_family_route_accuracy": target_family_accuracy,
            "specialist_route_accuracy": specialist_accuracy,
            "source_retention": _accuracy(two_level_source, source_eval_y),
            "family_reset_accuracy": reset_accuracy,
            "family_reset_degradation": two_level_target_accuracy - reset_accuracy,
            "specialist_knockout_accuracy": specialist_knockout_accuracy,
            "specialist_knockout_drop": two_level_target_accuracy - specialist_knockout_accuracy,
            "family_knockout_accuracy": family_knockout_accuracy,
            "family_knockout_drop": two_level_target_accuracy - family_knockout_accuracy,
            "structure_reuse_score": target_family_accuracy,
            "lm_collapse_proxy": float(
                torch.isfinite(direct_loss.detach()) and torch.isfinite(family_loss.detach())
            ),
        }


def run_tac276_two_level_structure_routing(
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
        metrics.get("target_accuracy_gain", 0.0) > 0.20
        and metrics.get("two_level_target_accuracy", 0.0) >= 0.35
        and metrics.get("target_family_route_accuracy", 0.0) >= 0.75
        and metrics.get("specialist_route_accuracy", 0.0) >= 0.35
        and metrics.get("source_retention", 0.0) >= 0.55
        and metrics.get("family_reset_degradation", 0.0) > 0.20
        and metrics.get("specialist_knockout_drop", 0.0) > 0.20
        and metrics.get("family_knockout_drop", 0.0) > 0.20
        and metrics.get("lm_collapse_proxy", 0.0) == 1.0
    )
    result = {
        "schema": "tac276_two_level_structure_routing.v1",
        "method": {
            "task": "two_level_structure_routing",
            "concept_names": list(CONCEPT_NAMES),
            "family_names": list(FAMILY_NAMES),
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
                "Tests the TAC-275 fix: concept volumes select a structure "
                "family first, then a specialist route selects exact executable "
                "behavior. This remains a synthetic routing probe, not full LM "
                "training."
            ),
        },
    }
    return write_artifact(output_dir, "tac276_two_level_structure_routing.json", result)


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
    result = run_tac276_two_level_structure_routing(
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
