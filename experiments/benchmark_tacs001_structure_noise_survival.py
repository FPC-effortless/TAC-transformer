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
    _sample_dataset,
    _true_concept_parameters,
)
from experiments.benchmark_tac275_volume_aware_routing import (
    SOURCE_CONCEPTS,
    TARGET_CONCEPTS,
    _accuracy,
    _subset_examples,
)
from experiments.benchmark_tac276_two_level_structure_routing import (
    CONCEPT_TO_FAMILY,
    FAMILY_NAMES,
    FAMILY_TO_CONCEPTS,
    _family_predict,
    _prototype_table,
    _specialist_predict,
)
from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    write_artifact,
)
from tac_transformer.research_directions import (
    StructureMemoryRecord,
    adaptive_concept_volume_loss,
    structure_memory_score,
    update_structure_memory,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tacs001_structure_noise_survival")


def _fit_two_level_state(
    *,
    seed: int,
    source_examples: int,
    target_shots: int,
    eval_examples: int,
    steps: int,
    learning_rate: float,
) -> dict[str, torch.Tensor]:
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

    true_means, _ = _true_concept_parameters()
    family_base = torch.stack(
        [
            true_means[list(FAMILY_TO_CONCEPTS[family_id])].mean(dim=0)
            for family_id in range(len(FAMILY_NAMES))
        ],
        dim=0,
    )
    generator = torch.Generator().manual_seed(seed + 1001)
    family_means = torch.nn.Parameter(
        family_base + 0.40 * torch.randn(family_base.shape, generator=generator)
    )
    family_log_vars = torch.nn.Parameter(torch.zeros_like(family_means))
    optimizer = torch.optim.Adam([family_means, family_log_vars], lr=learning_rate)
    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = adaptive_concept_volume_loss(
            train_x,
            train_family_y,
            family_means,
            family_log_vars,
        )
        loss.backward()
        optimizer.step()

    eval_x_all, eval_y_all = _sample_dataset(
        seed=seed + 40_000,
        examples_per_concept=eval_examples,
    )
    source_eval_x, source_eval_y = _subset_examples(eval_x_all, eval_y_all, SOURCE_CONCEPTS)
    target_eval_x, target_eval_y = _subset_examples(eval_x_all, eval_y_all, TARGET_CONCEPTS)
    return {
        "family_means": family_means.detach(),
        "family_log_vars": family_log_vars.detach(),
        "prototypes": _prototype_table(train_x, train_y, n_concepts=7),
        "source_eval_x": source_eval_x,
        "source_eval_y": source_eval_y,
        "target_eval_x": target_eval_x,
        "target_eval_y": target_eval_y,
    }


def _predict_two_level(
    examples: torch.Tensor,
    *,
    family_means: torch.Tensor,
    family_log_vars: torch.Tensor,
    prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    family = _family_predict(examples, family_means, family_log_vars)
    concept = _specialist_predict(examples, family, prototypes)
    return family, concept


def _row(
    *,
    seed: int,
    source_examples: int,
    target_shots: int,
    eval_examples: int,
    steps: int,
    learning_rate: float,
    noise_std: float,
) -> dict[str, float | int]:
    state = _fit_two_level_state(
        seed=seed,
        source_examples=source_examples,
        target_shots=target_shots,
        eval_examples=eval_examples,
        steps=steps,
        learning_rate=learning_rate,
    )
    generator = torch.Generator().manual_seed(seed + 2001)
    noisy_target_x = state["target_eval_x"] + noise_std * torch.randn(
        state["target_eval_x"].shape,
        generator=generator,
    )
    noisy_source_x = state["source_eval_x"] + noise_std * torch.randn(
        state["source_eval_x"].shape,
        generator=generator,
    )
    clean_target_family, clean_target = _predict_two_level(
        state["target_eval_x"],
        family_means=state["family_means"],
        family_log_vars=state["family_log_vars"],
        prototypes=state["prototypes"],
    )
    noisy_target_family, noisy_target = _predict_two_level(
        noisy_target_x,
        family_means=state["family_means"],
        family_log_vars=state["family_log_vars"],
        prototypes=state["prototypes"],
    )
    clean_source_family, clean_source = _predict_two_level(
        state["source_eval_x"],
        family_means=state["family_means"],
        family_log_vars=state["family_log_vars"],
        prototypes=state["prototypes"],
    )
    noisy_source_family, noisy_source = _predict_two_level(
        noisy_source_x,
        family_means=state["family_means"],
        family_log_vars=state["family_log_vars"],
        prototypes=state["prototypes"],
    )
    target_family_y = CONCEPT_TO_FAMILY.index_select(0, state["target_eval_y"].long())
    source_family_y = CONCEPT_TO_FAMILY.index_select(0, state["source_eval_y"].long())
    clean_target_accuracy = _accuracy(clean_target, state["target_eval_y"])
    noisy_target_accuracy = _accuracy(noisy_target, state["target_eval_y"])
    clean_family_accuracy = _accuracy(clean_target_family, target_family_y)
    noisy_family_accuracy = _accuracy(noisy_target_family, target_family_y)
    clean_source_accuracy = _accuracy(clean_source, state["source_eval_y"])
    noisy_source_accuracy = _accuracy(noisy_source, state["source_eval_y"])
    clean_source_family_accuracy = _accuracy(clean_source_family, source_family_y)
    noisy_source_family_accuracy = _accuracy(noisy_source_family, source_family_y)

    target_retention = noisy_target_accuracy / max(clean_target_accuracy, 1e-6)
    family_retention = noisy_family_accuracy / max(clean_family_accuracy, 1e-6)
    source_retention = noisy_source_accuracy / max(clean_source_accuracy, 1e-6)
    noise_recovery = (target_retention + family_retention + source_retention) / 3.0
    record = update_structure_memory(
        StructureMemoryRecord(structure_id="two_level_structure_router"),
        task_descriptor="noise_attack",
        success=noise_recovery >= 0.75,
        reset_drop=max(clean_target_accuracy - noisy_target_accuracy, 0.0),
        knockout_drop=max(clean_family_accuracy - noisy_family_accuracy, 0.0),
        transfer_to="source_structures",
        transfer_gain=max(source_retention - 0.75, 0.0),
    )
    return {
        "seed": int(seed),
        "clean_target_accuracy": clean_target_accuracy,
        "noisy_target_accuracy": noisy_target_accuracy,
        "target_noise_retention": target_retention,
        "clean_family_accuracy": clean_family_accuracy,
        "noisy_family_accuracy": noisy_family_accuracy,
        "family_noise_retention": family_retention,
        "clean_source_accuracy": clean_source_accuracy,
        "noisy_source_accuracy": noisy_source_accuracy,
        "source_noise_retention": source_retention,
        "clean_source_family_accuracy": clean_source_family_accuracy,
        "noisy_source_family_accuracy": noisy_source_family_accuracy,
        "noise_recovery_score": noise_recovery,
        "structure_memory_survival_score": record.survival_score,
        "noise_survival_score": structure_memory_score(record),
    }


def run_tacs001_structure_noise_survival(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    source_examples: int = 48,
    target_shots: int = 4,
    eval_examples: int = 48,
    steps: int = 180,
    learning_rate: float = 0.04,
    noise_std: float = 0.08,
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
        _row(
            seed=seed,
            source_examples=actual_source,
            target_shots=actual_target,
            eval_examples=actual_eval,
            steps=actual_steps,
            learning_rate=learning_rate,
            noise_std=noise_std,
        )
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("clean_target_accuracy", 0.0) >= 0.35
        and metrics.get("target_noise_retention", 0.0) >= 0.75
        and metrics.get("family_noise_retention", 0.0) >= 0.90
        and metrics.get("source_noise_retention", 0.0) >= 0.80
        and metrics.get("noise_recovery_score", 0.0) >= 0.82
        and metrics.get("noise_survival_score", 0.0) >= 0.35
    )
    result = {
        "schema": "tacs001_structure_noise_survival.v1",
        "method": {
            "task": "structure_noise_survival",
            "source_model": "tac276_two_level_structure_routing",
            "noise_std": float(noise_std),
            "seeds": list(seed_list),
            "source_examples": actual_source,
            "target_shots": actual_target,
            "eval_examples": actual_eval,
            "steps": actual_steps,
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Tests survival under embedding noise for the two-level structure "
                "router. This is a Stage 2 survival probe, not adversarial or "
                "distribution-shift survival."
            ),
        },
    }
    return write_artifact(output_dir, "tacs001_structure_noise_survival.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-examples", type=int, default=48)
    parser.add_argument("--target-shots", type=int, default=4)
    parser.add_argument("--eval-examples", type=int, default=48)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--noise-std", type=float, default=0.08)
    args = parser.parse_args()
    result = run_tacs001_structure_noise_survival(
        output_dir=args.output_dir,
        seeds=args.seeds,
        source_examples=args.source_examples,
        target_shots=args.target_shots,
        eval_examples=args.eval_examples,
        steps=args.steps,
        learning_rate=args.learning_rate,
        noise_std=args.noise_std,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
