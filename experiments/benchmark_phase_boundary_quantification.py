from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_persistent_identity_broader_tasks import TASK_FAMILIES
from experiments.benchmark_relaxed_identity_routing_memory import (
    RelaxedRoutingMemoryModel,
    _memories_by_horizon,
    _mixture_program_loss,
    _normalized_mutual_information,
    _predict_from_logits,
    _row_program_targets_by_horizon,
    _row_route_logits,
    _tensorize_suite,
    build_relaxed_identity_sequence_suite,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/phase_boundary_quantification_2026_06_05")
DEFAULT_MEMORY_LEVELS = [0, 1, 2, 3]
DEFAULT_ROUTING_LEVELS = [0, 2, 4]
DEFAULT_TASK_LEVELS = [0, 2, 4]
DEFAULT_HORIZON_LEVELS = [0, 2, 4]
BOUNDARY_GAP_THRESHOLD = 0.10

MEMORY_LEVELS = {
    0: {"name": "perfect_memory", "description": "clean support-derived memory"},
    1: {"name": "slight_noise", "description": "small Gaussian memory perturbation"},
    2: {"name": "partial_corruption", "description": "25% blend with shuffled memory"},
    3: {"name": "shuffled", "description": "identity memory rolled across rows"},
    4: {"name": "adversarial", "description": "anti-correlated negative memory"},
}
ROUTING_LEVELS = {
    0: {"name": "deterministic_argmax", "temperature": 0.01},
    1: {"name": "low_temperature", "temperature": 0.50},
    2: {"name": "moderate_temperature", "temperature": 1.00},
    3: {"name": "high_entropy", "temperature": 2.50},
    4: {"name": "uniform", "temperature": math.inf},
}
TASK_LEVELS = {
    0: {"name": "single_domain", "task_families": ["transfer_learning"]},
    1: {"name": "clustered_tasks", "task_families": ["transfer_learning", "agent_memory"]},
    2: {"name": "mixed_domains", "task_families": list(TASK_FAMILIES)},
    3: {
        "name": "distribution_shift",
        "task_families": ["multi_hop_reasoning", "language_like_instruction"],
    },
    4: {"name": "adversarially_mixed_tasks", "task_families": list(reversed(TASK_FAMILIES))},
}
HORIZON_LEVELS = {
    0: {"name": "single_step", "horizon_windows": 1},
    1: {"name": "two_step_dependency", "horizon_windows": 2},
    2: {"name": "five_step_memory_dependency", "horizon_windows": 5},
    3: {"name": "ten_step_chain_proxy", "horizon_windows": 8},
    4: {"name": "long_horizon_latent_dependency", "horizon_windows": 10},
}


def run_phase_boundary_quantification_harness(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    model_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
    training_steps: int,
    memory_levels: Sequence[int] = DEFAULT_MEMORY_LEVELS,
    routing_levels: Sequence[int] = DEFAULT_ROUTING_LEVELS,
    task_levels: Sequence[int] = DEFAULT_TASK_LEVELS,
    horizon_levels: Sequence[int] = DEFAULT_HORIZON_LEVELS,
    boundary_gap_threshold: float = BOUNDARY_GAP_THRESHOLD,
    learning_rate: float = 0.035,
) -> dict[str, Any]:
    _validate_levels(memory_levels, MEMORY_LEVELS, "memory")
    _validate_levels(routing_levels, ROUTING_LEVELS, "routing")
    _validate_levels(task_levels, TASK_LEVELS, "task")
    _validate_levels(horizon_levels, HORIZON_LEVELS, "horizon")
    max_horizon_windows = max(
        HORIZON_LEVELS[int(level)]["horizon_windows"] for level in horizon_levels
    )
    suite = {
        "train": build_relaxed_identity_sequence_suite(
            seeds=train_seeds,
            identities_per_seed=identities_per_seed,
            examples_per_task=examples_per_task,
            horizon_windows=max_horizon_windows,
            vocab_size=vocab_size,
        ),
        "eval": build_relaxed_identity_sequence_suite(
            seeds=eval_seeds,
            identities_per_seed=identities_per_seed,
            examples_per_task=examples_per_task,
            horizon_windows=max_horizon_windows,
            vocab_size=vocab_size,
        ),
    }
    train_batch = _tensorize_suite(suite["train"])
    eval_batch = _tensorize_suite(suite["eval"])
    seed_models = [
        _train_phase_model(
            train_batch,
            model_seed=int(model_seed),
            vocab_size=vocab_size,
            horizon_windows=max_horizon_windows,
            training_steps=training_steps,
            learning_rate=learning_rate,
        )
        for model_seed in model_seeds
    ]
    clean_routes = {
        int(model_seed): _clean_identity_routes(model, eval_batch, max_horizon_windows)
        for model_seed, model in seed_models
    }
    phase_grid = []
    for memory_level, routing_level, task_level, horizon_level in product(
        [int(level) for level in memory_levels],
        [int(level) for level in routing_levels],
        [int(level) for level in task_levels],
        [int(level) for level in horizon_levels],
    ):
        seed_cells = [
            _evaluate_phase_cell(
                model,
                eval_batch,
                model_seed=int(model_seed),
                clean_routes=clean_routes[int(model_seed)],
                memory_level=memory_level,
                routing_level=routing_level,
                task_level=task_level,
                horizon_level=horizon_level,
            )
            for model_seed, model in seed_models
        ]
        phase_grid.append(_aggregate_cell(seed_cells))
    phase_sharpness = compute_phase_sharpness(phase_grid)
    phase_boundaries = estimate_phase_boundaries(
        phase_grid,
        boundary_gap_threshold=boundary_gap_threshold,
    )
    heatmaps = build_phase_heatmaps(phase_grid)
    aggregate = _aggregate_grid(phase_grid, phase_boundaries)
    decision = _decision(phase_grid, phase_boundaries, aggregate)
    return {
        "schema": "phase_boundary_quantification.v1",
        "hypothesis": (
            "TAC-185's identity advantage occupies a bounded phase region in "
            "memory-coherence, routing-entropy, task-entropy, and horizon-depth "
            "space. TAC-186 maps where carried identity state stops helping and "
            "starts behaving like reset or harmful memory."
        ),
        "measurement_contract": {
            "optimizes_accuracy": False,
            "estimates_phase_boundary": True,
            "trains_once_then_perturbs_evaluation": True,
            "primary_order_parameter": "performance_gap = carried_accuracy - reset_accuracy",
            "collapse_index": "1 - shuffled_memory_accuracy / carried_accuracy",
        },
        "axis_definitions": {
            "memory": {str(k): v for k, v in MEMORY_LEVELS.items() if k in memory_levels},
            "routing": {str(k): v for k, v in ROUTING_LEVELS.items() if k in routing_levels},
            "task": {str(k): v for k, v in TASK_LEVELS.items() if k in task_levels},
            "horizon": {str(k): v for k, v in HORIZON_LEVELS.items() if k in horizon_levels},
        },
        "grid_summary": {
            "memory_levels": [int(level) for level in memory_levels],
            "routing_levels": [int(level) for level in routing_levels],
            "task_levels": [int(level) for level in task_levels],
            "horizon_levels": [int(level) for level in horizon_levels],
            "cell_count": len(phase_grid),
            "model_seed_count": len(model_seeds),
            "train_rows": len(suite["train"]["rows"]),
            "eval_rows": len(suite["eval"]["rows"]),
            "max_horizon_windows": int(max_horizon_windows),
        },
        "phase_grid": phase_grid,
        "phase_sharpness": phase_sharpness,
        "phase_boundaries": phase_boundaries,
        "phase_heatmaps": heatmaps,
        "aggregate_metrics": aggregate,
        "decision": decision,
        "boundary": (
            "This is a controlled measurement instrument over the TAC-185 local "
            "probe. It maps phase behavior under synthetic perturbations; it is "
            "not a full TACTransformerLM checkpoint result, not a real-world "
            "language benchmark, and not an accuracy-promotion run."
        ),
    }


def estimate_phase_boundaries(
    phase_grid: Sequence[dict[str, Any]],
    *,
    boundary_gap_threshold: float,
) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in phase_grid:
        key = (int(row["task_level"]), int(row["horizon_level"]))
        grouped.setdefault(key, []).append(row)
    result: dict[str, Any] = {}
    for (task_level, horizon_level), rows in grouped.items():
        by_memory: dict[int, list[float]] = {}
        for row in rows:
            by_memory.setdefault(int(row["memory_level"]), []).append(
                float(row["performance_gap"])
            )
        curve = [
            {
                "memory_level": memory_level,
                "mean_performance_gap": mean(values),
            }
            for memory_level, values in sorted(by_memory.items())
        ]
        boundary_level = None
        for point in curve:
            if point["mean_performance_gap"] <= float(boundary_gap_threshold):
                boundary_level = int(point["memory_level"])
                break
        key = f"{task_level}:{horizon_level}"
        result[key] = {
            "task_level": int(task_level),
            "horizon_level": int(horizon_level),
            "boundary_status": "crossed" if boundary_level is not None else "not_crossed",
            "memory_boundary_level": boundary_level,
            "gap_threshold": float(boundary_gap_threshold),
            "curve": curve,
        }
    return result


def compute_phase_sharpness(phase_grid: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "memory": _axis_sharpness(phase_grid, axis="memory_level"),
        "routing": _axis_sharpness(phase_grid, axis="routing_level"),
        "task": _axis_sharpness(phase_grid, axis="task_level"),
        "horizon": _axis_sharpness(phase_grid, axis="horizon_level"),
    }


def build_phase_heatmaps(phase_grid: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "memory_x_routing_gap": _projection(
            phase_grid,
            x_axis="memory_level",
            y_axis="routing_level",
            value="performance_gap",
        ),
        "memory_x_horizon_collapse_index": _projection(
            phase_grid,
            x_axis="memory_level",
            y_axis="horizon_level",
            value="collapse_index",
        ),
        "routing_x_task_accuracy_variance": _projection(
            phase_grid,
            x_axis="routing_level",
            y_axis="task_level",
            value="accuracy_variance",
        ),
    }


def format_phase_boundary_markdown(result: dict[str, Any]) -> str:
    metrics = result["aggregate_metrics"]
    lines = [
        "# Phase Boundary Quantification",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        "- Contract: measurement harness, not an accuracy optimizer.",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| cell count | {result['grid_summary']['cell_count']} |",
        f"| mean performance gap | {metrics['mean_performance_gap']:.4f} |",
        f"| min performance gap | {metrics['min_performance_gap']:.4f} |",
        f"| max phase sharpness | {metrics['max_phase_sharpness']:.4f} |",
        f"| harmful memory cells | {metrics['harmful_memory_cell_count']} |",
        f"| mapped boundaries | {metrics['mapped_boundary_count']} |",
        "",
        "## Axis Sharpness",
        "",
        "| Axis | Max slope | Mean slope |",
        "| --- | ---: | ---: |",
    ]
    for axis, sharpness in result["phase_sharpness"].items():
        lines.append(
            f"| {axis} | {sharpness['max_abs_slope']:.4f} | {sharpness['mean_abs_slope']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary Slices",
            "",
            "| Task:Horizon | Status | Memory boundary |",
            "| --- | --- | ---: |",
        ]
    )
    for key, boundary in sorted(result["phase_boundaries"].items()):
        lines.append(
            "| {key} | {status} | {level} |".format(
                key=key,
                status=boundary["boundary_status"],
                level=(
                    boundary["memory_boundary_level"]
                    if boundary["memory_boundary_level"] is not None
                    else "n/a"
                ),
            )
        )
    lines.extend(["", "## Boundary", "", result["boundary"], ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Map TAC-185 phase boundaries over memory/routing/task/horizon axes."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[101])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[5])
    parser.add_argument("--identities-per-seed", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--training-steps", type=int, default=160)
    parser.add_argument("--memory-levels", nargs="+", type=int, default=DEFAULT_MEMORY_LEVELS)
    parser.add_argument("--routing-levels", nargs="+", type=int, default=DEFAULT_ROUTING_LEVELS)
    parser.add_argument("--task-levels", nargs="+", type=int, default=DEFAULT_TASK_LEVELS)
    parser.add_argument("--horizon-levels", nargs="+", type=int, default=DEFAULT_HORIZON_LEVELS)
    parser.add_argument("--boundary-gap-threshold", type=float, default=BOUNDARY_GAP_THRESHOLD)
    args = parser.parse_args(argv)

    result = run_phase_boundary_quantification_harness(
        train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds,
        model_seeds=args.model_seeds,
        identities_per_seed=args.identities_per_seed,
        examples_per_task=args.examples_per_task,
        vocab_size=args.vocab_size,
        training_steps=args.training_steps,
        memory_levels=args.memory_levels,
        routing_levels=args.routing_levels,
        task_levels=args.task_levels,
        horizon_levels=args.horizon_levels,
        boundary_gap_threshold=args.boundary_gap_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_boundary_quantification.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "phase_heatmaps.json").write_text(
        json.dumps(result["phase_heatmaps"], indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_phase_boundary_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return result


def _train_phase_model(
    train_batch: dict[str, Any],
    *,
    model_seed: int,
    vocab_size: int,
    horizon_windows: int,
    training_steps: int,
    learning_rate: float,
) -> tuple[int, RelaxedRoutingMemoryModel]:
    torch.manual_seed(int(model_seed))
    model = RelaxedRoutingMemoryModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    for step_index in range(int(training_steps)):
        optimizer.zero_grad(set_to_none=True)
        memories = _memories_by_horizon(
            model,
            train_batch,
            horizon_windows=horizon_windows,
            memory_noise_std=0.01,
            model_seed=int(model_seed),
            step_index=int(step_index),
        )
        row_logits = _row_route_logits(model, memories, train_batch)
        program_targets, target_values = _row_program_targets_by_horizon(train_batch)
        loss = _mixture_program_loss(row_logits, program_targets, target_values)
        loss.backward()
        optimizer.step()
    return int(model_seed), model


def _evaluate_phase_cell(
    model: RelaxedRoutingMemoryModel,
    batch: dict[str, Any],
    *,
    model_seed: int,
    clean_routes: Sequence[int],
    memory_level: int,
    routing_level: int,
    task_level: int,
    horizon_level: int,
) -> dict[str, Any]:
    horizon_windows = int(HORIZON_LEVELS[horizon_level]["horizon_windows"])
    with torch.inference_mode():
        clean_memories = _memories_by_horizon(
            model,
            batch,
            horizon_windows=horizon_windows,
        )
        memories = [
            _perturb_memory(memory, level=memory_level, model_seed=model_seed + index)
            for index, memory in enumerate(clean_memories)
        ]
        filtered = _filtered_batch(batch, task_level=task_level, horizon_windows=horizon_windows)
        carried_logits = _row_route_logits(model, memories, filtered)
        carried_predictions = _predict_with_routing_level(
            carried_logits,
            filtered["program_targets_by_horizon"],
            routing_level=routing_level,
        )
        carried_correct = carried_predictions.eq(filtered["target_values_by_horizon"])

        reset_memory = torch.zeros_like(memories[0])
        reset_logits = model.route_logits(reset_memory)[filtered["identity_index"]]
        reset_predictions = _predict_with_routing_level(
            reset_logits,
            filtered["program_targets"],
            routing_level=routing_level,
        )
        reset_correct = reset_predictions.eq(filtered["target_values"])

        shuffled_memories = [memory.roll(shifts=1, dims=0) for memory in clean_memories]
        shuffled_logits = _row_route_logits(model, shuffled_memories, filtered)
        shuffled_predictions = _predict_with_routing_level(
            shuffled_logits,
            filtered["program_targets_by_horizon"],
            routing_level=routing_level,
        )
        shuffled_correct = shuffled_predictions.eq(filtered["target_values_by_horizon"])

        perturbed_routes = torch.argmax(model.route_logits(memories[0]), dim=-1).tolist()
        routing_stability = _normalized_mutual_information(clean_routes, perturbed_routes)
        carried_accuracy = float(carried_correct.float().mean().item())
        reset_accuracy = float(reset_correct.float().mean().item())
        shuffled_accuracy = float(shuffled_correct.float().mean().item())
        performance_gap = carried_accuracy - reset_accuracy
        collapse_index = 1.0 - (
            shuffled_accuracy / max(carried_accuracy, 1e-8)
        )
        accuracies = [carried_accuracy, reset_accuracy, shuffled_accuracy]
        return {
            "model_seed": int(model_seed),
            "memory_level": int(memory_level),
            "routing_level": int(routing_level),
            "task_level": int(task_level),
            "horizon_level": int(horizon_level),
            "memory_name": MEMORY_LEVELS[memory_level]["name"],
            "routing_name": ROUTING_LEVELS[routing_level]["name"],
            "task_name": TASK_LEVELS[task_level]["name"],
            "horizon_name": HORIZON_LEVELS[horizon_level]["name"],
            "horizon_windows": int(horizon_windows),
            "task_families": list(TASK_LEVELS[task_level]["task_families"]),
            "example_count": int(filtered["target_values"].numel()),
            "carried_accuracy": carried_accuracy,
            "reset_accuracy": reset_accuracy,
            "shuffled_memory_accuracy": shuffled_accuracy,
            "performance_gap": performance_gap,
            "collapse_index": collapse_index,
            "routing_stability": routing_stability,
            "accuracy_variance": float(pstdev(accuracies) ** 2),
            "memory_harmful": carried_accuracy < reset_accuracy,
        }


def _aggregate_cell(seed_cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first = seed_cells[0]
    numeric_keys = [
        "carried_accuracy",
        "reset_accuracy",
        "shuffled_memory_accuracy",
        "performance_gap",
        "collapse_index",
        "routing_stability",
        "accuracy_variance",
    ]
    result = {
        key: first[key]
        for key in [
            "memory_level",
            "routing_level",
            "task_level",
            "horizon_level",
            "memory_name",
            "routing_name",
            "task_name",
            "horizon_name",
            "horizon_windows",
            "task_families",
            "example_count",
        ]
    }
    for key in numeric_keys:
        result[key] = float(mean(float(cell[key]) for cell in seed_cells))
    result["memory_harmful"] = any(bool(cell["memory_harmful"]) for cell in seed_cells)
    result["seed_count"] = len(seed_cells)
    return result


def _clean_identity_routes(
    model: RelaxedRoutingMemoryModel,
    batch: dict[str, Any],
    horizon_windows: int,
) -> list[int]:
    with torch.inference_mode():
        memory = _memories_by_horizon(model, batch, horizon_windows=horizon_windows)[0]
        return torch.argmax(model.route_logits(memory), dim=-1).tolist()


def _perturb_memory(memory: torch.Tensor, *, level: int, model_seed: int) -> torch.Tensor:
    if level == 0:
        return memory
    generator = torch.Generator(device=memory.device).manual_seed(int(model_seed) * 7919 + int(level))
    if level == 1:
        noise = torch.randn(
            memory.shape,
            generator=generator,
            device=memory.device,
            dtype=memory.dtype,
        )
        return memory + noise * 0.08
    if level == 2:
        return memory * 0.75 + memory.roll(shifts=1, dims=0) * 0.25
    if level == 3:
        return memory.roll(shifts=1, dims=0)
    if level == 4:
        return -memory.roll(shifts=1, dims=0)
    raise ValueError(f"unknown memory level: {level}")


def _predict_with_routing_level(
    row_logits: torch.Tensor,
    program_targets: torch.Tensor,
    *,
    routing_level: int,
) -> torch.Tensor:
    if routing_level == 4:
        routes = torch.remainder(
            torch.arange(row_logits.shape[0], device=row_logits.device),
            row_logits.shape[-1],
        )
    else:
        temperature = float(ROUTING_LEVELS[int(routing_level)]["temperature"])
        scaled = row_logits / max(temperature, 1e-6)
        routes = torch.argmax(scaled, dim=-1)
        if routing_level in {2, 3}:
            # Deterministic entropy proxy: progressively replace some confident
            # choices with adjacent routes without sampling noise in tests.
            stride = 5 if routing_level == 2 else 2
            mask = torch.remainder(torch.arange(routes.numel(), device=routes.device), stride).eq(0)
            routes = torch.where(mask, torch.remainder(routes + 1, row_logits.shape[-1]), routes)
    return program_targets[torch.arange(program_targets.shape[0]), routes]


def _filtered_batch(
    batch: dict[str, Any],
    *,
    task_level: int,
    horizon_windows: int,
) -> dict[str, Any]:
    task_allowed = set(TASK_LEVELS[int(task_level)]["task_families"])
    row_mask = torch.tensor(
        [
            task in task_allowed and int(horizon) < int(horizon_windows)
            for task, horizon in zip(batch["task_families"], batch["horizon_window"].tolist())
        ],
        dtype=torch.bool,
    )
    filtered = {
        "support_inputs": batch["support_inputs"],
        "support_targets": batch["support_targets"],
        "true_rule_index": batch["true_rule_index"],
        "identity_index": batch["identity_index"][row_mask],
        "target_values": batch["target_values"][row_mask],
        "program_targets": batch["program_targets"][row_mask],
        "horizon_window": batch["horizon_window"][row_mask],
        "task_families": [
            task for task, keep in zip(batch["task_families"], row_mask.tolist()) if keep
        ],
    }
    program_targets, target_values = _row_program_targets_by_horizon(filtered)
    filtered["program_targets_by_horizon"] = program_targets
    filtered["target_values_by_horizon"] = target_values
    return filtered


def _axis_sharpness(phase_grid: Sequence[dict[str, Any]], *, axis: str) -> dict[str, Any]:
    other_axes = [
        name
        for name in ["memory_level", "routing_level", "task_level", "horizon_level"]
        if name != axis
    ]
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for row in phase_grid:
        key = tuple(int(row[name]) for name in other_axes)
        grouped.setdefault(key, []).append(row)
    slopes = []
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: int(row[axis]))
        for left, right in zip(ordered, ordered[1:]):
            dx = int(right[axis]) - int(left[axis])
            if dx == 0:
                continue
            slopes.append(abs((float(right["performance_gap"]) - float(left["performance_gap"])) / dx))
    return {
        "max_abs_slope": max(slopes) if slopes else 0.0,
        "mean_abs_slope": mean(slopes) if slopes else 0.0,
        "slope_count": len(slopes),
    }


def _projection(
    phase_grid: Sequence[dict[str, Any]],
    *,
    x_axis: str,
    y_axis: str,
    value: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in phase_grid:
        key = (int(row[x_axis]), int(row[y_axis]))
        grouped.setdefault(key, []).append(float(row[value]))
    return [
        {
            x_axis: x,
            y_axis: y,
            value: mean(values),
            "sample_count": len(values),
        }
        for (x, y), values in sorted(grouped.items())
    ]


def _aggregate_grid(
    phase_grid: Sequence[dict[str, Any]],
    phase_boundaries: dict[str, Any],
) -> dict[str, Any]:
    gaps = [float(row["performance_gap"]) for row in phase_grid]
    sharpness = compute_phase_sharpness(phase_grid)
    max_phase_sharpness = max(
        axis["max_abs_slope"] for axis in sharpness.values()
    )
    return {
        "mean_performance_gap": mean(gaps),
        "min_performance_gap": min(gaps),
        "max_performance_gap": max(gaps),
        "max_phase_sharpness": max_phase_sharpness,
        "harmful_memory_cell_count": sum(
            1 for row in phase_grid if bool(row["memory_harmful"]) or row["performance_gap"] < 0.0
        ),
        "mapped_boundary_count": sum(
            1 for boundary in phase_boundaries.values() if boundary["boundary_status"] == "crossed"
        ),
        "cell_count": len(phase_grid),
    }


def _decision(
    phase_grid: Sequence[dict[str, Any]],
    phase_boundaries: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    coverage_ok = len(phase_grid) > 0
    boundary_ok = aggregate["mapped_boundary_count"] > 0
    harmful_ok = aggregate["harmful_memory_cell_count"] > 0
    if coverage_ok and boundary_ok and harmful_ok:
        return {
            "status": "phase_boundary_mapped",
            "reason": (
                "The harness produced a non-empty 4D phase grid, found at least "
                "one memory-coherence boundary where identity advantage falls "
                "below threshold, and detected harmful-memory cells."
            ),
            "recommendation": (
                "Use TAC-186 as the measurement layer before future routing or "
                "memory changes. Do not optimize this harness for accuracy; use "
                "it to track boundary movement."
            ),
        }
    return {
        "status": "phase_boundary_not_mapped",
        "reason": (
            "The harness did not produce enough coverage, boundary crossings, or "
            "harmful-memory cells to quantify the phase surface."
        ),
        "recommendation": (
            "Increase perturbation strength or grid coverage before drawing "
            "architecture conclusions."
        ),
    }


def _validate_levels(
    levels: Sequence[int],
    definitions: dict[int, Any],
    axis: str,
) -> None:
    missing = [int(level) for level in levels if int(level) not in definitions]
    if missing:
        raise ValueError(f"unknown {axis} levels: {missing}")


if __name__ == "__main__":
    main()
