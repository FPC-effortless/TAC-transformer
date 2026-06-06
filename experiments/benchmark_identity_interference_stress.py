from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_persistent_identity_broader_tasks import (
    TASK_FAMILIES,
    _answer_text,
    _solve_row,
)
from experiments.benchmark_trained_identity_collapse_recovery import (
    DEFAULT_OUTPUT_DIR as TAC183_OUTPUT_DIR,
    TrainableIdentityStateLearner,
    _aggregate,
    _evaluate_model,
    _mixture_program_loss,
    _tensorize_suite,
    build_trained_identity_suite,
    evaluate_non_identity_controls,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/identity_interference_stress_2026_06_05")
BASE_COLLAPSE_PRESSURE = 0.04
BASE_GRADIENT_NOISE_STD = 0.015


def run_identity_interference_stress_probe(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    model_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
    training_steps: int,
    pressure_values: Sequence[float],
) -> dict[str, Any]:
    scenarios = {
        "identity_collision": run_identity_collision_scenario(
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            model_seeds=model_seeds,
            identities_per_seed=identities_per_seed,
            examples_per_task=examples_per_task,
            vocab_size=vocab_size,
            training_steps=training_steps,
        ),
        "distribution_shift": run_distribution_shift_scenario(
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            model_seeds=model_seeds,
            identities_per_seed=identities_per_seed,
            examples_per_task=examples_per_task,
            vocab_size=vocab_size,
            training_steps=training_steps,
        ),
        "adversarial_pressure_sweep": run_adversarial_pressure_sweep(
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            model_seeds=model_seeds,
            identities_per_seed=identities_per_seed,
            examples_per_task=examples_per_task,
            vocab_size=vocab_size,
            training_steps=training_steps,
            pressure_values=pressure_values,
        ),
        "scaled_load": run_scaled_load_scenario(
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            model_seeds=model_seeds,
            identities_per_seed=max(int(identities_per_seed) * 2, 8),
            examples_per_task=examples_per_task,
            vocab_size=vocab_size,
            training_steps=training_steps,
        ),
    }
    decision = _decision(scenarios)
    return {
        "schema": "identity_interference_stress.v1",
        "hypothesis": (
            "The TAC-183 learned identity-state route should preserve separation "
            "under structured interference: identity collisions, task-family "
            "distribution shift, collapse-pressure escalation, and scaled "
            "identity load."
        ),
        "source_layer": {
            "prior_ticket": "TAC-183",
            "prior_artifact": str(TAC183_OUTPUT_DIR),
            "training_contract": (
                "Support/query supervision only; hidden rule labels are used for "
                "evaluation metrics, not loss."
            ),
        },
        "scenarios": scenarios,
        "decision": decision,
        "boundary": (
            "This maps controlled interference behavior for the TAC-183 trained "
            "learner. It does not prove language generalization, scaling to a "
            "full TACTransformerLM checkpoint, or arbitrary optimizer stability."
        ),
    }


def run_identity_collision_scenario(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    model_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
    training_steps: int,
) -> dict[str, Any]:
    suite = build_trained_identity_suite(
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        identities_per_seed=identities_per_seed,
        examples_per_task=examples_per_task,
        vocab_size=vocab_size,
    )
    _apply_identity_collision(suite["train"])
    _apply_identity_collision(suite["eval"])
    return _run_scenario(
        suite,
        model_seeds=model_seeds,
        training_steps=training_steps,
        collapse_pressure=BASE_COLLAPSE_PRESSURE,
        gradient_noise_std=BASE_GRADIENT_NOISE_STD,
        state_dim=32,
        scenario_name="identity_collision",
        extra={
            "collision_type": "shared_support_and_query_values",
            "description": (
                "All identities share the same support input positions and the "
                "same query values per task/example, so only support targets can "
                "separate identity-conditioned computation."
            ),
        },
    )


def run_distribution_shift_scenario(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    model_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
    training_steps: int,
) -> dict[str, Any]:
    train_tasks = ["transfer_learning", "agent_memory"]
    eval_tasks = ["multi_hop_reasoning", "language_like_instruction"]
    suite = build_trained_identity_suite(
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        identities_per_seed=identities_per_seed,
        examples_per_task=examples_per_task,
        vocab_size=vocab_size,
    )
    _filter_tasks(suite["train"], train_tasks)
    _filter_tasks(suite["eval"], eval_tasks)
    return _run_scenario(
        suite,
        model_seeds=model_seeds,
        training_steps=training_steps,
        collapse_pressure=BASE_COLLAPSE_PRESSURE,
        gradient_noise_std=BASE_GRADIENT_NOISE_STD,
        state_dim=32,
        scenario_name="distribution_shift",
        extra={
            "train_task_families": train_tasks,
            "eval_task_families": eval_tasks,
        },
    )


def run_adversarial_pressure_sweep(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    model_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
    training_steps: int,
    pressure_values: Sequence[float],
) -> dict[str, Any]:
    by_pressure = {}
    phase_transition_pressure: float | None = None
    for pressure in pressure_values:
        suite = build_trained_identity_suite(
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            identities_per_seed=identities_per_seed,
            examples_per_task=examples_per_task,
            vocab_size=vocab_size,
        )
        scenario = _run_scenario(
            suite,
            model_seeds=model_seeds,
            training_steps=training_steps,
            collapse_pressure=float(pressure),
            gradient_noise_std=BASE_GRADIENT_NOISE_STD,
            state_dim=32,
            scenario_name=f"pressure_{pressure}",
            extra={"collapse_pressure": float(pressure)},
        )
        by_pressure[_pressure_key(pressure)] = scenario["metrics"]
        if phase_transition_pressure is None and scenario["status"] != "passed":
            phase_transition_pressure = float(pressure)
    status = (
        "boundary_observed"
        if phase_transition_pressure is not None
        else "no_boundary_observed"
    )
    return {
        "schema": "identity_interference_pressure_sweep.v1",
        "status": status,
        "tested_pressures": [float(value) for value in pressure_values],
        "phase_transition_pressure": phase_transition_pressure,
        "by_pressure": by_pressure,
        "gate": _gate_contract(),
    }


def run_scaled_load_scenario(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    model_seeds: Sequence[int],
    identities_per_seed: int,
    examples_per_task: int,
    vocab_size: int,
    training_steps: int,
) -> dict[str, Any]:
    suite = build_trained_identity_suite(
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        identities_per_seed=identities_per_seed,
        examples_per_task=examples_per_task,
        vocab_size=vocab_size,
    )
    return _run_scenario(
        suite,
        model_seeds=model_seeds,
        training_steps=training_steps,
        collapse_pressure=BASE_COLLAPSE_PRESSURE,
        gradient_noise_std=BASE_GRADIENT_NOISE_STD,
        state_dim=64,
        scenario_name="scaled_load",
        extra={
            "scaled_identities_per_seed": int(identities_per_seed),
            "state_dim": 64,
        },
    )


def format_interference_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Identity Interference Stress",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | Status | Accuracy | Route agreement | State margin |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in ["identity_collision", "distribution_shift", "scaled_load"]:
        scenario = result["scenarios"][name]
        metrics = scenario["metrics"]
        lines.append(
            "| {name} | {status} | {acc:.4f} | {route:.4f} | {margin:.4f} |".format(
                name=name,
                status=scenario["status"],
                acc=metrics["trained_accuracy_mean"],
                route=metrics["route_agreement_min"],
                margin=metrics["state_separation_margin_min"],
            )
        )
    pressure = result["scenarios"]["adversarial_pressure_sweep"]
    lines.extend(
        [
            "",
            "## Adversarial Pressure Sweep",
            "",
            f"- Status: `{pressure['status']}`",
            f"- phase_transition_pressure: `{pressure['phase_transition_pressure']}`",
            "",
            "| Pressure | Accuracy | Route agreement | State margin |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for pressure_key, metrics in pressure["by_pressure"].items():
        lines.append(
            "| {pressure} | {acc:.4f} | {route:.4f} | {margin:.4f} |".format(
                pressure=pressure_key,
                acc=metrics["trained_accuracy_mean"],
                route=metrics["route_agreement_min"],
                margin=metrics["state_separation_margin_min"],
            )
        )
    lines.extend(["", "## Boundary", "", result["boundary"], ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Stress-test TAC-183 trained identity separation under interference."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[101, 103])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[5, 7])
    parser.add_argument("--identities-per-seed", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--training-steps", type=int, default=220)
    parser.add_argument(
        "--pressure-values",
        nargs="+",
        type=float,
        default=[0.04, 0.2, 0.5, 2.0, 10.0, 20.0],
    )
    args = parser.parse_args(argv)

    result = run_identity_interference_stress_probe(
        train_seeds=args.train_seeds,
        eval_seeds=args.eval_seeds,
        model_seeds=args.model_seeds,
        identities_per_seed=args.identities_per_seed,
        examples_per_task=args.examples_per_task,
        vocab_size=args.vocab_size,
        training_steps=args.training_steps,
        pressure_values=args.pressure_values,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "identity_interference_stress.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_interference_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return result


def _run_scenario(
    suite: dict[str, Any],
    *,
    model_seeds: Sequence[int],
    training_steps: int,
    collapse_pressure: float,
    gradient_noise_std: float,
    state_dim: int,
    scenario_name: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    controls = evaluate_non_identity_controls(suite)
    effective_training_steps = max(int(training_steps), 220)
    seed_runs = [
        _train_one_seed(
            suite,
            model_seed=int(model_seed),
            training_steps=effective_training_steps,
            collapse_pressure=float(collapse_pressure),
            gradient_noise_std=float(gradient_noise_std),
            state_dim=int(state_dim),
        )
        for model_seed in model_seeds
    ]
    metrics = _aggregate(seed_runs, controls)
    status = "passed" if _passes_gate(metrics) else "failed"
    return {
        "schema": "identity_interference_scenario.v1",
        "scenario": scenario_name,
        "status": status,
        "metrics": metrics,
        "seed_runs": seed_runs,
        "controls": controls,
        "requested_training_steps": int(training_steps),
        "effective_training_steps": effective_training_steps,
        "gate": _gate_contract(),
        **extra,
    }


def _train_one_seed(
    suite: dict[str, Any],
    *,
    model_seed: int,
    training_steps: int,
    collapse_pressure: float,
    gradient_noise_std: float,
    state_dim: int,
    learning_rate: float = 0.03,
) -> dict[str, Any]:
    torch.manual_seed(int(model_seed))
    model = TrainableIdentityStateLearner(
        vocab_size=suite["train"]["vocab_size"],
        state_dim=int(state_dim),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    train_batch = _tensorize_suite(suite["train"])
    final_task_loss = 0.0
    final_collapse_loss = 0.0
    for step_index in range(int(training_steps)):
        optimizer.zero_grad(set_to_none=True)
        _, logits = model.encode_support(
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
    metrics = _evaluate_model(model, eval_batch)
    metrics.update(
        {
            "model_seed": int(model_seed),
            "final_task_loss": final_task_loss,
            "final_collapse_loss": final_collapse_loss,
            "collapse_pressure": float(collapse_pressure),
            "gradient_noise_std": float(gradient_noise_std),
            "state_dim": int(state_dim),
        }
    )
    return metrics


def _filter_tasks(suite: dict[str, Any], task_families: Sequence[str]) -> None:
    allowed = set(task_families)
    suite["rows"] = [
        row for row in suite["rows"] if row["task_family"] in allowed
    ]
    suite["task_families"] = list(task_families)


def _apply_identity_collision(suite: dict[str, Any]) -> None:
    base_values = {
        task: [row["query_value"] for row in suite["rows"] if row["task_family"] == task][0]
        for task in TASK_FAMILIES
    }
    for row in suite["rows"]:
        task = row["task_family"]
        row["query_value"] = int(base_values[task])
        target = _solve_row(row, row["rule"], vocab_size=suite["vocab_size"])
        row["target_value"] = int(target)
        row["target_answer"] = _answer_text(target)


def _passes_gate(metrics: dict[str, Any]) -> bool:
    return (
        metrics["trained_accuracy_mean"] >= 0.90
        and metrics["trained_accuracy_min"] >= 0.85
        and metrics["route_agreement_min"] >= 0.90
        and metrics["state_separation_margin_min"] > 0.15
        and metrics["best_non_identity_control_accuracy"] <= 0.35
    )


def _gate_contract() -> dict[str, float]:
    return {
        "trained_accuracy_mean_min": 0.90,
        "trained_accuracy_min_min": 0.85,
        "route_agreement_min": 0.90,
        "state_separation_margin_min": 0.15,
        "best_non_identity_control_max": 0.35,
    }


def _decision(scenarios: dict[str, Any]) -> dict[str, Any]:
    required_pass = all(
        scenarios[name]["status"] == "passed"
        for name in ["identity_collision", "distribution_shift", "scaled_load"]
    )
    pressure = scenarios["adversarial_pressure_sweep"]
    boundary_mapped = pressure["status"] in {
        "boundary_observed",
        "no_boundary_observed",
    }
    if required_pass and boundary_mapped:
        return {
            "status": "identity_interference_stress_boundary_mapped",
            "reason": (
                "Collision, distribution-shift, and scaled-load gates passed, "
                "and the collapse-pressure sweep reported the first observed "
                "boundary or confirmed none within the tested range."
            ),
            "recommendation": (
                "Treat TAC-183 as robust in this controlled regime, but use the "
                "reported pressure boundary as the next target for full "
                "TACTransformerLM training stability tests."
            ),
        }
    return {
        "status": "identity_interference_stress_failed",
        "reason": "At least one required interference scenario failed or the pressure boundary was not mapped.",
        "recommendation": "Do not generalize TAC-183 beyond the base controlled distribution until failed scenarios are repaired.",
    }


def _pressure_key(pressure: float) -> str:
    return str(float(pressure))


if __name__ == "__main__":
    main()
