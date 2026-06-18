from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_forced_identity_objective import (
    VALUE_COUNT,
    VALUE_START,
    forced_identity_config,
    knockout_program,
    make_batch,
    training_loss,
)
from tac_transformer import TACTransformerLM
from tac_transformer.training import count_parameters


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "runs"
    / "benchmarks"
    / "identity_readout_bridge_tac214_215_2026_06_07"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TAC-214/TAC-215 diagnostic: freeze a forced-state TAC base, train an "
            "oracle probe on identity readout vectors, then train a direct "
            "readout-to-logit injection bridge."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-steps", type=int, default=240)
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--bridge-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--n-pairs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_identity_readout_bridge(
        output_dir=args.output_dir,
        base_steps=args.base_steps,
        probe_steps=args.probe_steps,
        bridge_steps=args.bridge_steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        n_pairs=args.n_pairs,
        seeds=args.seeds,
        device=args.device,
        torch_threads=args.torch_threads,
    )
    print(json.dumps(report["decision"], indent=2), flush=True)


def run_identity_readout_bridge(
    *,
    output_dir: Path,
    base_steps: int = 240,
    probe_steps: int = 200,
    bridge_steps: int = 200,
    batch_size: int = 32,
    eval_batches: int = 8,
    n_pairs: int = 3,
    seeds: Sequence[int] = (7, 19, 31),
    device: str | torch.device = "cpu",
    torch_threads: int = 4,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_threads = torch.get_num_threads()
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    try:
        started = time.perf_counter()
        rows = [
            run_seed(
                seed=int(seed),
                base_steps=int(base_steps),
                probe_steps=int(probe_steps),
                bridge_steps=int(bridge_steps),
                batch_size=int(batch_size),
                eval_batches=int(eval_batches),
                n_pairs=int(n_pairs),
                device=torch.device(device),
            )
            for seed in seeds
        ]
        aggregate = aggregate_rows(rows)
        report = {
            "schema": "identity_readout_bridge.v1",
            "created_at": "2026-06-07",
            "question": (
                "Does the forced-state TAC identity readout contain answer "
                "information, and can a direct readout-to-logit bridge make that "
                "information drive output choices?"
            ),
            "protocol": {
                "base_steps": int(base_steps),
                "probe_steps": int(probe_steps),
                "bridge_steps": int(bridge_steps),
                "batch_size": int(batch_size),
                "eval_batches": int(eval_batches),
                "n_pairs": int(n_pairs),
                "seeds": [int(seed) for seed in seeds],
                "base_objective": "forced_state from TAC-213",
                "scoring": (
                    "All bridge/probe accuracies are forced-choice over the "
                    f"{VALUE_COUNT} value tokens."
                ),
            },
            "rows": rows,
            "aggregate": aggregate,
            "decision": decide(aggregate),
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output_dir / "identity_readout_bridge.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "RESULTS.md").write_text(
            format_markdown(report),
            encoding="utf-8",
        )
        return report
    finally:
        if torch_threads > 0:
            torch.set_num_threads(prior_threads)


def run_seed(
    *,
    seed: int,
    base_steps: int,
    probe_steps: int,
    bridge_steps: int,
    batch_size: int,
    eval_batches: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    config = forced_identity_config()
    model = TACTransformerLM(config).to(device)
    base_losses = train_forced_state_base(
        model,
        seed=seed,
        steps=base_steps,
        batch_size=batch_size,
        n_pairs=n_pairs,
        device=device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    probe = nn.Linear(config.d_model, VALUE_COUNT).to(device)
    bridge = nn.Linear(config.d_model, VALUE_COUNT).to(device)
    probe_losses = train_readout_head(
        model,
        probe,
        seed=seed * 3001 + 41,
        steps=probe_steps,
        batch_size=batch_size,
        n_pairs=n_pairs,
        device=device,
        mode="probe",
    )
    bridge_losses = train_readout_head(
        model,
        bridge,
        seed=seed * 4001 + 53,
        steps=bridge_steps,
        batch_size=batch_size,
        n_pairs=n_pairs,
        device=device,
        mode="bridge",
    )
    evaluation = evaluate_heads(
        model,
        probe,
        bridge,
        seed=seed * 5003 + 67,
        batch_size=batch_size,
        eval_batches=eval_batches,
        n_pairs=n_pairs,
        device=device,
    )
    knockout_summary = bridge_knockout_summary(
        model,
        bridge,
        seed=seed * 6007 + 79,
        batch_size=batch_size,
        eval_batches=max(2, eval_batches // 2),
        n_pairs=n_pairs,
        device=device,
    )
    return {
        "seed": int(seed),
        "base_training_loss_trace": trace(base_losses),
        "probe_training_loss_trace": trace(probe_losses),
        "bridge_training_loss_trace": trace(bridge_losses),
        "parameter_counts": count_parameters(model),
        "evaluation": evaluation,
        "program_knockout": knockout_summary,
    }


def train_forced_state_base(
    model: TACTransformerLM,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    n_pairs: int,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rng = random.Random(seed * 1009 + 17)
    losses = []
    model.train()
    for step in range(int(steps)):
        batch = make_batch(rng, batch_size=batch_size, n_pairs=n_pairs, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(model, batch, variant="forced_state")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if should_trace(step, steps):
            losses.append(float(loss.detach().cpu()))
    return losses


def train_readout_head(
    model: TACTransformerLM,
    head: nn.Linear,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    n_pairs: int,
    device: torch.device,
    mode: str,
) -> list[float]:
    if mode not in {"probe", "bridge"}:
        raise ValueError(f"unknown readout-head mode: {mode}")
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=0.0)
    rng = random.Random(seed)
    losses = []
    head.train()
    model.eval()
    for step in range(int(steps)):
        batch = make_batch(rng, batch_size=batch_size, n_pairs=n_pairs, device=device)
        with torch.no_grad():
            features = extract_bridge_features(model, batch)
        optimizer.zero_grad(set_to_none=True)
        logits = head(features["read_vector"])
        if mode == "bridge":
            logits = logits + features["base_value_logits"]
        loss = F.cross_entropy(logits, features["target_class"])
        loss.backward()
        optimizer.step()
        if should_trace(step, steps):
            losses.append(float(loss.detach().cpu()))
    return losses


def evaluate_heads(
    model: TACTransformerLM,
    probe: nn.Linear,
    bridge: nn.Linear,
    *,
    seed: int,
    batch_size: int,
    eval_batches: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(seed)
    model.eval()
    probe.eval()
    bridge.eval()
    base_carry = []
    base_reset = []
    direct_memory_read = []
    oracle_probe = []
    logit_bridge = []
    reset_bridge = []
    zero_bridge = []
    shuffled_bridge = []
    with torch.inference_mode():
        for _ in range(eval_batches):
            batch = make_batch(rng, batch_size=batch_size, n_pairs=n_pairs, device=device)
            features = extract_bridge_features(model, batch)
            base_carry.append(
                class_accuracy(features["base_value_logits"], features["target_class"])
            )
            base_reset.append(
                class_accuracy(features["reset_value_logits"], features["target_class"])
            )
            direct_memory_read.append(
                class_accuracy(
                    features["memory_read_value_logits"],
                    features["target_class"],
                )
            )
            oracle_probe.append(
                class_accuracy(
                    probe(features["read_vector"]),
                    features["target_class"],
                )
            )
            logit_bridge.append(
                class_accuracy(
                    features["base_value_logits"] + bridge(features["read_vector"]),
                    features["target_class"],
                )
            )
            reset_bridge.append(
                class_accuracy(
                    features["reset_value_logits"] + bridge(features["zero_read_vector"]),
                    features["target_class"],
                )
            )
            zero_bridge.append(
                class_accuracy(
                    features["base_value_logits"] + bridge(features["zero_read_vector"]),
                    features["target_class"],
                )
            )
            shuffled_bridge.append(
                class_accuracy(
                    features["base_value_logits"]
                    + bridge(features["shuffled_read_vector"]),
                    features["target_class"],
                )
            )
    return {
        "base_carry_value_accuracy": mean(base_carry),
        "base_reset_value_accuracy": mean(base_reset),
        "base_carry_minus_reset": mean(base_carry) - mean(base_reset),
        "direct_memory_read_accuracy": mean(direct_memory_read),
        "oracle_probe_accuracy": mean(oracle_probe),
        "logit_bridge_accuracy": mean(logit_bridge),
        "reset_bridge_accuracy": mean(reset_bridge),
        "zero_bridge_accuracy": mean(zero_bridge),
        "shuffled_bridge_accuracy": mean(shuffled_bridge),
        "bridge_minus_base_carry": mean(logit_bridge) - mean(base_carry),
        "bridge_minus_reset_bridge": mean(logit_bridge) - mean(reset_bridge),
        "bridge_minus_shuffled_bridge": mean(logit_bridge) - mean(shuffled_bridge),
    }


def bridge_knockout_summary(
    model: TACTransformerLM,
    bridge: nn.Linear,
    *,
    seed: int,
    batch_size: int,
    eval_batches: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, Any]:
    rng = random.Random(seed)
    batches = [
        make_batch(rng, batch_size=batch_size, n_pairs=n_pairs, device=device)
        for _ in range(eval_batches)
    ]
    baseline_scores = []
    with torch.inference_mode():
        for batch in batches:
            features = extract_bridge_features(model, batch)
            baseline_scores.append(
                class_accuracy(
                    features["base_value_logits"] + bridge(features["read_vector"]),
                    features["target_class"],
                )
            )
    baseline_accuracy = mean(baseline_scores)
    rows = []
    for program_id in range(model.config.n_programs):
        scores = []
        with knockout_program(model, program_id):
            with torch.inference_mode():
                for batch in batches:
                    features = extract_bridge_features(model, batch)
                    scores.append(
                        class_accuracy(
                            features["base_value_logits"]
                            + bridge(features["read_vector"]),
                            features["target_class"],
                        )
                    )
        knockout_accuracy = mean(scores)
        rows.append(
            {
                "program": int(program_id),
                "baseline_accuracy": baseline_accuracy,
                "knockout_accuracy": knockout_accuracy,
                "accuracy_drop": baseline_accuracy - knockout_accuracy,
            }
        )
    positive_drops = [max(0.0, row["accuracy_drop"]) for row in rows]
    total_positive = sum(positive_drops)
    return {
        "baseline_logit_bridge_accuracy": baseline_accuracy,
        "programs": rows,
        "max_logit_bridge_accuracy_drop": max(positive_drops, default=0.0),
        "mean_logit_bridge_accuracy_drop": mean([row["accuracy_drop"] for row in rows]),
        "drop_concentration": (
            0.0 if total_positive <= 0.0 else max(positive_drops) / total_positive
        ),
        "harmful_programs_ge_5pct": sum(
            1 for row in rows if row["accuracy_drop"] >= 0.05
        ),
    }


def extract_bridge_features(
    model: TACTransformerLM,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    value_ids = value_token_ids(batch["target"].device)
    support = model(
        batch["support"],
        collect_auxiliary=False,
        collect_metrics=False,
    )
    key_ids = batch["query"][:, -1]
    query = model(
        batch["query"],
        identity_states=support.identity_states,
        collect_auxiliary=False,
        collect_metrics=False,
    )
    reset_query = model(
        batch["query"],
        collect_auxiliary=False,
        collect_metrics=False,
    )
    read_vector = model.memory_read_vector(key_ids, support.identity_states)
    memory_read_logits = model.memory_read_logits(key_ids, support.identity_states)
    shuffled = read_vector[torch.randperm(read_vector.shape[0], device=read_vector.device)]
    return {
        "read_vector": read_vector.detach(),
        "zero_read_vector": torch.zeros_like(read_vector).detach(),
        "shuffled_read_vector": shuffled.detach(),
        "base_value_logits": query.logits[:, -1, value_ids].detach(),
        "reset_value_logits": reset_query.logits[:, -1, value_ids].detach(),
        "memory_read_value_logits": memory_read_logits[:, value_ids].detach(),
        "target_class": (batch["target"] - VALUE_START).detach(),
    }


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seeds": [row["seed"] for row in rows],
        "base_final_training_loss": mean_path(rows, ["base_training_loss_trace", -1]),
        "probe_final_training_loss": mean_path(rows, ["probe_training_loss_trace", -1]),
        "bridge_final_training_loss": mean_path(rows, ["bridge_training_loss_trace", -1]),
        "base_carry_value_accuracy": mean_path(
            rows,
            ["evaluation", "base_carry_value_accuracy"],
        ),
        "base_reset_value_accuracy": mean_path(
            rows,
            ["evaluation", "base_reset_value_accuracy"],
        ),
        "base_carry_minus_reset": mean_path(
            rows,
            ["evaluation", "base_carry_minus_reset"],
        ),
        "direct_memory_read_accuracy": mean_path(
            rows,
            ["evaluation", "direct_memory_read_accuracy"],
        ),
        "oracle_probe_accuracy": mean_path(
            rows,
            ["evaluation", "oracle_probe_accuracy"],
        ),
        "logit_bridge_accuracy": mean_path(
            rows,
            ["evaluation", "logit_bridge_accuracy"],
        ),
        "reset_bridge_accuracy": mean_path(
            rows,
            ["evaluation", "reset_bridge_accuracy"],
        ),
        "zero_bridge_accuracy": mean_path(
            rows,
            ["evaluation", "zero_bridge_accuracy"],
        ),
        "shuffled_bridge_accuracy": mean_path(
            rows,
            ["evaluation", "shuffled_bridge_accuracy"],
        ),
        "bridge_minus_base_carry": mean_path(
            rows,
            ["evaluation", "bridge_minus_base_carry"],
        ),
        "bridge_minus_reset_bridge": mean_path(
            rows,
            ["evaluation", "bridge_minus_reset_bridge"],
        ),
        "bridge_minus_shuffled_bridge": mean_path(
            rows,
            ["evaluation", "bridge_minus_shuffled_bridge"],
        ),
        "max_logit_bridge_knockout_drop": mean_path(
            rows,
            ["program_knockout", "max_logit_bridge_accuracy_drop"],
        ),
        "harmful_programs_ge_5pct": mean_path(
            rows,
            ["program_knockout", "harmful_programs_ge_5pct"],
        ),
    }


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    probe = aggregate["oracle_probe_accuracy"]
    bridge = aggregate["logit_bridge_accuracy"]
    bridge_gain = aggregate["bridge_minus_base_carry"]
    reset_gap = aggregate["bridge_minus_reset_bridge"]
    shuffled_gap = aggregate["bridge_minus_shuffled_bridge"]
    knockout_drop = aggregate["max_logit_bridge_knockout_drop"]
    if probe >= 0.80 and bridge >= 0.80 and reset_gap >= 0.25:
        status = "readout_contains_answer_and_logit_bridge_recovers_generation"
    elif probe >= 0.80 and bridge < 0.80:
        status = "readout_contains_answer_but_simple_logit_bridge_failed"
    elif bridge_gain >= 0.20 and reset_gap >= 0.10:
        status = "logit_bridge_partially_recovers_carried_state_answers"
    elif probe >= 0.30:
        status = "readout_signal_present_but_weak_for_generation"
    else:
        status = "readout_signal_too_weak_for_bridge"
    if bridge_gain >= 0.20 and knockout_drop < 0.05:
        locality = "bridge_improves_answers_without_localized_program_causality"
    elif knockout_drop >= 0.05:
        locality = "bridge_shows_localized_program_sensitivity"
    else:
        locality = "no_bridge_or_program_causal_signal"
    return {
        "status": status,
        "locality_status": locality,
        "oracle_probe_accuracy": probe,
        "logit_bridge_accuracy": bridge,
        "bridge_minus_base_carry": bridge_gain,
        "bridge_minus_reset_bridge": reset_gap,
        "bridge_minus_shuffled_bridge": shuffled_gap,
        "max_logit_bridge_knockout_drop": knockout_drop,
        "interpretation": (
            "TAC-214 estimates how much answer information is decodable from "
            "the frozen identity readout. TAC-215 tests whether adding a direct "
            "linear readout-to-value-logit channel makes carried state affect "
            "answers."
        ),
    }


def format_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# TAC-214/TAC-215 Identity Readout Bridge",
        "",
        f"Decision: `{report['decision']['status']}`.",
        f"Locality: `{report['decision']['locality_status']}`.",
        "",
        "| Metric | Mean |",
        "| --- | ---: |",
    ]
    for label, key in [
        ("Base carry value accuracy", "base_carry_value_accuracy"),
        ("Base reset value accuracy", "base_reset_value_accuracy"),
        ("Base carry - reset", "base_carry_minus_reset"),
        ("Direct memory-read accuracy", "direct_memory_read_accuracy"),
        ("Oracle readout probe accuracy", "oracle_probe_accuracy"),
        ("Logit bridge accuracy", "logit_bridge_accuracy"),
        ("Reset bridge accuracy", "reset_bridge_accuracy"),
        ("Zero-read bridge accuracy", "zero_bridge_accuracy"),
        ("Shuffled-read bridge accuracy", "shuffled_bridge_accuracy"),
        ("Bridge - base carry", "bridge_minus_base_carry"),
        ("Bridge - reset bridge", "bridge_minus_reset_bridge"),
        ("Bridge - shuffled bridge", "bridge_minus_shuffled_bridge"),
        ("Max bridge knockout drop", "max_logit_bridge_knockout_drop"),
    ]:
        lines.append(f"| {label} | {format_value(aggregate.get(key))} |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The base model is trained with TAC-213's forced-state objective, then frozen.",
            "- The oracle probe estimates the value-token information available in the frozen identity readout vector.",
            "- The logit bridge adds a learned linear value-token correction from the same readout to the frozen query logits.",
            "- Reset, zero-read, shuffled-read, and program-knockout controls test whether the bridge depends on carried identity state and localized programs.",
            "",
        ]
    )
    return "\n".join(lines)


def value_token_ids(device: torch.device) -> torch.Tensor:
    return torch.arange(VALUE_START, VALUE_START + VALUE_COUNT, device=device)


def class_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().detach().cpu())


def trace(losses: Sequence[float]) -> list[float]:
    return [float(loss) for loss in losses]


def should_trace(step: int, steps: int) -> bool:
    return step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 4) == 0


def mean(values: Sequence[float]) -> float:
    vals = [float(value) for value in values]
    return statistics.fmean(vals) if vals else 0.0


def mean_path(rows: Sequence[dict[str, Any]], path: Sequence[str | int]) -> float:
    values = []
    for row in rows:
        current: Any = row
        for key in path:
            current = current[key]
        if current is not None:
            values.append(float(current))
    return mean(values)


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
