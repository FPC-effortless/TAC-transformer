from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM
from tac_transformer.training import count_parameters


DEFAULT_OUTPUT_DIR = ROOT / "runs" / "benchmarks" / "forced_identity_objective_tac213_2026_06_07"
VARIANTS = ("context_visible_lm", "forced_state")
PAD = 0
BOS = 1
SEP = 2
QUERY = 3
KEY_START = 8
KEY_COUNT = 12
VALUE_START = 40
VALUE_COUNT = 12


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether forced identity-state dependence makes TAC programs causal."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--n-pairs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    report = run_forced_identity_objective(
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        n_pairs=args.n_pairs,
        seeds=args.seeds,
        variants=args.variants,
        device=args.device,
    )
    print(json.dumps(report["decision"], indent=2), flush=True)


def run_forced_identity_objective(
    *,
    output_dir: Path,
    steps: int = 240,
    batch_size: int = 32,
    eval_batches: int = 8,
    n_pairs: int = 3,
    seeds: Sequence[int] = (7, 19, 31),
    variants: Sequence[str] = VARIANTS,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        for seed in seeds:
            rows.append(
                run_variant_seed(
                    variant=variant,
                    seed=int(seed),
                    steps=int(steps),
                    batch_size=int(batch_size),
                    eval_batches=int(eval_batches),
                    n_pairs=int(n_pairs),
                    device=torch.device(device),
                )
            )
    aggregate = aggregate_rows(rows)
    report = {
        "schema": "forced_identity_objective.v1",
        "created_at": "2026-06-07",
        "question": (
            "Does replacing a context-visible LM-style objective with a forced "
            "identity-state objective make identity programs causally necessary?"
        ),
        "protocol": {
            "steps": int(steps),
            "batch_size": int(batch_size),
            "eval_batches": int(eval_batches),
            "n_pairs": int(n_pairs),
            "seeds": [int(seed) for seed in seeds],
            "variants": list(variants),
            "task": (
                "Support segment contains random key/value identity pairs. Query "
                "segment contains only QUERY,key. The forced_state objective must "
                "answer from carried state; reset state removes the only source of "
                "the per-example value."
            ),
        },
        "rows": rows,
        "aggregate": aggregate,
        "decision": decide(aggregate),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "forced_identity_objective.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "RESULTS.md").write_text(format_markdown(report), encoding="utf-8")
    return report


def run_variant_seed(
    *,
    variant: str,
    seed: int,
    steps: int,
    batch_size: int,
    eval_batches: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    config = forced_identity_config()
    model = TACTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rng = random.Random(seed * 1009 + 17)
    losses = []
    for step in range(int(steps)):
        batch = make_batch(rng, batch_size=batch_size, n_pairs=n_pairs, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(model, batch, variant=variant)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 4) == 0:
            losses.append(float(loss.detach().cpu()))

    eval_rng = random.Random(seed * 2003 + 29)
    eval_summary = evaluate_model(
        model,
        eval_rng,
        batch_size=batch_size,
        eval_batches=eval_batches,
        n_pairs=n_pairs,
        device=device,
    )
    gradient_summary = gradient_flow_summary(
        model,
        make_batch(eval_rng, batch_size=batch_size, n_pairs=n_pairs, device=device),
        variant=variant,
    )
    knockout_summary = knockout_summary_for_model(
        model,
        eval_rng,
        batch_size=batch_size,
        eval_batches=max(2, eval_batches // 2),
        n_pairs=n_pairs,
        device=device,
    )
    return {
        "variant": variant,
        "seed": int(seed),
        "training_loss_trace": losses,
        "final_training_loss": losses[-1],
        "parameter_counts": count_parameters(model),
        "evaluation": eval_summary,
        "gradient_flow": gradient_summary,
        "program_knockout": knockout_summary,
    }


def forced_identity_config() -> TACConfig:
    return TACConfig(
        vocab_size=80,
        d_model=32,
        n_heads=4,
        n_layers=2,
        n_programs=8,
        max_seq_len=32,
        beta=1.5,
        energy_budget=4.0,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        position_type="rope",
        program_compute_type="linear_expert",
        routing_type="base",
        memory_read_type="content_addressed",
        content_store_size=16,
        content_read_steps=2,
        content_read_gate_type="cue_match",
        content_read_cue_match_threshold=0.5,
        identity_attention_type="identity_first",
        memory_adapter_type="gated_residual",
        detach_identity_state=False,
        dropout=0.0,
    )


def make_batch(
    rng: random.Random,
    *,
    batch_size: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    support_rows = []
    query_rows = []
    full_rows = []
    targets = []
    for _ in range(batch_size):
        keys = rng.sample(range(KEY_START, KEY_START + KEY_COUNT), n_pairs)
        values = rng.sample(range(VALUE_START, VALUE_START + VALUE_COUNT), n_pairs)
        target_index = rng.randrange(n_pairs)
        support = [BOS]
        for key, value in zip(keys, values):
            support.extend([key, value])
        query = [QUERY, keys[target_index]]
        full = support + [SEP] + query
        support_rows.append(support)
        query_rows.append(query)
        full_rows.append(full)
        targets.append(values[target_index])
    return {
        "support": torch.tensor(support_rows, dtype=torch.long, device=device),
        "query": torch.tensor(query_rows, dtype=torch.long, device=device),
        "full": torch.tensor(full_rows, dtype=torch.long, device=device),
        "target": torch.tensor(targets, dtype=torch.long, device=device),
    }


def training_loss(
    model: TACTransformerLM,
    batch: dict[str, torch.Tensor],
    *,
    variant: str,
) -> torch.Tensor:
    if variant == "context_visible_lm":
        output = model(batch["full"], collect_auxiliary=True, collect_metrics=True)
        return F.cross_entropy(output.logits[:, -1, :], batch["target"])
    if variant == "forced_state":
        support = model(
            batch["support"],
            collect_auxiliary=False,
            collect_metrics=False,
        )
        query = model(
            batch["query"],
            identity_states=support.identity_states,
            collect_auxiliary=True,
            collect_metrics=True,
        )
        return F.cross_entropy(query.logits[:, -1, :], batch["target"])
    raise ValueError(f"unknown variant: {variant}")


def evaluate_model(
    model: TACTransformerLM,
    rng: random.Random,
    *,
    batch_size: int,
    eval_batches: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    full_context = []
    carry = []
    reset = []
    shuffled = []
    memory_read = []
    compression: dict[int, list[float]] = {2: [], 4: [], 8: [], 16: [], 24: []}
    with torch.inference_mode():
        for _ in range(eval_batches):
            batch = make_batch(rng, batch_size=batch_size, n_pairs=n_pairs, device=device)
            full_context.append(accuracy_from_logits(model(batch["full"]).logits[:, -1, :], batch["target"]))
            support = model(batch["support"], collect_auxiliary=False, collect_metrics=False)
            try:
                read_logits = model.memory_read_logits(
                    batch["query"][:, -1],
                    support.identity_states,
                )
                memory_read.append(accuracy_from_logits(read_logits, batch["target"]))
            except Exception:
                pass
            carry_output = model(batch["query"], identity_states=support.identity_states)
            carry.append(accuracy_from_logits(carry_output.logits[:, -1, :], batch["target"]))
            reset_output = model(batch["query"])
            reset.append(accuracy_from_logits(reset_output.logits[:, -1, :], batch["target"]))
            shuffled_states = shuffle_states(support.identity_states)
            shuffled_output = model(batch["query"], identity_states=shuffled_states)
            shuffled.append(accuracy_from_logits(shuffled_output.logits[:, -1, :], batch["target"]))
            for budget in compression:
                truncated = batch["full"][:, -budget:]
                compression[budget].append(
                    accuracy_from_logits(model(truncated).logits[:, -1, :], batch["target"])
                )
    model.train()
    return {
        "full_context_accuracy": mean(full_context),
        "carry_accuracy": mean(carry),
        "reset_accuracy": mean(reset),
        "shuffled_accuracy": mean(shuffled),
        "carry_minus_reset": mean(carry) - mean(reset),
        "carry_minus_shuffled": mean(carry) - mean(shuffled),
        "memory_read_accuracy": None if not memory_read else mean(memory_read),
        "active_context_accuracy": {
            str(budget): mean(values)
            for budget, values in sorted(compression.items())
        },
        "min_active_context_for_75pct": min_context_for_accuracy(compression, 0.75),
    }


def gradient_flow_summary(
    model: TACTransformerLM,
    batch: dict[str, torch.Tensor],
    *,
    variant: str,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    loss = training_loss(model, batch, variant=variant)
    loss.backward()
    identity_norms = []
    program_norms = []
    transformer_norms = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().norm().cpu())
        if ".identity_field." in name:
            identity_norms.append(norm)
            if any(
                marker in name
                for marker in (
                    "program_embeddings",
                    "program_expert",
                    "program_update",
                    "program_conditioned_update",
                )
            ):
                program_norms.append(norm)
        else:
            transformer_norms.append(norm)
    identity_total = sum(identity_norms)
    transformer_total = sum(transformer_norms)
    total = identity_total + transformer_total
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach().cpu()),
        "identity_grad_norm_sum": identity_total,
        "program_grad_norm_sum": sum(program_norms),
        "transformer_grad_norm_sum": transformer_total,
        "identity_grad_share": 0.0 if total <= 0.0 else identity_total / total,
        "program_grad_share": 0.0 if total <= 0.0 else sum(program_norms) / total,
    }


def knockout_summary_for_model(
    model: TACTransformerLM,
    rng: random.Random,
    *,
    batch_size: int,
    eval_batches: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, Any]:
    baseline = []
    knockout_rows = []
    batches = [
        make_batch(rng, batch_size=batch_size, n_pairs=n_pairs, device=device)
        for _ in range(eval_batches)
    ]
    with torch.inference_mode():
        for batch in batches:
            support = model(batch["support"], collect_auxiliary=False, collect_metrics=False)
            query = model(batch["query"], identity_states=support.identity_states)
            baseline.append(accuracy_from_logits(query.logits[:, -1, :], batch["target"]))
    baseline_accuracy = mean(baseline)
    for program_id in range(model.config.n_programs):
        scores = []
        with knockout_program(model, program_id):
            with torch.inference_mode():
                for batch in batches:
                    support = model(
                        batch["support"],
                        collect_auxiliary=False,
                        collect_metrics=False,
                    )
                    query = model(batch["query"], identity_states=support.identity_states)
                    scores.append(
                        accuracy_from_logits(query.logits[:, -1, :], batch["target"])
                    )
        knockout_accuracy = mean(scores)
        knockout_rows.append(
            {
                "program": int(program_id),
                "baseline_accuracy": baseline_accuracy,
                "knockout_accuracy": knockout_accuracy,
                "accuracy_drop": baseline_accuracy - knockout_accuracy,
            }
        )
    positive_drops = [max(0.0, row["accuracy_drop"]) for row in knockout_rows]
    total_drop = sum(positive_drops)
    return {
        "baseline_accuracy": baseline_accuracy,
        "programs": knockout_rows,
        "max_accuracy_drop": max(positive_drops, default=0.0),
        "mean_accuracy_drop": mean([row["accuracy_drop"] for row in knockout_rows]),
        "drop_concentration": (
            0.0 if total_drop <= 0.0 else max(positive_drops) / total_drop
        ),
        "harmful_programs_ge_5pct": sum(
            1 for row in knockout_rows if row["accuracy_drop"] >= 0.05
        ),
    }


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    aggregate = {}
    for variant, variant_rows in sorted(by_variant.items()):
        aggregate[variant] = {
            "seeds": [row["seed"] for row in variant_rows],
            "final_training_loss": mean_path(variant_rows, ["final_training_loss"]),
            "full_context_accuracy": mean_path(variant_rows, ["evaluation", "full_context_accuracy"]),
            "carry_accuracy": mean_path(variant_rows, ["evaluation", "carry_accuracy"]),
            "reset_accuracy": mean_path(variant_rows, ["evaluation", "reset_accuracy"]),
            "shuffled_accuracy": mean_path(variant_rows, ["evaluation", "shuffled_accuracy"]),
            "carry_minus_reset": mean_path(variant_rows, ["evaluation", "carry_minus_reset"]),
            "carry_minus_shuffled": mean_path(variant_rows, ["evaluation", "carry_minus_shuffled"]),
            "memory_read_accuracy": mean_path(variant_rows, ["evaluation", "memory_read_accuracy"]),
            "identity_grad_share": mean_path(variant_rows, ["gradient_flow", "identity_grad_share"]),
            "program_grad_share": mean_path(variant_rows, ["gradient_flow", "program_grad_share"]),
            "max_knockout_accuracy_drop": mean_path(variant_rows, ["program_knockout", "max_accuracy_drop"]),
            "knockout_drop_concentration": mean_path(variant_rows, ["program_knockout", "drop_concentration"]),
            "harmful_programs_ge_5pct": mean_path(variant_rows, ["program_knockout", "harmful_programs_ge_5pct"]),
            "min_active_context_for_75pct": min_optional(
                row["evaluation"]["min_active_context_for_75pct"]
                for row in variant_rows
            ),
        }
    if "context_visible_lm" in aggregate and "forced_state" in aggregate:
        aggregate["forced_minus_context"] = {
            key: aggregate["forced_state"].get(key, 0.0)
            - aggregate["context_visible_lm"].get(key, 0.0)
            for key in (
                "carry_accuracy",
                "reset_accuracy",
                "carry_minus_reset",
                "memory_read_accuracy",
                "identity_grad_share",
                "program_grad_share",
                "max_knockout_accuracy_drop",
            )
        }
    return aggregate


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    forced = aggregate.get("forced_state")
    context = aggregate.get("context_visible_lm")
    if not forced:
        return {
            "status": "forced_state_not_run",
            "reason": "forced_state variant is required for the TAC-213 decision.",
        }
    forced_causal = (
        forced["carry_minus_reset"] >= 0.25
        and forced["max_knockout_accuracy_drop"] >= 0.05
        and forced["identity_grad_share"] > 0.0
    )
    if forced_causal:
        status = "forced_identity_objective_makes_identity_causal"
    elif forced["carry_minus_reset"] >= 0.25:
        status = "forced_identity_objective_teaches_carry_without_localized_knockout"
    else:
        status = "forced_identity_objective_not_sufficient"
    return {
        "status": status,
        "forced_carry_accuracy": forced["carry_accuracy"],
        "forced_reset_accuracy": forced["reset_accuracy"],
        "forced_carry_minus_reset": forced["carry_minus_reset"],
        "forced_identity_grad_share": forced["identity_grad_share"],
        "forced_max_knockout_accuracy_drop": forced["max_knockout_accuracy_drop"],
        "context_carry_minus_reset": None
        if context is None
        else context["carry_minus_reset"],
        "interpretation": (
            "TAC-213 tests the objective bottleneck hypothesis from TAC-212: "
            "specialization may emerge without becoming causal unless the loss "
            "requires identity-state usage."
        ),
    }


def format_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# TAC-213 Forced Identity Objective",
        "",
        f"Decision: `{report['decision']['status']}`.",
        "",
        "| Metric | context-visible LM | forced-state |",
        "| --- | ---: | ---: |",
    ]
    context = aggregate.get("context_visible_lm", {})
    forced = aggregate.get("forced_state", {})
    for label, key in [
        ("Full-context accuracy", "full_context_accuracy"),
        ("Carry accuracy", "carry_accuracy"),
        ("Reset accuracy", "reset_accuracy"),
        ("Carry - reset", "carry_minus_reset"),
        ("Direct memory-read accuracy", "memory_read_accuracy"),
        ("Identity grad share", "identity_grad_share"),
        ("Program grad share", "program_grad_share"),
        ("Max knockout accuracy drop", "max_knockout_accuracy_drop"),
        ("Knockout drop concentration", "knockout_drop_concentration"),
    ]:
        lines.append(
            f"| {label} | {format_value(context.get(key))} | {format_value(forced.get(key))} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `context_visible_lm` can see the key/value support in the same sequence.",
            "- `forced_state` sees support only in a prefill segment; the query contains only `QUERY,key` and must use carried identity state.",
            "- A positive TAC-213 result requires carry > reset and a nonzero retrieval drop under program knockout.",
            "",
        ]
    )
    return "\n".join(lines)


@contextmanager
def knockout_program(model: TACTransformerLM, program_id: int) -> Iterator[None]:
    originals = []
    for block in model.blocks:
        identity = block.identity_field
        original = identity._compute_program_context
        originals.append((identity, original))

        def patched(
            self,
            hidden,
            selected_weights,
            selected_denominator,
            previous_memory,
            previous_engram_patterns,
            previous_engram_values,
            previous_engram_mask,
            previous_content_cues,
            previous_content_values,
            previous_content_mask,
            *,
            _original=original,
        ):
            masked_weights = selected_weights.clone()
            masked_weights[..., int(program_id)] = 0.0
            denominator = masked_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return _original(
                hidden,
                masked_weights,
                denominator,
                previous_memory,
                previous_engram_patterns,
                previous_engram_values,
                previous_engram_mask,
                previous_content_cues,
                previous_content_values,
                previous_content_mask,
            )

        identity._compute_program_context = MethodType(patched, identity)
    try:
        yield
    finally:
        for identity, original in originals:
            identity._compute_program_context = original


def shuffle_states(states):
    if not states:
        return states
    shuffled = []
    for state in states:
        batch_size = state.program_memory.shape[0]
        permutation = torch.randperm(batch_size, device=state.program_memory.device)
        shuffled.append(
            replace(
                state,
                stability=state.stability[permutation],
                program_memory=state.program_memory[permutation],
                stable_program_memory=shuffle_optional_batch(
                    state.stable_program_memory,
                    permutation,
                ),
                archival_program_memory=shuffle_optional_batch(
                    state.archival_program_memory,
                    permutation,
                ),
                program_age=shuffle_optional_batch(state.program_age, permutation),
                program_write_frequency=shuffle_optional_batch(
                    state.program_write_frequency,
                    permutation,
                ),
                engram_patterns=shuffle_optional_batch(state.engram_patterns, permutation),
                engram_values=shuffle_optional_batch(state.engram_values, permutation),
                engram_mask=shuffle_optional_batch(state.engram_mask, permutation),
                content_cues=shuffle_optional_batch(state.content_cues, permutation),
                content_values=shuffle_optional_batch(state.content_values, permutation),
                content_mask=shuffle_optional_batch(state.content_mask, permutation),
                content_cue_token_ids=shuffle_optional_batch(
                    state.content_cue_token_ids,
                    permutation,
                ),
                content_value_token_ids=shuffle_optional_batch(
                    state.content_value_token_ids,
                    permutation,
                ),
            )
        )
    return shuffled


def shuffle_optional_batch(value, permutation):
    if value is None:
        return None
    if value.shape[0] != permutation.shape[0]:
        return value
    return value[permutation]


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().detach().cpu())


def min_context_for_accuracy(compression: dict[int, list[float]], threshold: float) -> int | None:
    for budget, values in sorted(compression.items()):
        if mean(values) >= threshold:
            return int(budget)
    return None


def mean(values: Sequence[float]) -> float:
    vals = [float(value) for value in values]
    return statistics.fmean(vals) if vals else 0.0


def mean_path(rows: Sequence[dict[str, Any]], path: Sequence[str]) -> float:
    values = []
    for row in rows:
        current: Any = row
        for key in path:
            current = current[key]
        if current is not None:
            values.append(float(current))
    return mean(values)


def min_optional(values: Sequence[int | None]) -> int | None:
    real = [int(value) for value in values if value is not None]
    return min(real) if real else None


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
