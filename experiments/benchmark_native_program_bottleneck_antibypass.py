from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_native_program_causal_binding import (
    _native_field,
    _zero_native_program_parameters,
)
from experiments.benchmark_near_native_lm_head_binding import (
    _candidate_logits,
    _query_ids,
    _target_tokens,
)
from experiments.benchmark_state_query_binding_validation import (
    N_RULES,
    N_VALUES,
    OBSERVE,
    _aggregate,
    _invariance_metrics,
    _knockout_state_slot,
    _make_batch,
    _roll_states,
    _train_state_encoder,
)
from tac_transformer.training import count_parameters


DEFAULT_VARIANTS = (
    "program_only_bottleneck",
    "residual_hidden_dropout",
    "gated_program_residual",
    "input_program_bottleneck",
    "state_query_program_bottleneck",
    "slot_conditioned_program_bottleneck",
)


class ProgramBottleneckBridge(nn.Module):
    def __init__(self, *, variant: str, state_dim: int, d_model: int, hidden_keep_prob: float = 0.25):
        super().__init__()
        if variant not in DEFAULT_VARIANTS:
            raise ValueError(f"unknown bottleneck variant: {variant}")
        self.variant = variant
        self.hidden_keep_prob = hidden_keep_prob
        self.query_embedding = nn.Embedding(N_VALUES, d_model)
        self.state_to_model = nn.Linear(state_dim, d_model)
        self.program_input = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid(),
        )
        self.last_native_outputs: torch.Tensor | None = None

    def forward(
        self,
        encoder,
        states,
        state_embedding: torch.Tensor,
        query_x: torch.Tensor,
        *,
        training_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        field = _native_field(encoder)
        state_model = self.state_to_model(state_embedding)
        query_model = self.query_embedding(query_x.to(state_embedding.device))
        program_seed = self.program_input(
            torch.cat([state_model, query_model, state_model * query_model], dim=-1)
        )
        if self.variant in {
            "input_program_bottleneck",
            "state_query_program_bottleneck",
            "slot_conditioned_program_bottleneck",
        }:
            identity = field(
                program_seed.unsqueeze(1),
                previous_state=states[-1],
                collect_auxiliary=True,
                collect_metrics=True,
                update_identity_state=False,
            )
            hidden = program_seed
            selected_mask = identity.selected_program_mask
            if self.variant == "slot_conditioned_program_bottleneck":
                slot_memory = states[-1].program_memory
                native_outputs = field._program_axis_expert_outputs_for_batch(
                    program_seed[:, None, :] + slot_memory
                )
            else:
                expert_input = program_seed
                native_outputs = field._all_program_expert_outputs_for_batch(expert_input)
        else:
            query_output = encoder.base(
                _query_ids(query_x),
                identity_states=states,
                collect_auxiliary=True,
                collect_metrics=True,
            )
            hidden = query_output.hidden_states[:, -1, :]
            selected_mask = query_output.aux.selected_program_mask
            expert_input = program_seed if self.variant == "program_only_bottleneck" else hidden
            native_outputs = field._all_program_expert_outputs_for_batch(expert_input)
        selected = selected_mask[:, :N_RULES]
        route_weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        native_outputs = native_outputs[:, :N_RULES, :]
        self.last_native_outputs = native_outputs
        program_delta = torch.einsum("bp,bpd->bd", route_weights, native_outputs)

        if self.variant in {
            "program_only_bottleneck",
            "input_program_bottleneck",
            "state_query_program_bottleneck",
            "slot_conditioned_program_bottleneck",
        }:
            fused = program_delta
        elif self.variant == "residual_hidden_dropout":
            if training_mode:
                keep = torch.bernoulli(
                    hidden.new_full((hidden.shape[0], 1), self.hidden_keep_prob)
                )
                hidden_path = hidden * keep
            else:
                hidden_path = hidden * self.hidden_keep_prob
            correction = self.delta_mlp(
                torch.cat([hidden, state_model, program_delta, hidden * program_delta], dim=-1)
            )
            fused = hidden_path + correction
        elif self.variant == "gated_program_residual":
            gate = self.gate(torch.cat([hidden, state_model, program_delta], dim=-1))
            fused = hidden + gate * program_delta
        else:
            raise ValueError(f"unknown bottleneck variant: {self.variant}")
        return encoder.base.lm_head(fused), selected_mask


def _train_variant(
    encoder,
    bridge: ProgramBottleneckBridge,
    *,
    seed: int,
    bottleneck_steps: int,
    batch_size: int,
) -> None:
    torch.manual_seed(290_000 + seed)
    rng = random.Random(190_000 + seed)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(bridge.parameters()), lr=2e-3)
    encoder.train()
    bridge.train()
    for _ in range(bottleneck_steps):
        support_a, support_b, rules, query_x, transition_targets = _make_batch(
            rng,
            batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        states = encoder.encode(support_a, support_b, collect_auxiliary=False)
        state_embedding = encoder.state_embedding(states)
        answer_logits, selected_mask = bridge(
            encoder,
            states,
            state_embedding,
            query_x,
            training_mode=True,
        )
        answer_targets = _target_tokens(transition_targets)
        target_selected = selected_mask.gather(1, rules[:, None]).squeeze(1)
        selected_mass_loss = -target_selected.clamp_min(1e-4).log().mean()
        wrong_selected = selected_mask[:, :N_RULES].sum(dim=-1) - target_selected
        route_margin_loss = wrong_selected.clamp_min(0.0).mean()
        program_loss = answer_logits.new_zeros(())
        if bridge.last_native_outputs is not None:
            rows = torch.arange(rules.numel(), device=rules.device)
            true_program_hidden = bridge.last_native_outputs[rows, rules]
            true_program_logits = encoder.base.lm_head(true_program_hidden)
            program_loss = (
                F.cross_entropy(true_program_logits, answer_targets)
                + 0.50 * F.cross_entropy(
                    _candidate_logits(true_program_logits),
                    transition_targets,
                )
            )
        # The direct query-token path is deliberately not rewarded. The final
        # loss forces answer, routing, state, and native program output together.
        loss = (
            F.cross_entropy(answer_logits, answer_targets)
            + 0.75 * F.cross_entropy(_candidate_logits(answer_logits), transition_targets)
            + 1.25 * program_loss
            + 1.00 * selected_mass_loss
            + 0.50 * route_margin_loss
            + 0.75 * F.cross_entropy(encoder.rule_head(state_embedding), rules)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(bridge.parameters()), 1.0)
        optimizer.step()


@torch.no_grad()
def _evaluate_variant(
    encoder,
    bridge: ProgramBottleneckBridge,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
    rule: int | None = None,
    state_slot_knockout: int | None = None,
) -> dict[str, float]:
    rng = random.Random(200_000 + seed + (0 if rule is None else 997 * int(rule)))
    rule_correct = 0
    route_selected = 0
    carry_correct = 0
    reset_correct = 0
    shuffled_correct = 0
    full_vocab_correct = 0
    total = 0
    encoder.eval()
    bridge.eval()
    for _ in range(eval_batches):
        support_a, support_b, rules, query_x, transition_targets = _make_batch(
            rng,
            batch_size=batch_size,
            rule=rule,
        )
        states = encoder.encode(support_a, support_b, collect_auxiliary=True)
        if state_slot_knockout is not None:
            states = _knockout_state_slot(states, state_slot_knockout)
        reset_states = encoder.encode(
            torch.zeros_like(support_a).fill_(OBSERVE),
            torch.zeros_like(support_b).fill_(OBSERVE),
            collect_auxiliary=True,
        )
        shuffled_states = _roll_states(states)
        state_embedding = encoder.state_embedding(states)
        reset_embedding = encoder.state_embedding(reset_states)
        shuffled_embedding = encoder.state_embedding(shuffled_states)
        carry_logits, selected_mask = bridge(encoder, states, state_embedding, query_x)
        reset_logits, _ = bridge(encoder, reset_states, reset_embedding, query_x)
        shuffled_logits, _ = bridge(encoder, shuffled_states, shuffled_embedding, query_x)
        answer_targets = _target_tokens(transition_targets)
        target_selected = selected_mask.gather(1, rules[:, None]).squeeze(1) > 0.0
        rule_pred = encoder.rule_head(state_embedding).argmax(dim=-1)
        rule_correct += int((rule_pred == rules).sum())
        route_selected += int(target_selected.sum())
        carry_correct += int(
            (_candidate_logits(carry_logits).argmax(dim=-1) == transition_targets).sum()
        )
        reset_correct += int(
            (_candidate_logits(reset_logits).argmax(dim=-1) == transition_targets).sum()
        )
        shuffled_correct += int(
            (_candidate_logits(shuffled_logits).argmax(dim=-1) == transition_targets).sum()
        )
        full_vocab_correct += int((carry_logits.argmax(dim=-1) == answer_targets).sum())
        total += int(rules.numel())
    carry_accuracy = carry_correct / max(total, 1)
    reset_accuracy = reset_correct / max(total, 1)
    shuffled_accuracy = shuffled_correct / max(total, 1)
    return {
        "hidden_rule_accuracy": rule_correct / max(total, 1),
        "future_transition_accuracy": carry_accuracy,
        "carry_accuracy": carry_accuracy,
        "reset_accuracy": reset_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "state_advantage": carry_accuracy - reset_accuracy,
        "shuffle_drop": carry_accuracy - shuffled_accuracy,
        "internal_route_role_accuracy": route_selected / max(total, 1),
        "full_vocab_answer_accuracy": full_vocab_correct / max(total, 1),
    }


def _knockout_metrics(
    encoder,
    bridge: ProgramBottleneckBridge,
    *,
    seed: int,
    knockout_batches: int,
    batch_size: int,
) -> dict[str, float]:
    field = _native_field(encoder)
    correct_drops = []
    wrong_drops = []
    for rule in range(N_RULES):
        base = _evaluate_variant(
            encoder,
            bridge,
            seed=seed,
            eval_batches=knockout_batches,
            batch_size=batch_size,
            rule=rule,
        )["carry_accuracy"]
        with _zero_native_program_parameters(field, rule):
            correct = _evaluate_variant(
                encoder,
                bridge,
                seed=seed,
                eval_batches=knockout_batches,
                batch_size=batch_size,
                rule=rule,
            )["carry_accuracy"]
        with _zero_native_program_parameters(field, (rule + 1) % N_RULES):
            wrong = _evaluate_variant(
                encoder,
                bridge,
                seed=seed,
                eval_batches=knockout_batches,
                batch_size=batch_size,
                rule=rule,
            )["carry_accuracy"]
        correct_drops.append(base - correct)
        wrong_drops.append(base - wrong)

    base_all = _evaluate_variant(
        encoder,
        bridge,
        seed=seed,
        eval_batches=knockout_batches,
        batch_size=batch_size,
    )["carry_accuracy"]
    state_drops = []
    for program_index in range(encoder.base.config.n_programs):
        state_accuracy = _evaluate_variant(
            encoder,
            bridge,
            seed=seed,
            eval_batches=knockout_batches,
            batch_size=batch_size,
            state_slot_knockout=program_index,
        )["carry_accuracy"]
        state_drops.append(base_all - state_accuracy)

    correct_drop = mean(correct_drops)
    wrong_drop = mean(wrong_drops)
    return {
        "state_slot_knockout_drop": max(state_drops) if state_drops else 0.0,
        "state_slot_knockout_mean_drop": mean(state_drops) if state_drops else 0.0,
        "correct_program_parameter_knockout_drop": correct_drop,
        "wrong_program_parameter_knockout_drop": wrong_drop,
        "program_knockout_selectivity_gap": correct_drop - wrong_drop,
    }


def _evaluate_seed_variant(
    *,
    variant: str,
    seed: int,
    stage1_steps: int,
    bottleneck_steps: int,
    eval_batches: int,
    batch_size: int,
    knockout_batches: int,
) -> dict[str, float]:
    encoder = _train_state_encoder(
        seed=seed,
        stage1_steps=stage1_steps,
        batch_size=batch_size,
    )
    bridge = ProgramBottleneckBridge(
        variant=variant,
        state_dim=encoder.hidden_dim,
        d_model=encoder.base.config.d_model,
    )
    _train_variant(
        encoder,
        bridge,
        seed=seed,
        bottleneck_steps=bottleneck_steps,
        batch_size=batch_size,
    )
    metrics = _evaluate_variant(
        encoder,
        bridge,
        seed=seed,
        eval_batches=eval_batches,
        batch_size=batch_size,
    )
    metrics.update(_invariance_metrics(encoder, seed=seed, batch_size=batch_size))
    metrics.update(
        _knockout_metrics(
            encoder,
            bridge,
            seed=seed,
            knockout_batches=knockout_batches,
            batch_size=batch_size,
        )
    )
    metrics["parameter_count_total"] = float(count_parameters(encoder.base)["total"])
    metrics["seed"] = float(seed)
    return metrics


def _passes(metrics: dict[str, float], threshold: float) -> bool:
    return (
        metrics["hidden_rule_accuracy"] >= 0.85
        and metrics["carry_accuracy"] >= 0.80
        and metrics["future_transition_accuracy"] >= 0.80
        and metrics["reset_accuracy"] <= threshold
        and metrics["shuffled_accuracy"] <= threshold
        and metrics["state_advantage"] >= 0.50
        and metrics["state_slot_knockout_drop"] >= 0.30
        and metrics["correct_program_parameter_knockout_drop"] >= 0.30
        and metrics["wrong_program_parameter_knockout_drop"] < 0.10
        and metrics["internal_route_role_accuracy"] >= 0.80
    )


def run_native_program_bottleneck_antibypass(
    *,
    output_dir: Path,
    seeds: Iterable[int] = (7, 19, 31),
    stage1_steps: int = 250,
    bottleneck_steps: int = 240,
    eval_batches: int = 8,
    batch_size: int = 12,
    torch_threads: int = 4,
    knockout_batches: int = 2,
    variants: Iterable[str] = DEFAULT_VARIANTS,
) -> dict:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(torch_threads)))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = tuple(seeds)
    variant_list = tuple(variants)
    per_seed = {variant: [] for variant in variant_list}
    try:
        for variant in variant_list:
            for seed in seed_list:
                per_seed[variant].append(
                    _evaluate_seed_variant(
                        variant=variant,
                        seed=seed,
                        stage1_steps=stage1_steps,
                        bottleneck_steps=bottleneck_steps,
                        eval_batches=eval_batches,
                        batch_size=batch_size,
                        knockout_batches=knockout_batches,
                    )
                )
    finally:
        torch.set_num_threads(previous_threads)

    variant_metrics = {variant: _aggregate(rows) for variant, rows in per_seed.items()}
    near_chance_control_accuracy = 1.0 / N_RULES
    near_chance_tolerance = 0.05
    near_chance_threshold = near_chance_control_accuracy + near_chance_tolerance
    best_variant = max(
        variant_metrics,
        key=lambda name: (
            variant_metrics[name]["program_knockout_selectivity_gap"],
            variant_metrics[name]["carry_accuracy"],
            variant_metrics[name]["state_advantage"],
        ),
    )
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "task": "native_program_bottleneck_antibypass",
            "hypotheses": [
                "TAC-234 failed because simple fusion exposed state without making native selected program output mandatory.",
                "A valid anti-bypass bridge must preserve answer accuracy and make correct native program knockout damage performance.",
                "Program-only and hidden-dropout bottlenecks should outperform unconstrained fusion on causal gates.",
            ],
            "variants": list(variant_list),
            "stage1_steps": stage1_steps,
            "bottleneck_steps": bottleneck_steps,
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "knockout_batches": knockout_batches,
            "seeds": list(seed_list),
            "near_chance_control_accuracy": near_chance_control_accuracy,
            "near_chance_tolerance": near_chance_tolerance,
            "near_chance_threshold": near_chance_threshold,
        },
        "variants": variant_metrics,
        "per_seed": per_seed,
        "decision": {
            "status": "validated" if _passes(variant_metrics[best_variant], near_chance_threshold) else "not_validated",
            "best_variant": best_variant,
            "boundary": "Actual TAC query-token path with native selected-program bottleneck variants before normal lm_head answer logits.",
        },
    }
    artifact_path = output_dir / "native_program_bottleneck_antibypass.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/native_program_bottleneck_antibypass_tac235_2026_06_12"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--stage1-steps", type=int, default=250)
    parser.add_argument("--bottleneck-steps", type=int, default=240)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--knockout-batches", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    args = parser.parse_args()
    result = run_native_program_bottleneck_antibypass(
        output_dir=args.output_dir,
        seeds=args.seeds,
        stage1_steps=args.stage1_steps,
        bottleneck_steps=args.bottleneck_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        knockout_batches=args.knockout_batches,
        variants=args.variants,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
