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
    _set_native_programs_trainable,
    _zero_native_program_parameters,
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
    _set_encoder_trainable,
    _train_state_encoder,
)
from tac_transformer.training import count_parameters


class InternalRouteNativeProgramBinding(nn.Module):
    """Uses TAC's internal identity-field route to select native programs."""

    def __init__(self, *, state_dim: int, d_model: int):
        super().__init__()
        self.query_embedding = nn.Embedding(N_VALUES, state_dim)
        self.state_query_to_model = nn.Sequential(
            nn.Linear(state_dim * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.action_head = nn.Linear(d_model, N_VALUES)

    def forward(
        self,
        field,
        previous_state,
        state_embedding: torch.Tensor,
        query_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_embedding(query_x.to(state_embedding.device))
        hidden = self.state_query_to_model(
            torch.cat([state_embedding, query, state_embedding * query], dim=-1)
        ).unsqueeze(1)
        identity = field(
            hidden,
            previous_state=previous_state,
            collect_auxiliary=True,
            collect_metrics=True,
            update_identity_state=False,
        )
        logits = self.action_head(identity.program_context[:, 0, :])
        return logits, identity.selected_program_mask, identity.activations


def _train_internal_route_binding(
    encoder,
    *,
    seed: int,
    binding_steps: int,
    batch_size: int,
) -> InternalRouteNativeProgramBinding:
    torch.manual_seed(170_000 + seed)
    rng = random.Random(100_000 + seed)
    field = _native_field(encoder)
    head = InternalRouteNativeProgramBinding(
        state_dim=encoder.hidden_dim,
        d_model=encoder.base.config.d_model,
    )
    _set_encoder_trainable(encoder, False)
    native_program_params = _set_native_programs_trainable(encoder, True)
    optimizer = torch.optim.AdamW(list(head.parameters()) + native_program_params, lr=5e-3)
    encoder.eval()
    head.train()
    for _ in range(binding_steps):
        support_a, support_b, rules, query_x, transition_targets = _make_batch(
            rng,
            batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            states = encoder.encode(support_a, support_b, collect_auxiliary=False)
            state_embedding = encoder.state_embedding(states)
        logits, selected_mask, activations = head(field, states[-1], state_embedding, query_x)
        target_selected = selected_mask.gather(1, rules[:, None]).squeeze(1)
        selected_mass_loss = -target_selected.clamp_min(1e-4).log().mean()
        wrong_selected = selected_mask[:, :N_RULES].sum(dim=-1) - target_selected
        route_margin_loss = wrong_selected.clamp_min(0.0).mean()
        loss = (
            F.cross_entropy(logits, transition_targets)
            + 1.0 * F.cross_entropy(activations[:, :N_RULES], rules)
            + 0.75 * selected_mass_loss
            + 0.25 * route_margin_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + native_program_params, 1.0)
        optimizer.step()
    _set_native_programs_trainable(encoder, False)
    return head


@torch.no_grad()
def _evaluate_internal_route_variant(
    encoder,
    head: InternalRouteNativeProgramBinding,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
    rule: int | None = None,
    state_slot_knockout: int | None = None,
) -> dict[str, float]:
    rng = random.Random(110_000 + seed + (0 if rule is None else 997 * int(rule)))
    field = _native_field(encoder)
    rule_correct = 0
    route_selected = 0
    carry_correct = 0
    reset_correct = 0
    shuffled_correct = 0
    total = 0
    encoder.eval()
    head.eval()
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
        state_embedding = encoder.state_embedding(states)
        reset_embedding = encoder.state_embedding(reset_states)
        shuffled_states = _roll_states(states)
        shuffled_embedding = encoder.state_embedding(shuffled_states)

        carry_logits, selected_mask, _ = head(field, states[-1], state_embedding, query_x)
        reset_logits, _, _ = head(field, reset_states[-1], reset_embedding, query_x)
        shuffled_logits, _, _ = head(field, shuffled_states[-1], shuffled_embedding, query_x)
        rule_pred = encoder.rule_head(state_embedding).argmax(dim=-1)
        target_selected = selected_mask.gather(1, rules[:, None]).squeeze(1) > 0.0
        rule_correct += int((rule_pred == rules).sum())
        route_selected += int(target_selected.sum())
        carry_correct += int((carry_logits.argmax(dim=-1) == transition_targets).sum())
        reset_correct += int((reset_logits.argmax(dim=-1) == transition_targets).sum())
        shuffled_correct += int((shuffled_logits.argmax(dim=-1) == transition_targets).sum())
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
    }


def _internal_route_knockout_metrics(
    encoder,
    head: InternalRouteNativeProgramBinding,
    *,
    seed: int,
    knockout_batches: int,
    batch_size: int,
) -> dict[str, float]:
    field = _native_field(encoder)
    correct_drops = []
    wrong_drops = []
    for rule in range(N_RULES):
        base = _evaluate_internal_route_variant(
            encoder,
            head,
            seed=seed,
            eval_batches=knockout_batches,
            batch_size=batch_size,
            rule=rule,
        )["carry_accuracy"]
        with _zero_native_program_parameters(field, rule):
            correct = _evaluate_internal_route_variant(
                encoder,
                head,
                seed=seed,
                eval_batches=knockout_batches,
                batch_size=batch_size,
                rule=rule,
            )["carry_accuracy"]
        with _zero_native_program_parameters(field, (rule + 1) % N_RULES):
            wrong = _evaluate_internal_route_variant(
                encoder,
                head,
                seed=seed,
                eval_batches=knockout_batches,
                batch_size=batch_size,
                rule=rule,
            )["carry_accuracy"]
        correct_drops.append(base - correct)
        wrong_drops.append(base - wrong)

    base_all = _evaluate_internal_route_variant(
        encoder,
        head,
        seed=seed,
        eval_batches=knockout_batches,
        batch_size=batch_size,
    )["carry_accuracy"]
    state_drops = []
    for program_index in range(encoder.base.config.n_programs):
        state_accuracy = _evaluate_internal_route_variant(
            encoder,
            head,
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


def _evaluate_seed(
    *,
    seed: int,
    stage1_steps: int,
    binding_steps: int,
    eval_batches: int,
    batch_size: int,
    knockout_batches: int,
) -> dict[str, float]:
    encoder = _train_state_encoder(
        seed=seed,
        stage1_steps=stage1_steps,
        batch_size=batch_size,
    )
    head = _train_internal_route_binding(
        encoder,
        seed=seed,
        binding_steps=binding_steps,
        batch_size=batch_size,
    )
    metrics = _evaluate_internal_route_variant(
        encoder,
        head,
        seed=seed,
        eval_batches=eval_batches,
        batch_size=batch_size,
    )
    metrics.update(_invariance_metrics(encoder, seed=seed, batch_size=batch_size))
    metrics.update(
        _internal_route_knockout_metrics(
            encoder,
            head,
            seed=seed,
            knockout_batches=knockout_batches,
            batch_size=batch_size,
        )
    )
    metrics["parameter_count_total"] = float(count_parameters(encoder.base)["total"])
    metrics["seed"] = float(seed)
    return metrics


def run_internal_route_native_program_binding(
    *,
    output_dir: Path,
    seeds: Iterable[int] = (7, 19, 31),
    stage1_steps: int = 250,
    binding_steps: int = 240,
    eval_batches: int = 8,
    batch_size: int = 12,
    torch_threads: int = 4,
    knockout_batches: int = 2,
) -> dict:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(torch_threads)))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = tuple(seeds)
    rows = []
    try:
        for seed in seed_list:
            rows.append(
                _evaluate_seed(
                    seed=seed,
                    stage1_steps=stage1_steps,
                    binding_steps=binding_steps,
                    eval_batches=eval_batches,
                    batch_size=batch_size,
                    knockout_batches=knockout_batches,
                )
            )
    finally:
        torch.set_num_threads(previous_threads)

    metrics = _aggregate(rows)
    near_chance_control_accuracy = 1.0 / N_RULES
    near_chance_tolerance = 0.05
    near_chance_threshold = near_chance_control_accuracy + near_chance_tolerance
    validation_passed = (
        metrics["hidden_rule_accuracy"] >= 0.85
        and metrics["carry_accuracy"] >= 0.80
        and metrics["future_transition_accuracy"] >= 0.80
        and metrics["reset_accuracy"] <= near_chance_threshold
        and metrics["shuffled_accuracy"] <= near_chance_threshold
        and metrics["state_slot_knockout_drop"] >= 0.30
        and metrics["correct_program_parameter_knockout_drop"] >= 0.30
        and metrics["program_knockout_selectivity_gap"] >= 0.20
        and metrics["wrong_program_parameter_knockout_drop"] < 0.10
        and metrics["internal_route_role_accuracy"] >= 0.80
    )
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "task": "internal_route_native_program_binding",
            "hypotheses": [
                "TAC internal identity-field routing can replace TAC-231's external state-derived route head.",
                "Correct native program parameter knockout should damage performance much more than wrong program parameter knockout.",
                "Internal route-role accuracy should show that selected_program_mask includes the program matching the hidden rule.",
            ],
            "controls": [
                "reset_identity_state",
                "shuffled_identity_state",
                "state_slot_knockout",
                "correct_native_program_parameter_knockout",
                "wrong_native_program_parameter_knockout",
                "internal_selected_program_mask_route_role",
            ],
            "native_program_parameters": [
                "IdentityFieldLayer.program_expert_down",
                "IdentityFieldLayer.program_expert_up",
                "IdentityFieldLayer.program_expert_bias",
            ],
            "internal_route_source": "IdentityFieldLayer.forward(...).selected_program_mask",
            "stage1_steps": stage1_steps,
            "binding_steps": binding_steps,
            "eval_batches": eval_batches,
            "batch_size": batch_size,
            "knockout_batches": knockout_batches,
            "seeds": list(seed_list),
            "near_chance_control_accuracy": near_chance_control_accuracy,
            "near_chance_tolerance": near_chance_tolerance,
            "near_chance_threshold": near_chance_threshold,
        },
        "variants": {"internal_route_native_program_binding": metrics},
        "per_seed": {"internal_route_native_program_binding": rows},
        "decision": {
            "status": "validated" if validation_passed else "not_validated",
            "boundary": "Actual TAC state encoder training followed by state-query hidden input into TAC IdentityFieldLayer internal routing and native low-rank program parameters.",
        },
    }
    artifact_path = output_dir / "internal_route_native_program_binding.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/internal_route_native_program_binding_tac232_2026_06_12"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--stage1-steps", type=int, default=250)
    parser.add_argument("--binding-steps", type=int, default=240)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--knockout-batches", type=int, default=2)
    args = parser.parse_args()
    result = run_internal_route_native_program_binding(
        output_dir=args.output_dir,
        seeds=args.seeds,
        stage1_steps=args.stage1_steps,
        binding_steps=args.binding_steps,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        knockout_batches=args.knockout_batches,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
