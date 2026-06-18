from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_native_program_causal_binding import (
    _native_field,
    _zero_native_program_parameters,
)
from experiments.benchmark_state_query_binding_validation import (
    N_RULES,
    N_VALUES,
    OBSERVE,
    X_START,
    Y_START,
    _aggregate,
    _invariance_metrics,
    _knockout_state_slot,
    _make_batch,
    _roll_states,
    _train_state_encoder,
)
from tac_transformer.training import count_parameters


def _query_ids(query_x: torch.Tensor) -> torch.Tensor:
    return (X_START + query_x).view(-1, 1)


def _target_tokens(transition_targets: torch.Tensor) -> torch.Tensor:
    return Y_START + transition_targets


def _candidate_logits(vocab_logits: torch.Tensor) -> torch.Tensor:
    return vocab_logits[:, Y_START : Y_START + N_VALUES]


def _train_near_native_lm_path(
    encoder,
    *,
    seed: int,
    binding_steps: int,
    batch_size: int,
) -> None:
    torch.manual_seed(210_000 + seed)
    rng = random.Random(130_000 + seed)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-3)
    encoder.train()
    for _ in range(binding_steps):
        support_a, support_b, rules, query_x, transition_targets = _make_batch(
            rng,
            batch_size=batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        states = encoder.encode(support_a, support_b, collect_auxiliary=False)
        state_embedding = encoder.state_embedding(states)
        query_output = encoder.base(
            _query_ids(query_x),
            identity_states=states,
            collect_auxiliary=True,
            collect_metrics=True,
        )
        answer_logits = query_output.logits[:, -1, :]
        answer_targets = _target_tokens(transition_targets)
        selected_mask = query_output.aux.selected_program_mask
        target_selected = selected_mask.gather(1, rules[:, None]).squeeze(1)
        selected_mass_loss = -target_selected.clamp_min(1e-4).log().mean()
        wrong_selected = selected_mask[:, :N_RULES].sum(dim=-1) - target_selected
        route_margin_loss = wrong_selected.clamp_min(0.0).mean()
        loss = (
            F.cross_entropy(answer_logits, answer_targets)
            + 0.75 * F.cross_entropy(_candidate_logits(answer_logits), transition_targets)
            + 0.75 * F.cross_entropy(query_output.aux.program_activations[:, :N_RULES], rules)
            + 0.50 * selected_mass_loss
            + 0.25 * route_margin_loss
            + 0.50 * F.cross_entropy(encoder.rule_head(state_embedding), rules)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def _evaluate_near_native_variant(
    encoder,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
    rule: int | None = None,
    state_slot_knockout: int | None = None,
) -> dict[str, float]:
    rng = random.Random(140_000 + seed + (0 if rule is None else 997 * int(rule)))
    rule_correct = 0
    route_selected = 0
    carry_correct = 0
    reset_correct = 0
    shuffled_correct = 0
    full_vocab_correct = 0
    total = 0
    encoder.eval()
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

        query = _query_ids(query_x)
        carry_output = encoder.base(
            query,
            identity_states=states,
            collect_auxiliary=True,
            collect_metrics=True,
        )
        reset_output = encoder.base(
            query,
            identity_states=reset_states,
            collect_auxiliary=True,
            collect_metrics=True,
        )
        shuffled_output = encoder.base(
            query,
            identity_states=shuffled_states,
            collect_auxiliary=True,
            collect_metrics=True,
        )
        carry_logits = carry_output.logits[:, -1, :]
        reset_logits = reset_output.logits[:, -1, :]
        shuffled_logits = shuffled_output.logits[:, -1, :]
        rule_pred = encoder.rule_head(state_embedding).argmax(dim=-1)
        target_selected = (
            carry_output.aux.selected_program_mask.gather(1, rules[:, None]).squeeze(1)
            > 0.0
        )
        answer_targets = _target_tokens(transition_targets)
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


def _near_native_knockout_metrics(
    encoder,
    *,
    seed: int,
    knockout_batches: int,
    batch_size: int,
) -> dict[str, float]:
    field = _native_field(encoder)
    correct_drops = []
    wrong_drops = []
    for rule in range(N_RULES):
        base = _evaluate_near_native_variant(
            encoder,
            seed=seed,
            eval_batches=knockout_batches,
            batch_size=batch_size,
            rule=rule,
        )["carry_accuracy"]
        with _zero_native_program_parameters(field, rule):
            correct = _evaluate_near_native_variant(
                encoder,
                seed=seed,
                eval_batches=knockout_batches,
                batch_size=batch_size,
                rule=rule,
            )["carry_accuracy"]
        with _zero_native_program_parameters(field, (rule + 1) % N_RULES):
            wrong = _evaluate_near_native_variant(
                encoder,
                seed=seed,
                eval_batches=knockout_batches,
                batch_size=batch_size,
                rule=rule,
            )["carry_accuracy"]
        correct_drops.append(base - correct)
        wrong_drops.append(base - wrong)

    base_all = _evaluate_near_native_variant(
        encoder,
        seed=seed,
        eval_batches=knockout_batches,
        batch_size=batch_size,
    )["carry_accuracy"]
    state_drops = []
    for program_index in range(encoder.base.config.n_programs):
        state_accuracy = _evaluate_near_native_variant(
            encoder,
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
    _train_near_native_lm_path(
        encoder,
        seed=seed,
        binding_steps=binding_steps,
        batch_size=batch_size,
    )
    metrics = _evaluate_near_native_variant(
        encoder,
        seed=seed,
        eval_batches=eval_batches,
        batch_size=batch_size,
    )
    metrics.update(_invariance_metrics(encoder, seed=seed, batch_size=batch_size))
    metrics.update(
        _near_native_knockout_metrics(
            encoder,
            seed=seed,
            knockout_batches=knockout_batches,
            batch_size=batch_size,
        )
    )
    metrics["parameter_count_total"] = float(count_parameters(encoder.base)["total"])
    metrics["seed"] = float(seed)
    return metrics


def run_near_native_lm_head_binding(
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
            "task": "near_native_lm_head_binding",
            "hypotheses": [
                "The TAC-232 adapter/readout can be relaxed to ordinary TACTransformerLM query-token forward passes.",
                "The normal lm_head over vocabulary tokens can serve as the answer path while preserving internal routing and native program causality.",
                "Correct native program parameter knockout should damage performance much more than wrong program parameter knockout.",
            ],
            "controls": [
                "reset_identity_state",
                "shuffled_identity_state",
                "state_slot_knockout",
                "correct_native_program_parameter_knockout",
                "wrong_native_program_parameter_knockout",
                "internal_selected_program_mask_route_role",
            ],
            "answer_path": "TACTransformerLM.forward(query_token, identity_states=carried_state).logits[:, -1, :]",
            "answer_tokens": "Y_START + transition_target",
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
        "variants": {"near_native_lm_head_binding": metrics},
        "per_seed": {"near_native_lm_head_binding": rows},
        "decision": {
            "status": "validated" if validation_passed else "not_validated",
            "boundary": "Actual TAC query-token forward path with carried IdentityState, internal routing, native low-rank program parameters, and normal lm_head answer logits.",
        },
    }
    artifact_path = output_dir / "near_native_lm_head_binding.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/near_native_lm_head_binding_tac233_2026_06_12"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--stage1-steps", type=int, default=250)
    parser.add_argument("--binding-steps", type=int, default=240)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--knockout-batches", type=int, default=2)
    args = parser.parse_args()
    result = run_near_native_lm_head_binding(
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
