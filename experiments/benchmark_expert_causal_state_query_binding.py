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

from experiments.benchmark_state_query_binding_validation import (
    N_RULES,
    N_VALUES,
    OBSERVE,
    _aggregate,
    _evaluate_variant,
    _invariance_metrics,
    _knockout_state_slot,
    _make_batch,
    _roll_states,
    _set_encoder_trainable,
    _train_state_encoder,
)
from tac_transformer.training import count_parameters


class ExpertRoutedBindingHead(nn.Module):
    """State-query binding head with one transition specialist per latent rule."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_embedding = nn.Embedding(N_VALUES, hidden_dim)
        self.action_embedding = nn.Embedding(N_VALUES, hidden_dim)
        self.route_head = nn.Linear(hidden_dim, N_RULES)
        self.expert_matrix = nn.Parameter(torch.empty(N_RULES, hidden_dim, hidden_dim))
        self.expert_bias = nn.Parameter(torch.zeros(N_RULES, N_VALUES))
        self.hidden_dim = hidden_dim
        nn.init.xavier_uniform_(self.expert_matrix)

    def forward(
        self,
        state_embedding: torch.Tensor,
        query_x: torch.Tensor,
        *,
        knockout_expert: int | None = None,
        knockout_expert_by_sample: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_embedding(query_x.to(state_embedding.device))
        transformed = torch.einsum("bd,edh->beh", query, self.expert_matrix)
        expert_logits = torch.einsum("beh,ah->bea", transformed, self.action_embedding.weight)
        expert_logits = expert_logits + self.expert_bias[None, :, :]

        if knockout_expert is not None:
            expert_logits = expert_logits.clone()
            expert_logits[:, int(knockout_expert), :] = 0.0
        if knockout_expert_by_sample is not None:
            expert_logits = expert_logits.clone()
            rows = torch.arange(expert_logits.shape[0], device=expert_logits.device)
            expert_logits[rows, knockout_expert_by_sample.to(expert_logits.device), :] = 0.0

        route_logits = self.route_head(state_embedding)
        route_probs = F.softmax(route_logits, dim=-1)
        logits = torch.einsum("be,bea->ba", route_probs, expert_logits)
        return logits, route_logits, expert_logits


def _train_expert_binding_head(
    encoder,
    *,
    seed: int,
    binding_steps: int,
    batch_size: int,
) -> ExpertRoutedBindingHead:
    torch.manual_seed(90_000 + seed)
    rng = random.Random(40_000 + seed)
    head = ExpertRoutedBindingHead(encoder.hidden_dim)
    _set_encoder_trainable(encoder, False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-3)
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
        logits, route_logits, expert_logits = head(state_embedding, query_x)
        rows = torch.arange(rules.numel())
        true_expert_logits = expert_logits[rows, rules]
        loss = (
            F.cross_entropy(logits, transition_targets)
            + 1.5 * F.cross_entropy(route_logits, rules)
            + 1.0 * F.cross_entropy(true_expert_logits, transition_targets)
        )
        loss.backward()
        optimizer.step()
    return head


@torch.no_grad()
def _evaluate_expert_variant(
    encoder,
    head: ExpertRoutedBindingHead,
    *,
    seed: int,
    eval_batches: int,
    batch_size: int,
    state_slot_knockout: int | None = None,
    correct_expert_knockout: bool = False,
    wrong_expert_knockout: bool = False,
) -> dict[str, float]:
    rng = random.Random(50_000 + seed)
    rule_correct = 0
    route_correct = 0
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
        shuffled_embedding = encoder.state_embedding(_roll_states(states))

        knockout_by_sample = None
        if correct_expert_knockout:
            knockout_by_sample = rules
        elif wrong_expert_knockout:
            knockout_by_sample = (rules + 1) % N_RULES

        carry_logits, route_logits, _ = head(
            state_embedding,
            query_x,
            knockout_expert_by_sample=knockout_by_sample,
        )
        reset_logits, _, _ = head(reset_embedding, query_x)
        shuffled_logits, _, _ = head(shuffled_embedding, query_x)
        rule_pred = encoder.rule_head(state_embedding).argmax(dim=-1)
        route_pred = route_logits.argmax(dim=-1)
        rule_correct += int((rule_pred == rules).sum())
        route_correct += int((route_pred == rules).sum())
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
        "route_role_accuracy": route_correct / max(total, 1),
    }


def _expert_knockout_metrics(
    encoder,
    head: ExpertRoutedBindingHead,
    *,
    seed: int,
    knockout_batches: int,
    batch_size: int,
) -> dict[str, float]:
    base = _evaluate_expert_variant(
        encoder,
        head,
        seed=seed,
        eval_batches=knockout_batches,
        batch_size=batch_size,
    )["carry_accuracy"]
    correct = _evaluate_expert_variant(
        encoder,
        head,
        seed=seed,
        eval_batches=knockout_batches,
        batch_size=batch_size,
        correct_expert_knockout=True,
    )["carry_accuracy"]
    wrong = _evaluate_expert_variant(
        encoder,
        head,
        seed=seed,
        eval_batches=knockout_batches,
        batch_size=batch_size,
        wrong_expert_knockout=True,
    )["carry_accuracy"]
    state_drops = []
    for program_index in range(encoder.base.config.n_programs):
        state_accuracy = _evaluate_expert_variant(
            encoder,
            head,
            seed=seed,
            eval_batches=knockout_batches,
            batch_size=batch_size,
            state_slot_knockout=program_index,
        )["carry_accuracy"]
        state_drops.append(base - state_accuracy)
    correct_drop = base - correct
    wrong_drop = base - wrong
    return {
        "state_slot_knockout_drop": max(state_drops) if state_drops else 0.0,
        "state_slot_knockout_mean_drop": mean(state_drops) if state_drops else 0.0,
        "correct_expert_knockout_drop": correct_drop,
        "wrong_expert_knockout_drop": wrong_drop,
        "expert_knockout_selectivity_gap": correct_drop - wrong_drop,
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
    head = _train_expert_binding_head(
        encoder,
        seed=seed,
        binding_steps=binding_steps,
        batch_size=batch_size,
    )
    metrics = _evaluate_expert_variant(
        encoder,
        head,
        seed=seed,
        eval_batches=eval_batches,
        batch_size=batch_size,
    )
    metrics.update(_invariance_metrics(encoder, seed=seed, batch_size=batch_size))
    metrics.update(
        _expert_knockout_metrics(
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


def run_expert_causal_state_query_binding(
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
        and metrics["correct_expert_knockout_drop"] >= 0.20
        and metrics["expert_knockout_selectivity_gap"] >= 0.15
        and metrics["wrong_expert_knockout_drop"] <= 0.10
        and metrics["route_role_accuracy"] >= 0.80
    )
    result = {
        "method": {
            "experiment_type": "actual_tac_training",
            "task": "expert_causal_state_query_binding",
            "hypotheses": [
                "Expert causality can be tested after TAC-229 by giving each expert ownership of one transition family.",
                "Correct expert knockout should damage performance much more than wrong expert knockout.",
                "Route-role accuracy should show that state-derived routing chooses the expert matching the hidden rule.",
            ],
            "controls": [
                "reset_identity_state",
                "shuffled_identity_state",
                "state_slot_knockout",
                "correct_expert_knockout",
                "wrong_expert_knockout",
                "route_role_supervision",
            ],
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
        "variants": {"expert_routed_binding": metrics},
        "per_seed": {"expert_routed_binding": rows},
        "decision": {
            "status": "validated" if validation_passed else "not_validated",
            "boundary": "Actual TAC state encoder training followed by route-supervised expert state-query binding heads on a synthetic hidden-rule task.",
        },
    }
    artifact_path = output_dir / "expert_causal_state_query_binding.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/expert_causal_state_query_binding_tac230_2026_06_12"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--stage1-steps", type=int, default=250)
    parser.add_argument("--binding-steps", type=int, default=240)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--knockout-batches", type=int, default=2)
    args = parser.parse_args()
    result = run_expert_causal_state_query_binding(
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
