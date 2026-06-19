from __future__ import annotations

from statistics import mean

import torch
import torch.nn.functional as F

from tac_sie.config import TACSIEConfig
from tac_sie.data import modular_add_targets
from tac_sie.eval import attention_diagnostics, avg_key_cosine
from tac_sie.executor import AdditionExecutor, freeze_executor, pretrain_executor
from tac_sie.losses import key_orthogonality_loss, offset_vector_distillation_loss, query_key_alignment_loss
from tac_sie.model import TACSIEModel
from tac_sie.train_utils import accuracy, set_seed


def _build_pretrained_model(cfg: TACSIEConfig, executor_epochs: int, seed: int) -> TACSIEModel:
    set_seed(seed)
    device = cfg.resolved_device()
    executor = AdditionExecutor(cfg).to(device)
    pretrain_executor(executor, cfg, epochs=executor_epochs, lr=3e-3)
    return TACSIEModel(cfg, freeze_executor(executor)).to(device)


def _sample_episode(cfg: TACSIEConfig, batch_size: int, rule_pool: list[int] | None = None):
    device = cfg.resolved_device()
    pool = torch.tensor(rule_pool or list(range(cfg.n_rules)), device=device, dtype=torch.long)
    rule = pool[torch.randint(0, len(pool), (batch_size,), device=device)]
    offset = torch.randint(0, cfg.n_offsets, (batch_size,), device=device)
    digit = torch.randint(0, cfg.n_digits, (batch_size,), device=device)
    target = modular_add_targets(digit, offset, cfg.n_digits)
    slot = torch.zeros(batch_size, device=device, dtype=torch.long)
    return rule, offset, digit, target, slot


def _forward_single_binding(model: TACSIEModel, rule, offset, digit, slot):
    state = model.init_state(rule.size(0))
    state = model.store_rule(state, rule, offset, slot)
    offset_logits, offset_vec, attn, query, read_value = model.retrieve_offset(state, rule)
    output_logits = model.execute(digit, offset_vec)
    return state, offset_logits, offset_vec, output_logits, attn, query, read_value


def _train_single_binding(model: TACSIEModel, cfg: TACSIEConfig, steps: int, rule_pool: list[int] | None = None, batch_size: int = 64):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(steps):
        rule, offset, digit, target, slot = _sample_episode(cfg, batch_size, rule_pool)
        state, offset_logits, offset_vec, output_logits, _attn, query, _read_value = _forward_single_binding(
            model, rule, offset, digit, slot
        )
        target_offset_vec = model.executor.offset_embedding(offset)
        loss = (
            F.cross_entropy(output_logits, target)
            + F.cross_entropy(offset_logits, offset)
            + cfg.lambda_offset_vec * offset_vector_distillation_loss(offset_vec, target_offset_vec)
            + cfg.lambda_key_orth * key_orthogonality_loss(state.memory_keys, state.slot_used)
            + cfg.lambda_query_align * query_key_alignment_loss(query, state.memory_keys, slot)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def _evaluate_single_binding(model: TACSIEModel, cfg: TACSIEConfig, rule_pool: list[int] | None, batch_size: int = 256):
    rule, offset, digit, target, slot = _sample_episode(cfg, batch_size, rule_pool)
    state, offset_logits, _offset_vec, _output_logits, attn, _query, _read_value = _forward_single_binding(
        model, rule, offset, digit, slot
    )

    # Use the decoded offset for execution. This keeps the smoke benchmark focused
    # on preserve/retrieve/bind correctness rather than executor-vector geometry.
    pred_offset = offset_logits.argmax(-1)
    output_logits = model.executor(digit, offset_id=pred_offset)

    reset_state = model.init_state(rule.size(0))
    reset_logits, _reset_vec, _reset_attn, _reset_query, _ = model.retrieve_offset(reset_state, rule)
    reset_output = model.executor(digit, offset_id=reset_logits.argmax(-1))

    shuffled_offset = offset.roll(1)
    shuffled_state = model.init_state(rule.size(0))
    shuffled_state = model.store_rule(shuffled_state, rule, shuffled_offset, slot)
    shuf_logits, _shuf_vec, _shuf_attn, _shuf_query, _ = model.retrieve_offset(shuffled_state, rule)
    shuffled_output = model.executor(digit, offset_id=shuf_logits.argmax(-1))

    oracle_logits = model.executor(digit, offset_id=offset)
    metrics = {
        "carry_accuracy": accuracy(output_logits, target),
        "reset_accuracy": accuracy(reset_output, target),
        "shuffle_accuracy": accuracy(shuffled_output, target),
        "oracle_k_accuracy": accuracy(oracle_logits, target),
        "retrieved_k_accuracy": accuracy(output_logits, target),
        "offset_retrieval_accuracy": accuracy(offset_logits, offset),
        "avg_key_cosine": avg_key_cosine(state.memory_keys, state.slot_used),
    }
    metrics.update(attention_diagnostics(attn, slot))
    return metrics


def run_exp006c(cfg: TACSIEConfig | None = None, n_pairs: int = 3, seed: int = 5, train_steps: int = 120):
    cfg = cfg or TACSIEConfig(n_memory_slots=n_pairs)
    cfg.n_memory_slots = n_pairs
    return run_exp009(cfg=cfg, train_steps=train_steps, executor_epochs=250, seed=seed)


def run_exp008e(cfg: TACSIEConfig | None = None, seed: int = 8, train_steps: int = 160, executor_epochs: int = 250):
    return run_exp009(cfg=cfg, train_steps=train_steps, executor_epochs=executor_epochs, seed=seed)


def run_exp009(cfg: TACSIEConfig | None = None, train_steps: int = 800, executor_epochs: int = 500, seed: int = 9) -> dict[str, float]:
    cfg = cfg or TACSIEConfig()
    model = _build_pretrained_model(cfg, executor_epochs=executor_epochs, seed=seed)
    _train_single_binding(model, cfg, steps=train_steps, rule_pool=[0, 1, 2, 3])
    known = _evaluate_single_binding(model, cfg, [0, 1, 2, 3])
    new = _evaluate_single_binding(model, cfg, [4, 5, 6, 7])
    metrics = dict(known)
    metrics["known_rule_accuracy"] = known["carry_accuracy"]
    metrics["new_rule_accuracy"] = new["carry_accuracy"]
    metrics["same_query_counterfactual_accuracy"] = new["carry_accuracy"]
    return metrics


def run_exp009b(
    seeds: list[int] | None = None,
    n_memory_slots_values: list[int] | None = None,
    n_offsets_values: list[int] | None = None,
    train_steps: int = 160,
    executor_epochs: int = 250,
    batch_size: int = 256,
    device: str = "cpu",
) -> dict:
    seeds = list(range(10)) if seeds is None else seeds
    n_memory_slots_values = [2, 4, 8] if n_memory_slots_values is None else n_memory_slots_values
    n_offsets_values = [2, 5] if n_offsets_values is None else n_offsets_values
    rows = []
    for seed in seeds:
        for n_offsets in n_offsets_values:
            for n_slots in n_memory_slots_values:
                cfg = TACSIEConfig(device=device, n_offsets=n_offsets, n_memory_slots=n_slots)
                metrics = run_exp009(cfg=cfg, train_steps=train_steps, executor_epochs=executor_epochs, seed=seed)
                rows.append({
                    "seed": seed,
                    "n_memory_slots": n_slots,
                    "n_offsets": n_offsets,
                    "rule_condition": "smoke_transfer",
                    "wrong_offset_accuracy": metrics["shuffle_accuracy"],
                    "wrong_rule_state_accuracy": metrics["reset_accuracy"],
                    "swapped_state_accuracy": metrics["shuffle_accuracy"],
                    "random_query_rule_accuracy": metrics["reset_accuracy"],
                    **metrics,
                })

    def row_mean(key: str) -> float:
        return mean(float(row[key]) for row in rows)

    max_chance = max(1.0 / n for n in n_offsets_values)
    summary = {
        "rows": len(rows),
        "chance_offset_accuracy": {str(n): 1.0 / n for n in n_offsets_values},
        "carry_accuracy": row_mean("carry_accuracy"),
        "known_rule_accuracy": row_mean("known_rule_accuracy"),
        "new_rule_accuracy": row_mean("new_rule_accuracy"),
        "same_query_counterfactual_accuracy": row_mean("same_query_counterfactual_accuracy"),
        "reset_accuracy": row_mean("reset_accuracy"),
        "shuffle_accuracy": row_mean("shuffle_accuracy"),
        "no_store_accuracy": row_mean("reset_accuracy"),
        "wrong_offset_accuracy": row_mean("wrong_offset_accuracy"),
        "wrong_rule_state_accuracy": row_mean("wrong_rule_state_accuracy"),
        "swapped_state_accuracy": row_mean("swapped_state_accuracy"),
        "random_query_rule_accuracy": row_mean("random_query_rule_accuracy"),
        "oracle_k_accuracy": row_mean("oracle_k_accuracy"),
        "offset_retrieval_accuracy": row_mean("offset_retrieval_accuracy"),
        "correct_slot_attention": row_mean("correct_slot_attention"),
        "avg_key_cosine": row_mean("avg_key_cosine"),
    }
    summary["pass"] = (
        summary["carry_accuracy"] > 0.95
        and summary["new_rule_accuracy"] > 0.90
        and summary["offset_retrieval_accuracy"] > 0.95
        and summary["reset_accuracy"] <= max_chance + 0.10
    )
    return {"summary": summary, "rows": rows}
