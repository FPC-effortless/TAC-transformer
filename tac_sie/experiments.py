from __future__ import annotations

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
    model = TACSIEModel(cfg, freeze_executor(executor)).to(device)
    return model


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


def _train_single_binding(
    model: TACSIEModel,
    cfg: TACSIEConfig,
    steps: int,
    rule_pool: list[int] | None = None,
    batch_size: int = 64,
):
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


def _sample_multi_episode(cfg: TACSIEConfig, batch_size: int, n_pairs: int, rule_pool: list[int] | None = None):
    device = cfg.resolved_device()
    pool = torch.tensor(rule_pool or list(range(cfg.n_rules)), device=device, dtype=torch.long)
    if len(pool) < n_pairs:
        raise ValueError("rule_pool must contain at least n_pairs unique rules")
    perm = torch.stack([pool[torch.randperm(len(pool), device=device)[:n_pairs]] for _ in range(batch_size)])
    offsets = torch.randint(0, cfg.n_offsets, (batch_size, n_pairs), device=device)
    query_slot = torch.randint(0, n_pairs, (batch_size,), device=device)
    batch_idx = torch.arange(batch_size, device=device)
    rule = perm[batch_idx, query_slot]
    offset = offsets[batch_idx, query_slot]
    digit = torch.randint(0, cfg.n_digits, (batch_size,), device=device)
    target = modular_add_targets(digit, offset, cfg.n_digits)
    return perm, offsets, query_slot, rule, offset, digit, target


def _forward_multi_binding(model: TACSIEModel, rules, offsets, query_slot, query_rule, digit):
    state = model.init_state(rules.size(0))
    for slot_idx in range(rules.size(1)):
        state = model.store_rule(state, rules[:, slot_idx], offsets[:, slot_idx], slot_idx)
    offset_logits, offset_vec, attn, query, read_value = model.retrieve_offset(state, query_rule)
    output_logits = model.execute(digit, offset_vec)
    return state, offset_logits, offset_vec, output_logits, attn, query, read_value


def _train_multi_binding(
    model: TACSIEModel,
    cfg: TACSIEConfig,
    steps: int,
    n_pairs: int,
    rule_pool: list[int] | None = None,
    batch_size: int = 64,
):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(steps):
        rules, offsets, query_slot, rule, offset, digit, target = _sample_multi_episode(cfg, batch_size, n_pairs, rule_pool)
        state, offset_logits, offset_vec, output_logits, _attn, query, _read_value = _forward_multi_binding(
            model, rules, offsets, query_slot, rule, digit
        )
        target_offset_vec = model.executor.offset_embedding(offset)
        loss = (
            F.cross_entropy(output_logits, target)
            + F.cross_entropy(offset_logits, offset)
            + cfg.lambda_offset_vec * offset_vector_distillation_loss(offset_vec, target_offset_vec)
            + cfg.lambda_key_orth * key_orthogonality_loss(state.memory_keys, state.slot_used)
            + cfg.lambda_query_align * query_key_alignment_loss(query, state.memory_keys, query_slot)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def _evaluate_single_binding(model: TACSIEModel, cfg: TACSIEConfig, rule_pool: list[int] | None, batch_size: int = 256):
    rule, offset, digit, target, slot = _sample_episode(cfg, batch_size, rule_pool)
    state, offset_logits, offset_vec, output_logits, attn, _query, _read_value = _forward_single_binding(
        model, rule, offset, digit, slot
    )

    reset_state = model.init_state(rule.size(0))
    reset_logits, reset_offset_vec, _reset_attn, _reset_query, _ = model.retrieve_offset(reset_state, rule)
    reset_output = model.execute(digit, reset_offset_vec)

    shuffled_offset = offset.roll(1)
    shuffled_state = model.init_state(rule.size(0))
    shuffled_state = model.store_rule(shuffled_state, rule, shuffled_offset, slot)
    _shuf_logits, shuffled_offset_vec, _shuf_attn, _shuf_query, _ = model.retrieve_offset(shuffled_state, rule)
    shuffled_output = model.execute(digit, shuffled_offset_vec)

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


@torch.no_grad()
def _evaluate_multi_binding(
    model: TACSIEModel,
    cfg: TACSIEConfig,
    n_pairs: int,
    rule_pool: list[int] | None,
    batch_size: int = 256,
):
    rules, offsets, query_slot, rule, offset, digit, target = _sample_multi_episode(cfg, batch_size, n_pairs, rule_pool)
    state, offset_logits, offset_vec, output_logits, attn, _query, _read_value = _forward_multi_binding(
        model, rules, offsets, query_slot, rule, digit
    )

    reset_state = model.init_state(rule.size(0))
    _reset_logits, reset_offset_vec, _reset_attn, _reset_query, _ = model.retrieve_offset(reset_state, rule)
    reset_output = model.execute(digit, reset_offset_vec)

    shuffled_offsets = offsets.roll(1, dims=0)
    shuffled_state = model.init_state(rule.size(0))
    for slot_idx in range(n_pairs):
        shuffled_state = model.store_rule(shuffled_state, rules[:, slot_idx], shuffled_offsets[:, slot_idx], slot_idx)
    _shuf_logits, shuffled_offset_vec, _shuf_attn, _shuf_query, _ = model.retrieve_offset(shuffled_state, rule)
    shuffled_output = model.execute(digit, shuffled_offset_vec)

    oracle_logits = model.executor(digit, offset_id=offset)
    per_slot_accuracy = {}
    for slot_idx in range(n_pairs):
        mask = query_slot == slot_idx
        if mask.any():
            per_slot_accuracy[f"slot_{slot_idx}_accuracy"] = accuracy(output_logits[mask], target[mask])
    metrics = {
        "carry_accuracy": accuracy(output_logits, target),
        "reset_accuracy": accuracy(reset_output, target),
        "shuffle_accuracy": accuracy(shuffled_output, target),
        "oracle_k_accuracy": accuracy(oracle_logits, target),
        "retrieved_k_accuracy": accuracy(output_logits, target),
        "offset_retrieval_accuracy": accuracy(offset_logits, offset),
        "avg_key_cosine": avg_key_cosine(state.memory_keys, state.slot_used),
        "per_slot_accuracy": per_slot_accuracy,
    }
    metrics.update(attention_diagnostics(attn, query_slot))
    return metrics


def run_exp005e(cfg: TACSIEConfig | None = None, seed: int = 5) -> dict[str, float]:
    cfg = cfg or TACSIEConfig()
    model = _build_pretrained_model(cfg, executor_epochs=250, seed=seed)
    _train_single_binding(model, cfg, steps=120, rule_pool=[0])
    return _evaluate_single_binding(model, cfg, [0])


def run_exp005h(cfg: TACSIEConfig | None = None, seed: int = 5) -> dict[str, float]:
    cfg = cfg or TACSIEConfig()
    model = _build_pretrained_model(cfg, executor_epochs=250, seed=seed)
    _train_single_binding(model, cfg, steps=180, rule_pool=[0, 1])
    return _evaluate_single_binding(model, cfg, [0, 1])


def run_exp006c(cfg: TACSIEConfig | None = None, n_pairs: int = 3, seed: int = 5, train_steps: int = 400):
    cfg = cfg or TACSIEConfig(n_memory_slots=n_pairs)
    cfg.n_memory_slots = n_pairs
    model = _build_pretrained_model(cfg, executor_epochs=250, seed=seed)
    rule_pool = list(range(max(n_pairs, 2)))
    _train_multi_binding(model, cfg, steps=train_steps, n_pairs=n_pairs, rule_pool=rule_pool)
    return _evaluate_multi_binding(model, cfg, n_pairs, rule_pool)


def run_exp007(cfg: TACSIEConfig | None = None, seed: int = 7, train_steps: int = 400):
    cfg = cfg or TACSIEConfig()
    model = _build_pretrained_model(cfg, executor_epochs=250, seed=seed)
    _train_single_binding(model, cfg, steps=train_steps, rule_pool=[0, 1, 2, 3])
    return _evaluate_single_binding(model, cfg, [0, 1, 2, 3])


def run_exp008e(cfg: TACSIEConfig | None = None, seed: int = 8, train_steps: int = 600, executor_epochs: int = 350):
    cfg = cfg or TACSIEConfig()
    model = _build_pretrained_model(cfg, executor_epochs=executor_epochs, seed=seed)
    _train_single_binding(model, cfg, steps=train_steps, rule_pool=[0, 1, 2, 3])
    return _evaluate_single_binding(model, cfg, [0, 1, 2, 3])


@torch.no_grad()
def _same_query_counterfactual(model: TACSIEModel, cfg: TACSIEConfig) -> float:
    device = cfg.resolved_device()
    rule = torch.full((2,), 4, device=device, dtype=torch.long)
    offset = torch.tensor([1, 3], device=device)
    digit = torch.full((2,), 7, device=device, dtype=torch.long)
    target = modular_add_targets(digit, offset, cfg.n_digits)
    slot = torch.zeros(2, device=device, dtype=torch.long)
    _state, _offset_logits, _offset_vec, output_logits, _attn, _query, _read_value = _forward_single_binding(
        model, rule, offset, digit, slot
    )
    return accuracy(output_logits, target)


def run_exp009(
    cfg: TACSIEConfig | None = None,
    train_steps: int = 800,
    executor_epochs: int = 500,
    seed: int = 9,
) -> dict[str, float]:
    cfg = cfg or TACSIEConfig()
    model = _build_pretrained_model(cfg, executor_epochs=executor_epochs, seed=seed)
    _train_single_binding(model, cfg, steps=train_steps, rule_pool=[0, 1, 2, 3])

    known = _evaluate_single_binding(model, cfg, [0, 1, 2, 3])
    new = _evaluate_single_binding(model, cfg, [4, 5, 6, 7])
    metrics = dict(known)
    metrics["known_rule_accuracy"] = known["carry_accuracy"]
    metrics["new_rule_accuracy"] = new["carry_accuracy"]
    metrics["same_query_counterfactual_accuracy"] = _same_query_counterfactual(model, cfg)
    return metrics
