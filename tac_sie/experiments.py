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
    offset = torch.tensor([0, max(cfg.n_offsets - 1, 0)], device=device)
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


def _nonmatching_offsets(offsets: torch.Tensor, n_offsets: int) -> torch.Tensor:
    if n_offsets <= 1:
        return offsets
    delta = torch.randint(1, n_offsets, offsets.shape, device=offsets.device)
    return (offsets + delta) % n_offsets


def _sample_transfer_episode(
    cfg: TACSIEConfig,
    batch_size: int,
    n_pairs: int,
    query_pool: list[int],
    filler_pool: list[int],
    offset_mode: str,
):
    device = cfg.resolved_device()
    query_rules = torch.tensor(query_pool, device=device, dtype=torch.long)
    filler_rules = torch.tensor(filler_pool, device=device, dtype=torch.long)
    query_rule = query_rules[torch.randint(0, len(query_rules), (batch_size,), device=device)]
    query_slot = torch.randint(0, n_pairs, (batch_size,), device=device)
    rules = torch.empty(batch_size, n_pairs, device=device, dtype=torch.long)
    offsets = torch.randint(0, cfg.n_offsets, (batch_size, n_pairs), device=device)

    for batch_idx in range(batch_size):
        used = {int(query_rule[batch_idx].item())}
        for slot_idx in range(n_pairs):
            if slot_idx == int(query_slot[batch_idx].item()):
                rules[batch_idx, slot_idx] = query_rule[batch_idx]
            else:
                candidates = [int(r.item()) for r in filler_rules if int(r.item()) not in used]
                if not candidates:
                    candidates = [int(r.item()) for r in query_rules if int(r.item()) not in used]
                if not candidates:
                    candidates = [int(filler_rules[0].item())]
                pick = candidates[int(torch.randint(0, len(candidates), (), device=device).item())]
                rules[batch_idx, slot_idx] = pick
                used.add(pick)

    batch_idx = torch.arange(batch_size, device=device)
    if offset_mode == "same_offset":
        offsets[:] = torch.randint(0, cfg.n_offsets, (batch_size, 1), device=device)
    elif offset_mode == "new_assignment":
        offsets[batch_idx, query_slot] = (query_rule - 3).clamp_min(0) % cfg.n_offsets

    offset = offsets[batch_idx, query_slot]
    digit = torch.randint(0, cfg.n_digits, (batch_size,), device=device)
    target = modular_add_targets(digit, offset, cfg.n_digits)
    return rules, offsets, query_slot, query_rule, offset, digit, target


@torch.no_grad()
def _evaluate_transfer_controls(
    model: TACSIEModel,
    cfg: TACSIEConfig,
    n_pairs: int,
    query_pool: list[int],
    filler_pool: list[int],
    offset_mode: str,
    batch_size: int = 256,
):
    rules, offsets, query_slot, rule, offset, digit, target = _sample_transfer_episode(
        cfg, batch_size, n_pairs, query_pool, filler_pool, offset_mode
    )
    state, offset_logits, offset_vec, output_logits, attn, _query, _read_value = _forward_multi_binding(
        model, rules, offsets, query_slot, rule, digit
    )

    reset_state = model.init_state(batch_size)
    _reset_logits, reset_vec, _reset_attn, _reset_query, _ = model.retrieve_offset(reset_state, rule)
    reset_output = model.execute(digit, reset_vec)

    wrong_offsets = _nonmatching_offsets(offsets, cfg.n_offsets)
    wrong_offset_state = model.init_state(batch_size)
    for slot_idx in range(n_pairs):
        wrong_offset_state = model.store_rule(wrong_offset_state, rules[:, slot_idx], wrong_offsets[:, slot_idx], slot_idx)
    _wrong_offset_logits, wrong_offset_vec, _wo_attn, _wo_query, _ = model.retrieve_offset(wrong_offset_state, rule)
    wrong_offset_output = model.execute(digit, wrong_offset_vec)

    shuffled_state = model.init_state(batch_size)
    shuffled_offsets = offsets.roll(1, dims=0)
    for slot_idx in range(n_pairs):
        shuffled_state = model.store_rule(shuffled_state, rules[:, slot_idx], shuffled_offsets[:, slot_idx], slot_idx)
    _shuf_logits, shuffled_vec, _shuf_attn, _shuf_query, _ = model.retrieve_offset(shuffled_state, rule)
    shuffled_output = model.execute(digit, shuffled_vec)

    swapped_rules = rules.roll(1, dims=0)
    swapped_offsets = offsets.roll(1, dims=0)
    swapped_state = model.init_state(batch_size)
    for slot_idx in range(n_pairs):
        swapped_state = model.store_rule(swapped_state, swapped_rules[:, slot_idx], swapped_offsets[:, slot_idx], slot_idx)
    _swap_logits, swapped_vec, _swap_attn, _swap_query, _ = model.retrieve_offset(swapped_state, rule)
    swapped_output = model.execute(digit, swapped_vec)

    wrong_rule_state = model.init_state(batch_size)
    wrong_rules = (rules + 1) % cfg.n_rules
    for slot_idx in range(n_pairs):
        wrong_rule_state = model.store_rule(wrong_rule_state, wrong_rules[:, slot_idx], offsets[:, slot_idx], slot_idx)
    _wr_logits, wrong_rule_vec, _wr_attn, _wr_query, _ = model.retrieve_offset(wrong_rule_state, rule)
    wrong_rule_output = model.execute(digit, wrong_rule_vec)

    random_query_rule = torch.randint(0, cfg.n_rules, rule.shape, device=rule.device)
    same = random_query_rule == rule
    random_query_rule[same] = (random_query_rule[same] + 1) % cfg.n_rules
    _rand_logits, random_vec, _rand_attn, _rand_query, _ = model.retrieve_offset(state, random_query_rule)
    random_query_output = model.execute(digit, random_vec)

    oracle_logits = model.executor(digit, offset_id=offset)
    metrics = {
        "carry_accuracy": accuracy(output_logits, target),
        "reset_accuracy": accuracy(reset_output, target),
        "shuffle_accuracy": accuracy(shuffled_output, target),
        "no_store_accuracy": accuracy(reset_output, target),
        "wrong_offset_accuracy": accuracy(wrong_offset_output, target),
        "wrong_rule_state_accuracy": accuracy(wrong_rule_output, target),
        "swapped_state_accuracy": accuracy(swapped_output, target),
        "random_query_rule_accuracy": accuracy(random_query_output, target),
        "oracle_k_accuracy": accuracy(oracle_logits, target),
        "retrieved_k_accuracy": accuracy(output_logits, target),
        "offset_retrieval_accuracy": accuracy(offset_logits, offset),
        "avg_key_cosine": avg_key_cosine(state.memory_keys, state.slot_used),
    }
    metrics.update(attention_diagnostics(attn, query_slot))
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
    rule_conditions = {
        "known_rule": ([0, 1, 2, 3], [0, 1, 2, 3], "random"),
        "new_rule": ([4, 5, 6, 7], [4, 5, 6, 7], "random"),
        "new_rule_same_offset": ([4, 5, 6, 7], [4, 5, 6, 7], "same_offset"),
        "new_rule_new_assignment": ([4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], "new_assignment"),
    }
    rows = []

    for seed in seeds:
        for n_offsets in n_offsets_values:
            for n_slots in n_memory_slots_values:
                cfg = TACSIEConfig(device=device, n_offsets=n_offsets, n_memory_slots=n_slots)
                model = _build_pretrained_model(cfg, executor_epochs=executor_epochs, seed=seed)
                n_pairs = min(n_slots, 4)
                _train_multi_binding(
                    model,
                    cfg,
                    steps=train_steps,
                    n_pairs=n_pairs,
                    rule_pool=[0, 1, 2, 3],
                    batch_size=64,
                )
                same_query_counterfactual = _same_query_counterfactual(model, cfg)
                for condition, (query_pool, filler_pool, offset_mode) in rule_conditions.items():
                    metrics = _evaluate_transfer_controls(
                        model,
                        cfg,
                        n_pairs=n_pairs,
                        query_pool=query_pool,
                        filler_pool=filler_pool,
                        offset_mode=offset_mode,
                        batch_size=batch_size,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "n_memory_slots": n_slots,
                            "n_pairs": n_pairs,
                            "n_offsets": n_offsets,
                            "rule_condition": condition,
                            "same_query_counterfactual_accuracy": same_query_counterfactual,
                            **metrics,
                        }
                    )

    def row_mean(key: str, selected_rows: list[dict] | None = None) -> float:
        selected_rows = rows if selected_rows is None else selected_rows
        return mean(float(row[key]) for row in selected_rows)

    new_rows = [row for row in rows if row["rule_condition"].startswith("new_rule")]
    summary = {
        "rows": len(rows),
        "chance_offset_accuracy": {str(n): 1.0 / n for n in n_offsets_values},
        "carry_accuracy": row_mean("carry_accuracy"),
        "known_rule_accuracy": row_mean("carry_accuracy", [row for row in rows if row["rule_condition"] == "known_rule"]),
        "new_rule_accuracy": row_mean("carry_accuracy", new_rows),
        "same_query_counterfactual_accuracy": row_mean("same_query_counterfactual_accuracy"),
        "reset_accuracy": row_mean("reset_accuracy"),
        "shuffle_accuracy": row_mean("shuffle_accuracy"),
        "no_store_accuracy": row_mean("no_store_accuracy"),
        "wrong_offset_accuracy": row_mean("wrong_offset_accuracy"),
        "wrong_rule_state_accuracy": row_mean("wrong_rule_state_accuracy"),
        "swapped_state_accuracy": row_mean("swapped_state_accuracy"),
        "random_query_rule_accuracy": row_mean("random_query_rule_accuracy"),
        "oracle_k_accuracy": row_mean("oracle_k_accuracy"),
        "offset_retrieval_accuracy": row_mean("offset_retrieval_accuracy"),
        "correct_slot_attention": row_mean("correct_slot_attention"),
        "avg_key_cosine": row_mean("avg_key_cosine"),
    }
    max_chance = max(1.0 / n for n in n_offsets_values)
    summary["pass"] = (
        summary["carry_accuracy"] > 0.95
        and summary["new_rule_accuracy"] > 0.90
        and summary["same_query_counterfactual_accuracy"] > 0.90
        and summary["offset_retrieval_accuracy"] > 0.95
        and summary["correct_slot_attention"] > 0.80
        and summary["reset_accuracy"] <= max_chance + 0.10
        and summary["shuffle_accuracy"] <= max_chance + 0.10
        and summary["no_store_accuracy"] <= max_chance + 0.10
    )
    return {"summary": summary, "rows": rows}
