import torch
import torch.nn as nn
import torch.nn.functional as F

from tac_sie.types import IdentityState


def write_slot(state: IdentityState, slot_idx, key_vec: torch.Tensor, value_vec: torch.Tensor) -> IdentityState:
    new_keys = state.memory_keys.clone()
    new_values = state.memory_values.clone()
    new_used = state.slot_used.clone()

    if isinstance(slot_idx, int):
        new_keys[:, slot_idx, :] = key_vec
        new_values[:, slot_idx, :] = value_vec
        new_used[:, slot_idx] = 1.0
    else:
        batch_idx = torch.arange(key_vec.size(0), device=key_vec.device)
        new_keys[batch_idx, slot_idx, :] = key_vec
        new_values[batch_idx, slot_idx, :] = value_vec
        new_used[batch_idx, slot_idx] = 1.0

    return IdentityState(new_keys, new_values, new_used)


def read_memory(state: IdentityState, query_key: torch.Tensor, temperature: float = 1.0):
    q = F.normalize(query_key, dim=-1)
    k = F.normalize(state.memory_keys, dim=-1)
    logits = torch.einsum("bd,bnd->bn", q, k) / temperature
    logits = logits.masked_fill(state.slot_used <= 0, -1e9)
    attn = torch.softmax(logits, dim=-1)
    read_value = torch.einsum("bn,bnd->bd", attn, state.memory_values)
    return read_value, attn


class BindingMemoryIO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rule_embedding = nn.Embedding(cfg.n_rules, cfg.d_model)
        self.offset_embedding = nn.Embedding(cfg.n_offsets, cfg.d_model)
        self.write_key = nn.Linear(cfg.d_model, cfg.d_key)
        self.write_value = nn.Linear(cfg.d_model, cfg.d_value)
        self.query_key = nn.Linear(cfg.d_model, cfg.d_key)
        self.value_to_offset_logits = nn.Linear(cfg.d_value, cfg.n_offsets)

        nn.init.normal_(self.rule_embedding.weight, mean=0.0, std=1.0)
        self.rule_embedding.weight.requires_grad = False

    def encode_rule(self, rule_id: torch.Tensor) -> torch.Tensor:
        return self.rule_embedding(rule_id)

    def encode_offset(self, offset_id: torch.Tensor) -> torch.Tensor:
        return self.offset_embedding(offset_id)

    def make_write_key(self, rule_id: torch.Tensor) -> torch.Tensor:
        return self.write_key(self.encode_rule(rule_id))

    def make_write_value(self, offset_id: torch.Tensor) -> torch.Tensor:
        return self.write_value(self.encode_offset(offset_id))

    def make_query_key(self, rule_id: torch.Tensor) -> torch.Tensor:
        return self.query_key(self.encode_rule(rule_id))

    def decode_offset(self, read_value: torch.Tensor) -> torch.Tensor:
        return self.value_to_offset_logits(read_value)

