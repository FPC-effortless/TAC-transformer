from tac_sie.memory import BindingMemoryIO, read_memory, write_slot
from tac_sie.types import IdentityState

import torch.nn as nn


class TACSIEModel(nn.Module):
    def __init__(self, cfg, executor):
        super().__init__()
        self.cfg = cfg
        self.memory_io = BindingMemoryIO(cfg)
        self.executor = executor
        self.read_to_executor_offset = nn.Linear(cfg.d_value, cfg.d_model)

    def init_state(self, batch_size: int) -> IdentityState:
        return IdentityState.init(
            batch_size=batch_size,
            n_slots=self.cfg.n_memory_slots,
            d_key=self.cfg.d_key,
            d_value=self.cfg.d_value,
            device=self.cfg.resolved_device(),
        )

    def store_rule(self, state: IdentityState, rule_id, offset_id, slot_idx) -> IdentityState:
        key_vec = self.memory_io.make_write_key(rule_id)
        value_vec = self.memory_io.make_write_value(offset_id)
        return write_slot(state, slot_idx, key_vec, value_vec)

    def retrieve_offset(self, state: IdentityState, rule_id):
        query = self.memory_io.make_query_key(rule_id)
        read_value, attn = read_memory(state, query, temperature=self.cfg.read_temperature)
        offset_logits = self.memory_io.decode_offset(read_value)
        offset_vec = self.read_to_executor_offset(read_value)
        return offset_logits, offset_vec, attn, query, read_value

    def execute(self, x_digit, offset_vec):
        return self.executor(x_digit, offset_vec=offset_vec)

