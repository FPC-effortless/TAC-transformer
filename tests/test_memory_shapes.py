import torch

from tac_sie.config import TACSIEConfig
from tac_sie.memory import BindingMemoryIO, IdentityState, read_memory, write_slot


def test_identity_state_write_and_read_shapes():
    cfg = TACSIEConfig(device="cpu", n_memory_slots=3)
    batch = 4
    state = IdentityState.init(batch, cfg.n_memory_slots, cfg.d_key, cfg.d_value, cfg.device)
    io = BindingMemoryIO(cfg)

    rule_id = torch.tensor([0, 1, 2, 3])
    offset_id = torch.tensor([1, 2, 3, 4])
    key = io.make_write_key(rule_id)
    value = io.make_write_value(offset_id)

    state = write_slot(state, 1, key, value)
    read_value, attn = read_memory(state, key)

    assert state.memory_keys.shape == (batch, 3, cfg.d_key)
    assert state.memory_values.shape == (batch, 3, cfg.d_value)
    assert state.slot_used.shape == (batch, 3)
    assert read_value.shape == (batch, cfg.d_value)
    assert attn.shape == (batch, 3)
    assert torch.allclose(attn[:, 1], torch.ones(batch), atol=1e-5)

