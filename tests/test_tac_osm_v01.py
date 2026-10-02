import torch
from tac_osm import TACOSM, TACOSMConfig

def test_forward_and_state_carry():
    cfg = TACOSMConfig(input_dim=12, hidden_dim=24, structure_dim=10, state_slots=8, top_k=3)
    model = TACOSM(cfg)
    x = torch.randn(4, cfg.input_dim)
    first = model(x)
    assert first.decision_logits.shape == (4, cfg.decision_dim)
    assert first.verifier_logit.shape == (4,)
    assert first.route_weights.shape == (4, cfg.top_k)
    assert first.selected_indices.shape == (4, cfg.top_k)
    second = model(x, first.state)
    assert second.state.slots.shape == first.state.slots.shape
    assert torch.isfinite(second.decision_logits).all()

def test_reset_and_shuffle_interventions():
    cfg = TACOSMConfig(input_dim=8, hidden_dim=16, structure_dim=8, state_slots=6, top_k=2)
    model = TACOSM(cfg)
    out = model(torch.randn(2, cfg.input_dim))
    reset = model.reset_state(out.state)
    assert torch.count_nonzero(reset.slots) == 0
    assert torch.count_nonzero(reset.validity) == 0
    assert not torch.equal(out.state.slots.flip(1), out.state.slots)

def test_parameter_count_is_small():
    assert TACOSM(TACOSMConfig()).parameter_count() < 1_000_000
