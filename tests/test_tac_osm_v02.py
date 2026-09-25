import torch
from tac_osm import TACOSM, TACOSMConfig

def test_action_conditioned_forecast_surface():
    cfg = TACOSMConfig(input_dim=10, hidden_dim=20, structure_dim=12,
                       state_slots=6, top_k=2, action_dim=3, forecast_horizon=2)
    model = TACOSM(cfg)
    x = torch.randn(4, cfg.input_dim)
    actions = torch.randn(4, 5, cfg.forecast_horizon, cfg.action_dim)
    out = model(x, candidate_actions=actions)
    assert out.action_logits.shape == (4, cfg.action_dim)
    assert out.forecast.shape == (4, 5, cfg.forecast_horizon, cfg.input_dim)
    assert out.action_value.shape == (4, 5)
    assert torch.isfinite(out.forecast).all()
    assert torch.isfinite(out.action_value).all()

def test_action_conditioning_changes_prediction():
    cfg = TACOSMConfig(input_dim=6, hidden_dim=16, structure_dim=8,
                       state_slots=4, top_k=2, action_dim=2, forecast_horizon=1)
    model = TACOSM(cfg)
    x = torch.randn(2, cfg.input_dim)
    a0 = torch.zeros(2, 1, 1, cfg.action_dim)
    a1 = torch.ones(2, 1, 1, cfg.action_dim)
    y0 = model(x, candidate_actions=a0).forecast
    y1 = model(x, candidate_actions=a1).forecast
    assert not torch.allclose(y0, y1)

def test_v01_forward_remains_compatible():
    cfg = TACOSMConfig(input_dim=8, hidden_dim=16, structure_dim=8,
                       state_slots=6, top_k=2)
    out = TACOSM(cfg)(torch.randn(2, cfg.input_dim))
    assert out.decision_logits.shape == (2, cfg.decision_dim)
    assert out.verifier_logit.shape == (2,)
    assert out.forecast is None
