import torch

from tac_osm.environment import SyntheticOperationalWorld, WorldConfig
from tac_osm.experiment_v04 import (
    ControlledOperationalModel,
    ExperimentConfig,
    action_candidates,
    sample_batch,
)


def test_v04_all_cells_have_equal_capacity_and_forward():
    cfg = ExperimentConfig(train_steps=2, batch_size=8)
    models = {}
    for persistent in (False, True):
        for action_conditioned in (False, True):
            model = ControlledOperationalModel(cfg, persistent, action_conditioned)
            models[(persistent, action_conditioned)] = model
    counts = {m.parameter_count() for m in models.values()}
    assert len(counts) == 1

    world = SyntheticOperationalWorld(
        WorldConfig(cfg.input_dim, cfg.action_dim, 0.0, regime=0)
    )
    g = torch.Generator().manual_seed(7)
    for model in models.values():
        t0, t1, _, _, _, _ = sample_batch(world, 8, g, torch.device("cpu"))
        pred = model.forward_sequence(t0, t1, action_candidates(8, cfg.action_dim, torch.device("cpu")))
        assert pred.shape == (8, cfg.action_dim, 1, cfg.input_dim)
        assert torch.isfinite(pred).all()


def test_v04_context_is_absent_at_query_time():
    cfg = ExperimentConfig()
    world = SyntheticOperationalWorld(WorldConfig(cfg.input_dim, cfg.action_dim, 0.0))
    g = torch.Generator().manual_seed(11)
    t0, t1, context, _, _, _ = sample_batch(world, 16, g, torch.device("cpu"))
    assert torch.allclose(t1[:, -cfg.context_dim:], torch.zeros_like(t1[:, -cfg.context_dim:]))
    assert not torch.allclose(t0[:, -cfg.context_dim:], torch.zeros_like(t0[:, -cfg.context_dim:]))
    assert context.shape[-1] == cfg.context_dim


def test_v04_oracle_is_deterministic():
    cfg = ExperimentConfig()
    world = SyntheticOperationalWorld(WorldConfig(cfg.input_dim, cfg.action_dim, 0.0))
    state = torch.randn(32, cfg.input_dim)
    context = torch.randn(32, cfg.context_dim)
    a1, s1 = world.optimal_action(state, context=context)
    a2, s2 = world.optimal_action(state, context=context)
    assert torch.equal(a1, a2)
    assert torch.equal(s1, s2)


def test_v04_action_conditioning_changes_forecast_only_when_enabled():
    cfg = ExperimentConfig()
    world = SyntheticOperationalWorld(WorldConfig(cfg.input_dim, cfg.action_dim, 0.0))
    g = torch.Generator().manual_seed(19)
    t0, t1, _, _, _, _ = sample_batch(world, 8, g, torch.device("cpu"))
    cands = action_candidates(8, cfg.action_dim, torch.device("cpu"))

    a0 = ControlledOperationalModel(cfg, False, False)
    a1 = ControlledOperationalModel(cfg, False, True)
    p0 = a0.forward_sequence(t0, t1, cands)
    p1 = a1.forward_sequence(t0, t1, cands)
    assert torch.max(torch.abs(p0[:, 0] - p0[:, 1])).item() < 1e-7
    assert torch.max(torch.abs(p1[:, 0] - p1[:, 1])).item() > 1e-7


def test_v04_state_carry_changes_predictions():
    cfg = ExperimentConfig()
    world = SyntheticOperationalWorld(WorldConfig(cfg.input_dim, cfg.action_dim, 0.0))
    g = torch.Generator().manual_seed(23)
    t0, t1, _, _, _, _ = sample_batch(world, 8, g, torch.device("cpu"))
    model = ControlledOperationalModel(cfg, True, True)
    cands = action_candidates(8, cfg.action_dim, torch.device("cpu"))
    carry = model.forward_sequence(t0, t1, cands, mode="carry")
    reset = model.forward_sequence(t0, t1, cands, mode="reset")
    assert torch.mean(torch.abs(carry - reset)).item() > 1e-8


def test_v04_held_out_regime_is_distinct():
    cfg = ExperimentConfig()
    train = SyntheticOperationalWorld(WorldConfig(cfg.input_dim, cfg.action_dim, 0.0, regime=0))
    held = SyntheticOperationalWorld(WorldConfig(cfg.input_dim, cfg.action_dim, 0.0, regime=1))
    assert not torch.equal(train.effect, held.effect)
