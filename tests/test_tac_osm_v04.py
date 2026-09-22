import torch

from tac_osm.environment import SyntheticOperationalWorld, WorldConfig
from tac_osm.experiment_v04 import ControlledOperationalModel, ExperimentConfig, sample_batch


def test_v04_all_four_cells_forward():
    cfg = ExperimentConfig(train_steps=2, batch_size=8)
    world = SyntheticOperationalWorld(WorldConfig(cfg.input_dim, cfg.action_dim, 0.0))
    g = torch.Generator().manual_seed(7)
    for persistent in (False, True):
        for action_conditioned in (False, True):
            model = ControlledOperationalModel(cfg, persistent, action_conditioned)
            s, a, target, _ = sample_batch(world, 8, g, torch.device("cpu"))
            pred, logits, state = model(s, a, model.initial_state(8, torch.device("cpu")))
            assert pred.shape == target.shape
            assert logits.shape == (8, cfg.action_dim)
            assert torch.isfinite(pred).all()
            assert torch.isfinite(logits).all()
            if persistent:
                assert state is not None


def test_v04_reproducible_initialization():
    cfg = ExperimentConfig()
    torch.manual_seed(123)
    a = ControlledOperationalModel(cfg, True, True)
    torch.manual_seed(123)
    b = ControlledOperationalModel(cfg, True, True)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
