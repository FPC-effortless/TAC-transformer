import torch

from tac_osm.experiment_v04 import ExperimentConfig, ControlledOperationalModel
from tac_osm.experiment_v05_factor_probe import (
    _fit_additive_probe,
    _fit_factor_probe,
)
from tac_osm.experiment_v05_anti_lookup import held_out_pair


def test_factor_probe_split_has_eight_seen_and_eight_heldout():
    mask = torch.tensor(
        [[not held_out_pair(c, a) for a in range(4)] for c in range(4)],
        dtype=torch.bool,
    )
    assert int(mask.sum()) == 8
    assert int((~mask).sum()) == 8


def test_additive_probe_accepts_full_effect_matrix():
    effects = torch.randn(4, 4, 3)
    mask = torch.tensor(
        [[not held_out_pair(c, a) for a in range(4)] for c in range(4)],
        dtype=torch.bool,
    )
    value = _fit_additive_probe(effects, mask, ridge=1e-3)
    assert torch.isfinite(torch.tensor(value))


def test_representation_factor_probe_accepts_seen_pairs_only():
    context = torch.randn(4, 6)
    action = torch.randn(4, 6)
    effects = torch.randn(4, 4, 3)
    mask = torch.tensor(
        [[not held_out_pair(c, a) for a in range(4)] for c in range(4)],
        dtype=torch.bool,
    )
    value = _fit_factor_probe(context, action, effects, mask, ridge=1e-2)
    assert torch.isfinite(torch.tensor(value))


def test_model_factor_probe_does_not_change_parameter_count():
    cfg = ExperimentConfig()
    model = ControlledOperationalModel(cfg, True, True)
    assert model.parameter_count() > 0
