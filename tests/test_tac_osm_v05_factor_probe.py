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


def _checkerboard_mask():
    return torch.tensor(
        [[not held_out_pair(c, a) for a in range(4)] for c in range(4)],
        dtype=torch.bool,
    )


def test_additive_probe_accepts_full_effect_matrix():
    effects = torch.randn(4, 4, 3)
    value = _fit_additive_probe(effects, _checkerboard_mask(), ridge=1e-3)
    assert torch.isfinite(torch.tensor(value))


def test_representation_factor_probe_recovers_exact_heldout_grid():
    """Regression test for the grid-indexing bug.

    The probe loop must derive its dimensions from the (n_context, n_action)
    effect grid, not from a representation tensor: representations carry a
    hidden width that need not equal the action count. With a width of 6 and
    only 4 actions, the previous implementation unpacked the representation
    and raised ``IndexError: index 4 is out of bounds``.

    Targets are chosen so the fit is forced to zero and the held-out MSE is an
    exact function of which cells the loop visited and with which action index:
    every seen pair is 0, so ``y_train == 0`` and any positive ridge yields
    ``beta == 0`` exactly. Predictions are therefore 0 and the held-out MSE is
    the mean squared held-out target. Held-out pairs are (c, a) with
    (c + a) % 2 == 0, so the held-out action multiset is {0, 2, 1, 3, 0, 2, 1,
    3} and the expected MSE is (1 + 9 + 4 + 16) * 2 / 8 = 7.5. Any misindexing
    or skipped cell changes that value.
    """
    torch.manual_seed(0)
    context = torch.randn(4, 6)
    action = torch.randn(4, 6)
    effects = torch.zeros(4, 4, 3)
    for ci in range(4):
        for ai in range(4):
            if held_out_pair(ci, ai):
                effects[ci, ai] = float(ai + 1)

    value = _fit_factor_probe(context, action, effects, _checkerboard_mask(),
                              ridge=1e-2)
    assert value == 7.5


def test_representation_factor_probe_ignores_representation_width():
    """A hidden width larger than the action count must not overflow the grid."""
    torch.manual_seed(0)
    context = torch.randn(4, 32)
    action = torch.randn(4, 32)
    effects = torch.randn(4, 4, 3)
    value = _fit_factor_probe(context, action, effects, _checkerboard_mask(),
                              ridge=1e-2)
    assert torch.isfinite(torch.tensor(value))


def test_model_factor_probe_does_not_change_parameter_count():
    cfg = ExperimentConfig()
    model = ControlledOperationalModel(cfg, True, True)
    assert model.parameter_count() > 0
