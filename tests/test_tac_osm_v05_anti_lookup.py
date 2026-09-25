import torch
from tac_osm.environment import WorldConfig
from tac_osm.experiment_v04 import ExperimentConfig
from tac_osm.experiment_v05_anti_lookup import (
    AntiLookupWorld, held_out_pair, sample_batch,
)

def test_anti_lookup_split_exposes_all_individual_factors():
    assert all(any(not held_out_pair(c, a) for a in range(4)) for c in range(4))
    assert all(any(held_out_pair(c, a) for a in range(4)) for c in range(4))
    assert all(any(not held_out_pair(c, a) for c in range(4)) for a in range(4))
    assert all(any(held_out_pair(c, a) for c in range(4)) for a in range(4))

def test_anti_lookup_world_is_compositional():
    cfg = ExperimentConfig()
    world = AntiLookupWorld(
        WorldConfig(cfg.input_dim, cfg.action_dim, 0.0, cfg.context_dim, 0)
    )
    state = torch.zeros(4, cfg.input_dim)
    contexts = torch.eye(4)
    actions = torch.arange(4)
    out = world.transition(state, actions, context=contexts)
    expected = torch.stack([world.effect[c ^ a] for c, a in enumerate(actions)])
    assert torch.allclose(out, expected)

def test_anti_lookup_training_samples_never_contain_heldout_pairs():
    cfg = ExperimentConfig(batch_size=64)
    world = AntiLookupWorld(
        WorldConfig(cfg.input_dim, cfg.action_dim, 0.0, cfg.context_dim, 0)
    )
    gen = torch.Generator().manual_seed(1234)
    t0, _, action, _ = sample_batch(
        world, cfg.batch_size, gen, torch.device("cpu")
    )
    context = t0[:, -4:].argmax(-1)
    assert not torch.any(torch.tensor([
        held_out_pair(int(c), int(a))
        for c, a in zip(context, action)
    ]))
