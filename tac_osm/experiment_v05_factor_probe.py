"""Diagnostic probe for TAC-OSM v0.5 anti-lookup models.

This module does not change training, architecture, or the checkerboard split.
It asks whether held-out failure is already present in the learned factors or
appears when those factors must be composed at execution time.

For each seed it:
  * trains the unchanged anti-lookup model to 1000 steps;
  * extracts context and action representations independently;
  * evaluates all 16 context/action combinations on one fixed query state;
  * fits additive and representation-conditioned factor probes on the 8 seen
    pairs and evaluates them on the 8 held-out pairs.

The probes are diagnostics, not promotion criteria.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from .experiment_v04 import ControlledOperationalModel, ExperimentConfig
from .experiment_v05_anti_lookup import (
    TRAIN_STEPS,
    AntiLookupWorld,
    held_out_pair,
    make_world,
    sample_batch,
)


def _ridge_fit_predict(x_train, y_train, x_test, ridge=1e-3):
    """Fit a small ridge probe without changing the trained model."""
    ones_train = torch.ones(x_train.size(0), 1, device=x_train.device)
    ones_test = torch.ones(x_test.size(0), 1, device=x_test.device)
    xt = torch.cat([ones_train, x_train], -1)
    xv = torch.cat([ones_test, x_test], -1)
    eye = torch.eye(xt.size(1), device=xt.device)
    eye[0, 0] = 0.0
    beta = torch.linalg.solve(xt.T @ xt + ridge * eye, xt.T @ y_train)
    return xv @ beta


def _fit_factor_probe(context_repr, action_repr, effects, seen_mask, ridge):
    """Probe y(c,a) using [context, action, context*action] features."""
    c, a = context_repr.shape
    rows = []
    targets = []
    test_rows = []
    test_targets = []
    for ci in range(c):
        for ai in range(a):
            feat = torch.cat(
                [context_repr[ci], action_repr[ai],
                 context_repr[ci] * action_repr[ai]], -1
            )
            if seen_mask[ci, ai]:
                rows.append(feat)
                targets.append(effects[ci, ai])
            else:
                test_rows.append(feat)
                test_targets.append(effects[ci, ai])
    x_train = torch.stack(rows)
    y_train = torch.stack(targets)
    x_test = torch.stack(test_rows)
    y_test = torch.stack(test_targets)
    pred = _ridge_fit_predict(x_train, y_train, x_test, ridge=ridge)
    return torch.mean((pred - y_test) ** 2).item()


def _fit_additive_probe(effects, seen_mask, ridge):
    c, a, d = effects.shape
    x = []
    y = []
    xt = []
    yt = []
    for ci in range(c):
        for ai in range(a):
            feat = F.one_hot(torch.tensor(ci), c).float()
            feat = torch.cat([feat, F.one_hot(torch.tensor(ai), a).float()])
            if seen_mask[ci, ai]:
                x.append(feat)
                y.append(effects[ci, ai])
            else:
                xt.append(feat)
                yt.append(effects[ci, ai])
    pred = _ridge_fit_predict(
        torch.stack(x), torch.stack(y), torch.stack(xt), ridge=ridge
    )
    return torch.mean((pred - torch.stack(yt)) ** 2).item()


@torch.no_grad()
def collect_probe_data(model, cfg, seed, device):
    world = AntiLookupWorld(
        make_world(cfg, 0).cfg
    )
    gen = torch.Generator(device=device).manual_seed(seed + 9000)
    query_state = torch.randn(
        1, cfg.input_dim, generator=gen, device=device
    )

    # Context representation is produced before the query is observed.
    zero_state = model.initial_state(1, device)
    contexts = F.one_hot(
        torch.arange(cfg.context_dim, device=device),
        num_classes=cfg.context_dim,
    ).float()
    t0 = torch.cat(
        [torch.zeros(cfg.context_dim, cfg.input_dim, device=device), contexts], -1
    )
    context_hidden, context_states = model.encode(
        t0, model.initial_state(cfg.context_dim, device), write=True
    )
    context_repr = context_hidden.detach()

    actions = F.one_hot(
        torch.arange(cfg.action_dim, device=device), cfg.action_dim
    ).float()
    action_repr = model.action_encoder(actions).detach()

    # Full 4x4 prediction matrix at one identical query state.
    effects = []
    for ci in range(cfg.context_dim):
        context = contexts[ci:ci + 1]
        t1 = torch.cat(
            [query_state, torch.zeros(1, cfg.context_dim, device=device)], -1
        )
        hidden, state = model.encode(
            t1,
            (
                context_states[0][ci:ci + 1].clone(),
                context_states[1][ci:ci + 1].clone(),
            ),
            write=False,
        )
        candidates = actions.view(1, cfg.action_dim, 1, cfg.action_dim)
        pred = model.candidate_forecast(hidden, state, candidates)[0, :, 0]
        effects.append(pred)
    effects = torch.stack(effects)

    seen_mask = torch.tensor(
        [[not held_out_pair(ci, ai) for ai in range(cfg.action_dim)]
         for ci in range(cfg.context_dim)],
        device=device, dtype=torch.bool,
    )

    # Remove the common query-state baseline so the probe focuses on the
    # context/action effect. The same baseline is used for every pair.
    baseline = effects.mean(0, keepdim=True).mean(1, keepdim=True)
    effect_delta = effects - baseline

    additive_mse = _fit_additive_probe(effect_delta, seen_mask, ridge=1e-3)
    factor_mse = _fit_factor_probe(
        context_repr, action_repr, effect_delta, seen_mask, ridge=1e-3
    )

    # Representation diagnostics. Action representations are intentionally
    # context-independent in this architecture; report their geometry rather
    # than pretending it is learned context invariance.
    context_cos = F.cosine_similarity(
        context_repr[:, None, :], context_repr[None, :, :], dim=-1
    )
    action_cos = F.cosine_similarity(
        action_repr[:, None, :], action_repr[None, :, :], dim=-1
    )

    return {
        "context_repr_norm_mean": context_repr.norm(dim=-1).mean().item(),
        "action_repr_norm_mean": action_repr.norm(dim=-1).mean().item(),
        "offdiag_context_cosine_mean": (
            (context_cos.sum() - torch.diagonal(context_cos).sum()) / 12
        ).item(),
        "offdiag_action_cosine_mean": (
            (action_cos.sum() - torch.diagonal(action_cos).sum()) / 12
        ).item(),
        "seen_pair_effect_magnitude": effect_delta[seen_mask].norm(dim=-1).mean().item(),
        "heldout_pair_effect_magnitude": effect_delta[~seen_mask].norm(dim=-1).mean().item(),
        "additive_probe_heldout_mse": additive_mse,
        "representation_factor_probe_heldout_mse": factor_mse,
        "probe_ridge": 1e-3,
        "seen_pair_count": int(seen_mask.sum()),
        "heldout_pair_count": int((~seen_mask).sum()),
    }


def train_one(seed, cfg, device):
    torch.manual_seed(seed)
    world = make_world(cfg, 0)
    model = ControlledOperationalModel(cfg, True, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)
    last_loss = None
    for _ in range(cfg.train_steps):
        t0, t1, action, target = sample_batch(
            world, cfg.batch_size, gen, device
        )
        candidates = F.one_hot(action, cfg.action_dim).float().view(
            cfg.batch_size, 1, 1, cfg.action_dim
        )
        pred = model.forward_sequence(t0, t1, candidates)[:, 0, 0]
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
    return model, last_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        train_steps=args.steps,
        batch_size=args.batch,
    )
    device = torch.device(args.device)
    results = []
    for raw in args.seeds.split(","):
        if not raw.strip():
            continue
        seed = int(raw)
        model, train_loss = train_one(seed, cfg, device)
        probe = collect_probe_data(model, cfg, seed, device)
        probe.update({"seed": seed, "train_loss": train_loss})
        results.append(probe)

    print(json.dumps({
        "experiment": "TAC-OSM-v0.5-factor-probe",
        "parent_benchmark": "v0.5-anti-lookup",
        "training_architecture_unchanged": True,
        "training_objective_unchanged": True,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
