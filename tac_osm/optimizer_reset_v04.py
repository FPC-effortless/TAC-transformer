"""Optimizer-state reset diagnostic for TAC-OSM v0.4.

Tests whether the late P1-A1 capability basin is retained by model weights
alone or depends on accumulated AdamW optimizer state.

For each seed, a single continuous trajectory is trained to the reset points.
At each reset point, the exact model weights and RNG-generator state are
branched into:
  - continuous: restore the original optimizer state
  - reset:      create a fresh AdamW optimizer on identical model weights

Both branches consume identical training batches after the branch, isolating
optimizer state from data/order effects.
"""

import argparse
import copy
import json
from dataclasses import asdict

import torch

from .experiment_v04 import (
    ExperimentConfig,
    ControlledOperationalModel,
    evaluate,
    make_world,
    sample_batch,
    seed_all,
)


DEFAULT_RESETS = (1000, 1500, 2000)
DEFAULT_CHECKPOINTS = (1000, 1500, 2000, 2500, 3000)


def train_until(seed, cfg, device, checkpoints):
    seed_all(seed)
    world = make_world(cfg, regime=0)
    model = ControlledOperationalModel(cfg, True, True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    generator = torch.Generator(device=device).manual_seed(seed + 1000)

    snapshots = {}
    for step in range(1, max(checkpoints) + 1):
        t0, t1, context, action, next_state, _ = sample_batch(
            world, cfg.batch_size, generator, device
        )
        candidates = torch.nn.functional.one_hot(
            action, cfg.action_dim
        ).float().view(cfg.batch_size, 1, 1, cfg.action_dim)
        pred = model.forward_sequence(t0, t1, candidates)[:, 0, 0]
        loss = torch.mean((pred - next_state) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in checkpoints:
            snapshots[step] = {
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "generator": generator.get_state(),
            }

    return snapshots


def continue_from_snapshot(
    seed, reset_step, snapshot, cfg, device, reset_optimizer, checkpoints
):
    world = make_world(cfg, regime=0)
    model = ControlledOperationalModel(cfg, True, True).to(device)
    model.load_state_dict(snapshot["model"])

    if reset_optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        optimizer.load_state_dict(snapshot["optimizer"])

    generator = torch.Generator(device=device)
    generator.set_state(snapshot["generator"])

    rows = []
    for step in range(reset_step + 1, max(checkpoints) + 1):
        t0, t1, context, action, next_state, _ = sample_batch(
            world, cfg.batch_size, generator, device
        )
        candidates = torch.nn.functional.one_hot(
            action, cfg.action_dim
        ).float().view(cfg.batch_size, 1, 1, cfg.action_dim)
        pred = model.forward_sequence(t0, t1, candidates)[:, 0, 0]
        loss = torch.mean((pred - next_state) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in checkpoints:
            metrics = evaluate(model, cfg, seed + 5000, device)
            rows.append({
                "step": step,
                "train_loss": float(loss.item()),
                **metrics,
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4,5,9,12")
    parser.add_argument("--resets", default="1000,1500,2000")
    parser.add_argument("--checkpoints", default="1000,1500,2000,2500,3000")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    resets = tuple(int(x) for x in args.resets.split(",") if x.strip())
    checkpoints = tuple(int(x) for x in args.checkpoints.split(",") if x.strip())
    if not resets or not checkpoints:
        raise SystemExit("resets and checkpoints must be non-empty")
    if tuple(sorted(set(resets))) != resets:
        raise SystemExit("resets must be strictly increasing")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise SystemExit("checkpoints must be strictly increasing")
    if any(r >= checkpoints[-1] for r in resets):
        raise SystemExit("every reset point must precede the final checkpoint")

    device = torch.device(args.device)
    cfg = ExperimentConfig(train_steps=checkpoints[-1], batch_size=args.batch)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    results = []
    for seed in seeds:
        snapshot_steps = tuple(sorted(set(resets)))
        snapshots = train_until(seed, cfg, device, snapshot_steps)

        for reset_step in resets:
            snapshot = snapshots[reset_step]
            continuous = continue_from_snapshot(
                seed, reset_step, snapshot, cfg, device, False, checkpoints
            )
            reset = continue_from_snapshot(
                seed, reset_step, snapshot, cfg, device, True, checkpoints
            )
            results.append({
                "seed": seed,
                "reset_step": reset_step,
                "persistent": True,
                "action_conditioned": True,
                "parameters": ControlledOperationalModel(
                    cfg, True, True
                ).parameter_count(),
                "continuous": continuous,
                "optimizer_reset": reset,
            })

    print(json.dumps({
        "experiment": "TAC-OSM-v0.4-optimizer-reset",
        "purpose": (
            "test whether the late P1-A1 capability basin is retained by "
            "model weights or depends on accumulated AdamW optimizer state"
        ),
        "seeds": seeds,
        "reset_points": resets,
        "checkpoints": checkpoints,
        "config": asdict(cfg),
        "device": str(device),
        "optimizer": "AdamW",
        "learning_rate": cfg.lr,
        "branching_protocol": (
            "identical model weights and generator state; continuous branch "
            "restores optimizer state; reset branch uses fresh AdamW"
        ),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
