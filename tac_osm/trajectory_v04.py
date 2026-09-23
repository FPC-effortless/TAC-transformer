"""Training-trajectory diagnostic for TAC-OSM v0.4.

This diagnostic does not alter the canonical benchmark. Each seed trains once per
factor cell and evaluates at fixed checkpoints so we can locate basin transitions
rather than comparing only the terminal step.

Primary comparison:
  P0-A1: persistent=False, action_conditioned=True
  P1-A1: persistent=True, action_conditioned=True

The initialization, optimizer, batch generator, and evaluation seed protocol are
identical to experiment_v04.train_one.
"""

import argparse
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


CHECKPOINTS = (250, 500, 1000, 1500, 2000, 2500, 3000)


def train_trajectory(seed, persistent, action_conditioned, cfg, device, checkpoints):
    seed_all(seed)
    world = make_world(cfg, regime=0)
    model = ControlledOperationalModel(
        cfg, persistent, action_conditioned
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    generator = torch.Generator(device=device).manual_seed(seed + 1000)

    rows = []
    checkpoint_set = set(checkpoints)
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

        if step in checkpoint_set:
            metrics = evaluate(model, cfg, seed + 5000, device)
            rows.append({
                "step": step,
                "train_loss": float(loss.item()),
                **metrics,
            })

    return {
        "seed": seed,
        "persistent": persistent,
        "action_conditioned": action_conditioned,
        "parameters": model.parameter_count(),
        "checkpoints": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--steps", default="250,500,1000,1500,2000,2500,3000")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoints = tuple(
        int(x.strip()) for x in args.steps.split(",") if x.strip()
    )
    if not checkpoints or sorted(set(checkpoints)) != list(checkpoints):
        raise SystemExit("checkpoints must be a non-empty strictly increasing list")
    if checkpoints[-1] <= 0:
        raise SystemExit("final checkpoint must be positive")

    device = torch.device(args.device)
    cfg = ExperimentConfig(
        train_steps=checkpoints[-1],
        batch_size=args.batch,
    )
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    results = []
    for seed in seeds:
        for persistent in (False, True):
            results.append(
                train_trajectory(
                    seed,
                    persistent,
                    True,
                    cfg,
                    device,
                    checkpoints,
                )
            )

    print(json.dumps({
        "experiment": "TAC-OSM-v0.4-training-trajectory",
        "purpose": (
            "locate persistence-dependent basin transitions and distinguish "
            "early advantage from late stochastic transition"
        ),
        "seeds": seeds,
        "checkpoints": checkpoints,
        "config": asdict(cfg),
        "device": str(device),
        "factor_cells": ["P0-A1", "P1-A1"],
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
