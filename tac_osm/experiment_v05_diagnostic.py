"""TAC-OSM v0.5 optimization-path diagnostics.

This module does not change the v0.5 promotion gate. It diagnoses where
successful/failed seeds lose the observational context signal:
training fit, persistent-state encoding, t1 retrieval, or forecast binding.
"""
from __future__ import annotations

import argparse
import json

import torch

from .environment import SyntheticOperationalWorld
from .experiment_v05 import (
    ExperimentConfig,
    _context_pair,
    _forecast_with_context,
    action_candidates,
    make_world,
    seed_all,
    train_one,
)
from .experiment_v04 import sample_batch


@torch.no_grad()
def diagnose_seed(seed: int, cfg: ExperimentConfig, device: torch.device) -> dict:
    model = train_one(seed, cfg, device)
    world = make_world(cfg, 0)
    gen = torch.Generator(device=device).manual_seed(seed + 7000)

    t0, t1, _, action, next_state, _ = sample_batch(
        world, cfg.eval_batch, gen, device
    )
    candidates_single = torch.nn.functional.one_hot(
        action, cfg.action_dim
    ).float().view(cfg.eval_batch, 1, 1, cfg.action_dim)
    pred = model.forward_sequence(t0, t1, candidates_single)[:, 0, 0]
    train_fit_mse = torch.mean((pred - next_state) ** 2).item()

    ca, cb = _context_pair(cfg, cfg.eval_batch, device)
    zero_state = torch.zeros(cfg.eval_batch, cfg.input_dim, device=device)
    t0_a = torch.cat([zero_state, ca], -1)
    t0_b = torch.cat([zero_state, cb], -1)
    initial = model.initial_state(cfg.eval_batch, device)
    _, state_a = model.encode(t0_a, initial, write=True)
    _, state_b = model.encode(t0_b, initial, write=True)

    slots_a, valid_a = state_a
    slots_b, valid_b = state_b
    persistent_state_delta = torch.mean(torch.abs(slots_b - slots_a)).item()
    validity_delta = torch.mean(torch.abs(valid_b - valid_a)).item()

    hidden_a, read_a = model.encode(t1, state_a, write=False)
    hidden_b, read_b = model.encode(t1, state_b, write=False)
    read_slots_a, _ = read_a
    read_slots_b, _ = read_b
    retrieval_state_delta = torch.mean(torch.abs(read_slots_b - read_slots_a)).item()
    hidden_delta = torch.mean(torch.abs(hidden_b - hidden_a)).item()

    candidates = action_candidates(cfg.eval_batch, cfg.action_dim, device)
    pred_a = _forecast_with_context(model, t1, ca, candidates)
    pred_b = _forecast_with_context(model, t1, cb, candidates)
    forecast_delta = torch.mean(torch.abs(pred_b - pred_a)).item()

    return {
        "seed": seed,
        "train_fit_mse": train_fit_mse,
        "persistent_state_delta": persistent_state_delta,
        "validity_delta": validity_delta,
        "retrieval_state_delta": retrieval_state_delta,
        "hidden_delta": hidden_delta,
        "forecast_delta": forecast_delta,
        "parameters": model.parameter_count(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = ExperimentConfig(
        train_steps=args.steps,
        batch_size=args.batch,
        eval_batch=args.eval_batch,
    )
    device = torch.device(args.device)
    results = [
        diagnose_seed(int(seed), cfg, device)
        for seed in args.seeds.split(",")
        if seed.strip()
    ]
    print(json.dumps({
        "experiment": "TAC-OSM-v0.5-optimization-diagnostics",
        "promotion_gate": "unchanged",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
