"""Warmup-controlled optimizer reset diagnostic for TAC-OSM v0.4.

Keeps the v0.4 optimizer-reset experiment frozen and adds a fresh-AdamW
reset-shock control. At each reset point, the same weights and RNG state are
branched into continuous and reset trajectories. The reset trajectory is
compared with and without a short LR warmup. Gradient norm and optimizer
state/parameter norms are recorded around the reset.

The warmup is applied only to the fresh-optimizer branch. This tests whether
the degradation observed after a raw AdamW reset is a transient LR/state
mismatch rather than evidence that accumulated optimizer state is itself
required for the capability.
"""

import argparse
import copy
import json
import math
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


def _norm_parameters(model):
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        total = total + torch.sum(p.detach() ** 2)
    return float(torch.sqrt(total).item())


def _norm_gradients(model):
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            total = total + torch.sum(p.grad.detach() ** 2)
    return float(torch.sqrt(total).item())


def _norm_optimizer_state(optimizer):
    first = second = 0.0
    for state in optimizer.state.values():
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is not None:
            first += float(torch.sum(exp_avg.detach() ** 2).item())
        if exp_avg_sq is not None:
            second += float(torch.sum(exp_avg_sq.detach() ** 2).item())
    return math.sqrt(first), math.sqrt(second)


def _set_lr(optimizer, base_lr, warmup_index, warmup_steps):
    if warmup_steps <= 0:
        return
    scale = min(1.0, float(warmup_index) / float(warmup_steps))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


def train_until(seed, cfg, device, reset_points):
    seed_all(seed)
    world = make_world(cfg, regime=0)
    model = ControlledOperationalModel(cfg, True, True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    generator = torch.Generator(device=device).manual_seed(seed + 1000)

    snapshots = {}
    for step in range(1, max(reset_points) + 1):
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
        if step in reset_points:
            snapshots[step] = {
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "generator": generator.get_state(),
            }
    return snapshots


def continue_from_snapshot(
    seed,
    reset_step,
    snapshot,
    cfg,
    device,
    mode,
    checkpoints,
    warmup_steps,
):
    world = make_world(cfg, regime=0)
    model = ControlledOperationalModel(cfg, True, True).to(device)
    model.load_state_dict(snapshot["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    if mode == "continuous":
        optimizer.load_state_dict(snapshot["optimizer"])
    elif mode in {"reset", "reset_warmup"}:
        pass
    else:
        raise ValueError(mode)

    generator = torch.Generator(device=device)
    generator.set_state(snapshot["generator"])

    rows = []
    max_step = max(checkpoints)
    for step in range(reset_step + 1, max_step + 1):
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

        grad_norm = _norm_gradients(model)
        param_norm = _norm_parameters(model)
        opt_first, opt_second = _norm_optimizer_state(optimizer)

        if mode == "reset_warmup":
            warmup_index = step - reset_step
            _set_lr(optimizer, cfg.lr, warmup_index, warmup_steps)

        optimizer.step()

        if step in checkpoints:
            metrics = evaluate(model, cfg, seed + 5000, device)
            rows.append({
                "step": step,
                "steps_since_reset": step - reset_step,
                "train_loss": float(loss.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm": grad_norm,
                "parameter_norm": param_norm,
                "optimizer_exp_avg_norm": opt_first,
                "optimizer_exp_avg_sq_norm": opt_second,
                **metrics,
            })
        elif step - reset_step <= warmup_steps or step - reset_step in {1, 5, 10, 20, 50, 100}:
            # Keep the early transient observable even when not a formal
            # evaluation checkpoint.
            metrics = evaluate(model, cfg, seed + 5000, device)
            rows.append({
                "step": step,
                "steps_since_reset": step - reset_step,
                "train_loss": float(loss.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm": grad_norm,
                "parameter_norm": param_norm,
                "optimizer_exp_avg_norm": opt_first,
                "optimizer_exp_avg_sq_norm": opt_second,
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
    parser.add_argument("--warmup-steps", type=int, default=50)
    args = parser.parse_args()

    resets = tuple(int(x) for x in args.resets.split(",") if x.strip())
    checkpoints = tuple(int(x) for x in args.checkpoints.split(",") if x.strip())
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if any(r >= checkpoints[-1] for r in resets):
        raise SystemExit("every reset point must precede the final checkpoint")
    if args.warmup_steps < 1:
        raise SystemExit("warmup-steps must be positive")

    cfg = ExperimentConfig(train_steps=checkpoints[-1], batch_size=args.batch)
    device = torch.device(args.device)

    results = []
    for seed in seeds:
        snapshots = train_until(seed, cfg, device, resets)
        for reset_step in resets:
            snapshot = snapshots[reset_step]
            branches = {}
            for mode in ("continuous", "reset", "reset_warmup"):
                branches[mode] = continue_from_snapshot(
                    seed, reset_step, snapshot, cfg, device, mode,
                    checkpoints, args.warmup_steps
                )
            results.append({
                "seed": seed,
                "reset_step": reset_step,
                "persistent": True,
                "action_conditioned": True,
                "parameters": ControlledOperationalModel(
                    cfg, True, True
                ).parameter_count(),
                **branches,
            })

    print(json.dumps({
        "experiment": "TAC-OSM-v0.4-optimizer-reset-warmup-control",
        "purpose": (
            "distinguish raw AdamW reset shock from persistent dependence "
            "on accumulated optimizer state"
        ),
        "seeds": seeds,
        "reset_points": resets,
        "checkpoints": checkpoints,
        "warmup_steps": args.warmup_steps,
        "config": asdict(cfg),
        "device": str(device),
        "optimizer": "AdamW",
        "learning_rate": cfg.lr,
        "branching_protocol": (
            "identical model weights and generator state; continuous restores "
            "optimizer state; reset uses fresh AdamW; reset_warmup uses fresh "
            "AdamW plus linear LR warmup"
        ),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
