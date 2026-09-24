"""Narrow emergence-threshold follow-up for TAC-OSM v0.4.

Frozen factors:
  - same v0.4 architecture and data protocol
  - reset at step 2000
  - continuous optimizer-state control
  - fresh AdamW reset
  - fresh AdamW + 50-step linear LR warmup

New preregistered controls:
  - 16 seeds
  - 5000 training steps
  - dense 100-step checkpoints from 2000 through 5000
  - primary capability threshold: planning accuracy >= 0.80
  - secondary thresholds: 0.60 and 0.90
  - learning-rate scale 1.0x and 0.5x

The lower-LR arm is trained from the same seed-specific initialization/data
protocol at that LR before branching, so it is a genuine fixed-LR control
rather than a post-hoc LR change after the reset.

Dense checkpoints use a lightweight held-out planning-accuracy evaluation.
Early transient probes additionally record parameter update norm; unlike raw
gradient norm, update norm captures the optimizer/LR-induced reset shock.
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
    action_candidates,
    make_world,
    sample_batch,
    seed_all,
)

RESET_STEP = 2000
WARMUP_STEPS = 50
PRIMARY_THRESHOLD = 0.80
THRESHOLDS = (0.60, 0.80, 0.90)
EARLY_PROBES = frozenset({1, 5, 10, 20, 50, 100})
DENSE_CHECKPOINTS = tuple(range(2000, 5001, 100))


def _norm_gradients(model):
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            total = total + torch.sum(p.grad.detach() ** 2)
    return float(torch.sqrt(total).item())


def _norm_parameters(model):
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        total = total + torch.sum(p.detach() ** 2)
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
    scale = min(1.0, float(warmup_index) / float(warmup_steps))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * scale


@torch.no_grad()
def evaluate_planning_accuracy(model, cfg, seed, device):
    """Lightweight exact planning-accuracy evaluation on held-out regime."""
    held_world = make_world(cfg, regime=cfg.held_out_regime)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0, t1, context, _, _, _ = sample_batch(
        held_world, cfg.eval_batch, gen, device
    )
    candidates = action_candidates(cfg.eval_batch, cfg.action_dim, device)
    pred = model.forward_sequence(t0, t1, candidates)
    scores = held_world.objective(
        pred[:, :, -1].reshape(-1, cfg.input_dim),
        context=context.repeat_interleave(cfg.action_dim, 0),
    ).view(cfg.eval_batch, cfg.action_dim)
    selected = scores.argmax(-1)
    oracle_action, _ = held_world.optimal_action(
        t1[:, :cfg.input_dim], context=context
    )
    return float((selected == oracle_action).float().mean().item())


def train_until(seed, cfg, device, reset_step):
    seed_all(seed)
    world = make_world(cfg, regime=0)
    model = ControlledOperationalModel(cfg, True, True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    generator = torch.Generator(device=device).manual_seed(seed + 1000)

    for step in range(1, reset_step + 1):
        t0, t1, _, action, next_state, _ = sample_batch(
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

    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "generator": generator.get_state(),
    }


def continue_from_snapshot(
    seed,
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
    if RESET_STEP in checkpoints:
        rows.append({
            "step": RESET_STEP,
            "steps_since_reset": 0,
            "planning_accuracy_held_out": evaluate_planning_accuracy(
                model, cfg, seed + 5000, device
            ),
        })

    for step in range(RESET_STEP + 1, max(checkpoints) + 1):
        t0, t1, _, action, next_state, _ = sample_batch(
            world, cfg.batch_size, generator, device
        )
        candidates = torch.nn.functional.one_hot(
            action, cfg.action_dim
        ).float().view(cfg.batch_size, 1, 1, cfg.action_dim)
        pred = model.forward_sequence(t0, t1, candidates)[:, 0, 0]
        loss = torch.mean((pred - next_state) ** 2)
        optimizer.zero_grad()
        loss.backward()

        steps_since_reset = step - RESET_STEP
        is_early = (
            mode in {"reset", "reset_warmup"}
            and steps_since_reset in EARLY_PROBES
        )
        is_dense = step in checkpoints

        if is_early:
            before = [p.detach().clone() for p in model.parameters()]
            grad_norm = _norm_gradients(model)
            param_norm = _norm_parameters(model)
            opt_first, opt_second = _norm_optimizer_state(optimizer)
        else:
            before = None
            grad_norm = None
            param_norm = None
            opt_first = opt_second = None

        if mode == "reset_warmup":
            _set_lr(optimizer, cfg.lr, steps_since_reset, warmup_steps)

        optimizer.step()

        if is_early:
            update_sq = 0.0
            for p, old in zip(model.parameters(), before):
                update_sq += float(torch.sum((p.detach() - old) ** 2).item())
            update_norm = math.sqrt(update_sq)
        else:
            update_norm = None

        if is_early or is_dense:
            acc = evaluate_planning_accuracy(
                model, cfg, seed + 5000, device
            )
            row = {
                "step": step,
                "steps_since_reset": steps_since_reset,
                "planning_accuracy_held_out": acc,
            }
            if is_early:
                row.update({
                    "train_loss": float(loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "gradient_norm": grad_norm,
                    "parameter_norm_before_step": param_norm,
                    "optimizer_exp_avg_norm": opt_first,
                    "optimizer_exp_avg_sq_norm": opt_second,
                    "parameter_update_norm": update_norm,
                })
            rows.append(row)

    return rows


def first_crossing(rows, threshold):
    for row in rows:
        if row["planning_accuracy_held_out"] >= threshold:
            return row["step"]
    return None


def summarize(rows):
    dense = [r for r in rows if r["step"] in DENSE_CHECKPOINTS]
    return {
        "threshold_crossing_steps": {
            f"{threshold:.2f}": first_crossing(dense, threshold)
            for threshold in THRESHOLDS
        },
        "final_planning_accuracy": (
            dense[-1]["planning_accuracy_held_out"] if dense else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--reset", type=int, default=RESET_STEP)
    parser.add_argument("--lr-scales", default="1.0,0.5")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.reset != RESET_STEP:
        raise SystemExit("this preregistered study fixes reset at step 2000")
    if args.warmup_steps != WARMUP_STEPS:
        raise SystemExit("this preregistered study fixes warmup at 50 steps")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    lr_scales = [float(x) for x in args.lr_scales.split(",") if x.strip()]
    if not seeds or not lr_scales or any(x <= 0 for x in lr_scales):
        raise SystemExit("seeds and positive lr-scales are required")

    device = torch.device(args.device)
    results = []

    for lr_scale in lr_scales:
        cfg = ExperimentConfig(
            train_steps=5000,
            batch_size=args.batch,
            lr=ExperimentConfig().lr * lr_scale,
        )
        for seed in seeds:
            snapshot = train_until(seed, cfg, device, args.reset)
            for mode in ("continuous", "reset", "reset_warmup"):
                rows = continue_from_snapshot(
                    seed,
                    snapshot,
                    cfg,
                    device,
                    mode,
                    DENSE_CHECKPOINTS,
                    args.warmup_steps,
                )
                results.append({
                    "seed": seed,
                    "reset_step": args.reset,
                    "lr_scale": lr_scale,
                    "learning_rate": cfg.lr,
                    "mode": mode,
                    "persistent": True,
                    "action_conditioned": True,
                    "parameters": ControlledOperationalModel(
                        cfg, True, True
                    ).parameter_count(),
                    "summary": summarize(rows),
                    "rows": rows,
                })

    print(json.dumps({
        "experiment": "TAC-OSM-v0.4-emergence-threshold",
        "purpose": (
            "test whether optimizer reset/warmup changes the timing of a "
            "pre-registered capability-emergence threshold"
        ),
        "primary_threshold": PRIMARY_THRESHOLD,
        "thresholds": THRESHOLDS,
        "seeds": seeds,
        "reset_step": args.reset,
        "train_steps": 5000,
        "dense_checkpoints": DENSE_CHECKPOINTS,
        "warmup_steps": args.warmup_steps,
        "lr_scales": lr_scales,
        "base_learning_rate": ExperimentConfig().lr,
        "batch_size": args.batch,
        "device": str(device),
        "branching_protocol": (
            "for each seed and LR scale, train one trajectory to step 2000; "
            "branch identical weights and RNG into continuous, fresh-reset, "
            "and fresh-reset+warmup trajectories"
        ),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
