"""TAC-OSM v0.5 optimization-path diagnostics.

This module does not change the v0.5 promotion gate. It diagnoses where
successful/failed seeds lose the observational context signal:
encoding, routed retrieval, executor binding, or forecast binding.

The checkpoint trajectory diagnostic keeps one optimizer trajectory per seed
and measures the same model at successive training checkpoints. This is
distinct from the independent-duration sweep.
"""
from __future__ import annotations

import argparse
import json

import torch

from .experiment_v05 import (
    ExperimentConfig,
    _context_pair,
    _forecast_with_context,
    action_candidates,
    evaluate,
    make_world,
    seed_all,
    train_one,
)
from .experiment_v04 import ControlledOperationalModel, sample_batch


def _route(model, hidden, state):
    slots, validity = state
    q = model.route_query(hidden).unsqueeze(1)
    scores = (q * slots).sum(-1) / (model.cfg.structure_dim ** 0.5)
    weights = torch.softmax(
        scores.masked_fill(validity <= 1e-6, -1e9), -1
    )
    return (slots * weights.unsqueeze(-1)).sum(1), weights


def diagnose_model(
    model,
    seed: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict:
    """Measure pathway propagation for an already-trained model."""
    with torch.no_grad():
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

        t0_hidden_a = model.encoder(t0_a)
        t0_hidden_b = model.encoder(t0_b)
        t0_hidden_delta = torch.mean(
            torch.abs(t0_hidden_b - t0_hidden_a)
        ).item()

        _, state_a = model.encode(t0_a, initial, write=True)
        _, state_b = model.encode(t0_b, initial, write=True)

        slots_a, valid_a = state_a
        slots_b, valid_b = state_b
        persistent_state_delta = torch.mean(
            torch.abs(slots_b - slots_a)
        ).item()
        validity_delta = torch.mean(torch.abs(valid_b - valid_a)).item()

        hidden_a, _ = model.encode(t1, state_a, write=False)
        hidden_b, _ = model.encode(t1, state_b, write=False)

        structure_a, route_weights_a = _route(model, hidden_a, state_a)
        structure_b, route_weights_b = _route(model, hidden_b, state_b)
        routed_structure_delta = torch.mean(
            torch.abs(structure_b - structure_a)
        ).item()
        route_weight_delta = torch.mean(
            torch.abs(route_weights_b - route_weights_a)
        ).item()

        active_a = model.executor(
            torch.cat([hidden_a, structure_a], -1)
        )
        active_b = model.executor(
            torch.cat([hidden_b, structure_b], -1)
        )
        active_delta = torch.mean(torch.abs(active_b - active_a)).item()
        hidden_delta = torch.mean(torch.abs(hidden_b - hidden_a)).item()

        candidates = action_candidates(cfg.eval_batch, cfg.action_dim, device)
        pred_a = _forecast_with_context(model, t1, ca, candidates)
        pred_b = _forecast_with_context(model, t1, cb, candidates)
        forecast_delta = torch.mean(torch.abs(pred_b - pred_a)).item()

    return {
        "seed": seed,
        "train_fit_mse": train_fit_mse,
        "t0_hidden_delta": t0_hidden_delta,
        "persistent_state_delta": persistent_state_delta,
        "validity_delta": validity_delta,
        "routed_structure_delta": routed_structure_delta,
        "route_weight_delta": route_weight_delta,
        "hidden_delta": hidden_delta,
        "active_delta": active_delta,
        "forecast_delta": forecast_delta,
        "parameters": model.parameter_count(),
    }



def _parameter_group_snapshot(model):
    """Return parameter norms grouped by functional module."""
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = name.split(".", 1)[0]
        groups.setdefault(group, {"parameter_norm": 0.0, "parameters": 0})
        groups[group]["parameter_norm"] += float(torch.sum(param.detach() ** 2).item())
        groups[group]["parameters"] += param.numel()
    for group in groups:
        groups[group]["parameter_norm"] = groups[group]["parameter_norm"] ** 0.5
    return groups


def _gradient_group_snapshot(model):
    """Return L2 gradient norms grouped by functional module."""
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        group = name.split(".", 1)[0]
        groups.setdefault(group, 0.0)
        groups[group] += float(torch.sum(param.grad.detach() ** 2).item())
    return {group: value ** 0.5 for group, value in groups.items()}


def _parameter_drift(before, model):
    """Return L2 parameter drift from a saved snapshot, grouped by module."""
    drift = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = name.split(".", 1)[0]
        delta = param.detach() - before[name]
        drift[group] = drift.get(group, 0.0) + float(torch.sum(delta ** 2).item())
    return {group: value ** 0.5 for group, value in drift.items()}


def _train_transition_diagnostic(
    seed: int,
    cfg: ExperimentConfig,
    start_step: int,
    end_step: int,
    interval: int,
    device: torch.device,
) -> list[dict]:
    """Track parameter-group gradients/drift through one optimizer trajectory."""
    if start_step <= 0 or end_step <= start_step:
        raise ValueError("require 0 < start_step < end_step")
    if interval <= 0:
        raise ValueError("interval must be positive")

    seed_all(seed)
    world = make_world(cfg, 0)
    model = ControlledOperationalModel(cfg, True, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)

    baseline = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    rows = []
    grad_accum = {}
    completed = 0

    for step in range(1, end_step + 1):
        t0, t1, _, action, next_state, _ = sample_batch(
            world, cfg.batch_size, gen, device
        )
        candidates = torch.nn.functional.one_hot(
            action, cfg.action_dim
        ).float().view(cfg.batch_size, 1, 1, cfg.action_dim)
        pred = model.forward_sequence(t0, t1, candidates)[:, 0, 0]
        loss = torch.mean((pred - next_state) ** 2)
        opt.zero_grad()
        loss.backward()

        for group, value in _gradient_group_snapshot(model).items():
            grad_accum[group] = grad_accum.get(group, 0.0) + value

        opt.step()

        if step >= start_step and (step == end_step or (step - start_step) % interval == 0):
            current = _parameter_group_snapshot(model)
            drift = _parameter_drift(baseline, model)
            row = {
                "seed": seed,
                "train_steps": step,
                "train_loss": float(loss.detach().item()),
                "parameter_drift": drift,
                "parameter_norm": {
                    group: values["parameter_norm"] for group, values in current.items()
                },
                "cumulative_gradient_norm": dict(grad_accum),
                "gradient_norm_per_step": {
                    group: value / max(step - completed, 1)
                    for group, value in grad_accum.items()
                },
            }
            path = diagnose_model(model, seed, cfg, device)
            intervention = evaluate(model, cfg, seed + 5000, device)
            row.update({
                "routed_structure_delta": path["routed_structure_delta"],
                "persistent_state_delta": path["persistent_state_delta"],
                "forecast_delta": path["forecast_delta"],
                "normalized_contrast_mse": intervention["normalized_contrast_mse"],
                "context_flip_recall": intervention["context_flip_recall"],
            })
            rows.append(row)
            grad_accum = {}
            completed = step

    return rows

def diagnose_seed(seed: int, cfg: ExperimentConfig, device: torch.device) -> dict:
    # Training must retain autograd; only post-training measurements are
    # inference diagnostics.
    model = train_one(seed, cfg, device)
    return diagnose_model(model, seed, cfg, device)


def _train_continuous_checkpoints(
    seed: int,
    cfg: ExperimentConfig,
    checkpoints: list[int],
    device: torch.device,
) -> list[dict]:
    """Train one optimizer trajectory and measure it at each checkpoint."""
    if not checkpoints:
        return []
    if any(step <= 0 for step in checkpoints):
        raise ValueError("checkpoint steps must be positive")
    if checkpoints != sorted(set(checkpoints)):
        raise ValueError("checkpoint steps must be strictly increasing")

    seed_all(seed)
    world = make_world(cfg, 0)
    model = ControlledOperationalModel(cfg, True, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)

    rows = []
    completed = 0
    for target in checkpoints:
        for _ in range(completed, target):
            t0, t1, _, action, next_state, _ = sample_batch(
                world, cfg.batch_size, gen, device
            )
            candidates = torch.nn.functional.one_hot(
                action, cfg.action_dim
            ).float().view(cfg.batch_size, 1, 1, cfg.action_dim)
            pred = model.forward_sequence(t0, t1, candidates)[:, 0, 0]
            loss = torch.mean((pred - next_state) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        completed = target

        row = diagnose_model(model, seed, cfg, device)
        intervention = evaluate(model, cfg, seed + 5000, device)
        row.update({
            "train_steps": target,
            "normalized_contrast_mse": intervention["normalized_contrast_mse"],
            "context_flip_recall": intervention["context_flip_recall"],
            "state_intervention_delta": intervention["state_intervention_delta"],
        })
        rows.append(row)

    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument(
        "--steps-sweep",
        default=None,
        help="Independent models at each duration (legacy optimization-path sweep).",
    )
    p.add_argument(
        "--checkpoint-trajectory",
        default=None,
        help="Continuous per-seed trajectory, e.g. 500,1000,2000,4000.",
    )
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--transition-diagnostic",
        default=None,
        help="Continuous parameter-group diagnostic as start,end, e.g. 500,1000.",
    )
    p.add_argument("--transition-interval", type=int, default=50)
    args = p.parse_args()

    cfg = ExperimentConfig(
        train_steps=args.steps,
        batch_size=args.batch,
        eval_batch=args.eval_batch,
    )
    device = torch.device(args.device)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]

    if args.transition_diagnostic:
        parts = [int(x) for x in args.transition_diagnostic.split(",") if x.strip()]
        if len(parts) != 2:
            raise ValueError("--transition-diagnostic requires start,end")
        results = []
        for seed in seeds:
            results.extend(_train_transition_diagnostic(
                seed, cfg, parts[0], parts[1], args.transition_interval, device
            ))
        print(json.dumps({
            "experiment": "TAC-OSM-v0.5-parameter-transition-diagnostic",
            "variable": "parameter_group_gradients_and_drift",
            "transition": parts,
            "interval": args.transition_interval,
            "seeds": seeds,
            "architecture_unchanged": True,
            "objective_unchanged": True,
            "evaluation_unchanged": True,
            "promotion_gate": "unchanged",
            "results": results,
        }, indent=2))
        return

    if args.checkpoint_trajectory:
        checkpoints = [
            int(x) for x in args.checkpoint_trajectory.split(",") if x.strip()
        ]
        results = []
        for seed in seeds:
            results.extend(
                _train_continuous_checkpoints(
                    seed, cfg, checkpoints, device
                )
            )
        print(json.dumps({
            "experiment": "TAC-OSM-v0.5-continuous-checkpoint-trajectory",
            "variable": "single_optimizer_trajectory",
            "checkpoints": checkpoints,
            "seeds": seeds,
            "architecture_unchanged": True,
            "objective_unchanged": True,
            "promotion_gate": "unchanged",
            "results": results,
        }, indent=2))
        return

    if args.steps_sweep:
        durations = [
            int(x) for x in args.steps_sweep.split(",") if x.strip()
        ]
        sweep = []
        for steps in durations:
            sweep_cfg = ExperimentConfig(
                train_steps=steps,
                batch_size=args.batch,
                eval_batch=args.eval_batch,
            )
            for seed in seeds:
                row = diagnose_seed(seed, sweep_cfg, device)
                row["train_steps"] = steps
                sweep.append(row)
        print(json.dumps({
            "experiment": "TAC-OSM-v0.5-optimization-path-sweep",
            "variable": "observational_training_duration",
            "steps": durations,
            "seeds": seeds,
            "architecture_unchanged": True,
            "promotion_gate": "unchanged",
            "results": sweep,
        }, indent=2))
        return

    results = [diagnose_seed(seed, cfg, device) for seed in seeds]
    print(json.dumps({
        "experiment": "TAC-OSM-v0.5-optimization-diagnostics",
        "promotion_gate": "unchanged",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
