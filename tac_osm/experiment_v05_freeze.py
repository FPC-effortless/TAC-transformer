"""TAC-OSM v0.5 550->600 parameter-freeze ablation.

Diagnostic-only experiment. It preserves the v0.5 architecture,
observational objective, held-out intervention evaluation, and promotion
gate. A model is trained continuously to step 550, then cloned into a
control and one clone per functional parameter group. Each clone receives
the identical next 50 training batches; one selected group is frozen in the
ablated clone. This tests whether plasticity in that group is necessary for
the observed transition.
"""
from __future__ import annotations

import argparse
import copy
import json

import torch

from .experiment_v05 import ExperimentConfig, evaluate, make_world, seed_all
from .experiment_v05_diagnostic import diagnose_model
from .experiment_v04 import ControlledOperationalModel, sample_batch


GROUPS = (
    "encoder",
    "write",
    "slot_value",
    "route_query",
    "executor",
    "action_encoder",
    "transition",
    "state_head",
)


def _train_steps(model, opt, world, gen, steps, cfg, device):
    last_loss = None
    for _ in range(steps):
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
        last_loss = float(loss.detach().item())
    return last_loss


def _freeze_group(model, group):
    matched = 0
    for name, param in model.named_parameters():
        if name.split(".", 1)[0] == group:
            param.requires_grad_(False)
            matched += 1
    if matched == 0:
        raise ValueError(f"unknown parameter group: {group}")
    return matched


def _measure(model, seed, cfg, device, condition, frozen_group=None, train_loss=None):
    path = diagnose_model(model, seed, cfg, device)
    intervention = evaluate(model, cfg, seed + 5000, device)
    return {
        "seed": seed,
        "condition": condition,
        "frozen_group": frozen_group,
        "train_steps": 600,
        "train_loss": train_loss,
        "routed_structure_delta": path["routed_structure_delta"],
        "persistent_state_delta": path["persistent_state_delta"],
        "forecast_delta": path["forecast_delta"],
        "normalized_contrast_mse": intervention["normalized_contrast_mse"],
        "context_flip_recall": intervention["context_flip_recall"],
        "state_intervention_delta": intervention["state_intervention_delta"],
    }


def run_seed(seed, cfg, device, warmup_steps=550, transition_steps=50):
    seed_all(seed)
    world = make_world(cfg, 0)

    base = ControlledOperationalModel(cfg, True, True).to(device)
    base_opt = torch.optim.AdamW(base.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)

    warmup_loss = _train_steps(
        base, base_opt, world, gen, warmup_steps, cfg, device
    )

    # Freeze the exact 550-step model and optimizer state. Every branch then
    # consumes the identical 50 batches from the same generator position.
    conditions = ["control", *GROUPS]
    rows = []
    for condition in conditions:
        model = copy.deepcopy(base)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        opt.load_state_dict(copy.deepcopy(base_opt.state_dict()))

        frozen = None if condition == "control" else condition
        if frozen is not None:
            _freeze_group(model, frozen)

        # Each branch must use an independent generator at the same state.
        branch_gen = torch.Generator(device=device)
        branch_gen.set_state(gen.get_state())
        loss = _train_steps(
            model, opt, world, branch_gen, transition_steps, cfg, device
        )
        rows.append(
            _measure(
                model,
                seed,
                cfg,
                device,
                condition,
                frozen,
                loss,
            )
        )

    # Also retain the exact step-550 reference so the observed transition
    # itself is explicit in the output.
    rows.insert(
        0,
        _measure(
            base,
            seed,
            cfg,
            device,
            "checkpoint_550",
            None,
            warmup_loss,
        )
        | {"train_steps": 550},
    )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = ExperimentConfig(
        train_steps=600,
        batch_size=args.batch,
        eval_batch=args.eval_batch,
    )
    device = torch.device(args.device)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    results = []
    for seed in seeds:
        results.extend(run_seed(seed, cfg, device))

    print(json.dumps({
        "experiment": "TAC-OSM-v0.5-550-to-600-freeze-ablation",
        "transition": [550, 600],
        "seeds": seeds,
        "conditions": ["control", *GROUPS],
        "architecture_unchanged": True,
        "objective_unchanged": True,
        "evaluation_unchanged": True,
        "promotion_gate": "unchanged",
        "identical_transition_batches": True,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
