"""TAC-OSM v0.5 anti-lookup compositional generalization benchmark.

Diagnostic extension of v0.5. The architecture and observational next-state
objective are unchanged. Training withholds a checkerboard of context/action
pairs while every individual context and action remains observed. Evaluation
measures intervention contrasts on those unseen combinations.

The operational rule is compositional: mapped_action = context XOR action
(using two-bit codes). A successful held-out-pair result is evidence of
compositional generalization, not by itself proof of causality.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from .environment import SyntheticOperationalWorld, WorldConfig
from .experiment_v04 import ControlledOperationalModel, ExperimentConfig as V04Config


TRAIN_STEPS = 1000


class AntiLookupWorld(SyntheticOperationalWorld):
    def transition(self, state, action_index, context=None, generator=None):
        if context is None:
            context_index = torch.zeros(
                state.size(0), device=state.device, dtype=torch.long
            )
        else:
            context_index = context.argmax(-1).clamp_max(self.cfg.action_dim - 1)
        mapped_action = torch.bitwise_xor(context_index, action_index)
        action = F.one_hot(mapped_action, num_classes=self.cfg.action_dim).float()
        effect = action @ self.effect
        noise = torch.randn(
            state.shape, device=state.device, generator=generator
        ) * self.cfg.noise_std
        return state @ self.dynamics.T + effect + self.bias + noise


def make_world(cfg, regime):
    return AntiLookupWorld(
        WorldConfig(cfg.input_dim, cfg.action_dim, 0.0, cfg.context_dim, regime)
    )


def held_out_pair(context_index, action_index):
    return ((context_index + action_index) % 2) == 0


def sample_batch(world, batch_size, generator, device):
    state = torch.randn(
        batch_size, world.cfg.state_dim, generator=generator, device=device
    )
    context_index = torch.randint(
        0, world.cfg.context_dim, (batch_size,),
        generator=generator, device=device
    )
    actions = []
    for c in context_index.tolist():
        allowed = [a for a in range(world.cfg.action_dim)
                   if not held_out_pair(c, a)]
        j = torch.randint(0, len(allowed), (1,),
                          generator=generator, device=device).item()
        actions.append(allowed[j])
    action = torch.tensor(actions, device=device, dtype=torch.long)
    context = F.one_hot(context_index, num_classes=world.cfg.context_dim).float()
    target = world.transition(state, action, context=context, generator=generator)
    t0 = torch.cat([torch.zeros_like(state), context], -1)
    t1 = torch.cat([state, torch.zeros_like(context)], -1)
    return t0, t1, action, target


@torch.no_grad()
def forecast_with_context(model, query_state, context, candidate_actions):
    t1 = torch.cat([
        query_state,
        torch.zeros(query_state.size(0), context.size(1), device=query_state.device),
    ], -1)
    t0 = torch.cat([torch.zeros_like(query_state), context], -1)
    state = model.initial_state(query_state.size(0), query_state.device)
    _, state = model.encode(t0, state, write=True)
    hidden, state = model.encode(t1, state, write=False)
    return model.candidate_forecast(hidden, state, candidate_actions)


def train_one(seed, cfg, device):
    torch.manual_seed(seed)
    world = make_world(cfg, 0)
    model = ControlledOperationalModel(cfg, True, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)
    last_loss = None
    for _ in range(cfg.train_steps):
        t0, t1, action, target = sample_batch(world, cfg.batch_size, gen, device)
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


@torch.no_grad()
def evaluate(model, cfg, seed, device):
    world = make_world(cfg, cfg.held_out_regime)
    gen = torch.Generator(device=device).manual_seed(seed)
    query_state = torch.randn(
        cfg.eval_batch, cfg.input_dim, generator=gen, device=device
    )
    candidates = F.one_hot(
        torch.arange(cfg.action_dim, device=device), cfg.action_dim
    ).float().view(1, cfg.action_dim, 1, cfg.action_dim).expand(
        cfg.eval_batch, -1, -1, -1
    )

    heldout_errors = []
    for c in range(cfg.context_dim):
        context = F.one_hot(
            torch.full((cfg.eval_batch,), c, device=device, dtype=torch.long),
            num_classes=cfg.context_dim,
        ).float()
        pred = forecast_with_context(model, query_state, context, candidates)
        held_actions = [a for a in range(cfg.action_dim)
                        if held_out_pair(c, a)]
        true = torch.stack([
            world.transition(
                query_state,
                torch.full((cfg.eval_batch,), a, device=device, dtype=torch.long),
                context=context,
            )
            for a in held_actions
        ], 1)
        heldout_errors.append(
            torch.mean((pred[:, held_actions, 0, :] - true) ** 2)
        )
    heldout_forecast_mse = torch.stack(heldout_errors).mean().item()

    ca = F.one_hot(
        torch.zeros(cfg.eval_batch, dtype=torch.long, device=device),
        num_classes=cfg.context_dim,
    ).float()
    cb = F.one_hot(
        torch.full((cfg.eval_batch,), 2, dtype=torch.long, device=device),
        num_classes=cfg.context_dim,
    ).float()
    pa = forecast_with_context(model, query_state, ca, candidates)[:, :, 0, :]
    pb = forecast_with_context(model, query_state, cb, candidates)[:, :, 0, :]
    common_held = [
        a for a in range(cfg.action_dim)
        if held_out_pair(0, a) and held_out_pair(2, a)
    ]
    common_seen = [
        a for a in range(cfg.action_dim)
        if not held_out_pair(0, a) and not held_out_pair(2, a)
    ]
    pred_delta = pb[:, common_held] - pa[:, common_held]
    true_a = torch.stack([
        world.transition(
            query_state,
            torch.full((cfg.eval_batch,), a, device=device, dtype=torch.long),
            context=ca,
        ) for a in common_held
    ], 1)
    true_b = torch.stack([
        world.transition(
            query_state,
            torch.full((cfg.eval_batch,), a, device=device, dtype=torch.long),
            context=cb,
        ) for a in common_held
    ], 1)
    true_delta = true_b - true_a
    contrast_mse = torch.mean((pred_delta - true_delta) ** 2).item()
    scale = torch.mean(true_delta ** 2).item()

    def contrast_for_actions(action_ids):
        pred = pb[:, action_ids] - pa[:, action_ids]
        ta = torch.stack([
            world.transition(query_state, torch.full((cfg.eval_batch,), a, device=device, dtype=torch.long), context=ca)
            for a in action_ids
        ], 1)
        tb = torch.stack([
            world.transition(query_state, torch.full((cfg.eval_batch,), a, device=device, dtype=torch.long), context=cb)
            for a in action_ids
        ], 1)
        td = tb - ta
        mse = torch.mean((pred - td) ** 2).item()
        sc = torch.mean(td ** 2).item()
        return mse, mse / max(sc, 1e-12)

    seen_contrast_mse, seen_normalized = contrast_for_actions(common_seen)

    return {
        "heldout_pair_forecast_mse": heldout_forecast_mse,
        "heldout_intervention_contrast_mse": contrast_mse,
        "heldout_normalized_contrast_mse": contrast_mse / max(scale, 1e-12),
        "seen_intervention_contrast_mse": seen_contrast_mse,
        "seen_normalized_contrast_mse": seen_normalized,
        "heldout_to_seen_normalized_ratio": (contrast_mse / max(scale, 1e-12)) / max(seen_normalized, 1e-12),
        "null_normalized_contrast_mse": 1.0,
        "common_heldout_actions": common_held,
        "common_seen_actions": common_seen,
        "train_pair_count": 8,
        "heldout_pair_count": 8,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=TRAIN_STEPS)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    cfg = V04Config(train_steps=args.steps, batch_size=args.batch,
                    eval_batch=args.eval_batch)
    device = torch.device(args.device)
    results = []
    for raw in args.seeds.split(","):
        if raw.strip():
            seed = int(raw)
            model, train_loss = train_one(seed, cfg, device)
            metrics = evaluate(model, cfg, seed + 5000, device)
            metrics.update({"seed": seed, "train_loss": train_loss})
            results.append(metrics)
    print(json.dumps({
        "experiment": "TAC-OSM-v0.5-anti-lookup",
        "training": "observational_next_state_only",
        "split": "checkerboard_context_action_pairs",
        "rule": "two_bit_context_xor_action",
        "architecture_unchanged": True,
        "objective_unchanged": True,
        "intervention_supervision": False,
        "oracle_action_labels_used_in_training": False,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
