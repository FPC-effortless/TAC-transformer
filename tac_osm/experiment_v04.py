"""Falsification-grade TAC-OSM v0.4 controlled benchmark.

The benchmark tests two factors:
  P: temporal persistent state
  A: action-conditioned transition prediction

The primary decision path is:
  t0 context -> persistent state -> t1 candidate forecasts -> fixed objective -> argmax

There is no direct supervised optimal-action head in the primary training objective.
All four cells use the same parameterized architecture; factor-off cells mask inputs.
"""
import argparse
import json
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .environment import SyntheticOperationalWorld, WorldConfig


@dataclass(frozen=True)
class ExperimentConfig:
    input_dim: int = 8
    hidden_dim: int = 32
    structure_dim: int = 16
    state_slots: int = 8
    action_dim: int = 4
    context_dim: int = 4
    train_steps: int = 300
    batch_size: int = 128
    eval_batch: int = 1024
    lr: float = 3e-3
    held_out_regime: int = 1


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ControlledOperationalModel(nn.Module):
    """Capacity-matched model with explicit factor masks."""

    def __init__(self, cfg: ExperimentConfig, persistent: bool, action_conditioned: bool):
        super().__init__()
        self.persistent = persistent
        self.action_conditioned = action_conditioned
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim + cfg.context_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.write = nn.Linear(cfg.hidden_dim, cfg.state_slots)
        self.slot_value = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.structure_dim),
            nn.LayerNorm(cfg.structure_dim),
        )
        self.route_query = nn.Linear(cfg.hidden_dim, cfg.structure_dim)
        self.executor = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.structure_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.transition = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.structure_dim + cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.state_head = nn.Linear(cfg.hidden_dim, cfg.input_dim)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_state(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.cfg.state_slots, self.cfg.structure_dim, device=device),
            torch.zeros(batch_size, self.cfg.state_slots, device=device),
        )

    def encode(self, obs, state, write=True):
        hidden = self.encoder(obs)
        if not self.persistent:
            return hidden, self.initial_state(obs.size(0), obs.device)

        slots, validity = state
        if write:
            gate = torch.sigmoid(self.write(hidden))
            value = self.slot_value(hidden).unsqueeze(1)
            slots = slots + gate.unsqueeze(-1) * (value - slots)
            validity = torch.maximum(validity, gate).clamp(0, 1)

        q = self.route_query(hidden).unsqueeze(1)
        scores = (q * slots).sum(-1) / (self.cfg.structure_dim ** 0.5)
        scores = scores.masked_fill(validity <= 1e-6, -1e9)
        weights = torch.softmax(scores, -1)
        structure = (slots * weights.unsqueeze(-1)).sum(1)
        hidden = hidden + self.executor(torch.cat([hidden, structure], -1))
        return hidden, (slots, validity)

    def candidate_forecast(self, hidden, state, candidate_actions):
        slots, validity = state
        q = self.route_query(hidden).unsqueeze(1)
        scores = (q * slots).sum(-1) / (self.cfg.structure_dim ** 0.5)
        weights = torch.softmax(scores.masked_fill(validity <= 1e-6, -1e9), -1)
        structure = (slots * weights.unsqueeze(-1)).sum(1)

        b, n, h, a = candidate_actions.shape
        fa = candidate_actions.reshape(b * n, h, a)
        latent = hidden.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)
        fs = structure.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)

        predictions = []
        for t in range(h):
            action = fa[:, t]
            if not self.action_conditioned:
                action = torch.zeros_like(action)
            action_repr = self.action_encoder(action)
            latent = self.transition(torch.cat([latent, fs, action_repr], -1))
            predictions.append(self.state_head(latent))
        return torch.stack(predictions, 1).reshape(b, n, h, self.cfg.input_dim)

    def forward_sequence(self, t0_obs, t1_obs, candidate_actions, mode="carry"):
        state = self.initial_state(t0_obs.size(0), t0_obs.device)
        if mode == "carry" and self.persistent:
            _, state = self.encode(t0_obs, state, write=True)
        elif mode == "shuffle":
            _, state = self.encode(t0_obs.flip(0), state, write=True)
        elif mode == "corrupt":
            _, state = self.encode(t0_obs, state, write=True)
            state = (state[0] + torch.randn_like(state[0]) * 0.5, state[1])
        elif mode == "reset":
            state = self.initial_state(t0_obs.size(0), t0_obs.device)

        hidden, state = self.encode(t1_obs, state, write=False)
        return self.candidate_forecast(hidden, state, candidate_actions)


def make_world(cfg, regime):
    return SyntheticOperationalWorld(WorldConfig(cfg.input_dim, cfg.action_dim, 0.0, regime=regime))


def sample_batch(world, batch_size, generator, device):
    state0 = torch.randn(batch_size, world.cfg.state_dim, generator=generator, device=device)
    context_index = torch.randint(
        0, world.cfg.context_dim, (batch_size,), generator=generator, device=device
    )
    context = torch.nn.functional.one_hot(
        context_index, num_classes=world.cfg.context_dim
    ).float()
    query_state = torch.randn(batch_size, world.cfg.state_dim, generator=generator, device=device)
    t0 = torch.cat([state0, context], -1)
    t1 = torch.cat([query_state, torch.zeros_like(context)], -1)
    action = torch.randint(0, world.cfg.action_dim, (batch_size,), generator=generator, device=device)
    next_state = world.transition(query_state, action, context=context, generator=generator)
    target_action, _ = world.optimal_action(query_state, context=context)
    return t0, t1, context, action, next_state, target_action


def action_candidates(batch, action_dim, device):
    one_hot = torch.eye(action_dim, device=device)
    return one_hot.view(1, action_dim, 1, action_dim).expand(batch, -1, -1, -1)


def train_one(seed, persistent, action_conditioned, cfg, device):
    seed_all(seed)
    world = make_world(cfg, regime=0)
    model = ControlledOperationalModel(cfg, persistent, action_conditioned).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)

    start = time.perf_counter()
    for step in range(1, cfg.train_steps + 1):
        t0, t1, context, action, next_state, _ = sample_batch(
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

    elapsed = time.perf_counter() - start
    metrics = evaluate(model, cfg, seed + 5000, device)
    metrics.update({
        "seed": seed,
        "persistent": persistent,
        "action_conditioned": action_conditioned,
        "parameters": model.parameter_count(),
        "train_steps": cfg.train_steps,
        "wall_seconds": elapsed,
    })
    return metrics


@torch.no_grad()
def evaluate(model, cfg, seed, device):
    train_world = make_world(cfg, regime=0)
    held_world = make_world(cfg, regime=cfg.held_out_regime)
    gen = torch.Generator(device=device).manual_seed(seed)

    t0, t1, context, _, next_state, _ = sample_batch(
        held_world, cfg.eval_batch, gen, device
    )
    candidates = action_candidates(cfg.eval_batch, cfg.action_dim, device)
    pred = model.forward_sequence(t0, t1, candidates)
    true = torch.stack([
        held_world.transition(
            torch.cat([t1[:, :cfg.input_dim]], -1),
            torch.full((cfg.eval_batch,), a, device=device, dtype=torch.long),
            context=context,
            generator=None,
        )
        for a in range(cfg.action_dim)
    ], 1).unsqueeze(2)

    forecast_mse = torch.mean((pred - true) ** 2).item()
    predicted_scores = held_world.objective(
        pred[:, :, -1].reshape(-1, cfg.input_dim),
        context=context.repeat_interleave(cfg.action_dim, 0),
    ).view(cfg.eval_batch, cfg.action_dim)
    selected = predicted_scores.argmax(-1)

    oracle_action, oracle_scores = held_world.optimal_action(
        t1[:, :cfg.input_dim], context=context
    )
    achieved = held_world.transition(
        t1[:, :cfg.input_dim], selected, context=context, generator=None
    )
    achieved_objective = held_world.objective(achieved, context=context).mean().item()
    oracle_objective = oracle_scores.max(-1).values.mean().item()

    # Intervention probes use the same query/context and change only one factor.
    same_state = pred[0:1, :, 0]
    action_delta = torch.max(torch.abs(same_state[0] - same_state[0, 0])).item()
    reset_pred = model.forward_sequence(t0, t1, candidates, mode="reset")
    shuffle_pred = model.forward_sequence(t0, t1, candidates, mode="shuffle")
    corrupt_pred = model.forward_sequence(t0, t1, candidates, mode="corrupt")
    state_reset_delta = torch.mean(torch.abs(pred - reset_pred)).item()
    state_shuffle_delta = torch.mean(torch.abs(pred - shuffle_pred)).item()
    state_corrupt_delta = torch.mean(torch.abs(pred - corrupt_pred)).item()

    return {
        "forecast_mse_held_out_all_actions": forecast_mse,
        "planning_accuracy_held_out": (selected == oracle_action).float().mean().item(),
        "achieved_objective": achieved_objective,
        "oracle_objective": oracle_objective,
        "objective_regret": oracle_objective - achieved_objective,
        "action_forecast_delta": action_delta,
        "state_reset_delta": state_reset_delta,
        "state_shuffle_delta": state_shuffle_delta,
        "state_corrupt_delta": state_corrupt_delta,
        "oracle_label_source": "evaluation_only",
        "train_world_regime": 0,
        "eval_world_regime": cfg.held_out_regime,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = ExperimentConfig(train_steps=args.steps, batch_size=args.batch)
    results = []
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        for persistent in (False, True):
            for action_conditioned in (False, True):
                results.append(train_one(seed, persistent, action_conditioned, cfg, device))
    print(json.dumps({
        "experiment": "TAC-OSM-v0.4-falsification",
        "factors": ["temporal_persistent_state", "action_conditioned_transition"],
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
