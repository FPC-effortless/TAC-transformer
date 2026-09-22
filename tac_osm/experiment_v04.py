"""Controlled TAC-OSM v0.4 experiment.

This benchmark isolates two factors:
  P: persistent structured state
  A: action-conditioned transition prediction

It uses supervised transition targets from the explicit synthetic causal oracle.
No RL, no oracle scores are fed to the model, and train/validation/test states
are generated from independent deterministic seeds.
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
    train_steps: int = 300
    batch_size: int = 128
    eval_batch: int = 1024
    lr: float = 3e-3
    state_write_scale: float = 1.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ControlledOperationalModel(nn.Module):
    """Minimal matched substrate for the 2x2 factor experiment."""

    def __init__(self, cfg: ExperimentConfig, persistent: bool, action_conditioned: bool):
        super().__init__()
        self.persistent = persistent
        self.action_conditioned = action_conditioned
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.hidden_dim),
        )
        if persistent:
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

        action_in = cfg.action_dim if action_conditioned else 0
        self.action_encoder = (
            nn.Sequential(
                nn.Linear(cfg.action_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            if action_conditioned else None
        )
        context_dim = cfg.hidden_dim + (cfg.structure_dim if persistent else 0) + action_in
        self.transition = nn.Sequential(
            nn.Linear(context_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.state_head = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        self.action_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + (cfg.structure_dim if persistent else 0), cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

    def initial_state(self, batch_size, device):
        if not self.persistent:
            return None
        return (
            torch.zeros(batch_size, self.cfg.state_slots, self.cfg.structure_dim, device=device),
            torch.zeros(batch_size, self.cfg.state_slots, device=device),
        )

    def encode(self, obs, state):
        hidden = self.encoder(obs)
        if not self.persistent:
            return hidden, None

        slots, validity = state
        gate = torch.sigmoid(self.write(hidden)) * self.cfg.state_write_scale
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

    def forward(self, obs, action_indices, state=None):
        hidden, new_state = self.encode(obs, state)
        structure = None
        if self.persistent:
            slots, validity = new_state
            q = self.route_query(hidden).unsqueeze(1)
            scores = (q * slots).sum(-1) / (self.cfg.structure_dim ** 0.5)
            weights = torch.softmax(scores.masked_fill(validity <= 1e-6, -1e9), -1)
            structure = (slots * weights.unsqueeze(-1)).sum(1)

        action_repr = None
        if self.action_conditioned:
            one_hot = torch.nn.functional.one_hot(
                action_indices, self.cfg.action_dim
            ).float()
            action_repr = self.action_encoder(one_hot)

        parts = [hidden]
        if structure is not None:
            parts.append(structure)
        if action_repr is not None:
            parts.append(action_repr)
        latent = self.transition(torch.cat(parts, -1))
        prediction = self.state_head(latent)
        decision_parts = [hidden]
        if structure is not None:
            decision_parts.append(structure)
        action_logits = self.action_head(torch.cat(decision_parts, -1))
        return prediction, action_logits, new_state


def sample_batch(world, batch_size, generator, device):
    state = torch.randn(batch_size, world.cfg.state_dim, generator=generator, device=device)
    action = torch.randint(
        0, world.cfg.action_dim, (batch_size,), generator=generator, device=device
    )
    next_state = world.transition(state, action, generator=generator)
    target_action, _ = world.optimal_action(state, generator=generator)
    return state, action, next_state, target_action


def train_one(seed, persistent, action_conditioned, cfg, device):
    seed_all(seed)
    world = SyntheticOperationalWorld(WorldConfig(cfg.input_dim, cfg.action_dim, 0.02))
    model = ControlledOperationalModel(cfg, persistent, action_conditioned).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)

    start = time.perf_counter()
    first_half = None
    final_loss = None
    for step in range(1, cfg.train_steps + 1):
        state, action, next_state, target_action = sample_batch(
            world, cfg.batch_size, gen, device
        )
        pred, action_logits, _ = model(state, action, model.initial_state(cfg.batch_size, device))
        forecast_loss = torch.mean((pred - next_state) ** 2)
        decision_loss = torch.nn.functional.cross_entropy(action_logits, target_action)
        loss = forecast_loss + decision_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.detach())
        if step == cfg.train_steps // 2:
            first_half = final_loss

    elapsed = time.perf_counter() - start
    metrics = evaluate(model, world, cfg, seed + 5000, device)
    metrics.update({
        "seed": seed,
        "persistent": persistent,
        "action_conditioned": action_conditioned,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_steps": cfg.train_steps,
        "wall_seconds": elapsed,
        "final_train_loss": final_loss,
        "mid_train_loss": first_half,
    })
    return metrics


@torch.no_grad()
def evaluate(model, world, cfg, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    state = torch.randn(cfg.eval_batch, cfg.input_dim, generator=gen, device=device)
    target_action, oracle_scores = world.optimal_action(state, generator=gen)

    # Forecast quality is measured on every intervention, not only the selected action.
    all_actions = torch.arange(cfg.action_dim, device=device).repeat(cfg.eval_batch)
    repeated_state = state.repeat_interleave(cfg.action_dim, dim=0)
    init = model.initial_state(repeated_state.size(0), device)
    pred, _, _ = model(repeated_state, all_actions, init)
    true = world.transition(repeated_state, all_actions, generator=gen)
    forecast_mse = torch.mean((pred - true) ** 2).item()

    # Decision is evaluated independently from transition training samples.
    init2 = model.initial_state(cfg.eval_batch, device)
    _, direct_logits, _ = model(state, target_action, init2)
    direct_pred = direct_logits.argmax(-1)
    decision_accuracy = (direct_pred == target_action).float().mean().item()

    # For action-conditioned models, score each candidate by predicted objective.
    candidate_pred = pred.view(cfg.eval_batch, cfg.action_dim, -1)
    predicted_scores = -(
        candidate_pred[:, :, 0].abs()
        + 0.7 * candidate_pred[:, :, 1].abs()
        + 0.4 * candidate_pred[:, :, 2].abs()
        + 0.2 * candidate_pred[:, :, 3].abs()
    )
    selected = predicted_scores.argmax(-1) if model.action_conditioned else direct_pred
    achieved = world.transition(state, selected, generator=gen)
    achieved_objective = world.objective(achieved).mean().item()
    oracle_objective = oracle_scores.max(-1).values.mean().item()
    regret = oracle_objective - achieved_objective

    return {
        "forecast_mse_all_actions": forecast_mse,
        "decision_accuracy": decision_accuracy,
        "achieved_objective": achieved_objective,
        "oracle_objective": oracle_objective,
        "objective_regret": regret,
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
                results.append(
                    train_one(seed, persistent, action_conditioned, cfg, device)
                )
    print(json.dumps({
        "experiment": "TAC-OSM-v0.4-2x2",
        "factors": ["persistent_state", "action_conditioned_transition"],
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
