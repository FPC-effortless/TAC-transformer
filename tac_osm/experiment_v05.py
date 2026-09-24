"""TAC-OSM v0.5 causal intervention benchmark."""
from __future__ import annotations
import argparse, json, random
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .environment import SyntheticOperationalWorld, WorldConfig
from .experiment_v04 import ControlledOperationalModel, ExperimentConfig as V04Config, action_candidates, sample_batch

@dataclass(frozen=True)
class ExperimentConfig(V04Config):
    train_steps: int = 1000
    batch_size: int = 128
    eval_batch: int = 1024
    intervention_context_a: int = 0
    intervention_context_b: int = 1

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def make_world(cfg: ExperimentConfig, regime: int):
    return SyntheticOperationalWorld(WorldConfig(cfg.input_dim, cfg.action_dim, 0.0, cfg.context_dim, regime))

def _context_pair(cfg, batch, device):
    a = torch.nn.functional.one_hot(torch.full((batch,), cfg.intervention_context_a, device=device, dtype=torch.long), num_classes=cfg.context_dim).float()
    b = torch.nn.functional.one_hot(torch.full((batch,), cfg.intervention_context_b, device=device, dtype=torch.long), num_classes=cfg.context_dim).float()
    return a, b

def train_one(seed: int, cfg: ExperimentConfig, device: torch.device):
    seed_all(seed)
    world = make_world(cfg, 0)
    model = ControlledOperationalModel(cfg, True, True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)
    for _ in range(cfg.train_steps):
        t0, t1, _, action, next_state, _ = sample_batch(world, cfg.batch_size, gen, device)
        candidates = torch.nn.functional.one_hot(action, cfg.action_dim).float().view(cfg.batch_size, 1, 1, cfg.action_dim)
        pred = model.forward_sequence(t0, t1, candidates)[:, 0, 0]
        loss = torch.mean((pred - next_state) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def _forecast_with_context(model, t1, context, candidate_actions):
    context_dim = context.size(1)
    state_dim = t1.size(1) - context_dim
    t0 = torch.cat([torch.zeros(t1.size(0), state_dim, device=t1.device), context], -1)
    t1_query = torch.cat([t1[:, :state_dim], torch.zeros_like(context)], -1)
    state = model.initial_state(t1.size(0), t1.device)
    _, state = model.encode(t0, state, write=True)
    hidden, state = model.encode(t1_query, state, write=False)
    return model.candidate_forecast(hidden, state, candidate_actions)

@torch.no_grad()
def evaluate(model, cfg, seed, device):
    world = make_world(cfg, cfg.held_out_regime)
    gen = torch.Generator(device=device).manual_seed(seed)
    query_state = torch.randn(cfg.eval_batch, cfg.input_dim, generator=gen, device=device)
    ca, cb = _context_pair(cfg, cfg.eval_batch, device)
    t1 = torch.cat([query_state, torch.zeros_like(ca)], -1)
    candidates = action_candidates(cfg.eval_batch, cfg.action_dim, device)
    pred_a = _forecast_with_context(model, t1, ca, candidates)
    pred_b = _forecast_with_context(model, t1, cb, candidates)
    predicted_delta = pred_b[:, :, 0] - pred_a[:, :, 0]
    true_a = []; true_b = []
    for a in range(cfg.action_dim):
        actions = torch.full((cfg.eval_batch,), a, device=device, dtype=torch.long)
        true_a.append(world.transition(query_state, actions, context=ca))
        true_b.append(world.transition(query_state, actions, context=cb))
    true_a = torch.stack(true_a, 1); true_b = torch.stack(true_b, 1)
    true_delta = true_b - true_a
    contrast_mse = torch.mean((predicted_delta - true_delta) ** 2).item()
    scale = torch.mean(true_delta ** 2).item()
    norm_mse = contrast_mse / max(scale, 1e-12)
    score_a = world.objective(pred_a[:, :, -1].reshape(-1, cfg.input_dim), context=ca.repeat_interleave(cfg.action_dim, 0)).view(cfg.eval_batch, cfg.action_dim)
    score_b = world.objective(pred_b[:, :, -1].reshape(-1, cfg.input_dim), context=cb.repeat_interleave(cfg.action_dim, 0)).view(cfg.eval_batch, cfg.action_dim)
    selected_a = score_a.argmax(-1); selected_b = score_b.argmax(-1)
    oracle_a, _ = world.optimal_action(query_state, context=ca)
    oracle_b, _ = world.optimal_action(query_state, context=cb)
    oracle_flip = oracle_a != oracle_b; model_flip = selected_a != selected_b
    flip_cases = int(oracle_flip.sum().item())
    flip_recall = float((model_flip & oracle_flip).sum().item() / flip_cases) if flip_cases else 1.0
    return {
        "held_out_regime": cfg.held_out_regime,
        "contrast_mse": contrast_mse,
        "true_contrast_mse_scale": scale,
        "normalized_contrast_mse": norm_mse,
        "null_normalized_contrast_mse": 1.0,
        "state_intervention_delta": torch.mean(torch.abs(pred_b - pred_a)).item(),
        "oracle_context_flip_cases": flip_cases,
        "context_flip_recall": flip_recall,
        "oracle_action_disagreement": float((oracle_a != oracle_b).float().mean().item()),
    }

def run(seed, cfg, device):
    model = train_one(seed, cfg, device)
    metrics = evaluate(model, cfg, seed + 5000, device)
    metrics.update({"seed": seed, "parameters": model.parameter_count()})
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4"); p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=128); p.add_argument("--eval-batch", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    cfg = ExperimentConfig(train_steps=args.steps, batch_size=args.batch, eval_batch=args.eval_batch)
    device = torch.device(args.device)
    results = [run(int(s), cfg, device) for s in args.seeds.split(",") if s.strip()]
    print(json.dumps({"experiment":"TAC-OSM-v0.5-causal-intervention","training":"observational","evaluation":"held-out explicit context intervention","results":results}, indent=2))

if __name__ == "__main__":
    main()
