from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor, nn
from .config import TACOSMConfig

@dataclass
class PersistentStructureState:
    slots: Tensor
    validity: Tensor
    def detach(self):
        return PersistentStructureState(self.slots.detach(), self.validity.detach())
    def clone(self):
        return PersistentStructureState(self.slots.clone(), self.validity.clone())

@dataclass
class TACOSMOutput:
    decision_logits: Tensor
    verifier_logit: Tensor
    route_weights: Tensor
    selected_indices: Tensor
    active_compute: Tensor
    state: PersistentStructureState
    action_logits: Optional[Tensor] = None
    forecast: Optional[Tensor] = None
    action_value: Optional[Tensor] = None

class StructuralCompressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.structure_dim), nn.LayerNorm(cfg.structure_dim))
        self.write_gate = nn.Linear(cfg.hidden_dim, cfg.state_slots)
    def forward(self, hidden, state):
        gate = torch.sigmoid(self.write_gate(hidden))
        value = self.value(hidden).unsqueeze(1)
        slots = state.slots + gate.unsqueeze(-1) * (value - state.slots)
        validity = torch.maximum(state.validity, gate).clamp(0.0, 1.0)
        return PersistentStructureState(slots, validity)

class StructureRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.query = nn.Linear(cfg.hidden_dim, cfg.structure_dim)
        self.scale = cfg.structure_dim ** -0.5
        self.top_k = cfg.top_k
    def forward(self, hidden, state):
        q = self.query(hidden).unsqueeze(1)
        scores = (q * state.slots).sum(-1) * self.scale
        scores = scores.masked_fill(state.validity <= 1e-6, -1e9)
        vals, indices = torch.topk(scores, k=min(self.top_k, state.slots.size(1)), dim=-1)
        weights = torch.softmax(vals, dim=-1)
        gathered = state.slots.gather(1, indices.unsqueeze(-1).expand(-1, -1, state.slots.size(-1)))
        route = (gathered * weights.unsqueeze(-1)).sum(1)
        return route, weights, indices

class ConditionalExecutor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        joined_dim = cfg.hidden_dim + cfg.structure_dim
        self.modules_bank = nn.ModuleList([
            nn.Sequential(nn.Linear(joined_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            for _ in range(cfg.num_compute_modules)
        ])
        self.gate = nn.Linear(joined_dim, cfg.num_compute_modules)
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, hidden, structure):
        joined = torch.cat([hidden, structure], dim=-1)
        gate = torch.softmax(self.gate(joined), dim=-1)
        outputs = torch.stack([m(joined) for m in self.modules_bank], dim=1)
        return self.dropout((gate.unsqueeze(-1) * outputs).sum(1)), gate

class ActionConditionedForecaster(nn.Module):
    """Small state-space surface for P(S_future | S_t, A)."""
    def __init__(self, cfg):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim), nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        )
        self.transition = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2 + cfg.input_dim, cfg.hidden_dim),
            nn.GELU(), nn.Linear(cfg.hidden_dim, cfg.input_dim)
        )
    def step(self, state, structure, action):
        action_h = self.action_encoder(action)
        return self.transition(torch.cat([state, structure, action_h], dim=-1))
    def forward(self, state, structure, actions):
        if actions.dim() != 3:
            raise ValueError("actions must have shape [batch, horizon, action_dim]")
        current = state
        predictions = []
        for t in range(actions.size(1)):
            current = self.step(current, structure, actions[:, t])
            predictions.append(current)
        return torch.stack(predictions, dim=1)

class ActionDecisionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim), nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1)
        )
    def forward(self, forecast):
        return self.score(forecast[:, :, -1]).squeeze(-1)

class TACOSM(nn.Module):
    """Persistent Operational Structure Model v0.2.

    v0.2 adds optional action-conditioned forecasting and finite candidate
    action scoring while preserving the v0.1 forward/state interface.
    """
    def __init__(self, cfg: Optional[TACOSMConfig] = None):
        super().__init__()
        self.cfg = cfg or TACOSMConfig()
        self.encoder = nn.Sequential(
            nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim),
            nn.GELU(), nn.LayerNorm(self.cfg.hidden_dim)
        )
        self.compressor = StructuralCompressor(self.cfg)
        self.router = StructureRouter(self.cfg)
        self.executor = ConditionalExecutor(self.cfg)
        self.decision_head = nn.Linear(self.cfg.hidden_dim, self.cfg.decision_dim)
        self.verifier_head = nn.Linear(self.cfg.hidden_dim + self.cfg.structure_dim, 1)
        self.forecaster = ActionConditionedForecaster(self.cfg)
        self.action_decision = ActionDecisionHead(self.cfg)
        self.action_head = nn.Linear(self.cfg.hidden_dim, self.cfg.action_dim)

    def initial_state(self, batch_size, *, device=None):
        return PersistentStructureState(
            torch.zeros(batch_size, self.cfg.state_slots, self.cfg.structure_dim, device=device),
            torch.zeros(batch_size, self.cfg.state_slots, device=device),
        )
    def reset_state(self, state):
        return PersistentStructureState(torch.zeros_like(state.slots), torch.zeros_like(state.validity))

    def forward(self, observation, state=None, *, write=True, candidate_actions=None):
        if observation.dim() != 2 or observation.size(-1) != self.cfg.input_dim:
            raise ValueError(f"observation must have shape [batch, {self.cfg.input_dim}]")
        if state is None:
            state = self.initial_state(observation.size(0), device=observation.device)
        hidden = self.encoder(observation)
        new_state = self.compressor(hidden, state) if write else state
        structure, route_weights, indices = self.router(hidden, new_state)
        active, compute_gate = self.executor(hidden, structure)
        decision_logits = self.decision_head(active)
        verifier_logit = self.verifier_head(torch.cat([active, structure], dim=-1)).squeeze(-1)
        action_logits = self.action_head(active)

        forecast = None
        action_value = None
        if candidate_actions is not None:
            if candidate_actions.dim() != 4:
                raise ValueError("candidate_actions must have shape [batch, candidates, horizon, action_dim]")
            b, n, h, a = candidate_actions.shape
            if b != observation.size(0) or a != self.cfg.action_dim:
                raise ValueError("candidate_actions has incompatible batch/action dimensions")
            flat_actions = candidate_actions.reshape(b * n, h, a)
            flat_state = active.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)
            flat_structure = structure.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)
            flat_forecast = self.forecaster(flat_state, flat_structure, flat_actions)
            forecast = flat_forecast.reshape(b, n, h, self.cfg.input_dim)
            action_value = self.action_decision(forecast)

        return TACOSMOutput(
            decision_logits, verifier_logit, route_weights, indices,
            compute_gate, new_state, action_logits, forecast, action_value
        )

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
