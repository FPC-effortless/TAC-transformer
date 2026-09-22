"""Deterministic causal operational world for TAC-OSM v0.4."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class WorldConfig:
    state_dim: int = 8
    action_dim: int = 4
    noise_std: float = 0.0
    context_dim: int = 4
    regime: int = 0


class SyntheticOperationalWorld:
    def __init__(self, cfg=WorldConfig()):
        self.cfg = cfg
        if cfg.state_dim < 4 or cfg.action_dim < 2:
            raise ValueError("state_dim/action_dim too small")
        self.effect = torch.zeros(cfg.action_dim, cfg.state_dim)
        base = torch.tensor([
            [0.8, -0.5, 0.0, 0.0],
            [-0.6, 0.0, 0.7, 0.0],
            [0.0, 0.5, 0.0, -0.6],
            [0.0, 0.0, -0.5, 0.8],
        ])
        if cfg.regime == 0:
            self.effect[:, :4] = base
        elif cfg.regime == 1:
            self.effect[:, :4] = base.roll(1, dims=0)
            self.effect[:, 0] *= -1
        else:
            raise ValueError("unsupported regime")

        self.dynamics = torch.eye(cfg.state_dim) * 0.88
        self.bias = torch.zeros(cfg.state_dim)

    def transition(self, state, action_index, context=None, generator=None):
        if state.dim() != 2:
            raise ValueError("state must be [batch,state_dim]")
        action = torch.nn.functional.one_hot(
            action_index, num_classes=self.cfg.action_dim
        ).float()
        effect = action @ self.effect
        if context is not None:
            if context.shape[-1] != self.cfg.context_dim:
                raise ValueError("invalid context shape")
            scale = 1.0 + 0.35 * torch.tanh(context[:, :1])
            effect = effect * scale
        noise = torch.randn(
            state.shape, device=state.device, generator=generator
        ) * self.cfg.noise_std
        return state @ self.dynamics.T + effect + self.bias + noise

    def objective(self, state, context=None):
        del context
        return -(state[:, 0].abs() + 0.7 * state[:, 1].abs()
                 + 0.4 * state[:, 2].abs() + 0.2 * state[:, 3].abs())

    def optimal_action(self, state, context=None):
        scores = []
        for a in range(self.cfg.action_dim):
            actions = torch.full(
                (state.size(0),), a, device=state.device, dtype=torch.long
            )
            next_state = self.transition(
                state, actions, context=context, generator=None
            )
            scores.append(self.objective(next_state, context=context))
        stacked = torch.stack(scores, 1)
        return stacked.argmax(1), stacked

    def sample(self, batch_size, generator=None, device=None):
        state = torch.randn(
            batch_size, self.cfg.state_dim, device=device, generator=generator
        )
        context = torch.randn(
            batch_size, self.cfg.context_dim, device=device, generator=generator
        )
        target, scores = self.optimal_action(state, context=context)
        return state, context, target, scores
