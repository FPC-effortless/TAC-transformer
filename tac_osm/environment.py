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
        if cfg.regime in (0, 1):
            self.effect[:, :4] = base
        else:
            raise ValueError("unsupported regime")

        dynamics_scale = 0.88 if cfg.regime == 0 else 0.82
        self.dynamics = torch.eye(cfg.state_dim) * dynamics_scale
        self.bias = torch.zeros(cfg.state_dim)

    def transition(self, state, action_index, context=None, generator=None):
        if state.dim() != 2:
            raise ValueError("state must be [batch,state_dim]")
        action = torch.nn.functional.one_hot(
            action_index, num_classes=self.cfg.action_dim
        ).float()
        if context is None:
            context_index = torch.zeros(
                state.size(0), device=state.device, dtype=torch.long
            )
        else:
            if context.shape[-1] != self.cfg.context_dim:
                raise ValueError("invalid context shape")
            if context.shape[1] != self.cfg.action_dim:
                raise ValueError("context/action dimensions must match")
            context_index = context.argmax(-1).clamp_max(self.cfg.action_dim - 1)

        # Context is a discrete task mode revealed at t0. Each mode remaps
        # the action semantics, so the optimal action can change while the
        # query state is held fixed. This prevents a context-blind policy from
        # solving the decision merely by learning an average effect magnitude.
        permutations = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 0, 3, 2],
                [2, 3, 0, 1],
                [3, 2, 1, 0],
            ], device=state.device, dtype=torch.long
        )
        mapped_action = permutations[context_index, action_index]
        effect = torch.nn.functional.one_hot(
            mapped_action, num_classes=self.cfg.action_dim
        ).float() @ self.effect
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
        context_index = torch.randint(
            0, self.cfg.context_dim, (batch_size,), device=device, generator=generator
        )
        context = torch.nn.functional.one_hot(
            context_index, num_classes=self.cfg.context_dim
        ).float()
        target, scores = self.optimal_action(state, context=context)
        return state, context, target, scores
