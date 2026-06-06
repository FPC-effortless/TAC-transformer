from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class ScratchpadItem:
    item_id: str
    kind: str
    payload: str
    utility: float
    confidence: float
    verified: bool = False
    imagined: bool = False
    cost: float = 1.0
    age: int = 0

    def score(
        self,
        *,
        confidence_weight: float = 1.0,
        verified_bonus: float = 0.25,
        cost_weight: float = 0.05,
        age_weight: float = 0.01,
    ) -> float:
        return (
            self.utility
            + confidence_weight * self.confidence
            + (verified_bonus if self.verified else 0.0)
            - cost_weight * self.cost
            - age_weight * self.age
        )


@dataclass(frozen=True)
class SimulationBranch:
    branch_id: str
    actions: tuple[str, ...]
    predicted_reward: float
    cost: float
    risk: float = 0.0
    confidence: float = 0.5
    summary: str = ""

    def value(self, *, cost_weight: float, risk_weight: float) -> float:
        return self.predicted_reward - cost_weight * self.cost - risk_weight * self.risk

    def to_scratchpad_item(self) -> ScratchpadItem:
        payload = self.summary or " -> ".join(self.actions)
        return ScratchpadItem(
            item_id=self.branch_id,
            kind="simulation",
            payload=payload,
            utility=self.predicted_reward,
            confidence=self.confidence,
            imagined=True,
            cost=self.cost,
        )


@dataclass(frozen=True)
class AgenticTrajectoryStep:
    step_index: int
    action: str
    action_logprob: float
    route_id: str
    memory_read_ids: tuple[str, ...] = ()
    scratchpad_item_ids: tuple[str, ...] = ()
    verifier_score: float = 0.0
    reward: float = 0.0
    cost: float = 0.0


@dataclass(frozen=True)
class AgenticTrajectory:
    trajectory_id: str
    steps: tuple[AgenticTrajectoryStep, ...]
    final_reward: float
    cost_weight: float = 1.0
    metadata: Mapping[str, Any] | None = None

    @property
    def total_cost(self) -> float:
        return sum(float(step.cost) for step in self.steps)

    @property
    def action_logprob_sum(self) -> float:
        return sum(float(step.action_logprob) for step in self.steps)

    @property
    def cost_adjusted_reward(self) -> float:
        return float(self.final_reward) - float(self.cost_weight) * self.total_cost

    @property
    def verifier_mean(self) -> float:
        if not self.steps:
            return 0.0
        return sum(float(step.verifier_score) for step in self.steps) / len(self.steps)


@dataclass(frozen=True)
class AgenticProofThresholds:
    min_state_margin: float = 0.0
    min_scratchpad_gain: float = 0.02
    min_simulation_gain: float = 0.02
    min_teaching_gain: float = 0.02
    max_world_error: float = 0.10
    max_false_authority_rate: float = 0.01
    max_hypothesis_contamination_rate: float = 0.0
    min_cost_adjusted_reward_gain: float = 0.0


def cost_adjusted_rewards(
    rewards: Tensor,
    costs: Tensor | Mapping[str, Tensor | float],
    *,
    cost_weight: float = 1.0,
) -> Tensor:
    """Return verified utility after subtracting explicit cost."""

    cost_tensor = _total_cost_like(rewards, costs)
    return rewards - cost_weight * cost_tensor


def group_relative_advantages(
    rewards: Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-6,
) -> Tensor:
    """GRPO/RLOO-style normalized advantages within each sampled group."""

    centered = rewards - rewards.mean(dim=dim, keepdim=True)
    std = rewards.std(dim=dim, keepdim=True, unbiased=False)
    return torch.where(std > eps, centered / (std + eps), torch.zeros_like(centered))


def policy_gradient_loss(
    action_logits: Tensor,
    actions: Tensor,
    advantages: Tensor,
    *,
    mask: Tensor | None = None,
    entropy_weight: float = 0.0,
) -> Tensor:
    """Actor loss over discrete internal or external actions."""

    log_probs = F.log_softmax(action_logits, dim=-1)
    selected = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    weighted = -selected * advantages.detach()
    if mask is not None:
        mask = mask.to(dtype=weighted.dtype, device=weighted.device)
        loss = (weighted * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        loss = weighted.mean()
    if entropy_weight:
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        if mask is not None:
            entropy_term = (entropy * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            entropy_term = entropy.mean()
        loss = loss - entropy_weight * entropy_term
    return loss


def group_relative_trajectory_policy_loss(
    trajectories: Sequence[AgenticTrajectory],
    *,
    action_logits: Tensor,
    actions: Tensor,
    group_ids: Sequence[str] | None = None,
    entropy_weight: float = 0.0,
) -> dict[str, Any]:
    """Apply group-relative policy optimization to complete trajectories."""

    trajectory_count = len(trajectories)
    if action_logits.shape[0] != trajectory_count:
        raise ValueError("action_logits batch must match trajectories")
    if actions.shape[0] != trajectory_count:
        raise ValueError("actions batch must match trajectories")
    resolved_group_ids = (
        tuple(str(group_id) for group_id in group_ids)
        if group_ids is not None
        else tuple(str((trajectory.metadata or {}).get("prompt_id", "default")) for trajectory in trajectories)
    )
    if len(resolved_group_ids) != trajectory_count:
        raise ValueError("group_ids length must match trajectories")
    rewards = torch.tensor(
        [trajectory.cost_adjusted_reward for trajectory in trajectories],
        dtype=action_logits.dtype,
        device=action_logits.device,
    )
    advantages = torch.zeros_like(rewards)
    for group_id in sorted(set(resolved_group_ids)):
        indexes = [
            index
            for index, candidate in enumerate(resolved_group_ids)
            if candidate == group_id
        ]
        index_tensor = torch.tensor(indexes, dtype=torch.long, device=action_logits.device)
        group_rewards = rewards.index_select(0, index_tensor)
        advantages.index_copy_(
            0,
            index_tensor,
            group_relative_advantages(group_rewards, dim=0),
        )
    loss = policy_gradient_loss(
        action_logits,
        actions.to(device=action_logits.device),
        advantages,
        entropy_weight=entropy_weight,
    )
    return {
        "schema": "group_relative_trajectory_policy_loss.v1",
        "loss": loss,
        "rewards": rewards,
        "advantages": advantages,
        "group_ids": resolved_group_ids,
        "trajectory_ids": tuple(trajectory.trajectory_id for trajectory in trajectories),
    }


def gspo_sequence_policy_loss(
    current_logprobs: Tensor,
    reference_logprobs: Tensor,
    advantages: Tensor,
    *,
    mask: Tensor | None = None,
    clip_epsilon: float = 0.2,
    length_normalize: bool = True,
) -> dict[str, Tensor]:
    """GSPO-style clipped objective over complete action sequences."""

    if current_logprobs.shape != reference_logprobs.shape:
        raise ValueError("current_logprobs and reference_logprobs must match")
    if current_logprobs.ndim != 2:
        raise ValueError("logprobs must have shape [batch, steps]")
    if advantages.shape != current_logprobs.shape[:1]:
        raise ValueError("advantages must have shape [batch]")
    if clip_epsilon < 0.0:
        raise ValueError("clip_epsilon must be non-negative")
    logprob_delta = current_logprobs - reference_logprobs.to(
        dtype=current_logprobs.dtype,
        device=current_logprobs.device,
    )
    if mask is not None:
        if mask.shape != current_logprobs.shape:
            raise ValueError("mask must match logprobs shape")
        mask = mask.to(dtype=current_logprobs.dtype, device=current_logprobs.device)
        logprob_delta = logprob_delta * mask
        lengths = mask.sum(dim=-1).clamp_min(1.0)
    else:
        lengths = current_logprobs.new_full(
            (current_logprobs.shape[0],),
            float(current_logprobs.shape[1]),
        )
    sequence_delta = logprob_delta.sum(dim=-1)
    if length_normalize:
        sequence_delta = sequence_delta / lengths
    ratios = sequence_delta.exp()
    clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    detached_advantages = advantages.to(
        dtype=current_logprobs.dtype,
        device=current_logprobs.device,
    ).detach()
    unclipped = ratios * detached_advantages
    clipped = clipped_ratios * detached_advantages
    surrogate = torch.minimum(unclipped, clipped)
    return {
        "schema": "gspo_sequence_policy_loss.v1",
        "loss": -surrogate.mean(),
        "sequence_logprob_delta": sequence_delta,
        "sequence_ratios": ratios,
        "clipped_sequence_ratios": clipped_ratios,
        "surrogate": surrogate,
    }


def shaped_trajectory_rewards(
    trajectories: Sequence[AgenticTrajectory],
    *,
    cost_weight: float = 1.0,
    length_penalty: float = 0.0,
) -> Tensor:
    """Return reward shaped by explicit compute cost and trajectory length."""

    if cost_weight < 0.0:
        raise ValueError("cost_weight must be non-negative")
    if length_penalty < 0.0:
        raise ValueError("length_penalty must be non-negative")
    return torch.tensor(
        [
            float(trajectory.final_reward)
            - float(cost_weight) * trajectory.total_cost
            - float(length_penalty) * len(trajectory.steps)
            for trajectory in trajectories
        ],
        dtype=torch.float32,
    )


def dapo_dynamic_sampling_filter(
    trajectories: Sequence[AgenticTrajectory],
    *,
    group_ids: Sequence[str] | None = None,
    reward_values: Tensor | Sequence[float] | None = None,
    success_threshold: float = 0.5,
) -> dict[str, Any]:
    """Keep prompt groups with mixed success/failure trajectory outcomes."""

    if not 0.0 <= success_threshold <= 1.0:
        raise ValueError("success_threshold must be in [0, 1]")
    resolved_group_ids = (
        tuple(str(group_id) for group_id in group_ids)
        if group_ids is not None
        else tuple(str((trajectory.metadata or {}).get("prompt_id", "default")) for trajectory in trajectories)
    )
    if len(resolved_group_ids) != len(trajectories):
        raise ValueError("group_ids length must match trajectories")
    if reward_values is None:
        success_values = [float(trajectory.final_reward) for trajectory in trajectories]
    elif isinstance(reward_values, Tensor):
        if reward_values.ndim != 1 or reward_values.shape[0] != len(trajectories):
            raise ValueError("reward_values must have shape [trajectories]")
        success_values = [float(value) for value in reward_values.detach().cpu().tolist()]
    else:
        success_values = [float(value) for value in reward_values]
        if len(success_values) != len(trajectories):
            raise ValueError("reward_values length must match trajectories")
    selected_indexes: list[int] = []
    selected_groups: list[str] = []
    dropped_groups: list[str] = []
    for group_id in sorted(set(resolved_group_ids)):
        indexes = [
            index
            for index, candidate in enumerate(resolved_group_ids)
            if candidate == group_id
        ]
        successes = [
            success_values[index] >= success_threshold
            for index in indexes
        ]
        has_success = any(successes)
        has_failure = any(not success for success in successes)
        if has_success and has_failure:
            selected_indexes.extend(indexes)
            selected_groups.append(group_id)
        else:
            dropped_groups.append(group_id)
    return {
        "schema": "dapo_dynamic_sampling_filter.v1",
        "selected_indexes": selected_indexes,
        "selected_group_ids": selected_groups,
        "dropped_group_ids": dropped_groups,
        "selected_fraction": len(selected_indexes) / max(len(trajectories), 1),
        "success_values": success_values,
    }


def bounded_scratchpad_update(
    existing: Sequence[ScratchpadItem],
    candidates: Sequence[ScratchpadItem],
    *,
    budget: int,
) -> list[ScratchpadItem]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    merged = list(existing) + list(candidates)
    ranked = sorted(
        merged,
        key=lambda item: (item.score(), item.confidence, item.utility),
        reverse=True,
    )
    return ranked[:budget]


def commit_verified_scratchpad_items(
    scratchpad: Sequence[ScratchpadItem],
    *,
    verifier_supported_ids: Iterable[str],
    min_confidence: float = 0.5,
) -> list[ScratchpadItem]:
    supported = set(verifier_supported_ids)
    committed = []
    for item in scratchpad:
        verifier_support = item.item_id in supported or item.verified
        if item.imagined and not verifier_support:
            continue
        if verifier_support and item.confidence >= min_confidence:
            committed.append(item)
    return committed


def select_best_simulation_branch(
    branches: Sequence[SimulationBranch],
    *,
    cost_weight: float,
    risk_weight: float,
) -> SimulationBranch:
    if not branches:
        raise ValueError("branches must not be empty")
    return max(
        branches,
        key=lambda branch: branch.value(
            cost_weight=cost_weight,
            risk_weight=risk_weight,
        ),
    )


def process_trace_distillation_loss(
    step_logits: Tensor,
    step_targets: Tensor,
    *,
    step_mask: Tensor | None = None,
    verifier_scores: Tensor | None = None,
) -> Tensor:
    """Teach process steps while weighting verified steps more heavily."""

    if step_logits.ndim != 3:
        raise ValueError("step_logits must have shape [batch, steps, classes]")
    if step_targets.shape != step_logits.shape[:2]:
        raise ValueError("step_targets must match step_logits batch and step dimensions")
    flat_loss = F.cross_entropy(
        step_logits.reshape(-1, step_logits.shape[-1]),
        step_targets.reshape(-1),
        reduction="none",
    ).reshape_as(step_targets).to(dtype=step_logits.dtype)
    weights = torch.ones_like(flat_loss)
    if verifier_scores is not None:
        if verifier_scores.shape != step_targets.shape:
            raise ValueError("verifier_scores must match step_targets shape")
        weights = weights * verifier_scores.to(dtype=flat_loss.dtype, device=flat_loss.device)
    if step_mask is not None:
        if step_mask.shape != step_targets.shape:
            raise ValueError("step_mask must match step_targets shape")
        weights = weights * step_mask.to(dtype=flat_loss.dtype, device=flat_loss.device)
    return (flat_loss * weights).sum() / weights.sum().clamp_min(1.0)


def implicit_process_rewards(
    current_step_logprobs: Tensor,
    reference_step_logprobs: Tensor,
    *,
    verifier_scores: Tensor | None = None,
    mask: Tensor | None = None,
    beta: float = 1.0,
    verifier_weight: float = 1.0,
) -> Tensor:
    """PRIME-style process rewards from policy/reference ratios and verifier support."""

    if current_step_logprobs.shape != reference_step_logprobs.shape:
        raise ValueError("current and reference step logprobs must match")
    log_ratio = current_step_logprobs - reference_step_logprobs.to(
        dtype=current_step_logprobs.dtype,
        device=current_step_logprobs.device,
    )
    rewards = float(beta) * log_ratio
    if verifier_scores is not None:
        if verifier_scores.shape != current_step_logprobs.shape:
            raise ValueError("verifier_scores must match step logprobs shape")
        verifier_signal = 2.0 * verifier_scores.to(
            dtype=current_step_logprobs.dtype,
            device=current_step_logprobs.device,
        ) - 1.0
        rewards = rewards + float(verifier_weight) * verifier_signal
    if mask is not None:
        if mask.shape != current_step_logprobs.shape:
            raise ValueError("mask must match step logprobs shape")
        rewards = rewards * mask.to(
            dtype=current_step_logprobs.dtype,
            device=current_step_logprobs.device,
        )
    return rewards


def value_prediction_loss(
    values: Tensor,
    returns: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Masked value-function regression loss for trajectory/process returns."""

    if values.shape != returns.shape:
        raise ValueError("values and returns must have the same shape")
    squared_error = (
        values
        - returns.to(dtype=values.dtype, device=values.device)
    ).pow(2)
    if mask is None:
        return squared_error.mean()
    if mask.shape != values.shape:
        raise ValueError("mask must match values shape")
    mask = mask.to(dtype=values.dtype, device=values.device)
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


def identity_persistence_score(
    retrieval_recurrence: Tensor,
    memory_survival_rate: Tensor,
    reuse_frequency: Tensor,
    *,
    weights: Tensor | Sequence[float] | None = None,
) -> Tensor:
    """Score whether program identities persist as reusable structures."""

    if retrieval_recurrence.shape != memory_survival_rate.shape:
        raise ValueError("retrieval_recurrence and memory_survival_rate must match")
    if retrieval_recurrence.shape != reuse_frequency.shape:
        raise ValueError("retrieval_recurrence and reuse_frequency must match")
    stacked = torch.stack(
        [
            retrieval_recurrence,
            memory_survival_rate.to(
                dtype=retrieval_recurrence.dtype,
                device=retrieval_recurrence.device,
            ),
            reuse_frequency.to(
                dtype=retrieval_recurrence.dtype,
                device=retrieval_recurrence.device,
            ),
        ],
        dim=0,
    )
    if weights is None:
        return stacked.mean(dim=0)
    weight_tensor = (
        weights
        if isinstance(weights, Tensor)
        else torch.tensor(weights, dtype=retrieval_recurrence.dtype)
    ).to(dtype=retrieval_recurrence.dtype, device=retrieval_recurrence.device)
    if weight_tensor.shape != (3,):
        raise ValueError("weights must contain three values")
    if bool((weight_tensor < 0.0).any()):
        raise ValueError("weights must be non-negative")
    normalized = weight_tensor / weight_tensor.sum().clamp_min(1e-6)
    return (stacked * normalized.reshape(3, *([1] * retrieval_recurrence.ndim))).sum(dim=0)


def memory_overlap_graph(
    program_memory: Tensor,
    *,
    coactivation_window: Tensor | None = None,
    tau_link: float = 0.5,
    tau_time: float | None = None,
    include_self: bool = False,
    eps: float = 1e-6,
) -> dict[str, Tensor | str]:
    """Build a soft memory-link graph from identity overlap and time proximity."""

    if program_memory.ndim != 2:
        raise ValueError("program_memory must have shape [programs, dim]")
    normalized = program_memory / program_memory.norm(dim=-1, keepdim=True).clamp_min(eps)
    overlap = normalized @ normalized.transpose(0, 1)
    adjacency = overlap >= float(tau_link)
    if coactivation_window is not None:
        if coactivation_window.shape != overlap.shape:
            raise ValueError("coactivation_window must have shape [programs, programs]")
        if tau_time is None:
            raise ValueError("tau_time is required when coactivation_window is provided")
        adjacency = adjacency & (
            coactivation_window.to(device=overlap.device) <= float(tau_time)
        )
    if not include_self:
        diagonal = torch.eye(
            overlap.shape[0],
            dtype=torch.bool,
            device=overlap.device,
        )
        adjacency = adjacency & ~diagonal
    return {
        "schema": "memory_overlap_graph.v1",
        "overlap": overlap,
        "link_adjacency": adjacency,
        "link_weights": overlap.clamp_min(0.0) * adjacency.to(dtype=overlap.dtype),
    }


def memory_link_utility(
    seed_scores: Tensor,
    link_adjacency: Tensor,
    *,
    target_scores: Tensor,
    link_weights: Tensor | None = None,
    propagation_steps: int = 1,
    decay: float = 1.0,
) -> dict[str, Tensor | str]:
    """Measure whether memory links expose target programs missed by direct recall."""

    if seed_scores.ndim != 1:
        raise ValueError("seed_scores must have shape [programs]")
    if target_scores.shape != seed_scores.shape:
        raise ValueError("target_scores must match seed_scores")
    if link_adjacency.shape != (seed_scores.shape[0], seed_scores.shape[0]):
        raise ValueError("link_adjacency must have shape [programs, programs]")
    if propagation_steps < 0:
        raise ValueError("propagation_steps must be non-negative")
    adjacency = link_adjacency.to(dtype=seed_scores.dtype, device=seed_scores.device)
    if link_weights is not None:
        if link_weights.shape != link_adjacency.shape:
            raise ValueError("link_weights must match link_adjacency")
        adjacency = adjacency * link_weights.to(
            dtype=seed_scores.dtype,
            device=seed_scores.device,
        )
    propagated = seed_scores.to(dtype=seed_scores.dtype)
    frontier = propagated
    for _ in range(propagation_steps):
        frontier = frontier @ adjacency
        propagated = torch.maximum(propagated, float(decay) * frontier)
    targets = target_scores.to(dtype=seed_scores.dtype, device=seed_scores.device)
    direct_utility = (seed_scores * targets).sum()
    linked_utility = (propagated * targets).sum()
    return {
        "schema": "memory_link_utility.v1",
        "direct_utility": direct_utility,
        "linked_utility": linked_utility,
        "link_gain": linked_utility - direct_utility,
        "propagated_scores": propagated,
    }


def coalition_participation_metrics(
    selected_program_mask: Tensor,
    *,
    eps: float = 1e-6,
) -> dict[str, Tensor | str]:
    """Summarize program coalition participation and coactivation structure."""

    if selected_program_mask.ndim == 2:
        flat = selected_program_mask
    elif selected_program_mask.ndim == 3:
        flat = selected_program_mask.reshape(-1, selected_program_mask.shape[-1])
    else:
        raise ValueError("selected_program_mask must have shape [steps, programs] or [batch, steps, programs]")
    flat = flat.to(dtype=torch.float32)
    event_count = flat.shape[0]
    participation = flat.mean(dim=0)
    coactivation_counts = flat.transpose(0, 1) @ flat
    coactivation_matrix = coactivation_counts / max(event_count, 1)
    off_diagonal = coactivation_matrix.clone()
    off_diagonal.fill_diagonal_(0.0)
    coactivation_degree = off_diagonal.sum(dim=-1)
    probabilities = participation / participation.sum().clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return {
        "schema": "coalition_participation_metrics.v1",
        "participation": participation,
        "coactivation_matrix": coactivation_matrix,
        "coactivation_degree": coactivation_degree,
        "participation_entropy": entropy,
    }


def basal_apical_belief_state(
    basal: Tensor,
    apical: Tensor,
    *,
    gate: Tensor | None = None,
    eps: float = 1e-6,
) -> dict[str, Tensor | str]:
    """Integrate observed/retrieved evidence with top-down identity proposals."""

    if basal.shape != apical.shape:
        raise ValueError("basal and apical must have the same shape")
    agreement = F.cosine_similarity(basal, apical, dim=-1, eps=eps)
    disagreement = 1.0 - agreement
    if gate is None:
        apical_weight = ((agreement + 1.0) / 2.0).clamp(0.0, 1.0)
    else:
        if gate.shape != agreement.shape:
            raise ValueError("gate must match the leading basal/apical shape")
        apical_weight = gate.to(dtype=basal.dtype, device=basal.device).clamp(0.0, 1.0)
    weight = apical_weight.unsqueeze(-1)
    belief = (1.0 - weight) * basal + weight * apical.to(
        dtype=basal.dtype,
        device=basal.device,
    )
    return {
        "schema": "basal_apical_belief_state.v1",
        "belief": belief,
        "agreement": agreement,
        "disagreement": disagreement,
        "apical_weight": apical_weight,
    }


def phase_d_agentic_reward(
    metrics: Mapping[str, float | Tensor],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Combine task, identity, coalition, verifier, and cost terms for Phase D."""

    resolved_weights = {
        "task": 1.0,
        "verify": 1.0,
        "state": 1.0,
        "route": 1.0,
        "ips": 1.0,
        "link": 1.0,
        "coalition": 1.0,
        "world": 1.0,
        "false": 1.0,
        "contam": 1.0,
        "cost": 1.0,
    }
    resolved_weights.update(dict(weights or {}))
    positive_sources = {
        "task_success": ("task", "task_success"),
        "verification_pass": ("verify", "verification_pass"),
        "state_utility": ("state", "state_utility"),
        "route_utility": ("route", "route_utility"),
        "identity_persistence": ("ips", "identity_persistence"),
        "memory_link_utility": ("link", "memory_link_utility"),
        "coalition_utility": ("coalition", "coalition_utility"),
        "world_accuracy": ("world", "world_accuracy"),
    }
    penalty_sources = {
        "false_authority": ("false", "false_authority"),
        "hypothesis_contamination": ("contam", "hypothesis_contamination"),
        "cost": ("cost", "cost"),
    }
    positive_terms = {
        output_name: resolved_weights[weight_name] * _metric_float(metrics, metric_name)
        for output_name, (weight_name, metric_name) in positive_sources.items()
    }
    penalty_terms = {
        output_name: resolved_weights[weight_name] * _metric_float(metrics, metric_name)
        for output_name, (weight_name, metric_name) in penalty_sources.items()
    }
    reward = sum(positive_terms.values()) - sum(penalty_terms.values())
    return {
        "schema": "phase_d_agentic_reward.v1",
        "reward": reward,
        "positive_terms": positive_terms,
        "penalty_terms": penalty_terms,
        "weights": dict(resolved_weights),
    }


def build_agentic_trajectory(
    *,
    trajectory_id: str,
    steps: Sequence[AgenticTrajectoryStep],
    final_reward: float,
    cost_weight: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> AgenticTrajectory:
    """Create a validated trajectory record for RL or process supervision."""

    ordered_steps = tuple(sorted(steps, key=lambda step: int(step.step_index)))
    expected_indexes = list(range(len(ordered_steps)))
    actual_indexes = [int(step.step_index) for step in ordered_steps]
    if actual_indexes != expected_indexes:
        raise ValueError("trajectory step_index values must be contiguous from zero")
    if cost_weight < 0.0:
        raise ValueError("cost_weight must be non-negative")
    return AgenticTrajectory(
        trajectory_id=str(trajectory_id),
        steps=ordered_steps,
        final_reward=float(final_reward),
        cost_weight=float(cost_weight),
        metadata=dict(metadata or {}),
    )


def trajectory_to_training_record(trajectory: AgenticTrajectory) -> dict[str, Any]:
    """Serialize a trajectory into a stable training/audit record."""

    return {
        "schema": "agentic_trajectory_record.v1",
        "trajectory_id": trajectory.trajectory_id,
        "actions": [step.action for step in trajectory.steps],
        "action_logprobs": [float(step.action_logprob) for step in trajectory.steps],
        "action_logprob_sum": trajectory.action_logprob_sum,
        "route_ids": [step.route_id for step in trajectory.steps],
        "memory_read_ids": [list(step.memory_read_ids) for step in trajectory.steps],
        "scratchpad_item_ids": [
            list(step.scratchpad_item_ids) for step in trajectory.steps
        ],
        "verifier_scores": [float(step.verifier_score) for step in trajectory.steps],
        "step_rewards": [float(step.reward) for step in trajectory.steps],
        "step_costs": [float(step.cost) for step in trajectory.steps],
        "total_cost": trajectory.total_cost,
        "final_reward": float(trajectory.final_reward),
        "cost_weight": float(trajectory.cost_weight),
        "cost_adjusted_reward": trajectory.cost_adjusted_reward,
        "verifier_mean": trajectory.verifier_mean,
        "metadata": dict(trajectory.metadata or {}),
    }


def verifier_reward_from_authority_report(
    authority_report: Any,
    *,
    base_reward: float = 0.0,
    trusted_correct_bonus: float = 0.25,
    false_authority_penalty: float = 1.0,
    cross_domain_penalty: float = 0.5,
) -> dict[str, Any]:
    """Shape reward from the existing authority-report manifest contract."""

    manifest = (
        authority_report.to_manifest()
        if hasattr(authority_report, "to_manifest")
        else dict(authority_report)
    )
    trusted_event_count = int(manifest.get("trusted_event_count") or 0)
    trusted_correct_count = int(manifest.get("trusted_correct_count") or 0)
    false_authority_count = int(manifest.get("false_authority_count") or 0)
    cross_domain_count = int(
        manifest.get("cross_domain_authority_violation_count") or 0
    )
    correct_bonus = trusted_correct_bonus if trusted_correct_count > 0 else 0.0
    verifier_reward = (
        float(base_reward)
        + float(correct_bonus)
        - float(false_authority_penalty) * false_authority_count
        - float(cross_domain_penalty) * cross_domain_count
    )
    false_authority_rate = (
        false_authority_count / trusted_event_count
        if trusted_event_count
        else 0.0
    )
    return {
        "schema": "agentic_verifier_reward.v1",
        "verifier_reward": verifier_reward,
        "base_reward": float(base_reward),
        "trusted_correct_bonus": float(trusted_correct_bonus),
        "false_authority_penalty": float(false_authority_penalty),
        "cross_domain_penalty": float(cross_domain_penalty),
        "trusted_event_count": trusted_event_count,
        "trusted_correct_count": trusted_correct_count,
        "false_authority_count": false_authority_count,
        "cross_domain_authority_violation_count": cross_domain_count,
        "false_authority_rate": false_authority_rate,
        "trusted_accuracy": manifest.get("trusted_accuracy"),
    }


def agentic_promotion_decision(
    metrics: Mapping[str, float],
    thresholds: AgenticProofThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or AgenticProofThresholds()
    carry = float(metrics.get("carry_score", 0.0))
    reset = float(metrics.get("reset_score", 0.0))
    shuffled = float(metrics.get("shuffled_score", 0.0))
    baseline = float(metrics.get("baseline_score", 0.0))
    checks = {
        "carry_beats_reset": carry - reset > thresholds.min_state_margin,
        "carry_beats_shuffled": carry - shuffled > thresholds.min_state_margin,
        "carry_beats_baseline": carry - baseline > thresholds.min_state_margin,
        "scratchpad_beats_no_scratchpad": _gain(
            metrics,
            "scratchpad_score",
            "no_scratchpad_score",
        )
        >= thresholds.min_scratchpad_gain,
        "simulation_beats_no_simulation": _gain(
            metrics,
            "simulation_score",
            "no_simulation_score",
        )
        >= thresholds.min_simulation_gain,
        "teaching_beats_no_teaching": _gain(
            metrics,
            "teaching_score",
            "no_teaching_score",
        )
        >= thresholds.min_teaching_gain,
        "world_error_bounded": float(metrics.get("world_error", 1.0))
        <= thresholds.max_world_error,
        "false_authority_bounded": float(metrics.get("false_authority_rate", 1.0))
        <= thresholds.max_false_authority_rate,
        "hypothesis_contamination_blocked": float(
            metrics.get("hypothesis_contamination_rate", 1.0)
        )
        <= thresholds.max_hypothesis_contamination_rate,
        "cost_adjusted_reward_beats_baseline": _gain(
            metrics,
            "cost_adjusted_reward",
            "baseline_cost_adjusted_reward",
        )
        > thresholds.min_cost_adjusted_reward_gain,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "agentic_promotion_decision.v1",
        "status": "promotable" if not failed else "blocked",
        "checks": checks,
        "failed_checks": failed,
        "metrics": dict(metrics),
        "thresholds": asdict(thresholds),
    }


def _total_cost_like(
    rewards: Tensor,
    costs: Tensor | Mapping[str, Tensor | float],
) -> Tensor:
    if isinstance(costs, Tensor):
        return costs.to(dtype=rewards.dtype, device=rewards.device)
    total = torch.zeros_like(rewards)
    for value in costs.values():
        if isinstance(value, Tensor):
            total = total + value.to(dtype=rewards.dtype, device=rewards.device)
        else:
            total = total + float(value)
    return total


def _gain(metrics: Mapping[str, float], numerator: str, denominator: str) -> float:
    return float(metrics.get(numerator, 0.0)) - float(metrics.get(denominator, 0.0))


def _metric_float(metrics: Mapping[str, float | Tensor], name: str) -> float:
    value = metrics.get(name, 0.0)
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        return float(value.detach().cpu())
    return float(value)
