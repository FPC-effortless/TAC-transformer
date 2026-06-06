from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import IdentityState, TACConfig, TACTransformerLM, VanillaTransformerLM
from .optimization import TACOptimizerConfig, build_tac_optimizer
from .training import count_parameters, parameter_matched_baseline_config


@dataclass(frozen=True)
class AgenticControlBatch:
    context_inputs: Tensor
    query_inputs: Tensor
    action_targets: Tensor
    next_observation_targets: Tensor
    reward_targets: Tensor
    reflection_targets: Tensor
    key_index: int = 1
    observation_index: int = 2
    decision_index: int = 3


@dataclass
class AgenticControllerOutput:
    action_logits: Tensor
    next_observation_logits: Optional[Tensor]
    reward_logits: Optional[Tensor]
    reflection_logits: Optional[Tensor]
    identity_states: list[IdentityState]
    loss: Optional[Tensor]
    losses: dict[str, Tensor]


class AgenticControlBatcher:
    """No-leak action task where the correct tool/action depends on carried memory."""

    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        num_actions: int,
        seed: int = 0,
    ):
        if vocab_size < 24:
            raise ValueError("vocab_size must be at least 24")
        if seq_len < 6:
            raise ValueError("seq_len must be at least 6")
        if num_actions < 2:
            raise ValueError("num_actions must be at least 2")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_actions = num_actions
        self.context_token = 0
        self.query_token = 1
        self.observation_token = 2
        self.action_token = 3
        self.data_floor = 4
        self.rng = random.Random(seed)

    def next_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> AgenticControlBatch:
        context_rows = []
        query_rows = []
        action_targets = []
        next_observation_targets = []
        reward_targets = []
        reflection_targets = []
        for _ in range(batch_size):
            (
                context,
                query,
                action,
                next_observation,
                reward,
                reflection,
            ) = self._make_pair()
            context_rows.append(context)
            query_rows.append(query)
            action_targets.append(action)
            next_observation_targets.append(next_observation)
            reward_targets.append(reward)
            reflection_targets.append(reflection)

        return AgenticControlBatch(
            context_inputs=torch.tensor(context_rows, dtype=torch.long, device=device),
            query_inputs=torch.tensor(query_rows, dtype=torch.long, device=device),
            action_targets=torch.tensor(action_targets, dtype=torch.long, device=device),
            next_observation_targets=torch.tensor(
                next_observation_targets,
                dtype=torch.long,
                device=device,
            ),
            reward_targets=torch.tensor(reward_targets, dtype=torch.long, device=device),
            reflection_targets=torch.tensor(
                reflection_targets,
                dtype=torch.long,
                device=device,
            ),
        )

    def _make_pair(self) -> tuple[list[int], list[int], int, int, int, int]:
        key = self._data_token()
        value = self._data_token(exclude=key)
        observation = self._data_token(exclude=value)
        action = self._action_for(value, observation)
        next_observation = self._next_observation(observation, action)
        reward = 1 if action == self._action_for(value, observation) else 0
        reflection = 1 if (value + key + observation) % 2 == 0 else 0

        context = [self.context_token, key, value]
        query = [self.query_token, key, observation, self.action_token]
        while len(context) < self.seq_len:
            context.append(self._data_token())
        while len(query) < self.seq_len:
            query.append(self._data_token(exclude=value))
        return context, query, action, next_observation, reward, reflection

    def _action_for(self, value: int, observation: int) -> int:
        return ((value - self.data_floor) + (observation - self.data_floor)) % self.num_actions

    def _next_observation(self, observation: int, action: int) -> int:
        span = self.vocab_size - self.data_floor
        return self.data_floor + ((observation - self.data_floor + action + 1) % span)

    def _data_token(self, exclude: int | None = None) -> int:
        token = self.rng.randrange(self.data_floor, self.vocab_size)
        if exclude is not None and token == exclude:
            token = self.data_floor + ((token - self.data_floor + 1) % (self.vocab_size - self.data_floor))
        return token


class AgenticController(nn.Module):
    """Action/world/reflection heads over a TAC or vanilla LM backbone."""

    def __init__(
        self,
        backbone: TACTransformerLM | VanillaTransformerLM,
        *,
        num_actions: int,
        use_world_model: bool = False,
        use_reward_model: bool = False,
        use_reflection: bool = False,
        use_memory_adapter: bool = True,
        use_memory_action_readout: bool = False,
        use_recurrent_state: bool = False,
        use_modular_cognition: bool = False,
        use_memory_stores: bool = False,
        use_planner: bool = False,
        use_orchestration: bool = False,
        memory_adapter_weight: float = 6.0,
        memory_action_weight: float = 1.0,
        planner_weight: float = 1.0,
        orchestration_weight: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_actions = num_actions
        self.use_world_model = use_world_model
        self.use_reward_model = use_reward_model
        self.use_reflection = use_reflection
        self.use_memory_adapter = use_memory_adapter
        self.use_memory_action_readout = use_memory_action_readout
        self.use_recurrent_state = use_recurrent_state
        self.use_modular_cognition = use_modular_cognition
        self.use_memory_stores = use_memory_stores
        self.use_planner = use_planner
        self.use_orchestration = use_orchestration
        self.memory_adapter_weight = memory_adapter_weight
        self.memory_action_weight = memory_action_weight
        self.planner_weight = planner_weight
        self.orchestration_weight = orchestration_weight
        d_model = backbone.config.d_model
        if use_recurrent_state:
            self.recurrent_cell = nn.GRUCell(d_model, d_model)
            self.recurrent_gate = nn.Linear(d_model * 2, d_model)
        else:
            self.recurrent_cell = None
            self.recurrent_gate = None
        if use_modular_cognition:
            self.cognitive_gate = nn.Linear(d_model, 4)
            self.cognitive_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_model, d_model * 2),
                        nn.SiLU(),
                        nn.Linear(d_model * 2, d_model),
                    )
                    for _ in range(4)
                ]
            )
        else:
            self.cognitive_gate = None
            self.cognitive_experts = None
        if use_memory_stores:
            self.memory_store_gate = nn.Linear(d_model, 4)
            self.memory_store_projections = nn.ModuleList(
                [nn.Linear(d_model, d_model) for _ in range(4)]
            )
        else:
            self.memory_store_gate = None
            self.memory_store_projections = None
        self.action_head = nn.Linear(d_model, num_actions)
        self.memory_action_head = (
            nn.Linear(d_model, num_actions)
            if use_memory_action_readout
            else None
        )
        self.planner_head = nn.Linear(d_model, num_actions) if use_planner else None
        if use_orchestration:
            self.agent_policy_heads = nn.ModuleList(
                [nn.Linear(d_model, num_actions) for _ in range(3)]
            )
            self.agent_critic = nn.Linear(d_model, 3)
        else:
            self.agent_policy_heads = None
            self.agent_critic = None
        self.next_observation_head = (
            nn.Linear(d_model, backbone.config.vocab_size)
            if use_world_model
            else None
        )
        self.reward_head = nn.Linear(d_model, 2) if use_reward_model else None
        self.reflection_head = nn.Linear(d_model, 2) if use_reflection else None

    def forward(
        self,
        batch: AgenticControlBatch,
        *,
        identity_states: Optional[list[IdentityState]] = None,
        mode: str = "carry",
        world_loss_weight: float = 0.0,
        reward_loss_weight: float = 0.0,
        reflection_loss_weight: float = 0.0,
        memory_action_loss_weight: float = 0.0,
        planner_loss_weight: float = 0.0,
        orchestration_loss_weight: float = 0.0,
    ) -> AgenticControllerOutput:
        context_states = identity_states
        if context_states is None:
            context_output = self.backbone(batch.context_inputs)
            context_states = context_output.identity_states
        query_states = self._intervention_states(context_states, mode)
        query_output = self.backbone(
            batch.query_inputs,
            identity_states=query_states,
        )
        decision_hidden = query_output.hidden_states[:, batch.decision_index, :]
        memory_vector = self._memory_vector(
            batch,
            context_states if mode == "carry" else query_states,
        )
        memory_vector = self._apply_memory_stores(memory_vector)
        decision_hidden = self._apply_recurrent_state(
            decision_hidden,
            query_output.hidden_states,
            batch.decision_index,
        )
        decision_hidden = self._adapt_decision_hidden(decision_hidden, memory_vector)
        decision_hidden = self._apply_modular_cognition(decision_hidden)

        action_logits = self.action_head(decision_hidden)
        memory_action_logits = None
        if self.memory_action_head is not None and memory_vector is not None:
            memory_action_logits = self.memory_action_head(memory_vector)
            action_logits = action_logits + self.memory_action_weight * memory_action_logits
        planner_logits = None
        if self.planner_head is not None:
            planner_logits = self.planner_head(decision_hidden)
            action_logits = action_logits + self.planner_weight * planner_logits
        orchestration_logits = None
        if self.agent_policy_heads is not None and self.agent_critic is not None:
            policy_logits = torch.stack(
                [head(decision_hidden) for head in self.agent_policy_heads],
                dim=1,
            )
            critic_weights = F.softmax(self.agent_critic(decision_hidden), dim=-1)
            orchestration_logits = torch.einsum(
                "ba,ban->bn",
                critic_weights,
                policy_logits,
            )
            action_logits = action_logits + self.orchestration_weight * orchestration_logits
        next_logits = (
            self.next_observation_head(decision_hidden)
            if self.next_observation_head is not None
            else None
        )
        reward_logits = (
            self.reward_head(decision_hidden) if self.reward_head is not None else None
        )
        reflection_logits = (
            self.reflection_head(decision_hidden)
            if self.reflection_head is not None
            else None
        )

        losses = {"action": F.cross_entropy(action_logits, batch.action_targets)}
        total_loss = losses["action"]
        if memory_action_logits is not None:
            losses["memory_action"] = F.cross_entropy(
                memory_action_logits,
                batch.action_targets,
            )
            total_loss = total_loss + memory_action_loss_weight * losses["memory_action"]
        if planner_logits is not None:
            losses["planning"] = F.cross_entropy(planner_logits, batch.action_targets)
            total_loss = total_loss + planner_loss_weight * losses["planning"]
        if orchestration_logits is not None:
            losses["orchestration"] = F.cross_entropy(
                orchestration_logits,
                batch.action_targets,
            )
            total_loss = total_loss + orchestration_loss_weight * losses["orchestration"]
        if next_logits is not None:
            losses["world"] = F.cross_entropy(
                next_logits,
                batch.next_observation_targets,
            )
            total_loss = total_loss + world_loss_weight * losses["world"]
        if reward_logits is not None:
            losses["reward"] = F.cross_entropy(reward_logits, batch.reward_targets)
            total_loss = total_loss + reward_loss_weight * losses["reward"]
        if reflection_logits is not None:
            losses["reflection"] = F.cross_entropy(
                reflection_logits,
                batch.reflection_targets,
            )
            total_loss = total_loss + reflection_loss_weight * losses["reflection"]

        return AgenticControllerOutput(
            action_logits=action_logits,
            next_observation_logits=next_logits,
            reward_logits=reward_logits,
            reflection_logits=reflection_logits,
            identity_states=query_output.identity_states,
            loss=total_loss,
            losses=losses,
        )

    def _adapt_decision_hidden(
        self,
        decision_hidden: Tensor,
        memory_vector: Optional[Tensor],
    ) -> Tensor:
        if (
            not self.use_memory_adapter
            or memory_vector is None
            or self.backbone.memory_adapter is None
        ):
            return decision_hidden
        memory_update = self.backbone.memory_adapter(memory_vector)
        if self.backbone.memory_adapter_gate is not None:
            gate = torch.sigmoid(
                self.backbone.memory_adapter_gate(
                    torch.cat([decision_hidden, memory_vector], dim=-1)
                )
            )
            memory_update = gate * memory_update
        return decision_hidden + self.memory_adapter_weight * memory_update

    def _apply_recurrent_state(
        self,
        decision_hidden: Tensor,
        hidden_states: Tensor,
        decision_index: int,
    ) -> Tensor:
        if self.recurrent_cell is None or self.recurrent_gate is None:
            return decision_hidden
        state = torch.zeros_like(decision_hidden)
        for token_index in range(decision_index + 1):
            state = self.recurrent_cell(hidden_states[:, token_index, :], state)
        gate = torch.sigmoid(self.recurrent_gate(torch.cat([decision_hidden, state], dim=-1)))
        return decision_hidden + gate * state

    def _apply_modular_cognition(self, decision_hidden: Tensor) -> Tensor:
        if self.cognitive_gate is None or self.cognitive_experts is None:
            return decision_hidden
        scores = F.softmax(self.cognitive_gate(decision_hidden), dim=-1)
        top_values, top_indices = torch.topk(scores, k=2, dim=-1)
        expert_outputs = torch.stack(
            [expert(decision_hidden) for expert in self.cognitive_experts],
            dim=1,
        )
        sparse_weights = torch.zeros_like(scores).scatter(
            dim=-1,
            index=top_indices,
            src=top_values,
        )
        sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        update = torch.einsum("be,bed->bd", sparse_weights, expert_outputs)
        return decision_hidden + update

    def _apply_memory_stores(self, memory_vector: Optional[Tensor]) -> Optional[Tensor]:
        if (
            memory_vector is None
            or self.memory_store_gate is None
            or self.memory_store_projections is None
        ):
            return memory_vector
        store_vectors = torch.stack(
            [projection(memory_vector) for projection in self.memory_store_projections],
            dim=1,
        )
        store_weights = F.softmax(self.memory_store_gate(memory_vector), dim=-1)
        return torch.einsum("bs,bsd->bd", store_weights, store_vectors)

    def _memory_vector(
        self,
        batch: AgenticControlBatch,
        identity_states: list[IdentityState],
    ) -> Optional[Tensor]:
        if (
            not isinstance(self.backbone, TACTransformerLM)
            or not identity_states
            or self.backbone.config.memory_read_type
            not in {"program_memory", "content_addressed"}
        ):
            return None
        return self.backbone.memory_read_vector(
            batch.query_inputs[:, batch.key_index],
            identity_states,
        )

    def _intervention_states(
        self,
        states: list[IdentityState],
        mode: str,
    ) -> list[IdentityState]:
        if mode == "carry":
            return states
        if mode == "reset" or not states:
            return []
        if mode != "shuffled":
            raise ValueError("mode must be 'carry', 'reset', or 'shuffled'")
        permutation = torch.arange(states[0].stability.shape[0] - 1, -1, -1, device=states[0].stability.device)
        shuffled = []
        for state in states:
            shuffled.append(
                IdentityState(
                    stability=state.stability[permutation],
                    program_memory=state.program_memory[permutation],
                    stable_program_memory=(
                        state.stable_program_memory[permutation]
                        if state.stable_program_memory is not None
                        else None
                    ),
                    archival_program_memory=(
                        state.archival_program_memory[permutation]
                        if state.archival_program_memory is not None
                        else None
                    ),
                    program_age=(
                        state.program_age[permutation]
                        if state.program_age is not None
                        else None
                    ),
                    program_write_frequency=(
                        state.program_write_frequency[permutation]
                        if state.program_write_frequency is not None
                        else None
                    ),
                    engram_patterns=(
                        state.engram_patterns[permutation]
                        if state.engram_patterns is not None
                        else None
                    ),
                    engram_values=(
                        state.engram_values[permutation]
                        if state.engram_values is not None
                        else None
                    ),
                    engram_mask=(
                        state.engram_mask[permutation]
                        if state.engram_mask is not None
                        else None
                    ),
                    content_cues=(
                        state.content_cues[permutation]
                        if state.content_cues is not None
                        else None
                    ),
                    content_values=(
                        state.content_values[permutation]
                        if state.content_values is not None
                        else None
                    ),
                    content_mask=(
                        state.content_mask[permutation]
                        if state.content_mask is not None
                        else None
                    ),
                )
            )
        return shuffled


def train_agentic_controller(
    model: AgenticController,
    batcher: AgenticControlBatcher,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    world_loss_weight: float = 0.0,
    reward_loss_weight: float = 0.0,
    reflection_loss_weight: float = 0.0,
    memory_action_loss_weight: float = 0.0,
    planner_loss_weight: float = 0.0,
    orchestration_loss_weight: float = 0.0,
    memory_action_contrastive_weight: float = 0.0,
    contrastive_margin: float = 1.0,
    device: str | torch.device = "cpu",
    optimizer_config: Optional[TACOptimizerConfig] = None,
) -> dict[str, float]:
    model.to(device)
    model.train()
    optimizer = build_tac_optimizer(
        model,
        optimizer_config or TACOptimizerConfig(learning_rate=learning_rate),
    )
    started = time.perf_counter()
    last_losses: dict[str, float] = {}
    last_accuracy = 0.0
    for _ in range(steps):
        batch = batcher.next_batch(batch_size=batch_size, device=device)
        output = model(
            batch,
            world_loss_weight=world_loss_weight,
            reward_loss_weight=reward_loss_weight,
            reflection_loss_weight=reflection_loss_weight,
            memory_action_loss_weight=memory_action_loss_weight,
            planner_loss_weight=planner_loss_weight,
            orchestration_loss_weight=orchestration_loss_weight,
        )
        assert output.loss is not None
        loss = output.loss
        contrastive_loss = None
        if memory_action_contrastive_weight > 0.0:
            shuffled_output = model(batch, mode="shuffled")
            contrastive_loss = _correct_action_margin_loss(
                output.action_logits,
                shuffled_output.action_logits,
                batch.action_targets,
                margin=contrastive_margin,
            )
            loss = loss + memory_action_contrastive_weight * contrastive_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_losses = {
            name: float(loss.detach())
            for name, loss in output.losses.items()
        }
        if contrastive_loss is not None:
            last_losses["memory_action_contrastive"] = float(
                contrastive_loss.detach()
            )
        last_accuracy = _action_accuracy(output.action_logits, batch.action_targets)

    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "loss": float(loss.detach()) if steps else 0.0,
        "action_accuracy": last_accuracy,
        "steps": steps,
        "tokens_per_second": steps * batch_size * batcher.seq_len * 2 / elapsed,
        **{f"{name}_loss": value for name, value in last_losses.items()},
    }


@torch.no_grad()
def evaluate_agentic_controller(
    model: AgenticController,
    batcher: AgenticControlBatcher,
    *,
    batches: int,
    batch_size: int,
    mode: str = "carry",
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    model.to(device)
    model.eval()
    correct = 0.0
    total = 0
    world_correct = 0.0
    reward_correct = 0.0
    reflection_correct = 0.0
    started = time.perf_counter()
    for _ in range(batches):
        batch = batcher.next_batch(batch_size=batch_size, device=device)
        output = model(batch, mode=mode)
        correct += float((output.action_logits.argmax(dim=-1) == batch.action_targets).sum())
        total += batch.action_targets.numel()
        if output.next_observation_logits is not None:
            world_correct += float(
                (
                    output.next_observation_logits.argmax(dim=-1)
                    == batch.next_observation_targets
                ).sum()
            )
        if output.reward_logits is not None:
            reward_correct += float(
                (output.reward_logits.argmax(dim=-1) == batch.reward_targets).sum()
            )
        if output.reflection_logits is not None:
            reflection_correct += float(
                (
                    output.reflection_logits.argmax(dim=-1)
                    == batch.reflection_targets
                ).sum()
            )
    elapsed = max(time.perf_counter() - started, 1e-9)
    metrics = {
        "action_accuracy": correct / max(total, 1),
        "tokens_per_second": batches * batch_size * batcher.seq_len * 2 / elapsed,
    }
    if model.next_observation_head is not None:
        metrics["world_accuracy"] = world_correct / max(total, 1)
    if model.reward_head is not None:
        metrics["reward_accuracy"] = reward_correct / max(total, 1)
    if model.reflection_head is not None:
        metrics["reflection_accuracy"] = reflection_correct / max(total, 1)
    return metrics


def benchmark_agentic_control(
    config: TACConfig,
    *,
    num_actions: int,
    steps: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: Optional[int] = None,
    learning_rate: float = 3e-4,
    world_loss_weight: float = 0.0,
    reward_loss_weight: float = 0.0,
    reflection_loss_weight: float = 0.0,
    use_world_model: bool = False,
    use_reward_model: bool = False,
    use_reflection: bool = False,
    use_memory_action_readout: bool = False,
    use_recurrent_state: bool = False,
    use_modular_cognition: bool = False,
    use_memory_stores: bool = False,
    use_planner: bool = False,
    use_orchestration: bool = False,
    memory_action_loss_weight: float = 0.0,
    planner_loss_weight: float = 0.0,
    orchestration_loss_weight: float = 0.0,
    memory_action_contrastive_weight: float = 0.0,
    contrastive_margin: float = 1.0,
    memory_action_weight: float = 1.0,
    planner_weight: float = 1.0,
    orchestration_weight: float = 1.0,
    include_recurrent_baseline: bool = False,
    seed: int = 0,
    device: str | torch.device = "cpu",
    match_baseline_parameters: bool = True,
) -> dict[str, object]:
    torch.manual_seed(seed)
    eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    train_batcher = AgenticControlBatcher(
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
        num_actions=num_actions,
        seed=seed,
    )
    eval_seed = seed + 10_000
    tac = AgenticController(
        TACTransformerLM(config),
        num_actions=num_actions,
        use_world_model=use_world_model,
        use_reward_model=use_reward_model,
        use_reflection=use_reflection,
        use_memory_adapter=True,
        use_memory_action_readout=use_memory_action_readout,
        use_recurrent_state=use_recurrent_state,
        use_modular_cognition=use_modular_cognition,
        use_memory_stores=use_memory_stores,
        use_planner=use_planner,
        use_orchestration=use_orchestration,
        memory_action_weight=memory_action_weight,
        planner_weight=planner_weight,
        orchestration_weight=orchestration_weight,
    )
    baseline_config = (
        parameter_matched_baseline_config(config)
        if match_baseline_parameters
        else config
    )
    baseline = AgenticController(
        VanillaTransformerLM(baseline_config),
        num_actions=num_actions,
        use_world_model=use_world_model,
        use_reward_model=use_reward_model,
        use_reflection=use_reflection,
        use_memory_adapter=False,
        use_memory_action_readout=False,
        use_recurrent_state=use_recurrent_state,
        use_modular_cognition=use_modular_cognition,
        use_memory_stores=False,
        use_planner=use_planner,
        use_orchestration=use_orchestration,
        planner_weight=planner_weight,
        orchestration_weight=orchestration_weight,
    )
    tac_train = train_agentic_controller(
        tac,
        train_batcher,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        world_loss_weight=world_loss_weight,
        reward_loss_weight=reward_loss_weight,
        reflection_loss_weight=reflection_loss_weight,
        memory_action_loss_weight=memory_action_loss_weight,
        planner_loss_weight=planner_loss_weight,
        orchestration_loss_weight=orchestration_loss_weight,
        memory_action_contrastive_weight=memory_action_contrastive_weight,
        contrastive_margin=contrastive_margin,
        device=device,
    )
    baseline_train = train_agentic_controller(
        baseline,
        AgenticControlBatcher(
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            num_actions=num_actions,
            seed=seed,
        ),
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        world_loss_weight=world_loss_weight,
        reward_loss_weight=reward_loss_weight,
        reflection_loss_weight=reflection_loss_weight,
        memory_action_loss_weight=0.0,
        planner_loss_weight=planner_loss_weight,
        orchestration_loss_weight=orchestration_loss_weight,
        memory_action_contrastive_weight=0.0,
        contrastive_margin=contrastive_margin,
        device=device,
    )

    tac_eval_batcher = AgenticControlBatcher(
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
        num_actions=num_actions,
        seed=eval_seed,
    )
    baseline_eval_batcher = AgenticControlBatcher(
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
        num_actions=num_actions,
        seed=eval_seed,
    )
    tac_carry = evaluate_agentic_controller(
        tac,
        tac_eval_batcher,
        batches=eval_batches,
        batch_size=eval_batch_size,
        mode="carry",
        device=device,
    )
    tac_reset = evaluate_agentic_controller(
        tac,
        AgenticControlBatcher(
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            num_actions=num_actions,
            seed=eval_seed,
        ),
        batches=eval_batches,
        batch_size=eval_batch_size,
        mode="reset",
        device=device,
    )
    tac_shuffled = evaluate_agentic_controller(
        tac,
        AgenticControlBatcher(
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            num_actions=num_actions,
            seed=eval_seed,
        ),
        batches=eval_batches,
        batch_size=eval_batch_size,
        mode="shuffled",
        device=device,
    )
    baseline_carry = evaluate_agentic_controller(
        baseline,
        baseline_eval_batcher,
        batches=eval_batches,
        batch_size=eval_batch_size,
        mode="carry",
        device=device,
    )
    decision = {
        "status": (
            "effective"
            if tac_carry["action_accuracy"] > tac_reset["action_accuracy"]
            and tac_carry["action_accuracy"] > tac_shuffled["action_accuracy"]
            and tac_carry["action_accuracy"] >= baseline_carry["action_accuracy"]
            else "inconclusive"
        ),
        "action_accuracy_delta": tac_carry["action_accuracy"] - tac_reset["action_accuracy"],
        "shuffled_action_penalty": tac_carry["action_accuracy"] - tac_shuffled["action_accuracy"],
        "baseline_action_gap": tac_carry["action_accuracy"] - baseline_carry["action_accuracy"],
    }
    result: dict[str, object] = {
        "config": asdict(config),
        "baseline_config": asdict(baseline_config),
        "num_actions": num_actions,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "objectives": {
            "world_model": use_world_model,
            "reward_model": use_reward_model,
            "reflection": use_reflection,
            "memory_action_readout": use_memory_action_readout,
            "recurrent_state": use_recurrent_state,
            "modular_cognition": use_modular_cognition,
            "memory_stores": use_memory_stores,
            "planner": use_planner,
            "orchestration": use_orchestration,
            "world_loss_weight": world_loss_weight,
            "reward_loss_weight": reward_loss_weight,
            "reflection_loss_weight": reflection_loss_weight,
            "memory_action_loss_weight": memory_action_loss_weight,
            "planner_loss_weight": planner_loss_weight,
            "orchestration_loss_weight": orchestration_loss_weight,
            "memory_action_contrastive_weight": memory_action_contrastive_weight,
            "contrastive_margin": contrastive_margin,
            "memory_action_weight": memory_action_weight,
            "planner_weight": planner_weight,
            "orchestration_weight": orchestration_weight,
        },
        "decision": decision,
        "tac": {
            "parameter_counts": count_parameters(tac),
            "train": tac_train,
            "eval": {
                "carry": tac_carry,
                "reset": tac_reset,
                "shuffled": tac_shuffled,
            },
        },
        "baseline": {
            "parameter_counts": count_parameters(baseline),
            "train": baseline_train,
            "eval": {"carry": baseline_carry},
        },
    }
    if include_recurrent_baseline:
        recurrent = RecurrentAgenticBaseline(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_actions=num_actions,
        )
        recurrent_train = train_recurrent_agentic_baseline(
            recurrent,
            AgenticControlBatcher(
                vocab_size=config.vocab_size,
                seq_len=config.max_seq_len,
                num_actions=num_actions,
                seed=seed,
            ),
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )
        recurrent_carry = evaluate_recurrent_agentic_baseline(
            recurrent,
            AgenticControlBatcher(
                vocab_size=config.vocab_size,
                seq_len=config.max_seq_len,
                num_actions=num_actions,
                seed=eval_seed,
            ),
            batches=eval_batches,
            batch_size=eval_batch_size,
            mode="carry",
            device=device,
        )
        recurrent_reset = evaluate_recurrent_agentic_baseline(
            recurrent,
            AgenticControlBatcher(
                vocab_size=config.vocab_size,
                seq_len=config.max_seq_len,
                num_actions=num_actions,
                seed=eval_seed,
            ),
            batches=eval_batches,
            batch_size=eval_batch_size,
            mode="reset",
            device=device,
        )
        recurrent_shuffled = evaluate_recurrent_agentic_baseline(
            recurrent,
            AgenticControlBatcher(
                vocab_size=config.vocab_size,
                seq_len=config.max_seq_len,
                num_actions=num_actions,
                seed=eval_seed,
            ),
            batches=eval_batches,
            batch_size=eval_batch_size,
            mode="shuffled",
            device=device,
        )
        result["recurrent_baseline"] = {
            "parameter_counts": count_parameters(recurrent),
            "train": recurrent_train,
            "eval": {
                "carry": recurrent_carry,
                "reset": recurrent_reset,
                "shuffled": recurrent_shuffled,
            },
        }
    return result


def _action_accuracy(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().detach())


def _correct_action_margin_loss(
    carry_logits: Tensor,
    shuffled_logits: Tensor,
    targets: Tensor,
    *,
    margin: float,
) -> Tensor:
    carry_score = carry_logits.gather(dim=-1, index=targets[:, None]).squeeze(-1)
    shuffled_score = shuffled_logits.gather(dim=-1, index=targets[:, None]).squeeze(-1)
    return F.relu(margin - carry_score + shuffled_score).mean()


class RecurrentAgenticBaseline(nn.Module):
    """Small recurrent baseline that carries context state into the query."""

    def __init__(self, *, vocab_size: int, d_model: int, num_actions: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.GRU(d_model, d_model, batch_first=True)
        self.action_head = nn.Linear(d_model, num_actions)

    def forward(self, batch: AgenticControlBatch, *, mode: str = "carry") -> Tensor:
        context_embeddings = self.token_embedding(batch.context_inputs)
        _, context_state = self.encoder(context_embeddings)
        if mode == "reset":
            context_state = torch.zeros_like(context_state)
        elif mode == "shuffled":
            permutation = torch.arange(
                context_state.shape[1] - 1,
                -1,
                -1,
                device=context_state.device,
            )
            context_state = context_state[:, permutation, :]
        elif mode != "carry":
            raise ValueError("mode must be 'carry', 'reset', or 'shuffled'")
        query_embeddings = self.token_embedding(batch.query_inputs)
        query_hidden, _ = self.encoder(query_embeddings, context_state)
        decision_hidden = query_hidden[:, batch.decision_index, :]
        return self.action_head(decision_hidden)


def train_recurrent_agentic_baseline(
    model: RecurrentAgenticBaseline,
    batcher: AgenticControlBatcher,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: str | torch.device = "cpu",
    optimizer_config: Optional[TACOptimizerConfig] = None,
) -> dict[str, float]:
    model.to(device)
    model.train()
    optimizer = build_tac_optimizer(
        model,
        optimizer_config or TACOptimizerConfig(learning_rate=learning_rate),
    )
    started = time.perf_counter()
    loss = torch.tensor(0.0, device=device)
    accuracy = 0.0
    for _ in range(steps):
        batch = batcher.next_batch(batch_size=batch_size, device=device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.action_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        accuracy = _action_accuracy(logits, batch.action_targets)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "loss": float(loss.detach()) if steps else 0.0,
        "action_accuracy": accuracy,
        "steps": steps,
        "tokens_per_second": steps * batch_size * batcher.seq_len * 2 / elapsed,
    }


@torch.no_grad()
def evaluate_recurrent_agentic_baseline(
    model: RecurrentAgenticBaseline,
    batcher: AgenticControlBatcher,
    *,
    batches: int,
    batch_size: int,
    mode: str,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    model.to(device)
    model.eval()
    correct = 0.0
    total = 0
    started = time.perf_counter()
    for _ in range(batches):
        batch = batcher.next_batch(batch_size=batch_size, device=device)
        logits = model(batch, mode=mode)
        correct += float((logits.argmax(dim=-1) == batch.action_targets).sum())
        total += batch.action_targets.numel()
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "action_accuracy": correct / max(total, 1),
        "tokens_per_second": batches * batch_size * batcher.seq_len * 2 / elapsed,
    }
