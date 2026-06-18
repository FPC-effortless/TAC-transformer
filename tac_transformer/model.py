from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import sqrt
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


AUTHORITY_MODE_NAMES = (
    "exact_memory",
    "proposal_verified",
    "calibrated_fast_path",
    "fresh_repair",
    "system2_verify",
)
AUTHORITY_MODE_TO_INDEX = {
    mode_name: mode_index for mode_index, mode_name in enumerate(AUTHORITY_MODE_NAMES)
}
AUTHORITY_EXACT_MEMORY_INDEX = AUTHORITY_MODE_TO_INDEX["exact_memory"]
AUTHORITY_PROPOSAL_VERIFIED_INDEX = AUTHORITY_MODE_TO_INDEX["proposal_verified"]
AUTHORITY_CALIBRATED_FAST_PATH_INDEX = AUTHORITY_MODE_TO_INDEX["calibrated_fast_path"]
AUTHORITY_FRESH_REPAIR_INDEX = AUTHORITY_MODE_TO_INDEX["fresh_repair"]
AUTHORITY_SYSTEM2_VERIFY_INDEX = AUTHORITY_MODE_TO_INDEX["system2_verify"]
AUTHORITY_FEATURE_DIM = 6


class ContentWritePolicy(str, Enum):
    DENSE = "dense"
    DISABLED = "disabled"
    QUERY_SKIP = "query_skip"
    MASKED_PREFILL_QUERY_SKIP = "masked_prefill_query_skip"
    DECODE_STATE_SKIP = "decode_state_skip"


@dataclass(frozen=True)
class TACConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: Optional[int] = None
    n_layers: int = 2
    n_programs: int = 16
    max_seq_len: int = 256
    beta: float = 1.0
    energy_budget: float = 4.0
    state_decay: float = 0.8
    n_sink_programs: int = 0
    mlp_ratio: int = 4
    dropout: float = 0.0
    causal: bool = True
    norm_type: str = "layernorm"
    mlp_type: str = "gelu"
    position_type: str = "learned"
    rope_base: float = 10000.0
    rope_scale: float = 1.0
    rope_scaling_type: str = "none"
    original_context_length: Optional[int] = None
    target_context_length: Optional[int] = None
    program_compute_type: str = "embedding"
    program_expert_rank: Optional[int] = None
    routing_type: str = "energy"
    routing_top_k: int = 1
    program_activation_type: str = "sigmoid"
    decision_continuity_strength: float = 1.0
    decision_continuity_decay: float = 0.8
    state_update_type: str = "fixed"
    memory_write_type: str = "standard"
    memory_system_type: str = "flat"
    memory_retention_rate: float = 0.85
    memory_consolidation_rate: float = 0.25
    procedural_memory_rate: float = 0.20
    memory_bridge_type: str = "none"
    memory_bridge_weight: float = 1.0
    memory_tier_type: str = "flat"
    memory_lookup_type: str = "none"
    memory_lookup_slots: int = 64
    memory_read_type: str = "none"
    program_memory_update_type: str = "shared"
    pattern_store_size: int = 4
    content_store_size: int = 8
    content_read_steps: int = 1
    content_read_gate_type: str = "learned"
    content_read_confidence_margin: float = 0.05
    content_read_cue_match_threshold: float = 0.65
    content_read_query_top_k: Optional[int] = None
    coalition_context_type: str = "none"
    coalition_context_scale: float = 0.1
    memory_adapter_type: str = "none"
    program_residual_scale: float = 1.0
    coherence_attention_scale: float = 1.0
    memory_allocation_type: str = "stability"
    memory_allocation_k: int = 1
    creb_alpha: float = 1.0
    creb_beta: float = 1.0
    creb_gamma: float = 0.25
    creb_delta: float = 0.0
    creb_frequency_decay: float = 0.9
    identity_attention_type: str = "none"
    attention_window_size: Optional[int] = None
    residual_stream_type: str = "single"
    sequence_mixer_type: str = "attention"
    state_mixer_kernel_size: int = 4
    n_prediction_heads: int = 1
    multi_token_loss_weight: float = 0.0
    memory_separation_weight: float = 0.0
    content_cue_separation_weight: float = 0.0
    content_gate_entropy_weight: float = 0.0
    routing_load_balance_weight: float = 0.0
    decision_continuity_loss_weight: float = 0.05
    semantic_route_allowed_programs: Optional[tuple[int, ...]] = None
    semantic_route_suppressed_programs: Optional[tuple[int, ...]] = None
    authority_trusted_threshold: float = 0.95
    memory_reconsolidate: bool = False
    reconsolidate_gate_type: str = "linear"
    content_reconsolidate: bool = False
    content_reconsolidate_rate: float = 0.1
    detach_identity_state: bool = True
    # run5b_plus: TAC-218 decision memory (3D shape)
    program_embed_dim: Optional[int] = None
    # run5b_plus: EBM data energy head
    ebm_head_hidden_dim: int = 128
    # run5b_plus: identity compression losses
    activation_l1_weight: float = 0.0
    identity_norm_floor_weight: float = 0.0
    identity_norm_floor_threshold: float = 0.13
    # run5b_plus: hybrid sliding window — token IDs that always get global attention
    global_attention_token_ids: Optional[tuple[int, ...]] = None
    lm_readout_type: str = "hidden"
    tac_active_layer_start: int = 0


@dataclass
class IdentityState:
    stability: Tensor
    program_memory: Tensor
    working_state: Optional[Tensor] = None
    episodic_state: Optional[Tensor] = None
    semantic_state: Optional[Tensor] = None
    procedural_state: Optional[Tensor] = None
    memory_confidence: Optional[Tensor] = None
    decision_memory: Optional[Tensor] = None
    stable_program_memory: Optional[Tensor] = None
    archival_program_memory: Optional[Tensor] = None
    program_age: Optional[Tensor] = None
    program_write_frequency: Optional[Tensor] = None
    engram_patterns: Optional[Tensor] = None
    engram_values: Optional[Tensor] = None
    engram_mask: Optional[Tensor] = None
    content_cues: Optional[Tensor] = None
    content_values: Optional[Tensor] = None
    content_mask: Optional[Tensor] = None
    content_cue_token_ids: Optional[Tensor] = None
    content_value_token_ids: Optional[Tensor] = None
    # run5b_plus: 3D decision memory for TAC-218 (shape: [batch, n_programs, program_embed_dim])
    decision_memory_ebm: Optional[Tensor] = None


def _program_expert_rank(config: TACConfig) -> int:
    if config.program_expert_rank is not None:
        return int(config.program_expert_rank)
    return max(1, config.d_model // 4)


@dataclass
class IdentityFieldOutput:
    coherence: Tensor
    activations: Tensor
    program_assignments: Tensor
    program_identity: Tensor
    selected_program_mask: Tensor
    used_energy: Tensor
    program_context: Tensor
    state: IdentityState
    losses: dict[str, Tensor]
    metrics: dict[str, Tensor]
    token_activations: Optional[Tensor] = None
    token_selected_program_mask: Optional[Tensor] = None
    authority_logits: Optional[Tensor] = None
    authority_probs: Optional[Tensor] = None
    authority_indices: Optional[Tensor] = None
    verifier_required: Optional[Tensor] = None
    halt_probability: Optional[Tensor] = None
    token_authority_logits: Optional[Tensor] = None
    token_authority_probs: Optional[Tensor] = None
    token_verifier_required: Optional[Tensor] = None


@dataclass
class AttentionOutput:
    hidden: Tensor
    attention_probs: Tensor
    attention_logits: Tensor


@dataclass
class TACAuxiliaryOutput:
    coherence: Tensor
    program_activations: Tensor
    selected_program_mask: Tensor
    used_energy: Tensor
    attention_probs: Tensor
    losses: dict[str, Tensor]
    metrics: dict[str, Tensor]
    token_program_activations: Optional[Tensor] = None
    token_selected_program_mask: Optional[Tensor] = None
    authority_logits: Optional[Tensor] = None
    authority_probs: Optional[Tensor] = None
    authority_indices: Optional[Tensor] = None
    verifier_required: Optional[Tensor] = None
    halt_probability: Optional[Tensor] = None
    token_authority_logits: Optional[Tensor] = None
    token_authority_probs: Optional[Tensor] = None
    token_verifier_required: Optional[Tensor] = None
    # run5b_plus: scalar EBM energy per token [batch, seq]
    data_energy: Optional[Tensor] = None


@dataclass
class TACOutput:
    logits: Tensor
    identity_states: list[IdentityState]
    aux: TACAuxiliaryOutput
    loss: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    multi_token_logits: Optional[list[Tensor]] = None


class RMSNorm(nn.Module):
    """Root-mean-square normalization without mean centering."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, hidden: Tensor) -> Tensor:
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(variance + self.eps) * self.weight


class GELUFeedForward(nn.Module):
    def __init__(self, config: TACConfig):
        super().__init__()
        hidden_dim = config.d_model * config.mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.net(hidden)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: TACConfig):
        super().__init__()
        hidden_dim = config.d_model * config.mlp_ratio
        self.up_gate = nn.Linear(config.d_model, hidden_dim * 2)
        self.down = nn.Linear(hidden_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, value = self.up_gate(hidden).chunk(2, dim=-1)
        hidden = F.silu(gate) * value
        hidden = self.dropout(hidden)
        hidden = self.down(hidden)
        return self.dropout(hidden)


class CausalStateMixer(nn.Module):
    """Small causal state-space-style mixer used for hybrid ablations."""

    def __init__(self, config: TACConfig):
        super().__init__()
        if config.state_mixer_kernel_size < 1:
            raise ValueError("state_mixer_kernel_size must be at least 1")
        self.in_proj = nn.Linear(config.d_model, config.d_model * 2)
        self.depthwise = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.state_mixer_kernel_size,
            groups=config.d_model,
            padding=config.state_mixer_kernel_size - 1,
        )
        self.decay_logit = nn.Parameter(torch.zeros(config.d_model))
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        gate, value = self.in_proj(hidden).chunk(2, dim=-1)
        convolved = self.depthwise(value.transpose(1, 2))
        convolved = convolved[..., : hidden.shape[1]].transpose(1, 2)
        decay = torch.sigmoid(self.decay_logit)[None, :]
        state = hidden.new_zeros(hidden.shape[0], hidden.shape[-1])
        outputs = []
        for step in range(hidden.shape[1]):
            state = decay * state + (1.0 - decay) * convolved[:, step, :]
            outputs.append(state * F.silu(gate[:, step, :]))
        mixed = torch.stack(outputs, dim=1)
        return self.dropout(self.out_proj(mixed))


class SelectiveStateMixer(nn.Module):
    """Mamba-inspired input-selective recurrent mixer in pure PyTorch."""

    def __init__(self, config: TACConfig):
        super().__init__()
        self.in_proj = nn.Linear(config.d_model, config.d_model * 3)
        self.state_decay = nn.Parameter(torch.zeros(config.d_model))
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        delta, candidate, gate = self.in_proj(hidden).chunk(3, dim=-1)
        delta = torch.sigmoid(delta)
        candidate = torch.tanh(candidate)
        gate = torch.sigmoid(gate)
        base_decay = torch.sigmoid(self.state_decay)[None, :]
        state = hidden.new_zeros(hidden.shape[0], hidden.shape[-1])
        outputs = []
        for step in range(hidden.shape[1]):
            decay = base_decay * (1.0 - delta[:, step, :])
            write = delta[:, step, :] * candidate[:, step, :]
            state = decay * state + write
            outputs.append(gate[:, step, :] * state)
        return self.dropout(self.out_proj(torch.stack(outputs, dim=1)))


class RWKVTimeMixer(nn.Module):
    """RWKV-inspired time-mix recurrent weighted value path."""

    def __init__(self, config: TACConfig):
        super().__init__()
        self.time_mix_key = nn.Parameter(torch.full((1, 1, config.d_model), 0.5))
        self.time_mix_value = nn.Parameter(torch.full((1, 1, config.d_model), 0.5))
        self.time_mix_receptance = nn.Parameter(torch.full((1, 1, config.d_model), 0.5))
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.receptance = nn.Linear(config.d_model, config.d_model)
        self.time_decay = nn.Parameter(torch.zeros(config.d_model))
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        shifted = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], dim=1)
        mixed_key = hidden * self.time_mix_key + shifted * (1.0 - self.time_mix_key)
        mixed_value = hidden * self.time_mix_value + shifted * (1.0 - self.time_mix_value)
        mixed_receptance = (
            hidden * self.time_mix_receptance
            + shifted * (1.0 - self.time_mix_receptance)
        )
        key = torch.sigmoid(self.key(mixed_key))
        value = self.value(mixed_value)
        receptance = torch.sigmoid(self.receptance(mixed_receptance))
        decay = torch.sigmoid(self.time_decay)[None, :]
        state = hidden.new_zeros(hidden.shape[0], hidden.shape[-1])
        outputs = []
        for step in range(hidden.shape[1]):
            state = decay * state + (1.0 - decay) * key[:, step, :] * value[:, step, :]
            outputs.append(receptance[:, step, :] * state)
        return self.dropout(self.out_proj(torch.stack(outputs, dim=1)))


class XLSTMStyleMixer(nn.Module):
    """xLSTM-inspired gated recurrent mixer with exponential gates."""

    def __init__(self, config: TACConfig):
        super().__init__()
        self.gates = nn.Linear(config.d_model * 2, config.d_model * 4)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        shifted = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], dim=1)
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat([hidden, shifted], dim=-1)
        ).chunk(4, dim=-1)
        input_gate = torch.exp(torch.clamp(input_gate, max=4.0))
        forget_gate = torch.exp(torch.clamp(forget_gate, max=4.0))
        normalizer = (input_gate + forget_gate).clamp_min(1e-6)
        input_gate = input_gate / normalizer
        forget_gate = forget_gate / normalizer
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        state = hidden.new_zeros(hidden.shape[0], hidden.shape[-1])
        outputs = []
        for step in range(hidden.shape[1]):
            state = forget_gate[:, step, :] * state + input_gate[:, step, :] * candidate[:, step, :]
            outputs.append(output_gate[:, step, :] * state)
        return self.dropout(self.out_proj(torch.stack(outputs, dim=1)))


class DecisionContinuityHead(nn.Module):
    """TAC-218: Projects 3D decision_memory_ebm into routing logit space.

    decision_memory_ebm: [batch, n_programs, program_embed_dim]
    Collapses program dim via mean → linear projection → routing logit space [batch, n_programs].
    Shape contract: prev_projected.shape == curr_routing_logits.shape at runtime.
    """

    def __init__(self, program_embed_dim: int, n_programs: int):
        super().__init__()
        self.proj = nn.Linear(program_embed_dim, n_programs, bias=False)

    def forward(
        self,
        prev_decision_memory_ebm: Tensor,
        curr_routing_logits: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # prev_decision_memory_ebm: [batch, n_programs, program_embed_dim]
        # curr_routing_logits:      [batch, n_programs]
        prev_summary = prev_decision_memory_ebm.mean(dim=1)   # [batch, program_embed_dim]
        prev_projected = self.proj(prev_summary)               # [batch, n_programs]
        assert prev_projected.shape == curr_routing_logits.shape, (
            f"DecisionContinuityHead shape mismatch: "
            f"prev_projected {prev_projected.shape} vs "
            f"curr_routing_logits {curr_routing_logits.shape}"
        )
        prev_dist = F.softmax(prev_projected, dim=-1)
        curr_dist = F.softmax(curr_routing_logits, dim=-1)
        return prev_dist, curr_dist


class DataEnergyHead(nn.Module):
    """EBM head: assigns a scalar energy to each (hidden, identity) pair.

    Lower energy = model believes the pair is a valid (clean) example.
    Training: E(clean) < E(corrupt) + margin via hinge loss.
    CRITICAL: always call with hidden_state.detach() and selected_identity.detach()
    to prevent the contrastive loss from shaping backbone representations.
    """

    def __init__(self, d_model: int, program_embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model + program_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_state: Tensor, selected_identity_state: Tensor) -> Tensor:
        # hidden_state:            [batch, seq, d_model]
        # selected_identity_state: [batch, seq, program_embed_dim]
        features = torch.cat([hidden_state, selected_identity_state], dim=-1)
        return self.proj(features).squeeze(-1)  # [batch, seq]


def _validate_program_embed_dim(config: TACConfig) -> None:
    if config.program_embed_dim is None:
        return
    if config.program_embed_dim < 1:
        raise ValueError("program_embed_dim must be positive when set")
    if config.program_embed_dim > config.d_model:
        raise ValueError("program_embed_dim must be <= d_model")


class IdentityFieldLayer(nn.Module):
    """Persistent executable program layer that runs beside attention."""

    def __init__(self, config: TACConfig):
        super().__init__()
        self.config = config
        _validate_program_embed_dim(config)
        if config.program_compute_type not in {
            "embedding",
            "linear_expert",
            "sparse_linear_expert",
            "low_rank_linear_expert",
        }:
            raise ValueError(
                "program_compute_type must be 'embedding', 'linear_expert', 'sparse_linear_expert', or 'low_rank_linear_expert'"
            )
        if config.program_expert_rank is not None:
            if config.program_expert_rank < 1:
                raise ValueError("program_expert_rank must be positive when set")
            if config.program_expert_rank > config.d_model:
                raise ValueError("program_expert_rank must be <= d_model")
        if config.routing_type not in {
            "energy",
            "expert_choice",
            "base",
            "hash",
            "sparse_ensemble",
            "base_semantic",
            "base_semantic_soft",
            "authority_gated",
        }:
            raise ValueError(
                "routing_type must be 'energy', 'expert_choice', 'base', 'hash', 'sparse_ensemble', 'base_semantic', 'base_semantic_soft', or 'authority_gated'"
            )
        if config.routing_top_k < 1:
            raise ValueError("routing_top_k must be at least 1")
        if config.routing_top_k > config.n_programs:
            raise ValueError("routing_top_k must be <= n_programs")
        if config.decision_continuity_strength < 0.0:
            raise ValueError("decision_continuity_strength must be non-negative")
        if not 0.0 <= config.decision_continuity_decay <= 1.0:
            raise ValueError("decision_continuity_decay must be between 0 and 1")
        if config.program_activation_type not in {"sigmoid", "relu", "softplus"}:
            raise ValueError(
                "program_activation_type must be 'sigmoid', 'relu', or 'softplus'"
            )
        if config.state_update_type not in {"fixed", "gated"}:
            raise ValueError("state_update_type must be 'fixed' or 'gated'")
        if config.memory_write_type not in {"standard", "novelty_gated", "hebbian_outer"}:
            raise ValueError(
                "memory_write_type must be 'standard', 'novelty_gated', or 'hebbian_outer'"
            )
        if config.memory_system_type not in {"flat", "multi_timescale"}:
            raise ValueError(
                "memory_system_type must be 'flat' or 'multi_timescale'"
            )
        if not 0.0 <= config.memory_retention_rate <= 1.0:
            raise ValueError("memory_retention_rate must be between 0 and 1")
        if not 0.0 <= config.memory_consolidation_rate <= 1.0:
            raise ValueError("memory_consolidation_rate must be between 0 and 1")
        if not 0.0 <= config.procedural_memory_rate <= 1.0:
            raise ValueError("procedural_memory_rate must be between 0 and 1")
        if config.memory_tier_type not in {"flat", "hierarchical"}:
            raise ValueError("memory_tier_type must be 'flat' or 'hierarchical'")
        if config.memory_lookup_type not in {"none", "product_key"}:
            raise ValueError("memory_lookup_type must be 'none' or 'product_key'")
        if config.memory_lookup_slots < 1:
            raise ValueError("memory_lookup_slots must be at least 1")
        if config.pattern_store_size < 1:
            raise ValueError("pattern_store_size must be at least 1")
        if config.content_store_size < 1:
            raise ValueError("content_store_size must be at least 1")
        if config.content_read_steps < 1:
            raise ValueError("content_read_steps must be at least 1")
        if config.content_read_gate_type not in {
            "learned",
            "confidence",
            "confidence_margin",
            "cue_match",
            "synthesis",
        }:
            raise ValueError(
                "content_read_gate_type must be 'learned', 'confidence', 'confidence_margin', 'cue_match', or 'synthesis'"
            )
        if config.content_read_confidence_margin < 0.0:
            raise ValueError("content_read_confidence_margin must be non-negative")
        if config.content_read_cue_match_threshold < 0.0:
            raise ValueError("content_read_cue_match_threshold must be non-negative")
        if (
            config.content_read_query_top_k is not None
            and config.content_read_query_top_k < 1
        ):
            raise ValueError("content_read_query_top_k must be positive when set")
        if config.coalition_context_type not in {
            "none",
            "program_memory",
            "program_memory_graph",
            "program_memory_task_graph",
        }:
            raise ValueError(
                "coalition_context_type must be 'none', 'program_memory', 'program_memory_graph', or 'program_memory_task_graph'"
            )
        if config.coalition_context_scale < 0.0:
            raise ValueError("coalition_context_scale must be non-negative")
        if config.program_memory_update_type not in {"shared", "program_conditioned"}:
            raise ValueError(
                "program_memory_update_type must be 'shared' or 'program_conditioned'"
            )
        if config.program_residual_scale < 0.0:
            raise ValueError("program_residual_scale must be non-negative")
        if config.coherence_attention_scale < 0.0:
            raise ValueError("coherence_attention_scale must be non-negative")
        if config.memory_allocation_type not in {"stability", "creb"}:
            raise ValueError("memory_allocation_type must be 'stability' or 'creb'")
        if config.memory_allocation_k < 1:
            raise ValueError("memory_allocation_k must be at least 1")
        if config.creb_delta < 0.0:
            raise ValueError("creb_delta must be non-negative")
        if not 0.0 <= config.creb_frequency_decay <= 1.0:
            raise ValueError("creb_frequency_decay must be between 0 and 1")
        if config.reconsolidate_gate_type not in {"linear", "mlp"}:
            raise ValueError("reconsolidate_gate_type must be 'linear' or 'mlp'")
        if config.memory_separation_weight < 0.0:
            raise ValueError("memory_separation_weight must be non-negative")
        if config.content_cue_separation_weight < 0.0:
            raise ValueError("content_cue_separation_weight must be non-negative")
        if config.content_gate_entropy_weight < 0.0:
            raise ValueError("content_gate_entropy_weight must be non-negative")
        if config.routing_load_balance_weight < 0.0:
            raise ValueError("routing_load_balance_weight must be non-negative")
        if config.decision_continuity_loss_weight < 0.0:
            raise ValueError("decision_continuity_loss_weight must be non-negative")
        if not 0.0 <= config.authority_trusted_threshold <= 1.0:
            raise ValueError("authority_trusted_threshold must be between 0 and 1")
        if not 0.0 <= config.content_reconsolidate_rate <= 1.0:
            raise ValueError("content_reconsolidate_rate must be between 0 and 1")
        if config.n_sink_programs < 0 or config.n_sink_programs > config.n_programs:
            raise ValueError("n_sink_programs must be between 0 and n_programs")
        self._validate_program_filter(
            config.semantic_route_allowed_programs,
            "semantic_route_allowed_programs",
            allow_empty=False,
        )
        self._validate_program_filter(
            config.semantic_route_suppressed_programs,
            "semantic_route_suppressed_programs",
            allow_empty=True,
        )

        self.program_embeddings = nn.Parameter(
            torch.empty(config.n_programs, config.d_model)
        )
        nn.init.normal_(self.program_embeddings, mean=0.0, std=0.02)

        initial_costs = torch.linspace(0.7, 1.25, config.n_programs)
        self.raw_energy_costs = nn.Parameter(_inverse_softplus(initial_costs))
        self.program_update = nn.Linear(config.d_model, config.d_model)
        if config.program_memory_update_type == "program_conditioned":
            self.program_conditioned_update = nn.Linear(
                config.d_model * 2,
                config.d_model,
            )
        else:
            self.program_conditioned_update = None
        if config.state_update_type == "gated":
            self.stability_gate = nn.Linear(config.d_model, config.n_programs)
            self.memory_gate = nn.Linear(config.d_model, config.n_programs)
        else:
            self.stability_gate = None
            self.memory_gate = None
        if config.memory_write_type == "novelty_gated":
            self.memory_novelty_gate = nn.Linear(config.d_model * 2, 1)
        else:
            self.memory_novelty_gate = None
        if config.routing_type == "authority_gated":
            authority_hidden_dim = max(config.d_model, config.n_programs * 2)
            self.authority_router = nn.Sequential(
                nn.Linear(config.n_programs * 2 + AUTHORITY_FEATURE_DIM, authority_hidden_dim),
                nn.GELU(),
                nn.Linear(authority_hidden_dim, authority_hidden_dim),
                nn.GELU(),
            )
            self.authority_program_head = nn.Linear(authority_hidden_dim, config.n_programs)
            self.authority_mode_head = nn.Linear(authority_hidden_dim, len(AUTHORITY_MODE_NAMES))
            self.authority_halt_head = nn.Linear(authority_hidden_dim, 1)
        else:
            self.authority_router = None
            self.authority_program_head = None
            self.authority_mode_head = None
            self.authority_halt_head = None
        if config.program_compute_type in {"linear_expert", "sparse_linear_expert"}:
            self.program_expert_weight = nn.Parameter(
                torch.empty(config.n_programs, config.d_model, config.d_model)
            )
            self.program_expert_bias = nn.Parameter(
                torch.zeros(config.n_programs, config.d_model)
            )
            nn.init.xavier_uniform_(self.program_expert_weight)
            self.program_expert_down = None
            self.program_expert_up = None
        elif config.program_compute_type == "low_rank_linear_expert":
            expert_rank = _program_expert_rank(config)
            self.program_expert_weight = None
            self.program_expert_down = nn.Parameter(
                torch.empty(config.n_programs, config.d_model, expert_rank)
            )
            self.program_expert_up = nn.Parameter(
                torch.empty(config.n_programs, expert_rank, config.d_model)
            )
            self.program_expert_bias = nn.Parameter(
                torch.zeros(config.n_programs, config.d_model)
            )
            nn.init.xavier_uniform_(self.program_expert_down)
            nn.init.xavier_uniform_(self.program_expert_up)
        else:
            self.program_expert_weight = None
            self.program_expert_down = None
            self.program_expert_up = None
            self.program_expert_bias = None
        if config.coalition_context_type in {
            "program_memory",
            "program_memory_graph",
            "program_memory_task_graph",
        }:
            self.coalition_context_projection = nn.Linear(
                config.d_model,
                config.d_model,
            )
        else:
            self.coalition_context_projection = None
        if config.coalition_context_type in {
            "program_memory_graph",
            "program_memory_task_graph",
        }:
            self.coalition_source_key_projection = nn.Linear(
                config.d_model,
                config.d_model,
            )
            self.coalition_source_value_projection = nn.Linear(
                config.d_model,
                config.d_model,
            )
            self.coalition_target_query_projection = nn.Linear(
                config.d_model,
                config.d_model,
            )
            if config.coalition_context_type == "program_memory_task_graph":
                self.coalition_task_query_projection = nn.Linear(
                    config.d_model,
                    config.d_model,
                )
            else:
                self.coalition_task_query_projection = None
        else:
            self.coalition_source_key_projection = None
            self.coalition_source_value_projection = None
            self.coalition_target_query_projection = None
            self.coalition_task_query_projection = None
        if config.memory_lookup_type == "product_key":
            first_dim = config.d_model // 2
            second_dim = config.d_model - first_dim
            self.memory_lookup_query = nn.Linear(config.d_model, config.d_model)
            self.memory_lookup_key_a = nn.Parameter(
                torch.empty(config.memory_lookup_slots, first_dim)
            )
            self.memory_lookup_key_b = nn.Parameter(
                torch.empty(config.memory_lookup_slots, second_dim)
            )
            self.memory_lookup_values = nn.Parameter(
                torch.empty(config.memory_lookup_slots, config.d_model)
            )
            nn.init.normal_(self.memory_lookup_key_a, mean=0.0, std=0.02)
            nn.init.normal_(self.memory_lookup_key_b, mean=0.0, std=0.02)
            nn.init.normal_(self.memory_lookup_values, mean=0.0, std=0.02)
        else:
            self.memory_lookup_query = None
            self.memory_lookup_key_a = None
            self.memory_lookup_key_b = None
            self.memory_lookup_values = None
        if config.memory_reconsolidate:
            if config.reconsolidate_gate_type == "mlp":
                self.memory_reconsolidate_gate = nn.Sequential(
                    nn.Linear(config.d_model * 3, config.d_model),
                    nn.GELU(),
                    nn.Linear(config.d_model, 1),
                )
            else:
                self.memory_reconsolidate_gate = nn.Linear(config.d_model * 3, 1)
        else:
            self.memory_reconsolidate_gate = None
        if config.content_read_steps > 1 and config.content_read_gate_type == "learned":
            self.content_read_blend_gate = nn.Linear(config.d_model * 2 + 1, 1)
            nn.init.zeros_(self.content_read_blend_gate.weight)
            nn.init.constant_(self.content_read_blend_gate.bias, 1.0)
        else:
            self.content_read_blend_gate = None
        if config.content_read_steps > 1 and config.content_read_gate_type == "synthesis":
            self.content_read_synthesis = nn.Linear(config.d_model * 5, config.d_model)
            self.content_read_synthesis_gate = nn.Linear(config.d_model * 5, 1)
            nn.init.zeros_(self.content_read_synthesis.weight)
            with torch.no_grad():
                self.content_read_synthesis.weight[
                    :, config.d_model : config.d_model * 2
                ].copy_(torch.eye(config.d_model))
            nn.init.zeros_(self.content_read_synthesis.bias)
            nn.init.zeros_(self.content_read_synthesis_gate.weight)
            nn.init.constant_(self.content_read_synthesis_gate.bias, -1.0)
        else:
            self.content_read_synthesis = None
            self.content_read_synthesis_gate = None
        # run5b_plus: TAC-218 decision continuity head (only when program_embed_dim is set)
        if config.program_embed_dim is not None:
            self.decision_continuity_head: Optional[DecisionContinuityHead] = (
                DecisionContinuityHead(config.program_embed_dim, config.n_programs)
            )
        else:
            self.decision_continuity_head = None
        # Mutable state for adaptive L1 (updated from training loop every 500 steps)
        self._last_norm_floor_fire_rate: float = 0.0

    @property
    def energy_costs(self) -> Tensor:
        return F.softplus(self.raw_energy_costs) + 0.05

    def forward(
        self,
        hidden: Tensor,
        previous_state: Optional[IdentityState] = None,
        *,
        collect_auxiliary: bool = True,
        collect_metrics: bool = True,
        update_content_memory: bool = True,
        update_identity_state: bool = True,
        content_write_mask: Optional[Tensor] = None,
    ) -> IdentityFieldOutput:
        batch_size, seq_len, d_model = hidden.shape
        self._last_content_synthesis_gate = hidden.new_zeros(())
        self._last_content_gate_entropy_loss = hidden.new_zeros(())
        self._last_content_gate_entropy = hidden.new_zeros(())
        self._last_content_read_queries = hidden.new_zeros(())
        self._last_content_read_query_fraction = hidden.new_zeros(())
        self._last_coalition_context_norm = hidden.new_zeros(())
        self._last_authority_logits = None
        self._last_authority_probs = None
        self._last_authority_indices = None
        self._last_verifier_required = None
        self._last_halt_probability = None
        programs = F.normalize(self.program_embeddings, dim=-1)
        normalized_hidden = F.normalize(hidden, dim=-1)

        program_logits = torch.matmul(normalized_hidden, programs.T) * sqrt(d_model)

        if previous_state is None:
            previous_stability = torch.zeros(
                batch_size,
                self.config.n_programs,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            previous_memory = torch.zeros(
                batch_size,
                self.config.n_programs,
                self.config.d_model,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            previous_working_state = torch.zeros_like(previous_memory)
            previous_episodic_state = torch.zeros_like(previous_memory)
            previous_semantic_state = torch.zeros_like(previous_memory)
            previous_procedural_state = torch.zeros_like(previous_memory)
            previous_stable_memory = torch.zeros_like(previous_memory)
            previous_archival_memory = torch.zeros_like(previous_memory)
            previous_program_age = torch.zeros(
                batch_size,
                self.config.n_programs,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            previous_write_frequency = torch.zeros_like(previous_program_age)
            previous_memory_confidence = torch.zeros_like(previous_program_age)
            previous_decision_memory = torch.zeros_like(previous_program_age)
            previous_engram_patterns = torch.zeros(
                batch_size,
                self.config.pattern_store_size,
                self.config.n_programs,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            previous_engram_values = torch.zeros(
                batch_size,
                self.config.pattern_store_size,
                self.config.d_model,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            previous_engram_mask = torch.zeros(
                batch_size,
                self.config.pattern_store_size,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            previous_content_cues = torch.zeros(
                batch_size,
                self.config.content_store_size,
                self.config.d_model,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            previous_content_values = torch.zeros_like(previous_content_cues)
            previous_content_mask = torch.zeros(
                batch_size,
                self.config.content_store_size,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            # run5b_plus: 3D decision memory (zeros when no prior state)
            if self.config.program_embed_dim is not None:
                previous_decision_memory_ebm: Optional[Tensor] = torch.zeros(
                    batch_size,
                    self.config.n_programs,
                    self.config.program_embed_dim,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            else:
                previous_decision_memory_ebm = None
        else:
            previous_stability = previous_state.stability.to(hidden.device)
            previous_memory = previous_state.program_memory.to(hidden.device)
            previous_working_state = self._state_memory_or_zeros(
                previous_state.working_state,
                previous_memory,
            )
            previous_episodic_state = self._state_memory_or_zeros(
                previous_state.episodic_state,
                previous_memory,
            )
            previous_semantic_state = self._state_memory_or_zeros(
                previous_state.semantic_state,
                previous_memory,
            )
            previous_procedural_state = self._state_memory_or_zeros(
                previous_state.procedural_state,
                previous_memory,
            )
            previous_stable_memory = self._state_memory_or_zeros(
                previous_state.stable_program_memory,
                previous_memory,
            )
            previous_archival_memory = self._state_memory_or_zeros(
                previous_state.archival_program_memory,
                previous_memory,
            )
            previous_program_age = self._state_age_or_zeros(
                previous_state.program_age,
                previous_stability,
            )
            previous_write_frequency = self._state_age_or_zeros(
                previous_state.program_write_frequency,
                previous_stability,
            )
            previous_memory_confidence = self._state_age_or_zeros(
                previous_state.memory_confidence,
                previous_stability,
            )
            previous_decision_memory = self._state_age_or_zeros(
                previous_state.decision_memory,
                previous_stability,
            )
            previous_engram_patterns = self._state_pattern_store_or_zeros(
                previous_state.engram_patterns,
                batch_size,
                hidden.device,
                hidden.dtype,
            )
            previous_engram_values = self._state_value_store_or_zeros(
                previous_state.engram_values,
                batch_size,
                hidden.device,
                hidden.dtype,
            )
            previous_engram_mask = self._state_mask_store_or_zeros(
                previous_state.engram_mask,
                batch_size,
                self.config.pattern_store_size,
                hidden.device,
                hidden.dtype,
            )
            previous_content_cues = self._state_content_store_or_zeros(
                previous_state.content_cues,
                batch_size,
                hidden.device,
                hidden.dtype,
            )
            previous_content_values = self._state_content_store_or_zeros(
                previous_state.content_values,
                batch_size,
                hidden.device,
                hidden.dtype,
            )
            previous_content_mask = self._state_mask_store_or_zeros(
                previous_state.content_mask,
                batch_size,
                self.config.content_store_size,
                hidden.device,
                hidden.dtype,
            )
            # run5b_plus: restore 3D decision memory from state
            if self.config.program_embed_dim is not None:
                if (
                    previous_state.decision_memory_ebm is not None
                    and previous_state.decision_memory_ebm.shape
                    == (batch_size, self.config.n_programs, self.config.program_embed_dim)
                ):
                    previous_decision_memory_ebm = previous_state.decision_memory_ebm.to(
                        hidden.device
                    )
                else:
                    previous_decision_memory_ebm = torch.zeros(
                        batch_size,
                        self.config.n_programs,
                        self.config.program_embed_dim,
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
            else:
                previous_decision_memory_ebm = None

        if self.config.causal:
            activations_by_token = self._program_activations(program_logits)
            stability_by_token = []
            running_stability = previous_stability
            stability_gates = self._stability_gates(hidden)
            for token_index, token_activations in enumerate(activations_by_token.transpose(0, 1)):
                token_gate = stability_gates[:, token_index, :]
                running_stability = self._blend_state(
                    running_stability,
                    token_activations,
                    token_gate,
                )
                stability_by_token.append(running_stability)

            token_stability = torch.stack(stability_by_token, dim=1)
            activations = activations_by_token[:, -1, :]
            stability = token_stability[:, -1, :]
            stability_for_tokens = token_stability
        else:
            activations = self._program_activations(program_logits.mean(dim=1))
            stability = self._blend_state(
                previous_stability,
                activations,
                self._pooled_stability_gate(hidden),
            )
            stability_for_tokens = stability[:, None, :].expand(-1, seq_len, -1)

        token_program_weights = F.softmax(
            program_logits + torch.log(stability_for_tokens.clamp_min(1e-6)),
            dim=-1,
        )
        program_assignments = token_program_weights.argmax(dim=-1)
        program_identity = torch.matmul(token_program_weights, self.program_embeddings)
        coherence = torch.matmul(
            token_program_weights,
            token_program_weights.transpose(-1, -2),
        ).clamp(0.0, 1.0)

        if self.config.causal:
            decision_memory_for_tokens = previous_decision_memory[:, None, :].expand(
                -1,
                seq_len,
                -1,
            )
            selected_by_token = self._route_programs(
                stability_for_tokens.reshape(batch_size * seq_len, self.config.n_programs),
                activations=activations_by_token.reshape(
                    batch_size * seq_len,
                    self.config.n_programs,
                ),
                decision_memory=decision_memory_for_tokens.reshape(
                    batch_size * seq_len,
                    self.config.n_programs,
                ),
            ).reshape(batch_size, seq_len, self.config.n_programs)
            used_by_token = self._used_route_energy(selected_by_token)
            selected_weights = activations_by_token * selected_by_token
            selected_program_mask = selected_by_token[:, -1, :]
            used_energy = used_by_token[:, -1]
            token_activations = activations_by_token
            token_selected_program_mask = selected_by_token
            token_authority_logits = self._reshape_token_authority(
                self._last_authority_logits,
                batch_size,
                seq_len,
            )
            token_authority_probs = self._reshape_token_authority(
                self._last_authority_probs,
                batch_size,
                seq_len,
            )
            token_verifier_required = self._reshape_token_authority(
                self._last_verifier_required,
                batch_size,
                seq_len,
            )
        else:
            selected_program_mask = self._route_programs(
                stability,
                activations=activations,
                decision_memory=previous_decision_memory,
            )
            used_energy = self._used_route_energy(selected_program_mask)
            selected_weights = activations * selected_program_mask
            token_activations = activations[:, None, :].expand(-1, seq_len, -1)
            token_selected_program_mask = selected_program_mask[:, None, :].expand(
                -1,
                seq_len,
                -1,
            )
            token_authority_logits = self._expand_sequence_authority(
                self._last_authority_logits,
                seq_len,
            )
            token_authority_probs = self._expand_sequence_authority(
                self._last_authority_probs,
                seq_len,
            )
            token_verifier_required = self._expand_sequence_authority(
                self._last_verifier_required,
                seq_len,
            )

        authority_logits = self._last_token_authority(token_authority_logits)
        authority_probs = self._last_token_authority(token_authority_probs)
        authority_indices = (
            authority_probs.argmax(dim=-1) if authority_probs is not None else None
        )
        verifier_required = (
            token_verifier_required[:, -1]
            if token_verifier_required is not None
            else None
        )
        halt_probability = self._last_authority_halt(batch_size, seq_len)

        selected_denominator = selected_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        routed_weights = selected_weights / selected_denominator
        if collect_auxiliary:
            pattern_completion_hit = self._pattern_completion_hit(
                routed_weights,
                previous_engram_patterns,
                previous_engram_mask,
            )
            content_addressed_hit = self._content_addressed_hit(
                hidden,
                previous_content_cues,
                previous_content_mask,
            )
        else:
            pattern_completion_hit = hidden.new_zeros(())
            content_addressed_hit = hidden.new_zeros(())
        read_memory_source = self._read_memory_source(
            previous_memory,
            previous_stable_memory,
            previous_archival_memory,
        )
        read_memory_source = self._multi_timescale_read_memory_source(
            read_memory_source,
            previous_working_state,
            previous_episodic_state,
            previous_semantic_state,
            previous_procedural_state,
        )
        program_context = self._compute_program_context(
            hidden,
            selected_weights,
            selected_denominator,
            read_memory_source,
            previous_engram_patterns,
            previous_engram_values,
            previous_engram_mask,
            previous_content_cues,
            previous_content_values,
            previous_content_mask,
        )
        if update_content_memory:
            reconsolidated_content_cues, content_reconsolidation_gate = (
                self._reconsolidate_content_cues(
                    previous_content_cues,
                    previous_content_mask,
                    hidden,
                )
            )
        else:
            reconsolidated_content_cues = previous_content_cues
            content_reconsolidation_gate = hidden.new_zeros(())

        if update_identity_state:
            pooled_hidden = hidden.mean(dim=1, keepdim=True)
            raw_candidate_memory = self._candidate_program_memory(pooled_hidden)
            candidate_memory = raw_candidate_memory * stability[:, :, None]
            write_gate = self._memory_write_gate(hidden, previous_memory, candidate_memory)
            write_gate, allocation_mask = self._allocate_memory_write_gate(
                write_gate,
                stability,
                activations,
                previous_program_age,
                previous_write_frequency,
            )
            if self.config.memory_write_type == "hebbian_outer":
                program_memory = self._hebbian_outer_memory_update(
                    previous_memory,
                    raw_candidate_memory,
                    routed_weights,
                    write_gate,
                )
                hebbian_write_strength = (
                    program_memory - previous_memory * self.config.state_decay
                ).norm(dim=-1).mean()
            else:
                program_memory = self._blend_memory(
                    previous_memory,
                    candidate_memory,
                    write_gate,
                )
                hebbian_write_strength = hidden.new_zeros(())
            program_memory, reconsolidate_gate = self._reconsolidate_memory(
                program_memory,
                read_memory_source,
                raw_candidate_memory,
                selected_program_mask,
            )
            stable_program_memory, archival_program_memory = self._update_memory_tiers(
                hidden,
                program_memory,
                previous_stable_memory,
                previous_archival_memory,
            )
            (
                working_state,
                episodic_state,
                semantic_state,
                procedural_state,
                memory_confidence,
            ) = self._update_multi_timescale_memory(
                previous_working_state,
                previous_episodic_state,
                previous_semantic_state,
                previous_procedural_state,
                previous_memory_confidence,
                candidate_memory,
                program_memory,
                write_gate,
                selected_program_mask,
                activations,
            )
            program_age = self._update_program_age(previous_program_age, write_gate)
            program_write_frequency = self._update_program_write_frequency(
                previous_write_frequency,
                write_gate,
            )
            engram_patterns, engram_values, engram_mask = self._update_engram_store(
                previous_engram_patterns,
                previous_engram_values,
                previous_engram_mask,
                routed_weights,
                program_memory,
            )
        else:
            write_gate = previous_program_age.new_zeros(previous_program_age.shape)
            allocation_mask = write_gate
            program_memory = previous_memory
            hebbian_write_strength = hidden.new_zeros(())
            reconsolidate_gate = hidden.new_zeros(batch_size, self.config.n_programs)
            stable_program_memory = (
                previous_state.stable_program_memory.to(hidden.device)
                if previous_state is not None
                and previous_state.stable_program_memory is not None
                else None
            )
            working_state = (
                previous_working_state
                if self.config.memory_system_type == "multi_timescale"
                else None
            )
            episodic_state = (
                previous_episodic_state
                if self.config.memory_system_type == "multi_timescale"
                else None
            )
            semantic_state = (
                previous_semantic_state
                if self.config.memory_system_type == "multi_timescale"
                else None
            )
            procedural_state = (
                previous_procedural_state
                if self.config.memory_system_type == "multi_timescale"
                else None
            )
            memory_confidence = (
                previous_memory_confidence
                if self.config.memory_system_type == "multi_timescale"
                else None
            )
            archival_program_memory = (
                previous_state.archival_program_memory.to(hidden.device)
                if previous_state is not None
                and previous_state.archival_program_memory is not None
                else None
            )
            program_age = previous_program_age
            program_write_frequency = previous_write_frequency
            engram_patterns = (
                previous_state.engram_patterns.to(hidden.device)
                if previous_state is not None and previous_state.engram_patterns is not None
                else None
            )
            engram_values = (
                previous_state.engram_values.to(hidden.device)
                if previous_state is not None and previous_state.engram_values is not None
                else None
            )
            engram_mask = (
                previous_state.engram_mask.to(hidden.device)
                if previous_state is not None and previous_state.engram_mask is not None
                else None
            )
        if update_content_memory:
            content_cues, content_values, content_mask = self._update_content_store(
                reconsolidated_content_cues,
                previous_content_values,
                previous_content_mask,
                hidden,
                content_write_mask=content_write_mask,
            )
        elif not update_identity_state and previous_state is not None:
            content_cues = (
                previous_state.content_cues.to(hidden.device)
                if previous_state.content_cues is not None
                else None
            )
            content_values = (
                previous_state.content_values.to(hidden.device)
                if previous_state.content_values is not None
                else None
            )
            content_mask = (
                previous_state.content_mask.to(hidden.device)
                if previous_state.content_mask is not None
                else None
            )
        else:
            content_cues = previous_content_cues
            content_values = previous_content_values
            content_mask = previous_content_mask

        decision_memory = (
            self._update_decision_memory(
                previous_decision_memory,
                token_selected_program_mask,
            )
            if update_identity_state
            else previous_decision_memory
        )

        # run5b_plus: update 3D decision memory with selected program embeddings
        # decision_memory_ebm accumulates a weighted sum of program embeddings for the
        # programs selected this sequence; decay controls how quickly old choices fade.
        if self.config.program_embed_dim is not None and previous_decision_memory_ebm is not None:
            decision_distribution = self._decision_distribution(selected_weights)
            selected_emb_by_program = (
                decision_distribution[..., None] * self._program_identity_embeddings()[None, :, :]
            )
            if update_identity_state:
                decay = self.config.decision_continuity_decay
                decision_memory_ebm: Optional[Tensor] = (
                    decay * previous_decision_memory_ebm
                    + (1.0 - decay) * selected_emb_by_program
                )
            else:
                decision_memory_ebm = previous_decision_memory_ebm
        else:
            decision_memory_ebm = None

        if collect_auxiliary:
            decision_continuity_loss = self._decision_continuity_loss(
                token_selected_program_mask,
                previous_decision_memory,
            )
            # run5b_plus: EBM decision continuity loss (TAC-218)
            if (
                self.decision_continuity_head is not None
                and previous_decision_memory_ebm is not None
                and bool((previous_decision_memory_ebm.abs().sum(dim=(1, 2)) > 1e-6).any())
            ):
                prev_dist, curr_dist = self.decision_continuity_head(
                    previous_decision_memory_ebm,
                    program_logits[:, -1, :] if program_logits.ndim == 3 else program_logits,
                )
                ebm_agreement = F.cosine_similarity(prev_dist, curr_dist, dim=-1).mean()
                ebm_decision_continuity_loss = -ebm_agreement
            else:
                ebm_decision_continuity_loss = hidden.new_zeros(())
                ebm_agreement = hidden.new_zeros(())

            # run5b_plus: compression losses (identity layers only, NOT backbone)
            # Compute selected identity state norm for this sequence
            selected_identity_state = torch.matmul(
                token_selected_program_mask,      # [batch, seq, n_programs]
                self._program_identity_embeddings(),
            )  # [batch, seq, program_embed_dim or d_model]
            activation_l1_loss = (
                selected_identity_state.abs().mean()
                if self.config.activation_l1_weight > 0.0
                else hidden.new_zeros(())
            )
            selected_norm = selected_identity_state.norm(dim=-1)  # [batch, seq]
            norm_floor_penalty = (
                F.relu(self.config.identity_norm_floor_threshold - selected_norm).mean()
                if self.config.identity_norm_floor_weight > 0.0
                else hidden.new_zeros(())
            )
            # Track fire rate for adaptive L1 adjustment (non-differentiable metric)
            norm_floor_fire_rate = float(
                (selected_norm < self.config.identity_norm_floor_threshold).float().mean().item()
            )
            self._last_norm_floor_fire_rate = norm_floor_fire_rate

            losses = {
                "coherence": (1.0 - coherence).pow(2).mean(),
                "program_reuse": (1.0 - activations).mean(),
                "energy": used_energy.mean() / max(self.config.energy_budget, 1e-6),
                "separation": self._program_memory_separation_loss(program_memory),
                "content_cue_separation": self._content_cue_separation_loss(
                    content_cues,
                    content_mask,
                ),
                "content_gate_entropy": self._last_content_gate_entropy_loss,
                "routing_load_balance": self._routing_load_balance_loss(
                    token_selected_program_mask,
                ),
                "decision_continuity": decision_continuity_loss,
                # run5b_plus losses
                "ebm_decision_continuity": ebm_decision_continuity_loss,
                "activation_l1": activation_l1_loss,
                "identity_norm_floor": norm_floor_penalty,
            }
            if collect_metrics:
                metrics = self._compute_metrics(selected_program_mask, selected_weights)
                metrics["routing_load_balance"] = losses["routing_load_balance"]
                metrics["decision_continuity_agreement"] = (
                    self._decision_continuity_agreement(
                        token_selected_program_mask,
                        previous_decision_memory,
                    )
                )
                metrics["decision_continuity_memory_mass"] = (
                    self._decision_continuity_memory_mass(previous_decision_memory)
                )
                metrics["program_memory_cosine"] = self._program_memory_cosine_metric(
                    program_memory,
                )
                metrics["program_ortho"] = losses["separation"]
                metrics["memory_reconsolidation_gate"] = reconsolidate_gate.mean()
                metrics["memory_allocation_type"] = selected_program_mask.new_tensor(
                    1.0 if self.config.memory_allocation_type == "creb" else 0.0
                )
                metrics["memory_allocation_dead_rate"] = (
                    program_memory.norm(dim=-1) <= 1e-6
                ).float().mean()
                metrics["memory_allocation_age"] = program_age.mean()
                metrics["memory_allocation_load_std"] = allocation_mask.float().mean(dim=0).std()
                metrics["memory_allocation_write_frequency"] = program_write_frequency.mean()
                metrics["pattern_completion_hit"] = pattern_completion_hit
                metrics["content_addressed_hit"] = content_addressed_hit
                metrics["content_read_queries"] = self._last_content_read_queries
                metrics["content_read_query_fraction"] = self._last_content_read_query_fraction
                metrics["content_read_skipped_fraction"] = (
                    1.0 - self._last_content_read_query_fraction
                )
                metrics["coalition_context_norm"] = self._last_coalition_context_norm
                metrics["content_synthesis_gate"] = self._last_content_synthesis_gate
                metrics["content_gate_entropy"] = self._last_content_gate_entropy
                metrics["content_cue_cosine"] = self._content_cue_cosine_metric(
                    content_cues,
                    content_mask,
                )
                metrics["content_reconsolidation_gate"] = content_reconsolidation_gate
                metrics["identity_sparse_density"] = (
                    program_assignments[:, :, None] == program_assignments[:, None, :]
                ).float().mean()
                metrics.update(
                    self._authority_metrics(
                        authority_probs,
                        verifier_required,
                        halt_probability,
                        selected_program_mask,
                    )
                )
                # run5b_plus metrics
                metrics["ebm_decision_agreement"] = ebm_agreement
                metrics["activation_l1_weight"] = hidden.new_tensor(
                    float(self.config.activation_l1_weight)
                )
                metrics["norm_floor_fire_rate"] = hidden.new_tensor(norm_floor_fire_rate)
                metrics["selected_identity_state_norm"] = selected_norm.mean()
                metrics["activation_density"] = (
                    token_activations > 1e-6
                ).float().mean()
                metrics["hebbian_write_strength"] = hebbian_write_strength
                metrics["multi_timescale_memory_mass"] = (
                    self._multi_timescale_memory_mass(
                        working_state,
                        episodic_state,
                        semantic_state,
                        procedural_state,
                        hidden,
                    )
                )
                metrics["memory_confidence"] = (
                    memory_confidence.mean()
                    if memory_confidence is not None
                    else hidden.new_zeros(())
                )
                metrics["decision_memory_ebm_mass"] = (
                    decision_memory_ebm.norm(dim=-1).mean()
                    if decision_memory_ebm is not None
                    else hidden.new_zeros(())
                )
            else:
                metrics = self._minimal_metrics(selected_program_mask, hidden.new_zeros(()))
        else:
            zero = hidden.new_zeros(())
            losses = {
                "coherence": zero,
                "program_reuse": zero,
                "energy": zero,
                "separation": zero,
                "content_cue_separation": zero,
                "content_gate_entropy": zero,
                "routing_load_balance": zero,
                "decision_continuity": zero,
                # run5b_plus (zero when not collecting auxiliary)
                "ebm_decision_continuity": zero,
                "activation_l1": zero,
                "identity_norm_floor": zero,
            }
            metrics = self._minimal_metrics(selected_program_mask, zero)

        state_stability = stability if update_identity_state else previous_stability
        return IdentityFieldOutput(
            coherence=coherence,
            activations=activations,
            program_assignments=program_assignments,
            program_identity=program_identity,
            selected_program_mask=selected_program_mask,
            used_energy=used_energy,
            program_context=program_context,
            state=self._make_identity_state(
                state_stability,
                program_memory,
                decision_memory,
                decision_memory_ebm,
                working_state,
                episodic_state,
                semantic_state,
                procedural_state,
                memory_confidence,
                stable_program_memory,
                archival_program_memory,
                program_age,
                program_write_frequency,
                engram_patterns,
                engram_values,
                engram_mask,
                content_cues,
                content_values,
                content_mask,
            ),
            losses=losses,
            metrics=metrics,
            token_activations=token_activations,
            token_selected_program_mask=token_selected_program_mask,
            authority_logits=authority_logits,
            authority_probs=authority_probs,
            authority_indices=authority_indices,
            verifier_required=verifier_required,
            halt_probability=halt_probability,
            token_authority_logits=token_authority_logits,
            token_authority_probs=token_authority_probs,
            token_verifier_required=token_verifier_required,
        )

    def _minimal_metrics(
        self,
        selected_program_mask: Tensor,
        zero: Tensor,
    ) -> dict[str, Tensor]:
        return {
            "active_expert_parameters": zero,
            "total_expert_parameters": zero,
            "active_expert_fraction": zero,
            "sink_programs": selected_program_mask.new_tensor(float(self.config.n_sink_programs)),
            "memory_tiers": selected_program_mask.new_tensor(
                1.0 if self.config.memory_tier_type == "hierarchical" else 0.0
            ),
            "routing_type": selected_program_mask.new_tensor(float(self._routing_type_id())),
            "routing_load_std": zero,
            "memory_lookup_slots": selected_program_mask.new_tensor(
                float(self.config.memory_lookup_slots)
                if self.config.memory_lookup_type == "product_key"
                else 0.0
            ),
            "residual_streams": selected_program_mask.new_tensor(
                2.0 if self.config.residual_stream_type == "dual_stream" else 1.0
            ),
            "sequence_mixer_type": selected_program_mask.new_tensor(
                float(_sequence_mixer_type_id(self.config.sequence_mixer_type))
            ),
            "decision_continuity_agreement": zero,
            "decision_continuity_memory_mass": zero,
            "program_memory_cosine": zero,
            "program_ortho": zero,
            "memory_reconsolidation_gate": zero,
            "memory_allocation_type": zero,
            "memory_allocation_dead_rate": zero,
            "memory_allocation_age": zero,
            "memory_allocation_load_std": zero,
            "memory_allocation_write_frequency": zero,
            "pattern_completion_hit": zero,
            "content_addressed_hit": zero,
            "content_read_queries": zero,
            "content_read_query_fraction": zero,
            "content_read_skipped_fraction": zero,
            "coalition_context_norm": zero,
            "content_synthesis_gate": zero,
            "content_gate_entropy": zero,
            "content_cue_cosine": zero,
            "content_reconsolidation_gate": zero,
            "identity_sparse_density": zero,
        }

    def _program_identity_embeddings(self) -> Tensor:
        if self.config.program_embed_dim is None:
            return self.program_embeddings
        return self.program_embeddings[:, : self.config.program_embed_dim]

    def _reshape_token_authority(
        self,
        authority: Optional[Tensor],
        batch_size: int,
        seq_len: int,
    ) -> Optional[Tensor]:
        if authority is None:
            return None
        return authority.reshape(batch_size, seq_len, *authority.shape[1:])

    def _expand_sequence_authority(
        self,
        authority: Optional[Tensor],
        seq_len: int,
    ) -> Optional[Tensor]:
        if authority is None:
            return None
        return authority[:, None, ...].expand(-1, seq_len, *authority.shape[1:])

    def _last_token_authority(self, authority: Optional[Tensor]) -> Optional[Tensor]:
        if authority is None:
            return None
        if authority.ndim >= 3:
            return authority[:, -1, ...]
        return authority

    def _last_authority_halt(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Optional[Tensor]:
        if self._last_halt_probability is None:
            return None
        if self.config.causal:
            return self._last_halt_probability.reshape(batch_size, seq_len)[:, -1]
        return self._last_halt_probability

    def _authority_metrics(
        self,
        authority_probs: Optional[Tensor],
        verifier_required: Optional[Tensor],
        halt_probability: Optional[Tensor],
        selected_program_mask: Tensor,
    ) -> dict[str, Tensor]:
        if (
            authority_probs is None
            or verifier_required is None
            or halt_probability is None
        ):
            return {}
        return {
            "authority_exact_memory_prob": authority_probs[:, 0].mean(),
            "authority_proposal_verified_prob": authority_probs[:, 1].mean(),
            "authority_calibrated_fast_path_prob": authority_probs[:, 2].mean(),
            "authority_fresh_repair_prob": authority_probs[:, 3].mean(),
            "authority_system2_verify_prob": authority_probs[:, 4].mean(),
            "authority_verifier_required_rate": verifier_required.float().mean(),
            "authority_halt_probability": halt_probability.mean(),
            "authority_active_programs": selected_program_mask.float().sum(dim=-1).mean(),
        }

    def _update_decision_memory(
        self,
        previous_decision_memory: Tensor,
        token_selected_program_mask: Tensor,
    ) -> Tensor:
        current = self._decision_distribution(token_selected_program_mask)
        if self.config.decision_continuity_decay <= 0.0:
            return current
        return (
            self.config.decision_continuity_decay * previous_decision_memory
            + (1.0 - self.config.decision_continuity_decay) * current
        )

    def _decision_distribution(self, token_selected_program_mask: Tensor) -> Tensor:
        selected = token_selected_program_mask.clamp_min(0.0)
        if selected.dim() == 3:
            selected = selected.mean(dim=1)
        return selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _decision_continuity_loss(
        self,
        token_selected_program_mask: Tensor,
        previous_decision_memory: Tensor,
    ) -> Tensor:
        active = previous_decision_memory.sum(dim=-1) > 1e-6
        if not bool(active.any()):
            return token_selected_program_mask.new_zeros(())
        current = self._decision_distribution(token_selected_program_mask)
        previous = previous_decision_memory / previous_decision_memory.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        overlap = torch.minimum(current, previous).sum(dim=-1)
        return (1.0 - overlap[active]).mean()

    def _decision_continuity_agreement(
        self,
        token_selected_program_mask: Tensor,
        previous_decision_memory: Tensor,
    ) -> Tensor:
        active = previous_decision_memory.sum(dim=-1) > 1e-6
        if not bool(active.any()):
            return token_selected_program_mask.new_zeros(())
        current = self._decision_distribution(token_selected_program_mask)
        previous = previous_decision_memory / previous_decision_memory.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        return torch.minimum(current, previous).sum(dim=-1)[active].mean()

    def _decision_continuity_memory_mass(
        self,
        previous_decision_memory: Tensor,
    ) -> Tensor:
        return previous_decision_memory.sum(dim=-1).mean()

    def _multi_timescale_read_memory_source(
        self,
        base_memory: Tensor,
        working_state: Tensor,
        episodic_state: Tensor,
        semantic_state: Tensor,
        procedural_state: Tensor,
    ) -> Tensor:
        if self.config.memory_system_type != "multi_timescale":
            return base_memory
        return (
            0.35 * base_memory
            + 0.20 * working_state
            + 0.20 * episodic_state
            + 0.15 * semantic_state
            + 0.10 * procedural_state
        )

    def _update_multi_timescale_memory(
        self,
        previous_working_state: Tensor,
        previous_episodic_state: Tensor,
        previous_semantic_state: Tensor,
        previous_procedural_state: Tensor,
        previous_memory_confidence: Tensor,
        candidate_memory: Tensor,
        program_memory: Tensor,
        write_gate: Tensor,
        selected_program_mask: Tensor,
        activations: Tensor,
    ) -> tuple[
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        if self.config.memory_system_type != "multi_timescale":
            return None, None, None, None, None
        importance = (selected_program_mask * activations).clamp(0.0, 1.0)
        write = write_gate.clamp(0.0, 1.0)
        retain = self.config.memory_retention_rate + (
            1.0 - self.config.memory_retention_rate
        ) * importance
        retain = retain.clamp(0.0, 1.0)

        working_state = candidate_memory
        episodic_state = (
            retain[:, :, None] * previous_episodic_state
            + write[:, :, None] * program_memory
        )
        consolidation_gate = (
            self.config.memory_consolidation_rate * importance
        ).clamp(0.0, 1.0)
        semantic_state = (
            (1.0 - consolidation_gate[:, :, None]) * previous_semantic_state
            + consolidation_gate[:, :, None] * episodic_state
        )
        procedural_value = self.program_embeddings[None, :, :].expand_as(
            previous_procedural_state
        )
        procedural_gate = (
            self.config.procedural_memory_rate * importance
        ).clamp(0.0, 1.0)
        procedural_state = (
            (1.0 - procedural_gate[:, :, None]) * previous_procedural_state
            + procedural_gate[:, :, None] * procedural_value
        )
        memory_confidence = (
            retain * previous_memory_confidence
            + (1.0 - retain) * importance
        ).clamp(0.0, 1.0)
        return (
            working_state,
            episodic_state,
            semantic_state,
            procedural_state,
            memory_confidence,
        )

    def _multi_timescale_memory_mass(
        self,
        working_state: Optional[Tensor],
        episodic_state: Optional[Tensor],
        semantic_state: Optional[Tensor],
        procedural_state: Optional[Tensor],
        reference: Tensor,
    ) -> Tensor:
        if (
            working_state is None
            or episodic_state is None
            or semantic_state is None
            or procedural_state is None
        ):
            return reference.new_zeros(())
        return torch.stack(
            [
                working_state.norm(dim=-1).mean(),
                episodic_state.norm(dim=-1).mean(),
                semantic_state.norm(dim=-1).mean(),
                procedural_state.norm(dim=-1).mean(),
            ]
        ).mean()

    def _make_identity_state(
        self,
        stability: Tensor,
        program_memory: Tensor,
        decision_memory: Tensor,
        decision_memory_ebm: Optional[Tensor],
        working_state: Optional[Tensor],
        episodic_state: Optional[Tensor],
        semantic_state: Optional[Tensor],
        procedural_state: Optional[Tensor],
        memory_confidence: Optional[Tensor],
        stable_program_memory: Optional[Tensor],
        archival_program_memory: Optional[Tensor],
        program_age: Optional[Tensor],
        program_write_frequency: Optional[Tensor],
        engram_patterns: Optional[Tensor],
        engram_values: Optional[Tensor],
        engram_mask: Optional[Tensor],
        content_cues: Optional[Tensor],
        content_values: Optional[Tensor],
        content_mask: Optional[Tensor],
    ) -> IdentityState:
        keep_engram_state = self.config.memory_read_type == "pattern_completion"
        keep_content_state = self.config.memory_read_type == "content_addressed"
        if self.config.detach_identity_state:
            return IdentityState(
                stability=stability.detach(),
                program_memory=program_memory.detach(),
                working_state=(
                    working_state.detach() if working_state is not None else None
                ),
                episodic_state=(
                    episodic_state.detach() if episodic_state is not None else None
                ),
                semantic_state=(
                    semantic_state.detach() if semantic_state is not None else None
                ),
                procedural_state=(
                    procedural_state.detach() if procedural_state is not None else None
                ),
                memory_confidence=(
                    memory_confidence.detach()
                    if memory_confidence is not None
                    else None
                ),
                decision_memory=decision_memory.detach(),
                decision_memory_ebm=(
                    decision_memory_ebm.detach()
                    if decision_memory_ebm is not None
                    else None
                ),
                stable_program_memory=(
                    stable_program_memory.detach()
                    if stable_program_memory is not None
                    else None
                ),
                archival_program_memory=(
                    archival_program_memory.detach()
                    if archival_program_memory is not None
                    else None
                ),
                program_age=program_age.detach() if program_age is not None else None,
                program_write_frequency=(
                    program_write_frequency.detach()
                    if program_write_frequency is not None
                    else None
                ),
                engram_patterns=(
                    engram_patterns.detach()
                    if keep_engram_state and engram_patterns is not None
                    else None
                ),
                engram_values=(
                    engram_values.detach()
                    if keep_engram_state and engram_values is not None
                    else None
                ),
                engram_mask=(
                    engram_mask.detach()
                    if keep_engram_state and engram_mask is not None
                    else None
                ),
                content_cues=(
                    content_cues.detach()
                    if keep_content_state and content_cues is not None
                    else None
                ),
                content_values=(
                    content_values.detach()
                    if keep_content_state and content_values is not None
                    else None
                ),
                content_mask=(
                    content_mask.detach()
                    if keep_content_state and content_mask is not None
                    else None
                ),
            )
        return IdentityState(
            stability=stability,
            program_memory=program_memory,
            working_state=working_state,
            episodic_state=episodic_state,
            semantic_state=semantic_state,
            procedural_state=procedural_state,
            memory_confidence=memory_confidence,
            decision_memory=decision_memory,
            decision_memory_ebm=decision_memory_ebm,
            stable_program_memory=stable_program_memory,
            archival_program_memory=archival_program_memory,
            program_age=program_age,
            program_write_frequency=program_write_frequency,
            engram_patterns=engram_patterns if keep_engram_state else None,
            engram_values=engram_values if keep_engram_state else None,
            engram_mask=engram_mask if keep_engram_state else None,
            content_cues=content_cues if keep_content_state else None,
            content_values=content_values if keep_content_state else None,
            content_mask=content_mask if keep_content_state else None,
        )

    def _state_memory_or_zeros(
        self,
        maybe_memory: Optional[Tensor],
        reference: Tensor,
    ) -> Tensor:
        if maybe_memory is None:
            return torch.zeros_like(reference)
        return maybe_memory.to(reference.device)

    def _state_age_or_zeros(
        self,
        maybe_age: Optional[Tensor],
        reference: Tensor,
    ) -> Tensor:
        if maybe_age is None:
            return torch.zeros_like(reference)
        return maybe_age.to(reference.device)

    def _state_pattern_store_or_zeros(
        self,
        maybe_patterns: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if maybe_patterns is None:
            return torch.zeros(
                batch_size,
                self.config.pattern_store_size,
                self.config.n_programs,
                device=device,
                dtype=dtype,
            )
        return maybe_patterns.to(device=device, dtype=dtype)

    def _state_value_store_or_zeros(
        self,
        maybe_values: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if maybe_values is None:
            return torch.zeros(
                batch_size,
                self.config.pattern_store_size,
                self.config.d_model,
                device=device,
                dtype=dtype,
            )
        return maybe_values.to(device=device, dtype=dtype)

    def _state_mask_store_or_zeros(
        self,
        maybe_mask: Optional[Tensor],
        batch_size: int,
        store_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if maybe_mask is None:
            return torch.zeros(
                batch_size,
                store_size,
                device=device,
                dtype=dtype,
            )
        return maybe_mask.to(device=device, dtype=dtype)

    def _state_content_store_or_zeros(
        self,
        maybe_values: Optional[Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if maybe_values is None:
            return torch.zeros(
                batch_size,
                self.config.content_store_size,
                self.config.d_model,
                device=device,
                dtype=dtype,
            )
        return maybe_values.to(device=device, dtype=dtype)

    def _stability_gates(self, hidden: Tensor) -> Tensor:
        if self.stability_gate is None:
            return hidden.new_full(
                (hidden.shape[0], hidden.shape[1], self.config.n_programs),
                1.0 - self.config.state_decay,
            )
        return torch.sigmoid(self.stability_gate(hidden))

    def _pooled_stability_gate(self, hidden: Tensor) -> Tensor:
        if self.stability_gate is None:
            return hidden.new_full(
                (hidden.shape[0], self.config.n_programs),
                1.0 - self.config.state_decay,
            )
        return torch.sigmoid(self.stability_gate(hidden.mean(dim=1)))

    def _pooled_memory_gate(self, hidden: Tensor) -> Tensor:
        if self.memory_gate is None:
            return hidden.new_full(
                (hidden.shape[0], self.config.n_programs),
                1.0 - self.config.state_decay,
            )
        return torch.sigmoid(self.memory_gate(hidden.mean(dim=1)))

    def _candidate_program_memory(self, pooled_hidden: Tensor) -> Tensor:
        if self.program_conditioned_update is None:
            return self.program_update(pooled_hidden).expand(
                -1,
                self.config.n_programs,
                -1,
            )
        program_context = self.program_embeddings[None, :, :].expand(
            pooled_hidden.shape[0],
            -1,
            -1,
        )
        hidden_context = pooled_hidden.expand_as(program_context)
        return self.program_conditioned_update(
            torch.cat([hidden_context, program_context], dim=-1)
        )

    def _program_activations(self, program_logits: Tensor) -> Tensor:
        if self.config.program_activation_type == "relu":
            return F.relu(program_logits)
        if self.config.program_activation_type == "softplus":
            return F.softplus(program_logits)
        return torch.sigmoid(program_logits)

    def _memory_write_gate(
        self,
        hidden: Tensor,
        previous_memory: Tensor,
        candidate_memory: Tensor,
    ) -> Tensor:
        update_gate = self._pooled_memory_gate(hidden)
        if self.memory_novelty_gate is None:
            return update_gate
        novelty_features = torch.cat([candidate_memory, previous_memory], dim=-1)
        novelty_gate = torch.sigmoid(self.memory_novelty_gate(novelty_features)).squeeze(-1)
        return update_gate * novelty_gate

    def _blend_state(self, previous: Tensor, candidate: Tensor, update_gate: Tensor) -> Tensor:
        return (1.0 - update_gate) * previous + update_gate * candidate

    def _blend_memory(
        self,
        previous: Tensor,
        candidate: Tensor,
        update_gate: Tensor,
    ) -> Tensor:
        return (1.0 - update_gate[:, :, None]) * previous + update_gate[:, :, None] * candidate

    def _hebbian_outer_memory_update(
        self,
        previous: Tensor,
        value_state: Tensor,
        routed_weights: Tensor,
        update_gate: Tensor,
    ) -> Tensor:
        key_state = self._chunk_route_pattern(routed_weights)
        hebbian_delta = key_state[:, :, None] * value_state
        return (
            self.config.state_decay * previous
            + update_gate[:, :, None] * hebbian_delta
        )

    def _allocate_memory_write_gate(
        self,
        write_gate: Tensor,
        stability: Tensor,
        activations: Tensor,
        program_age: Tensor,
        program_write_frequency: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.config.memory_allocation_type == "stability":
            return write_gate, (write_gate > 0).to(write_gate.dtype)
        write_score = (
            self.config.creb_alpha * (1.0 - stability)
            + self.config.creb_beta * activations
            - self.config.creb_gamma * program_age
            - self.config.creb_delta * program_write_frequency
        )
        top_k = min(self.config.memory_allocation_k, self.config.n_programs)
        write_targets = write_score.topk(k=top_k, dim=-1).indices
        allocation_mask = torch.zeros_like(write_gate).scatter(
            dim=-1,
            index=write_targets,
            value=1.0,
        )
        return write_gate * allocation_mask, allocation_mask

    def _update_program_age(self, previous_age: Tensor, write_gate: Tensor) -> Tensor:
        written = (write_gate > 1e-6).to(write_gate.dtype)
        return (previous_age + 1.0) * (1.0 - written)

    def _update_program_write_frequency(
        self,
        previous_frequency: Tensor,
        write_gate: Tensor,
    ) -> Tensor:
        written = (write_gate > 1e-6).to(write_gate.dtype)
        decay = self.config.creb_frequency_decay
        return decay * previous_frequency + (1.0 - decay) * written

    def _update_engram_store(
        self,
        previous_patterns: Tensor,
        previous_values: Tensor,
        previous_mask: Tensor,
        routed_weights: Tensor,
        program_memory: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.memory_read_type != "pattern_completion":
            return previous_patterns, previous_values, previous_mask
        pattern = self._chunk_route_pattern(routed_weights)
        value = torch.einsum("bp,bpd->bd", pattern, program_memory)
        patterns = torch.roll(previous_patterns, shifts=-1, dims=1)
        values = torch.roll(previous_values, shifts=-1, dims=1)
        mask = torch.roll(previous_mask, shifts=-1, dims=1)
        patterns[:, -1, :] = pattern
        values[:, -1, :] = value
        mask[:, -1] = 1.0
        return patterns, values, mask

    def _chunk_route_pattern(self, routed_weights: Tensor) -> Tensor:
        if routed_weights.dim() == 3:
            pattern = routed_weights.mean(dim=1)
        else:
            pattern = routed_weights
        return pattern / pattern.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _update_content_store(
        self,
        previous_cues: Tensor,
        previous_values: Tensor,
        previous_mask: Tensor,
        hidden: Tensor,
        *,
        content_write_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.memory_read_type != "content_addressed" or hidden.shape[1] < 2:
            return previous_cues, previous_values, previous_mask

        cues = hidden[:, :-1, :].detach()
        values = hidden[:, 1:, :].detach()
        pair_slots = cues.shape[1]
        if content_write_mask is None:
            pair_count = min(pair_slots, self.config.content_store_size)
            cues = cues[:, :pair_count, :]
            values = values[:, :pair_count, :]
            stored_cues = torch.roll(previous_cues, shifts=-pair_count, dims=1)
            stored_values = torch.roll(previous_values, shifts=-pair_count, dims=1)
            stored_mask = torch.roll(previous_mask, shifts=-pair_count, dims=1)
            stored_cues[:, -pair_count:, :] = cues
            stored_values[:, -pair_count:, :] = values
            stored_mask[:, -pair_count:] = 1.0
            return stored_cues, stored_values, stored_mask

        if content_write_mask.shape != cues.shape[:2]:
            raise ValueError("content_write_mask must have shape (batch, seq_len - 1)")
        write_mask = content_write_mask.to(device=hidden.device, dtype=torch.bool)
        stored_cues = previous_cues.clone()
        stored_values = previous_values.clone()
        stored_mask = previous_mask.clone()
        for batch_index in range(hidden.shape[0]):
            selected = torch.nonzero(write_mask[batch_index], as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            selected = selected[: self.config.content_store_size]
            count = int(selected.numel())
            stored_cues[batch_index] = torch.roll(
                stored_cues[batch_index],
                shifts=-count,
                dims=0,
            )
            stored_values[batch_index] = torch.roll(
                stored_values[batch_index],
                shifts=-count,
                dims=0,
            )
            stored_mask[batch_index] = torch.roll(
                stored_mask[batch_index],
                shifts=-count,
                dims=0,
            )
            stored_cues[batch_index, -count:, :] = cues[batch_index, selected, :]
            stored_values[batch_index, -count:, :] = values[batch_index, selected, :]
            stored_mask[batch_index, -count:] = 1.0
        return stored_cues, stored_values, stored_mask

    def _reconsolidate_content_cues(
        self,
        cues: Tensor,
        mask: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            not self.config.content_reconsolidate
            or self.config.memory_read_type != "content_addressed"
            or hidden.numel() == 0
        ):
            return cues, hidden.new_zeros(())
        valid = mask > 0
        if not bool(valid.any()):
            return cues, hidden.new_zeros(())

        with torch.no_grad():
            query = hidden.detach()
            query_norm = F.normalize(query, dim=-1)
            cue_norm = F.normalize(cues.detach(), dim=-1)
            scores = torch.einsum("bsd,bkd->bsk", query_norm, cue_norm)
            scores = scores.masked_fill(~valid[:, None, :], -1e4)
            token_to_cue = F.softmax(scores, dim=-1) * mask[:, None, :]
            assignment_mass = token_to_cue.sum(dim=1)
            cue_updates = torch.einsum("bsk,bsd->bkd", token_to_cue, query)
            cue_updates = cue_updates / assignment_mass[..., None].clamp_min(1e-6)
            active_updates = (assignment_mass > 1e-6).to(cues.dtype) * mask
            gate = (
                self.config.content_reconsolidate_rate
                * active_updates[..., None]
            )
            reconsolidated = (1.0 - gate) * cues + gate * cue_updates
        return reconsolidated.detach(), gate.mean()

    def _reconsolidate_memory(
        self,
        program_memory: Tensor,
        retrieved_memory: Tensor,
        current_context: Tensor,
        selected_program_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.memory_reconsolidate_gate is None:
            return program_memory, program_memory.new_zeros(selected_program_mask.shape)
        gate_features = torch.cat(
            [
                retrieved_memory,
                current_context,
                retrieved_memory - current_context,
            ],
            dim=-1,
        )
        reconsolidate_gate = torch.sigmoid(
            self.memory_reconsolidate_gate(gate_features)
        ).squeeze(-1)
        update_gate = reconsolidate_gate * selected_program_mask
        return (
            self._blend_memory(program_memory, current_context, update_gate),
            update_gate,
        )

    def _update_memory_tiers(
        self,
        hidden: Tensor,
        recent_memory: Tensor,
        previous_stable_memory: Tensor,
        previous_archival_memory: Tensor,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if self.config.memory_tier_type == "flat":
            return None, None
        stable_gate = 0.25 * self._pooled_memory_gate(hidden)
        stable_memory = self._blend_memory(
            previous_stable_memory,
            recent_memory,
            stable_gate,
        )
        archival_gate = 0.10 * stable_gate
        archival_memory = self._blend_memory(
            previous_archival_memory,
            stable_memory,
            archival_gate,
        )
        return stable_memory, archival_memory

    def _read_memory_source(
        self,
        recent_memory: Tensor,
        stable_memory: Tensor,
        archival_memory: Tensor,
    ) -> Tensor:
        if self.config.memory_tier_type == "flat":
            return recent_memory
        return 0.6 * recent_memory + 0.3 * stable_memory + 0.1 * archival_memory

    def _program_memory_separation_loss(self, program_memory: Tensor) -> Tensor:
        off_diagonal = self._program_memory_off_diagonal_cosine(program_memory)
        return off_diagonal.pow(2).mean()

    def _program_memory_cosine_metric(self, program_memory: Tensor) -> Tensor:
        off_diagonal = self._program_memory_off_diagonal_cosine(program_memory)
        return off_diagonal.abs().mean()

    def _program_memory_off_diagonal_cosine(self, program_memory: Tensor) -> Tensor:
        normalized = F.normalize(program_memory, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
        eye = torch.eye(
            self.config.n_programs,
            device=program_memory.device,
            dtype=program_memory.dtype,
        )
        return similarity * (1.0 - eye)

    def _content_cue_separation_loss(self, cues: Tensor, mask: Tensor) -> Tensor:
        off_diagonal, pair_mask = self._content_cue_off_diagonal_cosine(cues, mask)
        return (off_diagonal.pow(2) * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    def _content_cue_cosine_metric(self, cues: Tensor, mask: Tensor) -> Tensor:
        off_diagonal, pair_mask = self._content_cue_off_diagonal_cosine(cues, mask)
        return (off_diagonal.abs() * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    def _content_cue_off_diagonal_cosine(
        self,
        cues: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normalized = F.normalize(cues, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
        store_size = cues.shape[1]
        eye = torch.eye(
            store_size,
            device=cues.device,
            dtype=cues.dtype,
        )
        pair_mask = mask[:, :, None] * mask[:, None, :] * (1.0 - eye)
        return similarity * pair_mask, pair_mask

    def _compute_metrics(
        self,
        selected_program_mask: Tensor,
        selected_weights: Tensor,
    ) -> dict[str, Tensor]:
        if self.config.program_compute_type == "embedding":
            zero = selected_program_mask.new_zeros(())
            return {
                "active_expert_parameters": zero,
                "total_expert_parameters": zero,
                "active_expert_fraction": zero,
                "sink_programs": selected_program_mask.new_tensor(float(self.config.n_sink_programs)),
                "memory_tiers": selected_program_mask.new_tensor(
                    3.0 if self.config.memory_tier_type == "hierarchical" else 1.0
                ),
                "routing_type": selected_program_mask.new_tensor(
                    float(self._routing_type_id())
                ),
                "routing_load_std": selected_program_mask.float().mean(dim=0).std(),
                "memory_lookup_slots": selected_program_mask.new_tensor(
                    float(
                        self.config.memory_lookup_slots
                        if self.config.memory_lookup_type == "product_key"
                        else 0
                    )
                ),
                "residual_streams": selected_program_mask.new_tensor(
                    2.0 if self.config.residual_stream_type == "dual_stream" else 1.0
                ),
                "sequence_mixer_type": selected_program_mask.new_tensor(
                    float(_sequence_mixer_type_id(self.config.sequence_mixer_type))
                ),
            }

        if self.config.program_compute_type == "low_rank_linear_expert":
            expert_rank = _program_expert_rank(self.config)
            expert_parameters_per_program = (
                2 * self.config.d_model * expert_rank + self.config.d_model
            )
        else:
            expert_parameters_per_program = (
                self.config.d_model * self.config.d_model + self.config.d_model
            )
        total_expert_parameters = selected_program_mask.new_tensor(
            float(self.config.n_programs * expert_parameters_per_program)
        )
        if self.config.program_compute_type == "sparse_linear_expert":
            active_programs = (selected_weights > 0).float().sum(dim=-1).mean()
            active_expert_parameters = active_programs * expert_parameters_per_program
        else:
            active_expert_parameters = total_expert_parameters
        return {
            "active_expert_parameters": active_expert_parameters,
            "total_expert_parameters": total_expert_parameters,
            "active_expert_fraction": active_expert_parameters / total_expert_parameters.clamp_min(1.0),
            "sink_programs": selected_program_mask.new_tensor(float(self.config.n_sink_programs)),
            "memory_tiers": selected_program_mask.new_tensor(
                3.0 if self.config.memory_tier_type == "hierarchical" else 1.0
            ),
            "routing_type": selected_program_mask.new_tensor(float(self._routing_type_id())),
            "routing_load_std": selected_program_mask.float().mean(dim=0).std(),
            "memory_lookup_slots": selected_program_mask.new_tensor(
                float(
                    self.config.memory_lookup_slots
                    if self.config.memory_lookup_type == "product_key"
                    else 0
                )
            ),
            "residual_streams": selected_program_mask.new_tensor(
                2.0 if self.config.residual_stream_type == "dual_stream" else 1.0
            ),
            "sequence_mixer_type": selected_program_mask.new_tensor(
                float(_sequence_mixer_type_id(self.config.sequence_mixer_type))
            ),
        }

    def _route_programs(
        self,
        stability: Tensor,
        activations: Optional[Tensor] = None,
        decision_memory: Optional[Tensor] = None,
    ) -> Tensor:
        conditioned_stability = self._condition_route_signal(
            stability,
            decision_memory,
        )
        conditioned_activations = (
            self._condition_route_signal(activations, decision_memory)
            if activations is not None
            else None
        )
        if self.config.routing_type == "expert_choice":
            return self._expert_choice_route(conditioned_stability)
        if self.config.routing_type == "base":
            return self._base_route(conditioned_stability)
        if self.config.routing_type == "hash":
            return self._hash_route(conditioned_stability)
        if self.config.routing_type == "sparse_ensemble":
            return self._sparse_ensemble_route(conditioned_stability)
        if self.config.routing_type == "base_semantic":
            return self._base_semantic_route(
                conditioned_stability,
                conditioned_activations,
            )
        if self.config.routing_type == "base_semantic_soft":
            return self._base_semantic_soft_route(
                conditioned_stability,
                conditioned_activations,
            )
        if self.config.routing_type == "authority_gated":
            return self._authority_gated_route(
                conditioned_stability,
                conditioned_activations,
            )
        return self._energy_route(conditioned_stability)

    def _condition_route_signal(
        self,
        signal: Tensor,
        decision_memory: Optional[Tensor],
    ) -> Tensor:
        if (
            decision_memory is None
            or self.config.decision_continuity_strength <= 0.0
        ):
            return signal
        memory = decision_memory.to(device=signal.device, dtype=signal.dtype)
        active = memory.sum(dim=-1, keepdim=True) > 1e-6
        if not bool(active.any()):
            return signal
        normalized = memory / memory.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        biased = signal + self.config.decision_continuity_strength * normalized
        return torch.where(active, biased, signal)

    def _authority_gated_route(
        self,
        stability: Tensor,
        activations: Optional[Tensor],
    ) -> Tensor:
        if (
            self.authority_router is None
            or self.authority_program_head is None
            or self.authority_mode_head is None
            or self.authority_halt_head is None
        ):
            raise RuntimeError("authority-gated routing modules are not initialized")

        activation_source = activations if activations is not None else stability
        authority_features = self._authority_features(stability, activation_source)
        router_input = torch.cat([stability, activation_source, authority_features], dim=-1)
        router_state = self.authority_router(router_input)
        authority_program_score = self.authority_program_head(router_state)
        authority_logits = self._apply_authority_bias(
            self.authority_mode_head(router_state),
            authority_features,
        )
        authority_probs = F.softmax(authority_logits, dim=-1)
        authority_indices = authority_probs.argmax(dim=-1)
        halt_probability = torch.sigmoid(self.authority_halt_head(router_state)).squeeze(-1)
        verifier_required = self._authority_verifier_required(
            authority_indices,
            authority_features,
        )

        self._last_authority_logits = authority_logits
        self._last_authority_probs = authority_probs
        self._last_authority_indices = authority_indices
        self._last_verifier_required = verifier_required
        self._last_halt_probability = halt_probability

        costs = self.energy_costs.detach().to(stability.dtype)
        route_score = (
            authority_program_score
            + activation_source.detach()
            + stability.detach()
        ) / costs
        route_score = self._mask_semantic_route_score(route_score)
        adaptive_start = self.config.n_sink_programs
        adaptive_programs = self.config.n_programs - adaptive_start
        if adaptive_programs <= 0:
            routed = torch.zeros_like(stability)
            routed[..., :adaptive_start] = 1.0
            return routed

        k = min(max(1, self.config.routing_top_k), adaptive_programs)
        top_values, top_indices = route_score.topk(k=k, dim=-1)
        routed = torch.zeros_like(stability)
        routed = routed.scatter(
            dim=-1,
            index=top_indices,
            src=torch.isfinite(top_values).to(stability.dtype),
        )
        routed = self._trim_route_to_budget(routed, route_score, costs)
        routed = self._ensure_at_least_one_adaptive_route(routed, route_score)
        if adaptive_start:
            routed[..., :adaptive_start] = 1.0
        return self._straight_through_route(routed, route_score)

    def _authority_features(self, stability: Tensor, activations: Tensor) -> Tensor:
        stability_conf = stability.detach().clamp(0.0, 1.0)
        activation_conf = activations.detach().clamp(0.0, 1.0)
        max_stability = stability_conf.max(dim=-1).values
        max_activation = activation_conf.max(dim=-1).values
        activation_mass = activation_conf.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        activation_probs = activation_conf / activation_mass
        entropy = -(
            activation_probs * torch.log(activation_probs.clamp_min(1e-6))
        ).sum(dim=-1)
        normalized_entropy = entropy / torch.log(
            activation_conf.new_tensor(float(max(self.config.n_programs, 2)))
        )
        exact_verified = (max_stability * max_activation).clamp(0.0, 1.0)
        proposal_available = (max_activation - max_stability).clamp(0.0, 1.0)
        calibrated_confidence = torch.maximum(max_activation, max_stability).clamp(0.0, 1.0)
        contamination_risk = normalized_entropy.clamp(0.0, 1.0)
        verifier_confidence = (
            calibrated_confidence * (1.0 - contamination_risk)
        ).clamp(0.0, 1.0)
        memory_distance = (1.0 - max_stability).clamp(0.0, 1.0)
        return torch.stack(
            [
                exact_verified,
                proposal_available,
                calibrated_confidence,
                contamination_risk,
                verifier_confidence,
                memory_distance,
            ],
            dim=-1,
        )

    def _apply_authority_bias(self, logits: Tensor, features: Tensor) -> Tensor:
        exact = features[:, 0]
        proposal = features[:, 1]
        calibrated = features[:, 2]
        contamination = features[:, 3]
        verifier_confidence = features[:, 4]
        memory_distance = features[:, 5]
        biased = logits.clone()
        exact_index = AUTHORITY_EXACT_MEMORY_INDEX
        proposal_index = AUTHORITY_PROPOSAL_VERIFIED_INDEX
        fast_path_index = AUTHORITY_CALIBRATED_FAST_PATH_INDEX
        repair_index = AUTHORITY_FRESH_REPAIR_INDEX
        verify_index = AUTHORITY_SYSTEM2_VERIFY_INDEX
        biased[:, exact_index] = (
            biased[:, exact_index]
            + 8.0 * exact
            - 8.0 * contamination
            - 2.0 * memory_distance
        )
        biased[:, proposal_index] = (
            biased[:, proposal_index]
            + 5.0 * proposal
            + 2.0 * verifier_confidence
            - 6.0 * contamination
        )
        biased[:, fast_path_index] = (
            biased[:, fast_path_index]
            + 4.0 * calibrated
            - 6.0 * contamination
        )
        biased[:, repair_index] = (
            biased[:, repair_index]
            + 2.0 * contamination
            + 1.0 * (1.0 - verifier_confidence)
        )
        biased[:, verify_index] = (
            biased[:, verify_index]
            + 3.0 * contamination
            + 1.0 * (1.0 - calibrated)
        )
        return biased

    def _authority_verifier_required(
        self,
        authority_indices: Tensor,
        features: Tensor,
    ) -> Tensor:
        exact = authority_indices == AUTHORITY_EXACT_MEMORY_INDEX
        calibrated_fast_path = authority_indices == AUTHORITY_CALIBRATED_FAST_PATH_INDEX
        trusted_fast_path = (
            calibrated_fast_path
            & (features[:, 2] >= self.config.authority_trusted_threshold)
            & (features[:, 3] <= 0.05)
        )
        return ~(exact | trusted_fast_path)

    def _straight_through_route(self, routed: Tensor, route_score: Tensor) -> Tensor:
        if not route_score.requires_grad:
            return routed

        finite = torch.isfinite(route_score)
        safe_score = route_score.masked_fill(
            ~finite,
            torch.finfo(route_score.dtype).min,
        )
        soft_route = F.softmax(safe_score, dim=-1)
        soft_route = torch.where(
            finite,
            soft_route,
            torch.zeros_like(soft_route),
        )
        route_mass = soft_route.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        soft_route = soft_route / route_mass
        hard_adaptive_mass = (routed.detach() * finite.to(routed.dtype)).sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        soft_route = soft_route * hard_adaptive_mass
        return routed + soft_route - soft_route.detach()

    def _energy_route(self, stability: Tensor) -> Tensor:
        costs = self.energy_costs.detach()
        route_score = stability.detach() / costs
        if self.config.n_sink_programs:
            route_score = route_score.clone()
            route_score[..., : self.config.n_sink_programs] = float("-inf")
        order = torch.argsort(route_score, dim=-1, descending=True)
        sorted_costs = costs.to(stability.dtype)[order]
        sorted_mask = torch.zeros_like(stability)
        remaining = torch.full(
            (stability.shape[0],),
            self.config.energy_budget,
            device=stability.device,
            dtype=stability.dtype,
        )

        for rank in range(stability.shape[-1]):
            candidate_cost = sorted_costs[:, rank]
            can_select = candidate_cost <= remaining
            sorted_mask[:, rank] = can_select.to(stability.dtype)
            remaining = remaining - torch.where(
                can_select,
                candidate_cost,
                torch.zeros_like(candidate_cost),
            )

        routed = torch.zeros_like(stability).scatter(dim=-1, index=order, src=sorted_mask)
        if self.config.n_sink_programs:
            routed[..., : self.config.n_sink_programs] = 1.0
        return routed

    def _expert_choice_route(self, stability: Tensor) -> Tensor:
        costs = self.energy_costs.detach()
        route_score = stability.detach() / costs
        adaptive_start = self.config.n_sink_programs
        if adaptive_start:
            route_score = route_score.clone()
            route_score[..., :adaptive_start] = float("-inf")

        routed = torch.zeros_like(stability)
        adaptive_programs = max(self.config.n_programs - adaptive_start, 1)
        target_active = max(1, int(self.config.energy_budget / costs.mean().item()))
        capacity = max(1, (stability.shape[0] * target_active + adaptive_programs - 1) // adaptive_programs)
        for program_index in range(adaptive_start, self.config.n_programs):
            k = min(capacity, stability.shape[0])
            chosen_tokens = torch.topk(route_score[:, program_index], k=k, dim=0).indices
            routed[chosen_tokens, program_index] = 1.0
        routed = self._ensure_at_least_one_adaptive_route(routed, route_score)
        routed = self._trim_route_to_budget(routed, route_score, costs)
        if adaptive_start:
            routed[..., :adaptive_start] = 1.0
        return routed

    def _base_route(self, stability: Tensor) -> Tensor:
        costs = self.energy_costs.detach()
        routed = torch.zeros_like(stability)
        adaptive_start = self.config.n_sink_programs
        adaptive_programs = max(self.config.n_programs - adaptive_start, 1)
        row_ids = torch.arange(stability.shape[0], device=stability.device)
        primary = adaptive_start + (row_ids % adaptive_programs)
        routed[row_ids, primary] = 1.0
        route_score = stability.detach() / costs
        routed = self._trim_route_to_budget(routed, route_score, costs)
        routed = self._ensure_at_least_one_adaptive_route(routed, route_score)
        if adaptive_start:
            routed[..., :adaptive_start] = 1.0
        return routed

    def _sparse_ensemble_route(self, stability: Tensor) -> Tensor:
        routed = self._base_route(stability)
        adaptive_start = self.config.n_sink_programs
        adaptive_programs = self.config.n_programs - adaptive_start
        if adaptive_programs <= 1 or self.config.routing_top_k <= 1:
            return routed

        costs = self.energy_costs.detach()
        route_score = stability.detach() / costs
        if adaptive_start:
            route_score = route_score.clone()
            route_score[..., :adaptive_start] = float("-inf")

        extra_k = min(self.config.routing_top_k - 1, adaptive_programs - 1)
        candidate_score = route_score.masked_fill(routed.bool(), float("-inf"))
        extra = candidate_score.topk(k=extra_k, dim=-1).indices
        routed = routed.scatter(dim=-1, index=extra, value=1.0)
        routed = self._trim_route_to_budget(routed, route_score, costs)
        routed = self._ensure_at_least_one_adaptive_route(routed, route_score)
        if adaptive_start:
            routed[..., :adaptive_start] = 1.0
        return routed

    def _base_semantic_route(
        self,
        stability: Tensor,
        activations: Optional[Tensor],
    ) -> Tensor:
        routed = self._base_route(stability)
        adaptive_start = self.config.n_sink_programs
        adaptive_programs = self.config.n_programs - adaptive_start
        if adaptive_programs <= 1 or self.config.routing_top_k <= 1:
            return routed

        costs = self.energy_costs.detach()
        semantic_source = activations if activations is not None else stability
        semantic_score = semantic_source / costs
        semantic_score = self._mask_semantic_route_score(semantic_score)

        extra_k = min(self.config.routing_top_k - 1, adaptive_programs - 1)
        candidate_score = semantic_score.masked_fill(routed.bool(), float("-inf"))
        extra_values, extra = candidate_score.topk(k=extra_k, dim=-1)
        extra_mask = torch.isfinite(extra_values).to(routed.dtype)
        routed = routed.scatter(dim=-1, index=extra, src=extra_mask)
        routed = self._trim_route_to_budget(routed, semantic_score, costs)
        routed = self._ensure_at_least_one_adaptive_route(routed, semantic_score)
        if adaptive_start:
            routed[..., :adaptive_start] = 1.0
        return self._straight_through_route(routed, candidate_score)

    def _base_semantic_soft_route(
        self,
        stability: Tensor,
        activations: Optional[Tensor],
    ) -> Tensor:
        routed = self._base_route(stability)
        adaptive_start = self.config.n_sink_programs
        adaptive_programs = self.config.n_programs - adaptive_start
        if adaptive_programs <= 1:
            return routed

        semantic_source = activations if activations is not None else stability
        costs = self.energy_costs.detach().to(stability.dtype)
        semantic_score = semantic_source / costs
        semantic_score = self._mask_semantic_route_score(semantic_score)
        candidate_score = semantic_score.masked_fill(routed.bool(), float("-inf"))
        semantic_probs = torch.softmax(candidate_score, dim=-1)
        semantic_probs = torch.where(
            torch.isfinite(candidate_score),
            semantic_probs,
            torch.zeros_like(semantic_probs),
        )
        semantic_mass = float(max(self.config.routing_top_k - 1, 1))
        routed = routed + semantic_probs * semantic_mass
        if adaptive_start:
            routed[..., :adaptive_start] = 1.0
        return routed

    def _mask_semantic_route_score(self, semantic_score: Tensor) -> Tensor:
        allowed = self.config.semantic_route_allowed_programs
        suppressed = self.config.semantic_route_suppressed_programs
        adaptive_start = self.config.n_sink_programs
        if allowed is None and suppressed is None and adaptive_start == 0:
            return semantic_score

        masked = semantic_score.clone()
        if allowed is not None:
            allowed_mask = torch.zeros(
                self.config.n_programs,
                device=masked.device,
                dtype=torch.bool,
            )
            allowed_mask[list(allowed)] = True
            masked = masked.masked_fill(~allowed_mask, float("-inf"))
        if suppressed is not None:
            masked[..., list(suppressed)] = float("-inf")
        if adaptive_start:
            masked[..., :adaptive_start] = float("-inf")
        return masked

    def _validate_program_filter(
        self,
        programs: Optional[tuple[int, ...]],
        name: str,
        *,
        allow_empty: bool,
    ) -> None:
        if programs is None:
            return
        if not allow_empty and len(programs) == 0:
            raise ValueError(f"{name} must not be empty when provided")
        seen = set()
        for program_id in programs:
            if program_id in seen:
                raise ValueError(f"{name} must not contain duplicate program IDs")
            seen.add(program_id)
            if program_id < 0 or program_id >= self.config.n_programs:
                raise ValueError(
                    f"{name} entries must be between 0 and n_programs - 1"
                )

    def _hash_route(self, stability: Tensor) -> Tensor:
        costs = self.energy_costs.detach()
        routed = torch.zeros_like(stability)
        adaptive_start = self.config.n_sink_programs
        adaptive_programs = max(self.config.n_programs - adaptive_start, 1)
        row_ids = torch.arange(stability.shape[0], device=stability.device)
        top_program = stability.detach()[..., adaptive_start:].argmax(dim=-1)
        hashed = (row_ids * 1103515245 + top_program * 12345 + 97) % adaptive_programs
        chosen = adaptive_start + hashed
        routed[row_ids, chosen] = 1.0
        route_score = stability.detach() / costs
        routed = self._trim_route_to_budget(routed, route_score, costs)
        routed = self._ensure_at_least_one_adaptive_route(routed, route_score)
        if adaptive_start:
            routed[..., :adaptive_start] = 1.0
        return routed

    def _ensure_at_least_one_adaptive_route(
        self,
        routed: Tensor,
        route_score: Tensor,
    ) -> Tensor:
        adaptive_start = self.config.n_sink_programs
        adaptive = routed[..., adaptive_start:]
        if adaptive.shape[-1] == 0:
            return routed
        missing = adaptive.sum(dim=-1) == 0
        if not bool(missing.any()):
            return routed
        best = route_score[..., adaptive_start:].argmax(dim=-1) + adaptive_start
        routed = routed.clone()
        routed[missing, best[missing]] = 1.0
        return routed

    def _trim_route_to_budget(
        self,
        routed: Tensor,
        route_score: Tensor,
        costs: Tensor,
    ) -> Tensor:
        adaptive_start = self.config.n_sink_programs
        if adaptive_start >= self.config.n_programs:
            return routed
        score = route_score.clone()
        score[..., :adaptive_start] = float("-inf")
        score = score.masked_fill(routed <= 0, float("-inf"))
        order = torch.argsort(score, dim=-1, descending=True)
        sorted_costs = costs.to(routed.dtype)[order]
        sorted_selected = routed.gather(dim=-1, index=order)
        sorted_mask = torch.zeros_like(routed)
        remaining = torch.full(
            (routed.shape[0],),
            self.config.energy_budget,
            device=routed.device,
            dtype=routed.dtype,
        )
        for rank in range(routed.shape[-1]):
            candidate_cost = sorted_costs[:, rank]
            can_select = (sorted_selected[:, rank] > 0) & (candidate_cost <= remaining)
            sorted_mask[:, rank] = can_select.to(routed.dtype)
            remaining = remaining - torch.where(
                can_select,
                candidate_cost,
                torch.zeros_like(candidate_cost),
            )
        trimmed = torch.zeros_like(routed).scatter(dim=-1, index=order, src=sorted_mask)
        return trimmed

    def _routing_type_id(self) -> int:
        return {
            "energy": 0,
            "expert_choice": 1,
            "base": 2,
            "hash": 3,
            "sparse_ensemble": 4,
            "base_semantic": 5,
            "base_semantic_soft": 6,
            "authority_gated": 7,
        }[self.config.routing_type]

    def _routing_load_balance_loss(self, token_selected_program_mask: Tensor) -> Tensor:
        if token_selected_program_mask.numel() == 0:
            return token_selected_program_mask.new_zeros(())
        load = token_selected_program_mask.float().mean(dim=tuple(range(token_selected_program_mask.ndim - 1)))
        target = load.new_full(load.shape, 1.0 / max(load.numel(), 1))
        normalized = load / load.sum().clamp_min(1e-6)
        return (normalized - target).pow(2).mean()

    def _used_route_energy(self, selected_program_mask: Tensor) -> Tensor:
        if self.config.n_sink_programs == 0:
            return torch.matmul(selected_program_mask, self.energy_costs)
        adaptive_mask = selected_program_mask.clone()
        adaptive_mask[..., : self.config.n_sink_programs] = 0.0
        return torch.matmul(adaptive_mask, self.energy_costs)

    def _compute_program_context(
        self,
        hidden: Tensor,
        selected_weights: Tensor,
        selected_denominator: Tensor,
        previous_memory: Tensor,
        previous_engram_patterns: Tensor,
        previous_engram_values: Tensor,
        previous_engram_mask: Tensor,
        previous_content_cues: Tensor,
        previous_content_values: Tensor,
        previous_content_mask: Tensor,
    ) -> Tensor:
        routed_weights = selected_weights / selected_denominator
        if self.config.program_compute_type == "embedding":
            program_context = torch.matmul(routed_weights, self.program_embeddings)
            coalition_context = self._compute_coalition_context(
                hidden,
                routed_weights,
                previous_memory,
            )
            context = program_context + self._read_program_memory(
                hidden,
                routed_weights,
                previous_memory,
                previous_engram_patterns,
                previous_engram_values,
                previous_engram_mask,
                previous_content_cues,
                previous_content_values,
                previous_content_mask,
            )
            return context + coalition_context + self._lookup_sparse_memory(hidden)

        if self.config.program_compute_type not in {
            "linear_expert",
            "sparse_linear_expert",
            "low_rank_linear_expert",
        }:
            raise ValueError(
                "program_compute_type must be 'embedding', 'linear_expert', 'sparse_linear_expert', or 'low_rank_linear_expert'"
            )
        if self.config.program_compute_type == "low_rank_linear_expert":
            if (
                self.program_expert_down is None
                or self.program_expert_up is None
                or self.program_expert_bias is None
            ):
                raise RuntimeError("low-rank linear expert parameters are not initialized")
        elif self.program_expert_weight is None or self.program_expert_bias is None:
            raise RuntimeError("linear expert parameters are not initialized")

        if self.config.program_compute_type == "sparse_linear_expert":
            hidden_for_experts = self._apply_coalition_context(
                hidden,
                routed_weights,
                previous_memory,
            )
            program_context = self._compute_sparse_program_context(
                hidden_for_experts,
                routed_weights,
            )
            context = program_context + self._read_program_memory(
                hidden,
                routed_weights,
                previous_memory,
                previous_engram_patterns,
                previous_engram_values,
                previous_engram_mask,
                previous_content_cues,
                previous_content_values,
                previous_content_mask,
            )
            return context + self._lookup_sparse_memory(hidden)

        if routed_weights.dim() == 3:
            program_specific_context = self._compute_program_specific_coalition_context(
                hidden,
                routed_weights,
                previous_memory,
            )
            if program_specific_context is not None:
                hidden_for_experts = hidden[:, :, None, :] + program_specific_context
                expert_outputs = self._program_axis_expert_outputs_for_sequence(
                    hidden_for_experts
                )
                program_context = (expert_outputs * routed_weights[..., None]).sum(dim=-2)
                context = program_context + self._read_program_memory(
                    hidden,
                    routed_weights,
                    previous_memory,
                    previous_engram_patterns,
                    previous_engram_values,
                    previous_engram_mask,
                    previous_content_cues,
                    previous_content_values,
                    previous_content_mask,
                )
                return context + self._lookup_sparse_memory(hidden)

            hidden_for_experts = self._apply_coalition_context(
                hidden,
                routed_weights,
                previous_memory,
            )
            expert_outputs = self._all_program_expert_outputs_for_sequence(
                hidden_for_experts
            )
            program_context = (expert_outputs * routed_weights[..., None]).sum(dim=-2)
            context = program_context + self._read_program_memory(
                hidden,
                routed_weights,
                previous_memory,
                previous_engram_patterns,
                previous_engram_values,
                previous_engram_mask,
                previous_content_cues,
                previous_content_values,
                previous_content_mask,
            )
            return context + self._lookup_sparse_memory(hidden)

        pooled_hidden = hidden.mean(dim=1)
        program_specific_context = self._compute_program_specific_coalition_context(
            pooled_hidden,
            routed_weights,
            previous_memory,
        )
        if program_specific_context is not None:
            pooled_hidden_for_experts = pooled_hidden[:, None, :] + program_specific_context
            expert_outputs = self._program_axis_expert_outputs_for_batch(
                pooled_hidden_for_experts
            )
        else:
            pooled_hidden = self._apply_coalition_context(
                pooled_hidden,
                routed_weights,
                previous_memory,
            )
            expert_outputs = self._all_program_expert_outputs_for_batch(pooled_hidden)
        program_context = (expert_outputs * routed_weights[..., None]).sum(dim=-2)
        context = program_context + self._read_program_memory(
            hidden,
            routed_weights,
            previous_memory,
            previous_engram_patterns,
            previous_engram_values,
            previous_engram_mask,
            previous_content_cues,
            previous_content_values,
            previous_content_mask,
        )
        return context + self._lookup_sparse_memory(hidden)

    def _program_axis_expert_outputs_for_sequence(self, hidden_for_experts: Tensor) -> Tensor:
        if self.config.program_compute_type == "low_rank_linear_expert":
            if self.program_expert_down is None or self.program_expert_up is None:
                raise RuntimeError("low-rank linear expert parameters are not initialized")
            reduced = torch.einsum(
                "bspd,pdr->bspr",
                hidden_for_experts,
                self.program_expert_down,
            )
            outputs = torch.einsum("bspr,prf->bspf", reduced, self.program_expert_up)
        else:
            if self.program_expert_weight is None:
                raise RuntimeError("linear expert parameters are not initialized")
            outputs = torch.einsum(
                "bspd,pdf->bspf",
                hidden_for_experts,
                self.program_expert_weight,
            )
        if self.program_expert_bias is None:
            raise RuntimeError("linear expert bias is not initialized")
        return outputs + self.program_expert_bias[None, None, :, :]

    def _all_program_expert_outputs_for_sequence(self, hidden_for_experts: Tensor) -> Tensor:
        if self.config.program_compute_type == "low_rank_linear_expert":
            if self.program_expert_down is None or self.program_expert_up is None:
                raise RuntimeError("low-rank linear expert parameters are not initialized")
            reduced = torch.einsum(
                "bsd,pdr->bspr",
                hidden_for_experts,
                self.program_expert_down,
            )
            outputs = torch.einsum("bspr,prf->bspf", reduced, self.program_expert_up)
        else:
            if self.program_expert_weight is None:
                raise RuntimeError("linear expert parameters are not initialized")
            outputs = torch.einsum(
                "bsd,pdf->bspf",
                hidden_for_experts,
                self.program_expert_weight,
            )
        if self.program_expert_bias is None:
            raise RuntimeError("linear expert bias is not initialized")
        return outputs + self.program_expert_bias[None, None, :, :]

    def _program_axis_expert_outputs_for_batch(self, hidden_for_experts: Tensor) -> Tensor:
        if self.config.program_compute_type == "low_rank_linear_expert":
            if self.program_expert_down is None or self.program_expert_up is None:
                raise RuntimeError("low-rank linear expert parameters are not initialized")
            reduced = torch.einsum(
                "bpd,pdr->bpr",
                hidden_for_experts,
                self.program_expert_down,
            )
            outputs = torch.einsum("bpr,prf->bpf", reduced, self.program_expert_up)
        else:
            if self.program_expert_weight is None:
                raise RuntimeError("linear expert parameters are not initialized")
            outputs = torch.einsum(
                "bpd,pdf->bpf",
                hidden_for_experts,
                self.program_expert_weight,
            )
        if self.program_expert_bias is None:
            raise RuntimeError("linear expert bias is not initialized")
        return outputs + self.program_expert_bias[None, :, :]

    def _all_program_expert_outputs_for_batch(self, hidden_for_experts: Tensor) -> Tensor:
        if self.config.program_compute_type == "low_rank_linear_expert":
            if self.program_expert_down is None or self.program_expert_up is None:
                raise RuntimeError("low-rank linear expert parameters are not initialized")
            reduced = torch.einsum(
                "bd,pdr->bpr",
                hidden_for_experts,
                self.program_expert_down,
            )
            outputs = torch.einsum("bpr,prf->bpf", reduced, self.program_expert_up)
        else:
            if self.program_expert_weight is None:
                raise RuntimeError("linear expert parameters are not initialized")
            outputs = torch.einsum(
                "bd,pdf->bpf",
                hidden_for_experts,
                self.program_expert_weight,
            )
        if self.program_expert_bias is None:
            raise RuntimeError("linear expert bias is not initialized")
        return outputs + self.program_expert_bias[None, :, :]

    def _apply_coalition_context(
        self,
        hidden: Tensor,
        routed_weights: Tensor,
        previous_memory: Tensor,
    ) -> Tensor:
        return hidden + self._compute_coalition_context(
            hidden,
            routed_weights,
            previous_memory,
        )

    def _compute_coalition_context(
        self,
        hidden: Tensor,
        routed_weights: Tensor,
        previous_memory: Tensor,
    ) -> Tensor:
        if self.coalition_context_projection is None:
            self._last_coalition_context_norm = hidden.new_zeros(())
            return torch.zeros_like(hidden)
        if self.config.coalition_context_type in {
            "program_memory_graph",
            "program_memory_task_graph",
        }:
            program_context = self._compute_program_specific_coalition_context(
                hidden,
                routed_weights,
                previous_memory,
            )
            if program_context is None:
                self._last_coalition_context_norm = hidden.new_zeros(())
                return torch.zeros_like(hidden)
            if routed_weights.dim() == 3:
                return (program_context * routed_weights[..., None]).sum(dim=-2)
            return (program_context * routed_weights[..., None]).sum(dim=-2)
        if routed_weights.dim() == 3:
            coalition_memory = torch.einsum(
                "bsp,bpd->bsd",
                routed_weights,
                previous_memory,
            )
        else:
            coalition_memory = torch.einsum(
                "bp,bpd->bd",
                routed_weights,
                previous_memory,
            )
        coalition_context = self.coalition_context_projection(coalition_memory)
        coalition_context = coalition_context * self.config.coalition_context_scale
        self._last_coalition_context_norm = coalition_context.norm(dim=-1).mean()
        return coalition_context

    def _compute_program_specific_coalition_context(
        self,
        hidden: Tensor,
        routed_weights: Tensor,
        previous_memory: Tensor,
    ) -> Optional[Tensor]:
        if self.config.coalition_context_type not in {
            "program_memory_graph",
            "program_memory_task_graph",
        }:
            return None
        if (
            self.coalition_context_projection is None
            or self.coalition_source_key_projection is None
            or self.coalition_source_value_projection is None
            or self.coalition_target_query_projection is None
        ):
            raise RuntimeError("graph coalition context parameters are not initialized")
        if (
            self.config.coalition_context_type == "program_memory_task_graph"
            and self.coalition_task_query_projection is None
        ):
            raise RuntimeError("task-conditioned graph coalition parameters are not initialized")

        source_keys = self.coalition_source_key_projection(previous_memory)
        source_values = self.coalition_source_value_projection(previous_memory)
        target_queries = self.coalition_target_query_projection(self.program_embeddings)

        if routed_weights.dim() == 3:
            if self.config.coalition_context_type == "program_memory_task_graph":
                task_queries = self.coalition_task_query_projection(hidden)
                conditioned_target_queries = (
                    target_queries[None, None, :, :] + task_queries[:, :, None, :]
                )
                adjacency_logits = torch.einsum(
                    "bspd,bqd->bspq",
                    conditioned_target_queries,
                    source_keys,
                ) / sqrt(self.config.d_model)
                adjacency = F.softmax(adjacency_logits, dim=-1)
            else:
                adjacency_logits = torch.einsum(
                    "pd,bqd->bpq",
                    target_queries,
                    source_keys,
                ) / sqrt(self.config.d_model)
                adjacency = F.softmax(adjacency_logits, dim=-1)[:, None, :, :]
            source_weights = adjacency * routed_weights[:, :, None, :]
            source_weights = source_weights / source_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)
            coalition_memory = torch.einsum(
                "bspq,bqd->bspd",
                source_weights,
                source_values,
            )
        else:
            if self.config.coalition_context_type == "program_memory_task_graph":
                task_queries = self.coalition_task_query_projection(hidden)
                conditioned_target_queries = (
                    target_queries[None, :, :] + task_queries[:, None, :]
                )
                adjacency_logits = torch.einsum(
                    "bpd,bqd->bpq",
                    conditioned_target_queries,
                    source_keys,
                ) / sqrt(self.config.d_model)
            else:
                adjacency_logits = torch.einsum(
                    "pd,bqd->bpq",
                    target_queries,
                    source_keys,
                ) / sqrt(self.config.d_model)
            adjacency = F.softmax(adjacency_logits, dim=-1)
            source_weights = adjacency * routed_weights[:, None, :]
            source_weights = source_weights / source_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)
            coalition_memory = torch.einsum(
                "bpq,bqd->bpd",
                source_weights,
                source_values,
            )

        coalition_context = self.coalition_context_projection(coalition_memory)
        coalition_context = coalition_context * self.config.coalition_context_scale
        self._last_coalition_context_norm = coalition_context.norm(dim=-1).mean()
        return coalition_context

    def _lookup_sparse_memory(self, hidden: Tensor) -> Tensor:
        if self.config.memory_lookup_type == "none":
            return torch.zeros_like(hidden)
        if (
            self.memory_lookup_query is None
            or self.memory_lookup_key_a is None
            or self.memory_lookup_key_b is None
            or self.memory_lookup_values is None
        ):
            raise RuntimeError("product-key memory parameters are not initialized")
        query = self.memory_lookup_query(hidden)
        first_dim = self.memory_lookup_key_a.shape[-1]
        query_a = F.normalize(query[..., :first_dim], dim=-1)
        query_b = F.normalize(query[..., first_dim:], dim=-1)
        key_a = F.normalize(self.memory_lookup_key_a, dim=-1)
        key_b = F.normalize(self.memory_lookup_key_b, dim=-1)
        scores = torch.matmul(query_a, key_a.T) + torch.matmul(query_b, key_b.T)
        top_k = min(4, self.config.memory_lookup_slots)
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
        weights = F.softmax(top_scores, dim=-1)
        values = self.memory_lookup_values[top_indices]
        return (values * weights[..., None]).sum(dim=-2)

    def _compute_sparse_program_context(
        self,
        hidden: Tensor,
        routed_weights: Tensor,
    ) -> Tensor:
        if self.program_expert_weight is None or self.program_expert_bias is None:
            raise RuntimeError("linear expert parameters are not initialized")

        if routed_weights.dim() == 3:
            batch_size, seq_len, d_model = hidden.shape
            flat_hidden = hidden.reshape(batch_size * seq_len, d_model)
            flat_weights_by_program = routed_weights.reshape(
                batch_size * seq_len,
                self.config.n_programs,
            )
            flat_context = self._batched_sparse_expert_context(
                flat_hidden,
                flat_weights_by_program,
            )
            return flat_context.reshape(batch_size, seq_len, d_model)

        batch_size, d_model = hidden.shape[0], hidden.shape[-1]
        pooled_hidden = hidden.mean(dim=1)
        return self._batched_sparse_expert_context(pooled_hidden, routed_weights)

    def _batched_sparse_expert_context(
        self,
        flat_hidden: Tensor,
        flat_weights_by_program: Tensor,
    ) -> Tensor:
        if self.program_expert_weight is None or self.program_expert_bias is None:
            raise RuntimeError("linear expert parameters are not initialized")
        active_mask = flat_weights_by_program > 0
        max_active = int(active_mask.sum(dim=-1).max().item())
        self.latest_sparse_dispatch_size = max_active
        if max_active == 0:
            return flat_hidden.new_zeros(flat_hidden.shape)

        selected_weights, selected_programs = torch.topk(
            flat_weights_by_program,
            k=max_active,
            dim=-1,
        )
        selected_expert_weight = self.program_expert_weight[selected_programs]
        selected_expert_bias = self.program_expert_bias[selected_programs]
        selected_outputs = (
            torch.einsum("nd,nkdf->nkf", flat_hidden, selected_expert_weight)
            + selected_expert_bias
        )
        return (selected_outputs * selected_weights[..., None]).sum(dim=1)

    def _read_program_memory(
        self,
        hidden: Tensor,
        routed_weights: Tensor,
        previous_memory: Tensor,
        previous_engram_patterns: Tensor,
        previous_engram_values: Tensor,
        previous_engram_mask: Tensor,
        previous_content_cues: Tensor,
        previous_content_values: Tensor,
        previous_content_mask: Tensor,
    ) -> Tensor:
        if self.memory_gate is None:
            return torch.zeros_like(hidden)
        if self.config.memory_read_type == "content_addressed":
            if hidden.dim() == 3:
                memory_gate = torch.sigmoid(self.memory_gate(hidden)).mean(
                    dim=-1,
                    keepdim=True,
                )
                content_read = self._content_addressed_read_for_budgeted_queries(
                    hidden,
                    memory_gate,
                    previous_content_cues,
                    previous_content_values,
                    previous_content_mask,
                )
                return memory_gate * content_read
            content_read, _ = self._content_addressed_iterative_read(
                hidden,
                previous_content_cues,
                previous_content_values,
                previous_content_mask,
            )
            self._record_content_read_queries(
                hidden,
                query_count=hidden.shape[0],
                full_query_count=hidden.shape[0],
                mask=previous_content_mask,
            )
            memory_gate = self._pooled_memory_gate(hidden[:, None, :]).mean(
                dim=-1,
                keepdim=True,
            )
            return memory_gate * content_read
        if self.config.memory_read_type == "pattern_completion":
            pattern_read, _ = self._pattern_completion_read(
                routed_weights,
                previous_engram_patterns,
                previous_engram_values,
                previous_engram_mask,
            )
            if routed_weights.dim() == 3:
                memory_gate = torch.sigmoid(self.memory_gate(hidden)).mean(
                    dim=-1,
                    keepdim=True,
                )
                return memory_gate * pattern_read
            memory_gate = self._pooled_memory_gate(hidden).mean(dim=-1, keepdim=True)
            return memory_gate * pattern_read
        if routed_weights.dim() == 3:
            memory_gate = torch.sigmoid(self.memory_gate(hidden))
            memory_weights = routed_weights * memory_gate
            return torch.einsum("bsp,bpd->bsd", memory_weights, previous_memory)

        memory_gate = self._pooled_memory_gate(hidden)
        memory_weights = routed_weights * memory_gate
        return torch.einsum("bp,bpd->bd", memory_weights, previous_memory)

    def _content_addressed_read_for_budgeted_queries(
        self,
        hidden: Tensor,
        memory_gate: Tensor,
        cues: Tensor,
        values: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if hidden.dim() != 3:
            raise ValueError("budgeted content reads require sequence hidden states")
        batch_size, seq_len, d_model = hidden.shape
        top_k = self.config.content_read_query_top_k
        if top_k is None or top_k >= seq_len:
            content_read, _ = self._content_addressed_iterative_read(
                hidden,
                cues,
                values,
                mask,
            )
            self._record_content_read_queries(
                hidden,
                query_count=batch_size * seq_len,
                full_query_count=batch_size * seq_len,
                mask=mask,
            )
            return content_read

        selected_indices = memory_gate.squeeze(-1).topk(k=top_k, dim=1).indices
        gather_indices = selected_indices[:, :, None].expand(-1, -1, d_model)
        selected_hidden = hidden.gather(dim=1, index=gather_indices)
        selected_read, _ = self._content_addressed_iterative_read(
            selected_hidden,
            cues,
            values,
            mask,
        )
        content_read = hidden.new_zeros(hidden.shape)
        content_read = content_read.scatter(dim=1, index=gather_indices, src=selected_read)
        self._record_content_read_queries(
            hidden,
            query_count=batch_size * top_k,
            full_query_count=batch_size * seq_len,
            mask=mask,
        )
        return content_read

    def _record_content_read_queries(
        self,
        reference: Tensor,
        *,
        query_count: int,
        full_query_count: int,
        mask: Tensor,
    ) -> None:
        valid_batches = (mask > 0).any(dim=1).to(reference.dtype).sum()
        batch_size = max(mask.shape[0], 1)
        valid_fraction = valid_batches / reference.new_tensor(float(batch_size))
        actual = reference.new_tensor(float(query_count)) * valid_fraction
        full = reference.new_tensor(float(full_query_count)) * valid_fraction
        self._last_content_read_queries = actual
        self._last_content_read_query_fraction = actual / full.clamp_min(1.0)

    def _pattern_completion_read(
        self,
        routed_weights: Tensor,
        patterns: Tensor,
        values: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        current = routed_weights
        squeeze = False
        if current.dim() == 2:
            current = current[:, None, :]
            squeeze = True
        valid = mask > 0
        if not bool(valid.any()):
            read = current.new_zeros((*current.shape[:2], self.config.d_model))
            hit = current.new_zeros(())
            return (read[:, 0, :] if squeeze else read), hit

        current_norm = F.normalize(current, dim=-1)
        pattern_norm = F.normalize(patterns, dim=-1)
        scores = torch.einsum("bsp,bkp->bsk", current_norm, pattern_norm)
        scores = scores.masked_fill(~valid[:, None, :], -1e4)
        weights = F.softmax(scores, dim=-1) * mask[:, None, :]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        read = torch.einsum("bsk,bkd->bsd", weights, values)
        hit = scores.max(dim=-1).values.mean()
        return (read[:, 0, :] if squeeze else read), hit

    def _pattern_completion_hit(
        self,
        routed_weights: Tensor,
        patterns: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if self.config.memory_read_type != "pattern_completion":
            return routed_weights.new_zeros(())
        dummy_values = routed_weights.new_zeros(
            routed_weights.shape[0],
            patterns.shape[1],
            self.config.d_model,
        )
        _, hit = self._pattern_completion_read(
            routed_weights,
            patterns,
            dummy_values,
            mask,
        )
        return hit

    def _content_addressed_read(
        self,
        query_hidden: Tensor,
        cues: Tensor,
        values: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        read, hit, _ = self._content_addressed_read_with_token_hit(
            query_hidden,
            cues,
            values,
            mask,
        )
        return read, hit

    def _content_addressed_read_with_token_hit(
        self,
        query_hidden: Tensor,
        cues: Tensor,
        values: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        current = query_hidden
        squeeze = False
        if current.dim() == 2:
            current = current[:, None, :]
            squeeze = True
        valid = mask > 0
        if not bool(valid.any()):
            read = current.new_zeros(current.shape)
            hit = current.new_zeros(())
            token_hit = current.new_zeros(current.shape[:2])
            return (read[:, 0, :] if squeeze else read), hit, (token_hit[:, 0] if squeeze else token_hit)

        current_norm = F.normalize(current, dim=-1)
        cue_norm = F.normalize(cues, dim=-1)
        scores = torch.einsum("bsd,bkd->bsk", current_norm, cue_norm)
        scores = scores.masked_fill(~valid[:, None, :], -1e4)
        weights = F.softmax(scores, dim=-1) * mask[:, None, :]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        read = torch.einsum("bsk,bkd->bsd", weights, values)
        token_hit = scores.max(dim=-1).values
        hit = token_hit.mean()
        return (
            read[:, 0, :] if squeeze else read,
            hit,
            token_hit[:, 0] if squeeze else token_hit,
        )

    def _content_addressed_iterative_read(
        self,
        query_hidden: Tensor,
        cues: Tensor,
        values: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        first_read, first_hit, first_token_hit = self._content_addressed_read_with_token_hit(
            query_hidden,
            cues,
            values,
            mask,
        )
        if self.config.content_read_steps == 1:
            return first_read, first_hit

        second_read, second_hit, second_token_hit = self._content_addressed_read_with_token_hit(
            first_read,
            cues,
            values,
            mask,
        )
        current = query_hidden
        first = first_read
        squeeze = False
        if current.dim() == 2:
            current = current[:, None, :]
            first = first[:, None, :]
            second = second_read[:, None, :]
            squeeze = True
        else:
            second = second_read
        if self.config.content_read_gate_type == "confidence":
            first_confidence = first_token_hit[:, None] if squeeze else first_token_hit
            second_confidence = second_token_hit[:, None] if squeeze else second_token_hit
            continue_gate = torch.sigmoid(
                4.0 * (second_confidence - first_confidence)
            ).unsqueeze(-1)
            blended = (1.0 - continue_gate) * first + continue_gate * second
            self._record_content_gate_stats(continue_gate)
        elif self.config.content_read_gate_type == "confidence_margin":
            first_confidence = first_token_hit[:, None] if squeeze else first_token_hit
            second_confidence = second_token_hit[:, None] if squeeze else second_token_hit
            confidence_gap = (first_confidence - second_confidence).abs()
            margin = current.new_tensor(float(self.config.content_read_confidence_margin))
            continue_gate = torch.sigmoid(
                512.0 * (margin - confidence_gap)
            ).unsqueeze(-1)
            blended = (1.0 - continue_gate) * first + continue_gate * second
            self._record_content_gate_stats(continue_gate)
        elif self.config.content_read_gate_type == "cue_match":
            second_confidence = second_token_hit[:, None] if squeeze else second_token_hit
            threshold = current.new_tensor(
                float(self.config.content_read_cue_match_threshold)
            )
            continue_gate = torch.sigmoid(
                64.0 * (second_confidence - threshold)
            ).unsqueeze(-1)
            blended = (1.0 - continue_gate) * first + continue_gate * second
            self._record_content_gate_stats(continue_gate)
        elif (
            self.config.content_read_gate_type == "synthesis"
            and self.content_read_synthesis is not None
            and self.content_read_synthesis_gate is not None
        ):
            synthesis_input = torch.cat(
                [
                    current,
                    first,
                    second,
                    first - second,
                    first * second,
                ],
                dim=-1,
            )
            synthesized = self.content_read_synthesis(synthesis_input)
            synthesis_gate = torch.sigmoid(
                self.content_read_synthesis_gate(synthesis_input)
            )
            blended = (1.0 - synthesis_gate) * first + synthesis_gate * synthesized
            self._record_content_gate_stats(synthesis_gate)
        elif self.content_read_blend_gate is not None:
            confidence = first_hit.expand(*current.shape[:2], 1)
            gate_input = torch.cat([current, first, confidence], dim=-1)
            first_gate = torch.sigmoid(self.content_read_blend_gate(gate_input))
            blended = first_gate * first + (1.0 - first_gate) * second
            self._record_content_gate_stats(1.0 - first_gate)
        else:
            blended = first
        hit = 0.5 * (first_hit + second_hit)
        return (blended[:, 0, :] if squeeze else blended), hit

    def _record_content_gate_stats(self, gate: Tensor) -> None:
        eps = 1e-6
        gate = gate.clamp(eps, 1.0 - eps)
        entropy = -(
            gate * gate.log()
            + (1.0 - gate) * (1.0 - gate).log()
        )
        max_entropy = gate.new_tensor(0.6931471805599453)
        self._last_content_synthesis_gate = gate.mean()
        self._last_content_gate_entropy = entropy.mean() / max_entropy
        self._last_content_gate_entropy_loss = (
            1.0 - entropy / max_entropy
        ).mean()

    def _content_addressed_hit(
        self,
        hidden: Tensor,
        cues: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if self.config.memory_read_type != "content_addressed":
            return hidden.new_zeros(())
        dummy_values = hidden.new_zeros(
            hidden.shape[0],
            cues.shape[1],
            self.config.d_model,
        )
        _, hit = self._content_addressed_iterative_read(hidden, cues, dummy_values, mask)
        return hit


class IdentityAugmentedSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        causal: bool = False,
        position_type: str = "learned",
        rope_base: float = 10000.0,
        rope_scale: float = 1.0,
        rope_scaling_type: str = "none",
        original_context_length: Optional[int] = None,
        target_context_length: Optional[int] = None,
        n_kv_heads: Optional[int] = None,
        attention_window_size: Optional[int] = None,
        use_identity_key_value: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_kv_heads < 1:
            raise ValueError("n_kv_heads must be at least 1")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if position_type not in {"learned", "rope"}:
            raise ValueError("position_type must be 'learned' or 'rope'")
        if position_type == "rope" and (d_model // n_heads) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if rope_base <= 0.0:
            raise ValueError("rope_base must be positive")
        if rope_scale <= 0.0:
            raise ValueError("rope_scale must be positive")
        if rope_scaling_type not in {"none", "linear", "yarn"}:
            raise ValueError("rope_scaling_type must be 'none', 'linear', or 'yarn'")
        if attention_window_size is not None and attention_window_size < 1:
            raise ValueError("attention_window_size must be positive when set")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.position_type = position_type
        self.rope_base = rope_base
        self.rope_scale = rope_scale
        self.rope_scaling_type = rope_scaling_type
        self.original_context_length = original_context_length
        self.target_context_length = target_context_length
        self.attention_window_size = attention_window_size
        self.query = nn.Linear(d_model, d_model)
        self.key_value = nn.Linear(d_model, self.n_kv_heads * self.head_dim * 2)
        self.identity_key_value = (
            nn.Linear(d_model * 2, self.n_kv_heads * self.head_dim * 2)
            if use_identity_key_value
            else None
        )
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        *,
        coherence: Optional[Tensor] = None,
        compressed_memory: Optional[Tensor] = None,
        identity_context: Optional[Tensor] = None,
        identity_sparse_mask: Optional[Tensor] = None,
        beta: float = 1.0,
        # run5b_plus: hybrid sliding window mask, [seq] or [batch, seq].
        global_token_mask: Optional[Tensor] = None,
    ) -> AttentionOutput:
        batch_size, seq_len, _ = hidden.shape
        query = self.query(hidden)
        if identity_context is not None:
            if self.identity_key_value is None:
                raise ValueError("identity-aware K/V projection is not initialized")
            if identity_context.shape != hidden.shape:
                raise ValueError("identity_context must match hidden shape")
            key, value = self.identity_key_value(
                torch.cat([hidden, identity_context], dim=-1)
            ).chunk(2, dim=-1)
        else:
            key, value = self.key_value(hidden).chunk(2, dim=-1)

        query = self._split_query_heads(query)
        key = self._split_key_value_heads(key)
        value = self._split_key_value_heads(value)

        if self.position_type == "rope":
            query, key = _apply_rope(
                query,
                key,
                base=self.rope_base,
                scale=self.rope_scale,
                scaling_type=self.rope_scaling_type,
                original_context_length=self.original_context_length,
                target_context_length=self.target_context_length,
            )

        memory_key = None
        memory_value = None
        memory_len = 0
        if compressed_memory is not None:
            memory_len = compressed_memory.shape[1]
            memory_key, memory_value = self.key_value(compressed_memory).chunk(2, dim=-1)
            memory_key = self._split_key_value_heads(memory_key)
            memory_value = self._split_key_value_heads(memory_value)

        key = self._repeat_key_value_heads(key)
        value = self._repeat_key_value_heads(value)

        if (
            self.attention_window_size is not None
            and compressed_memory is None
            and self.causal
            and global_token_mask is None
        ):
            return self._forward_causal_sliding_window(
                query,
                key,
                value,
                coherence=coherence,
                identity_sparse_mask=identity_sparse_mask,
                beta=beta,
            )

        # run5b_plus: hybrid sliding window — build combined mask where global positions
        # can attend to all prior tokens, and non-global positions use sliding window.
        if (
            self.attention_window_size is not None
            and compressed_memory is None
            and self.causal
            and global_token_mask is not None
        ):
            # True marks positions that should bypass the local attention window.
            positions = torch.arange(seq_len, device=hidden.device)
            window_mask = positions[None, :] < (
                positions[:, None] - self.attention_window_size + 1
            )  # True = masked-OUT in causal sliding window
            global_token_mask = global_token_mask.to(
                device=hidden.device,
                dtype=torch.bool,
            )
            if global_token_mask.dim() == 1:
                if global_token_mask.shape != (seq_len,):
                    raise ValueError("global_token_mask must have shape [seq] or [batch, seq]")
                global_row = global_token_mask.unsqueeze(1)
                global_col = global_token_mask.unsqueeze(0)
                hybrid_window_mask = window_mask & ~global_row & ~global_col
            elif global_token_mask.dim() == 2:
                if global_token_mask.shape != (batch_size, seq_len):
                    raise ValueError("global_token_mask must have shape [seq] or [batch, seq]")
                global_row = global_token_mask[:, :, None]
                global_col = global_token_mask[:, None, :]
                hybrid_window_mask = window_mask[None, :, :] & ~global_row & ~global_col
            else:
                raise ValueError("global_token_mask must have shape [seq] or [batch, seq]")
            _hybrid_window_override: Optional[Tensor] = hybrid_window_mask
        else:
            _hybrid_window_override = None

        if memory_key is not None and memory_value is not None:
            memory_key = self._repeat_key_value_heads(memory_key)
            memory_value = self._repeat_key_value_heads(memory_value)
            key = torch.cat([memory_key, key], dim=-2)
            value = torch.cat([memory_value, value], dim=-2)

        attention_logits = torch.matmul(query, key.transpose(-1, -2)) / sqrt(
            self.head_dim
        )

        if coherence is not None and beta != 0:
            token_attention_logits = attention_logits[..., memory_len:]
            token_attention_logits = token_attention_logits + beta * coherence[:, None, :, :]
            attention_logits = torch.cat(
                [attention_logits[..., :memory_len], token_attention_logits],
                dim=-1,
            )

        if identity_sparse_mask is not None:
            if identity_sparse_mask.shape != (batch_size, seq_len, seq_len):
                raise ValueError("identity_sparse_mask must have shape [batch, seq, seq]")
            token_attention_logits = attention_logits[..., memory_len:]
            token_attention_logits = token_attention_logits.masked_fill(
                ~identity_sparse_mask[:, None, :, :].to(torch.bool),
                float("-inf"),
            )
            attention_logits = torch.cat(
                [attention_logits[..., :memory_len], token_attention_logits],
                dim=-1,
            )

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool),
                diagonal=1,
            )
            token_attention_logits = attention_logits[..., memory_len:]
            token_attention_logits = token_attention_logits.masked_fill(
                causal_mask,
                float("-inf"),
            )
            attention_logits = torch.cat(
                [attention_logits[..., :memory_len], token_attention_logits],
                dim=-1,
            )

        if self.attention_window_size is not None:
            if _hybrid_window_override is not None:
                # run5b_plus hybrid path: precomputed mask already accounts for global tokens
                effective_window_mask = _hybrid_window_override
            else:
                positions = torch.arange(seq_len, device=hidden.device)
                if self.causal:
                    effective_window_mask = positions[None, :] < (
                        positions[:, None] - self.attention_window_size + 1
                    )
                else:
                    effective_window_mask = (
                        positions[None, :] - positions[:, None]
                    ).abs() >= self.attention_window_size
            token_attention_logits = attention_logits[..., memory_len:]
            if effective_window_mask.dim() == 3:
                effective_window_mask = effective_window_mask[:, None, :, :]
            token_attention_logits = token_attention_logits.masked_fill(
                effective_window_mask,
                float("-inf"),
            )
            attention_logits = torch.cat(
                [attention_logits[..., :memory_len], token_attention_logits],
                dim=-1,
            )

        attention_probs = F.softmax(attention_logits, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attended = torch.matmul(attention_probs, value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        return AttentionOutput(
            hidden=self.out(attended),
            attention_probs=attention_probs,
            attention_logits=attention_logits,
        )

    def _split_query_heads(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(
            1, 2
        )

    def _split_key_value_heads(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(
            batch_size,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _repeat_key_value_heads(self, tensor: Tensor) -> Tensor:
        if self.n_kv_heads == self.n_heads:
            return tensor
        repeat_factor = self.n_heads // self.n_kv_heads
        return tensor.repeat_interleave(repeat_factor, dim=1)

    def _forward_causal_sliding_window(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        coherence: Optional[Tensor],
        identity_sparse_mask: Optional[Tensor],
        beta: float,
    ) -> AttentionOutput:
        batch_size, _, seq_len, _ = query.shape
        window_size = min(self.attention_window_size or seq_len, seq_len)
        key_windows = _causal_token_windows(key, window_size)
        value_windows = _causal_token_windows(value, window_size)
        attention_logits = (query[:, :, :, None, :] * key_windows).sum(dim=-1) / sqrt(
            self.head_dim
        )
        if coherence is not None and beta != 0:
            local_coherence = _causal_coherence_windows(coherence, window_size)
            attention_logits = attention_logits + beta * local_coherence[:, None, :, :]
        if identity_sparse_mask is not None:
            if identity_sparse_mask.shape != (batch_size, seq_len, seq_len):
                raise ValueError("identity_sparse_mask must have shape [batch, seq, seq]")
            local_sparse_mask = _causal_mask_windows(
                identity_sparse_mask.to(torch.bool),
                window_size,
            )
            attention_logits = attention_logits.masked_fill(
                ~local_sparse_mask[:, None, :, :],
                float("-inf"),
            )
        valid_mask = _causal_window_valid_mask(
            seq_len,
            window_size,
            device=query.device,
        )
        attention_logits = attention_logits.masked_fill(
            ~valid_mask[None, None, :, :],
            float("-inf"),
        )
        attention_probs = F.softmax(attention_logits, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attended = (attention_probs[..., None] * value_windows).sum(dim=-2)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        return AttentionOutput(
            hidden=self.out(attended),
            attention_probs=attention_probs,
            attention_logits=attention_logits,
        )


def _causal_token_windows(tensor: Tensor, window_size: int) -> Tensor:
    padded = F.pad(tensor, (0, 0, window_size - 1, 0))
    windows = padded.unfold(dimension=2, size=window_size, step=1)
    return windows.permute(0, 1, 2, 4, 3).contiguous()


def _causal_coherence_windows(coherence: Tensor, window_size: int) -> Tensor:
    batch_size, seq_len, _ = coherence.shape
    offsets = torch.arange(window_size, device=coherence.device)
    positions = torch.arange(seq_len, device=coherence.device)
    indices = positions[:, None] - window_size + 1 + offsets[None, :]
    gather_indices = indices.clamp_min(0)[None, :, :].expand(
        batch_size,
        seq_len,
        window_size,
    )
    return coherence.gather(dim=2, index=gather_indices)


def _causal_mask_windows(mask: Tensor, window_size: int) -> Tensor:
    batch_size, seq_len, _ = mask.shape
    offsets = torch.arange(window_size, device=mask.device)
    positions = torch.arange(seq_len, device=mask.device)
    indices = positions[:, None] - window_size + 1 + offsets[None, :]
    gather_indices = indices.clamp_min(0)[None, :, :].expand(
        batch_size,
        seq_len,
        window_size,
    )
    return mask.gather(dim=2, index=gather_indices)


def _causal_window_valid_mask(
    seq_len: int,
    window_size: int,
    *,
    device: torch.device,
) -> Tensor:
    offsets = torch.arange(window_size, device=device)
    positions = torch.arange(seq_len, device=device)
    invalid_prefix = (window_size - 1 - positions).clamp_min(0)
    return offsets[None, :] >= invalid_prefix[:, None]


def _build_norm(config: TACConfig) -> nn.Module:
    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.d_model)
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model)
    raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")


def _build_mlp(config: TACConfig) -> nn.Module:
    if config.mlp_type == "gelu":
        return GELUFeedForward(config)
    if config.mlp_type == "swiglu":
        return SwiGLUFeedForward(config)
    raise ValueError("mlp_type must be 'gelu' or 'swiglu'")


def _build_position_embedding(config: TACConfig) -> nn.Embedding | None:
    if config.position_type == "learned":
        return nn.Embedding(config.max_seq_len, config.d_model)
    if config.position_type == "rope":
        return None
    raise ValueError("position_type must be 'learned' or 'rope'")


def _uses_attention_mixer(config: TACConfig, layer_index: int) -> bool:
    if config.sequence_mixer_type in {"attention", "hybrid"}:
        return True
    if config.sequence_mixer_type == "alternating":
        return layer_index % 2 == 0
    return False


def _uses_state_mixer(config: TACConfig, layer_index: int) -> bool:
    if config.sequence_mixer_type in {
        "state",
        "hybrid",
        "selective_state",
        "rwkv",
        "xlstm",
    }:
        return True
    if config.sequence_mixer_type == "alternating":
        return layer_index % 2 == 1
    return False


def _build_sequence_state_mixer(config: TACConfig) -> nn.Module:
    if config.sequence_mixer_type in {"state", "hybrid", "alternating"}:
        return CausalStateMixer(config)
    if config.sequence_mixer_type == "selective_state":
        return SelectiveStateMixer(config)
    if config.sequence_mixer_type == "rwkv":
        return RWKVTimeMixer(config)
    if config.sequence_mixer_type == "xlstm":
        return XLSTMStyleMixer(config)
    raise ValueError(f"unsupported state mixer type: {config.sequence_mixer_type}")


def _sequence_mixer_type_id(sequence_mixer_type: str) -> int:
    return {
        "attention": 1,
        "state": 2,
        "hybrid": 3,
        "alternating": 4,
        "selective_state": 5,
        "rwkv": 6,
        "xlstm": 7,
    }[sequence_mixer_type]


def _empty_attention_output(hidden: Tensor, n_heads: int) -> AttentionOutput:
    batch_size, seq_len, _ = hidden.shape
    empty = hidden.new_zeros(batch_size, n_heads, seq_len, seq_len)
    return AttentionOutput(
        hidden=torch.zeros_like(hidden),
        attention_probs=empty,
        attention_logits=empty,
    )


def _apply_token_positions(hidden: Tensor, position_embedding: nn.Embedding | None) -> Tensor:
    if position_embedding is None:
        return hidden
    seq_len = hidden.shape[1]
    positions = torch.arange(seq_len, device=hidden.device)[None, :]
    return hidden + position_embedding(positions)


def _resolve_content_write_policy(
    write_policy: ContentWritePolicy | str | None,
) -> ContentWritePolicy | None:
    if write_policy is None:
        return None
    if isinstance(write_policy, ContentWritePolicy):
        return write_policy
    try:
        return ContentWritePolicy(write_policy)
    except ValueError as exc:
        valid = ", ".join(policy.value for policy in ContentWritePolicy)
        raise ValueError(f"unknown write_policy {write_policy!r}; expected one of: {valid}") from exc


def _resolve_content_update_enabled(
    write_policy: ContentWritePolicy | str | None,
    *,
    seq_len: int,
    update_content_memory: bool,
) -> bool:
    policy = _resolve_content_write_policy(write_policy)
    if policy is None:
        return update_content_memory
    if policy == ContentWritePolicy.DENSE:
        return True
    if policy == ContentWritePolicy.DISABLED:
        return False
    if policy in {
        ContentWritePolicy.QUERY_SKIP,
        ContentWritePolicy.MASKED_PREFILL_QUERY_SKIP,
        ContentWritePolicy.DECODE_STATE_SKIP,
    }:
        return seq_len > 1
    raise ValueError(f"unknown write_policy {write_policy!r}")


def _resolve_identity_state_update_enabled(
    write_policy: ContentWritePolicy | str | None,
    *,
    seq_len: int,
) -> bool:
    policy = _resolve_content_write_policy(write_policy)
    if policy == ContentWritePolicy.DECODE_STATE_SKIP:
        return seq_len > 1
    return True


def _apply_rope(
    query: Tensor,
    key: Tensor,
    *,
    base: float = 10000.0,
    scale: float = 1.0,
    scaling_type: str = "none",
    original_context_length: Optional[int] = None,
    target_context_length: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    seq_len = query.shape[-2]
    head_dim = query.shape[-1]
    if base <= 0.0:
        raise ValueError("base must be positive")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if scaling_type not in {"none", "linear", "yarn"}:
        raise ValueError("scaling_type must be 'none', 'linear', or 'yarn'")
    effective_scale = _effective_rope_scale(
        scale=scale,
        original_context_length=original_context_length,
        target_context_length=target_context_length,
    )
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=query.device, dtype=query.dtype)
            / head_dim
        )
    )
    positions = torch.arange(seq_len, device=query.device, dtype=query.dtype)
    if scaling_type == "none" or effective_scale == 1.0:
        freqs = torch.outer(positions, inv_freq)
    elif scaling_type == "linear":
        freqs = torch.outer(positions / effective_scale, inv_freq)
    else:
        freqs = _yarn_scaled_rope_frequencies(
            positions,
            inv_freq,
            scale=effective_scale,
        )
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    return _rotate_half_pairs(query, cos, sin), _rotate_half_pairs(key, cos, sin)


def _effective_rope_scale(
    *,
    scale: float,
    original_context_length: Optional[int],
    target_context_length: Optional[int],
) -> float:
    if scale != 1.0:
        return scale
    if original_context_length is None or target_context_length is None:
        return scale
    if original_context_length <= 0 or target_context_length <= 0:
        raise ValueError("context lengths must be positive when set")
    return max(float(target_context_length) / float(original_context_length), 1.0)


def _yarn_scaled_rope_frequencies(
    positions: Tensor,
    inv_freq: Tensor,
    *,
    scale: float,
) -> Tensor:
    if inv_freq.numel() == 1:
        return torch.outer(positions / scale, inv_freq)
    ramp = torch.linspace(
        0.0,
        1.0,
        steps=inv_freq.numel(),
        device=inv_freq.device,
        dtype=inv_freq.dtype,
    )
    interpolation_weight = (1.0 - ramp).clamp(0.0, 1.0)
    dim_scale = scale ** interpolation_weight
    return positions[:, None] * inv_freq[None, :] / dim_scale[None, :]


def _rotate_half_pairs(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class TACTransformerBlock(nn.Module):
    def __init__(self, config: TACConfig, layer_index: int = 0):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        if config.identity_attention_type not in {
            "none",
            "compressed_memory",
            "coherence_sparse",
            "coherence_sparse_compressed",
            "identity_first",
        }:
            raise ValueError(
                "identity_attention_type must be 'none', 'compressed_memory', 'coherence_sparse', 'coherence_sparse_compressed', or 'identity_first'"
            )
        if config.attention_window_size is not None and config.attention_window_size < 1:
            raise ValueError("attention_window_size must be positive when set")
        if config.residual_stream_type not in {"single", "dual_stream"}:
            raise ValueError("residual_stream_type must be 'single' or 'dual_stream'")
        if config.sequence_mixer_type not in {
            "attention",
            "state",
            "hybrid",
            "alternating",
            "selective_state",
            "rwkv",
            "xlstm",
        }:
            raise ValueError(
                "sequence_mixer_type must be 'attention', 'state', 'hybrid', 'alternating', 'selective_state', 'rwkv', or 'xlstm'"
            )
        self.norm_attention = _build_norm(config)
        self.norm_mlp = _build_norm(config)
        self.identity_field = IdentityFieldLayer(config)
        self.uses_attention = _uses_attention_mixer(config, layer_index)
        self.uses_state_mixer = _uses_state_mixer(config, layer_index)
        self.attention = (
            IdentityAugmentedSelfAttention(
                config.d_model,
                config.n_heads,
                dropout=config.dropout,
                causal=config.causal,
                position_type=config.position_type,
                rope_base=config.rope_base,
                rope_scale=config.rope_scale,
                rope_scaling_type=config.rope_scaling_type,
                original_context_length=config.original_context_length,
                target_context_length=config.target_context_length,
                n_kv_heads=config.n_kv_heads,
                attention_window_size=config.attention_window_size,
                use_identity_key_value=config.identity_attention_type
                == "identity_first",
            )
            if self.uses_attention
            else None
        )
        self.state_mixer = (
            _build_sequence_state_mixer(config) if self.uses_state_mixer else None
        )
        self.program_projection = nn.Linear(config.d_model, config.d_model)
        if config.residual_stream_type == "dual_stream":
            self.content_stream_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.identity_stream_gate = nn.Linear(config.d_model * 2, config.d_model)
        else:
            self.content_stream_gate = None
            self.identity_stream_gate = None
        self.mlp = _build_mlp(config)

    def forward(
        self,
        hidden: Tensor,
        previous_state: Optional[IdentityState] = None,
        *,
        collect_auxiliary: bool = True,
        collect_metrics: bool = True,
        update_content_memory: bool = True,
        update_identity_state: bool = True,
        content_write_mask: Optional[Tensor] = None,
        global_token_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, IdentityState, IdentityFieldOutput, AttentionOutput]:
        normalized = self.norm_attention(hidden)
        if self.layer_index >= self.config.tac_active_layer_start:
            identity = self.identity_field(
                normalized,
                previous_state,
                collect_auxiliary=collect_auxiliary,
                collect_metrics=collect_metrics,
                update_content_memory=update_content_memory,
                update_identity_state=update_identity_state,
                content_write_mask=content_write_mask,
            )
        else:
            identity = self._inactive_identity_output(normalized, previous_state)
        compressed_memory = None
        if (
            self.config.identity_attention_type
            in {"compressed_memory", "coherence_sparse_compressed"}
            and previous_state is not None
        ):
            compressed_memory = self._compressed_identity_memory(
                previous_state,
                hidden.device,
            )
        identity_context = None
        if self.config.identity_attention_type == "identity_first":
            identity_context = identity.program_identity
        identity_sparse_mask = None
        if self.config.identity_attention_type in {
            "coherence_sparse",
            "coherence_sparse_compressed",
        }:
            identity_sparse_mask = (
                identity.program_assignments[:, :, None]
                == identity.program_assignments[:, None, :]
            )
        if self.attention is not None:
            attention = self.attention(
                normalized,
                coherence=identity.coherence,
                compressed_memory=compressed_memory,
                identity_context=identity_context,
                identity_sparse_mask=identity_sparse_mask,
                beta=self.config.beta * self.config.coherence_attention_scale,
                global_token_mask=global_token_mask,
            )
            content_update = attention.hidden
        else:
            attention = _empty_attention_output(normalized, self.config.n_heads)
            content_update = torch.zeros_like(hidden)
        if self.state_mixer is not None:
            content_update = content_update + self.state_mixer(normalized)
        program_bias = self.program_projection(identity.program_context)
        if program_bias.dim() == 2:
            program_bias = program_bias[:, None, :]
        program_bias = self.config.program_residual_scale * program_bias
        if self.config.residual_stream_type == "dual_stream":
            if self.content_stream_gate is None or self.identity_stream_gate is None:
                raise RuntimeError("dual-stream residual gates are not initialized")
            content_gate = torch.sigmoid(
                self.content_stream_gate(torch.cat([hidden, content_update], dim=-1))
            )
            identity_gate = torch.sigmoid(
                self.identity_stream_gate(torch.cat([hidden, program_bias], dim=-1))
            )
            hidden = hidden + content_gate * content_update + identity_gate * program_bias
        else:
            hidden = hidden + content_update + program_bias
        hidden = hidden + self.mlp(self.norm_mlp(hidden))
        return hidden, identity.state, identity, attention

    def _inactive_identity_output(
        self,
        hidden: Tensor,
        previous_state: Optional[IdentityState],
    ) -> IdentityFieldOutput:
        batch_size, seq_len, _ = hidden.shape
        zero = hidden.new_zeros(())
        stability = hidden.new_zeros(batch_size, self.config.n_programs)
        program_memory = hidden.new_zeros(
            batch_size,
            self.config.n_programs,
            self.config.d_model,
        )
        if previous_state is not None:
            stability = previous_state.stability.to(hidden.device)
            program_memory = previous_state.program_memory.to(hidden.device)
        selected = hidden.new_zeros(batch_size, self.config.n_programs)
        token_selected = hidden.new_zeros(batch_size, seq_len, self.config.n_programs)
        losses = {
            "coherence": zero,
            "program_reuse": zero,
            "energy": zero,
            "separation": zero,
            "content_cue_separation": zero,
            "content_gate_entropy": zero,
            "routing_load_balance": zero,
            "decision_continuity": zero,
            "ebm_decision_continuity": zero,
            "activation_l1": zero,
            "identity_norm_floor": zero,
        }
        metrics = self.identity_field._minimal_metrics(selected, zero)
        metrics["tac_layer_active"] = zero
        return IdentityFieldOutput(
            coherence=hidden.new_zeros(batch_size, seq_len, seq_len),
            activations=selected,
            program_assignments=torch.zeros(
                batch_size,
                dtype=torch.long,
                device=hidden.device,
            ),
            program_identity=hidden.new_zeros(
                batch_size,
                seq_len,
                self.config.d_model,
            ),
            selected_program_mask=selected,
            used_energy=hidden.new_zeros(batch_size),
            program_context=hidden.new_zeros(batch_size, self.config.d_model),
            state=IdentityState(stability=stability, program_memory=program_memory),
            losses=losses,
            metrics=metrics,
            token_activations=token_selected,
            token_selected_program_mask=token_selected,
        )

    def _compressed_identity_memory(
        self,
        previous_state: IdentityState,
        device: torch.device,
    ) -> Tensor:
        base = previous_state.program_memory.to(device)
        if self.config.memory_system_type != "multi_timescale":
            return base
        working = (
            previous_state.working_state.to(device)
            if previous_state.working_state is not None
            else torch.zeros_like(base)
        )
        episodic = (
            previous_state.episodic_state.to(device)
            if previous_state.episodic_state is not None
            else torch.zeros_like(base)
        )
        semantic = (
            previous_state.semantic_state.to(device)
            if previous_state.semantic_state is not None
            else torch.zeros_like(base)
        )
        procedural = (
            previous_state.procedural_state.to(device)
            if previous_state.procedural_state is not None
            else torch.zeros_like(base)
        )
        return (
            0.25 * base
            + 0.15 * working
            + 0.20 * episodic
            + 0.20 * semantic
            + 0.20 * procedural
        )


class VanillaTransformerBlock(nn.Module):
    def __init__(self, config: TACConfig, layer_index: int = 0):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        if config.attention_window_size is not None and config.attention_window_size < 1:
            raise ValueError("attention_window_size must be positive when set")
        if config.sequence_mixer_type not in {
            "attention",
            "state",
            "hybrid",
            "alternating",
            "selective_state",
            "rwkv",
            "xlstm",
        }:
            raise ValueError(
                "sequence_mixer_type must be 'attention', 'state', 'hybrid', 'alternating', 'selective_state', 'rwkv', or 'xlstm'"
            )
        self.norm_attention = _build_norm(config)
        self.norm_mlp = _build_norm(config)
        self.uses_attention = _uses_attention_mixer(config, layer_index)
        self.uses_state_mixer = _uses_state_mixer(config, layer_index)
        self.attention = (
            IdentityAugmentedSelfAttention(
                config.d_model,
                config.n_heads,
                dropout=config.dropout,
                causal=config.causal,
                position_type=config.position_type,
                rope_base=config.rope_base,
                rope_scale=config.rope_scale,
                rope_scaling_type=config.rope_scaling_type,
                original_context_length=config.original_context_length,
                target_context_length=config.target_context_length,
                n_kv_heads=config.n_kv_heads,
                attention_window_size=config.attention_window_size,
            )
            if self.uses_attention
            else None
        )
        self.state_mixer = (
            _build_sequence_state_mixer(config) if self.uses_state_mixer else None
        )
        self.mlp = _build_mlp(config)

    def forward(
        self,
        hidden: Tensor,
        *,
        global_token_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, AttentionOutput]:
        normalized = self.norm_attention(hidden)
        if self.attention is not None:
            attention = self.attention(
                normalized,
                global_token_mask=global_token_mask,
            )
            content_update = attention.hidden
        else:
            attention = _empty_attention_output(normalized, self.config.n_heads)
            content_update = torch.zeros_like(hidden)
        if self.state_mixer is not None:
            content_update = content_update + self.state_mixer(normalized)
        hidden = hidden + content_update
        hidden = hidden + self.mlp(self.norm_mlp(hidden))
        return hidden, attention


class TACTransformerLM(nn.Module):
    def __init__(self, config: TACConfig):
        super().__init__()
        _validate_program_embed_dim(config)
        if config.n_prediction_heads < 1:
            raise ValueError("n_prediction_heads must be at least 1")
        if config.memory_read_type not in {
            "none",
            "program_memory",
            "pattern_completion",
            "content_addressed",
        }:
            raise ValueError(
                "memory_read_type must be 'none', 'program_memory', 'pattern_completion', or 'content_addressed'"
            )
        if config.content_read_steps < 1:
            raise ValueError("content_read_steps must be at least 1")
        if config.content_read_gate_type not in {
            "learned",
            "confidence",
            "confidence_margin",
            "cue_match",
            "synthesis",
        }:
            raise ValueError(
                "content_read_gate_type must be 'learned', 'confidence', 'confidence_margin', 'cue_match', or 'synthesis'"
            )
        if config.content_read_confidence_margin < 0.0:
            raise ValueError("content_read_confidence_margin must be non-negative")
        if config.content_read_cue_match_threshold < 0.0:
            raise ValueError("content_read_cue_match_threshold must be non-negative")
        if (
            config.content_read_query_top_k is not None
            and config.content_read_query_top_k < 1
        ):
            raise ValueError("content_read_query_top_k must be positive when set")
        if config.coalition_context_type not in {
            "none",
            "program_memory",
            "program_memory_graph",
            "program_memory_task_graph",
        }:
            raise ValueError(
                "coalition_context_type must be 'none', 'program_memory', 'program_memory_graph', or 'program_memory_task_graph'"
            )
        if config.coalition_context_scale < 0.0:
            raise ValueError("coalition_context_scale must be non-negative")
        if config.memory_separation_weight < 0.0:
            raise ValueError("memory_separation_weight must be non-negative")
        if config.content_cue_separation_weight < 0.0:
            raise ValueError("content_cue_separation_weight must be non-negative")
        if config.content_gate_entropy_weight < 0.0:
            raise ValueError("content_gate_entropy_weight must be non-negative")
        if config.routing_load_balance_weight < 0.0:
            raise ValueError("routing_load_balance_weight must be non-negative")
        if config.decision_continuity_strength < 0.0:
            raise ValueError("decision_continuity_strength must be non-negative")
        if not 0.0 <= config.decision_continuity_decay <= 1.0:
            raise ValueError("decision_continuity_decay must be between 0 and 1")
        if config.decision_continuity_loss_weight < 0.0:
            raise ValueError("decision_continuity_loss_weight must be non-negative")
        if config.tac_active_layer_start < 0 or config.tac_active_layer_start > config.n_layers:
            raise ValueError("tac_active_layer_start must be between 0 and n_layers")
        if not 0.0 <= config.content_reconsolidate_rate <= 1.0:
            raise ValueError("content_reconsolidate_rate must be between 0 and 1")
        if config.memory_adapter_type not in {"none", "residual", "gated_residual"}:
            raise ValueError(
                "memory_adapter_type must be 'none', 'residual', or 'gated_residual'"
            )
        if config.memory_bridge_type not in {
            "none",
            "multi_timescale_readout",
            "semantic_procedural_readout",
        }:
            raise ValueError(
                "memory_bridge_type must be 'none', 'multi_timescale_readout', or 'semantic_procedural_readout'"
            )
        if config.memory_bridge_weight < 0.0:
            raise ValueError("memory_bridge_weight must be non-negative")
        if config.program_residual_scale < 0.0:
            raise ValueError("program_residual_scale must be non-negative")
        if config.coherence_attention_scale < 0.0:
            raise ValueError("coherence_attention_scale must be non-negative")
        if config.lm_readout_type not in {
            "hidden",
            "slot_conditioned_program_bottleneck",
        }:
            raise ValueError(
                "lm_readout_type must be 'hidden' or 'slot_conditioned_program_bottleneck'"
            )
        if (
            config.lm_readout_type == "slot_conditioned_program_bottleneck"
            and config.program_compute_type
            not in {"linear_expert", "low_rank_linear_expert"}
        ):
            raise ValueError(
                "slot_conditioned_program_bottleneck requires linear or low-rank program experts"
            )
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = _build_position_embedding(config)
        self.blocks = nn.ModuleList(
            [
                TACTransformerBlock(config, layer_index=index)
                for index in range(config.n_layers)
            ]
        )
        self.final_norm = _build_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.multi_token_heads = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.vocab_size, bias=False)
                for _ in range(config.n_prediction_heads - 1)
            ]
        )
        if config.memory_adapter_type == "residual":
            self.memory_adapter = nn.Linear(config.d_model, config.d_model)
            self.memory_adapter_gate = None
        elif config.memory_adapter_type == "gated_residual":
            adapter_hidden_dim = config.d_model * config.mlp_ratio
            self.memory_adapter = nn.Sequential(
                nn.Linear(config.d_model, adapter_hidden_dim),
                nn.SiLU(),
                nn.Linear(adapter_hidden_dim, config.d_model),
            )
            self.memory_adapter_gate = nn.Linear(config.d_model * 2, config.d_model)
        else:
            self.memory_adapter = None
            self.memory_adapter_gate = None
        if config.memory_bridge_type != "none":
            self.memory_bridge_key_projection = nn.Linear(
                config.d_model,
                config.d_model,
                bias=False,
            )
            self.memory_bridge_value_projection = nn.Linear(
                config.d_model,
                config.d_model,
                bias=False,
            )
            self.memory_bridge_output_projection = nn.Linear(
                config.d_model,
                config.d_model,
                bias=False,
            )
            self.memory_bridge_gate = nn.Linear(config.d_model * 2, config.d_model)
        else:
            self.memory_bridge_key_projection = None
            self.memory_bridge_value_projection = None
            self.memory_bridge_output_projection = None
            self.memory_bridge_gate = None
        # run5b_plus: EBM data energy head (only when program_embed_dim is set)
        if config.program_embed_dim is not None:
            self.data_energy_head: Optional[DataEnergyHead] = DataEnergyHead(
                config.d_model,
                config.program_embed_dim,
                hidden_dim=config.ebm_head_hidden_dim,
            )
        else:
            self.data_energy_head = None

    def forward(
        self,
        input_ids: Tensor,
        *,
        identity_states: Optional[list[IdentityState]] = None,
        labels: Optional[Tensor] = None,
        collect_auxiliary: bool = True,
        collect_metrics: bool = True,
        update_content_memory: bool = True,
        content_write_mask: Optional[Tensor] = None,
        write_policy: ContentWritePolicy | str | None = None,
        authority_mode_targets: Optional[Tensor] = None,
        verifier_required_targets: Optional[Tensor] = None,
        authority_halt_targets: Optional[Tensor] = None,
    ) -> TACOutput:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        update_content_memory = _resolve_content_update_enabled(
            write_policy,
            seq_len=seq_len,
            update_content_memory=update_content_memory,
        )
        update_identity_state = _resolve_identity_state_update_enabled(
            write_policy,
            seq_len=seq_len,
        )

        hidden = _apply_token_positions(
            self.token_embedding(input_ids),
            self.position_embedding,
        )

        next_states = []
        identity_outputs = []
        attention_outputs = []
        global_token_mask = self._global_token_mask(input_ids)

        for index, block in enumerate(self.blocks):
            previous_state = identity_states[index] if identity_states else None
            hidden, state, identity, attention = block(
                hidden,
                previous_state,
                collect_auxiliary=collect_auxiliary,
                collect_metrics=collect_metrics,
                update_content_memory=update_content_memory,
                update_identity_state=update_identity_state,
                content_write_mask=content_write_mask,
                global_token_mask=global_token_mask,
            )
            state = self._update_content_token_memory(
                state,
                previous_state,
                input_ids,
                update_content_memory=update_content_memory,
                content_write_mask=content_write_mask,
            )
            identity.state = state
            next_states.append(state)
            identity_outputs.append(identity)
            attention_outputs.append(attention)

        hidden = self.final_norm(hidden)
        hidden, bridge_metrics = self._apply_memory_tier_bridge(
            hidden,
            identity_states,
        )
        hidden, readout_metrics = self._apply_lm_program_bottleneck(
            hidden,
            next_states,
            identity_outputs,
        )
        logits = self.lm_head(hidden)
        multi_token_logits = [head(hidden) for head in self.multi_token_heads]
        aux = self._merge_auxiliary(identity_outputs, attention_outputs)
        aux.metrics.update(bridge_metrics)
        aux.metrics.update(readout_metrics)
        if collect_auxiliary:
            self._add_data_energy_aux(aux, hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, self.config.vocab_size),
                labels.reshape(batch_size * seq_len),
            )
        aux.losses["multi_token"] = self._multi_token_loss(
            multi_token_logits,
            labels,
            logits,
        )
        self._add_authority_supervision_losses(
            aux,
            authority_mode_targets=authority_mode_targets,
            verifier_required_targets=verifier_required_targets,
            authority_halt_targets=authority_halt_targets,
        )

        return TACOutput(
            logits=logits,
            identity_states=next_states,
            aux=aux,
            loss=loss,
            hidden_states=hidden,
            multi_token_logits=multi_token_logits,
        )

    def _apply_lm_program_bottleneck(
        self,
        hidden: Tensor,
        next_states: list[IdentityState],
        identity_outputs: list[IdentityFieldOutput],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        zero = hidden.new_zeros(())
        metrics = {
            "lm_readout_type": hidden.new_tensor(
                1.0
                if self.config.lm_readout_type
                == "slot_conditioned_program_bottleneck"
                else 0.0
            ),
            "lm_program_bottleneck_delta_norm": zero,
            "lm_program_bottleneck_selected_mass": zero,
        }
        if self.config.lm_readout_type == "hidden":
            return hidden, metrics
        if not next_states or not identity_outputs:
            return hidden, metrics

        field = self.blocks[-1].identity_field
        state = next_states[-1]
        identity = identity_outputs[-1]
        selected = identity.token_selected_program_mask
        if selected is None:
            selected = identity.selected_program_mask[:, None, :].expand(
                hidden.shape[0],
                hidden.shape[1],
                identity.selected_program_mask.shape[-1],
            )
        selected = selected.to(dtype=hidden.dtype)
        routed_weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expert_inputs = hidden[:, :, None, :] + state.program_memory[:, None, :, :].to(
            dtype=hidden.dtype
        )
        expert_outputs = field._program_axis_expert_outputs_for_sequence(expert_inputs)
        bottleneck_hidden = (expert_outputs * routed_weights[..., None]).sum(dim=-2)
        metrics["lm_program_bottleneck_delta_norm"] = (
            bottleneck_hidden - hidden
        ).norm(dim=-1).mean()
        metrics["lm_program_bottleneck_selected_mass"] = selected.sum(dim=-1).mean()
        return bottleneck_hidden, metrics

    def _global_token_mask(self, input_ids: Tensor) -> Optional[Tensor]:
        if not self.config.global_attention_token_ids:
            return None
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.config.global_attention_token_ids:
            mask = mask | (input_ids == int(token_id))
        return mask

    def _apply_memory_tier_bridge(
        self,
        hidden: Tensor,
        identity_states: Optional[list[IdentityState]],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        zero = hidden.new_zeros(())
        metrics = {
            "memory_bridge_update_norm": zero,
            "memory_bridge_tier_entropy": zero,
        }
        if (
            self.config.memory_bridge_type == "none"
            or not identity_states
            or self.memory_bridge_key_projection is None
            or self.memory_bridge_value_projection is None
            or self.memory_bridge_output_projection is None
            or self.memory_bridge_gate is None
        ):
            return hidden, metrics
        state = identity_states[-1]
        tier_memory = self._memory_bridge_tier_memory(state, hidden)
        if tier_memory is None:
            return hidden, metrics

        keys = self.memory_bridge_key_projection(tier_memory)
        values = self.memory_bridge_value_projection(tier_memory)
        scores = torch.einsum("bsd,bmd->bsm", hidden, keys) / sqrt(self.config.d_model)
        weights = F.softmax(scores, dim=-1)
        read = torch.einsum("bsm,bmd->bsd", weights, values)
        gate = torch.sigmoid(self.memory_bridge_gate(torch.cat([hidden, read], dim=-1)))
        update = gate * self.memory_bridge_output_projection(read)
        bridged_hidden = hidden + self.config.memory_bridge_weight * update
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        metrics["memory_bridge_update_norm"] = update.norm(dim=-1).mean()
        metrics["memory_bridge_tier_entropy"] = entropy
        return bridged_hidden, metrics

    def _memory_bridge_tier_memory(
        self,
        state: IdentityState,
        hidden: Tensor,
    ) -> Optional[Tensor]:
        base = state.program_memory.to(device=hidden.device, dtype=hidden.dtype)
        zeros = torch.zeros_like(base)
        if self.config.memory_bridge_type == "semantic_procedural_readout":
            tiers = [
                state.semantic_state.to(device=hidden.device, dtype=hidden.dtype)
                if state.semantic_state is not None
                else zeros,
                state.procedural_state.to(device=hidden.device, dtype=hidden.dtype)
                if state.procedural_state is not None
                else zeros,
            ]
        elif self.config.memory_bridge_type == "multi_timescale_readout":
            tiers = [
                state.working_state.to(device=hidden.device, dtype=hidden.dtype)
                if state.working_state is not None
                else zeros,
                state.episodic_state.to(device=hidden.device, dtype=hidden.dtype)
                if state.episodic_state is not None
                else zeros,
                state.semantic_state.to(device=hidden.device, dtype=hidden.dtype)
                if state.semantic_state is not None
                else zeros,
                state.procedural_state.to(device=hidden.device, dtype=hidden.dtype)
                if state.procedural_state is not None
                else zeros,
            ]
        else:
            return None
        return torch.cat(tiers, dim=1)

    def _add_data_energy_aux(
        self,
        aux: TACAuxiliaryOutput,
        hidden: Tensor,
    ) -> None:
        if self.data_energy_head is None or aux.token_selected_program_mask is None:
            return
        last_field = self.blocks[-1].identity_field
        selected_identity = torch.matmul(
            aux.token_selected_program_mask,
            last_field._program_identity_embeddings(),
        )
        clean_energy = self.data_energy_head(
            hidden.detach(),
            selected_identity.detach(),
        )
        if selected_identity.shape[1] > 1:
            corrupt_identity = selected_identity.roll(shifts=1, dims=1)
        else:
            corrupt_identity = selected_identity.roll(shifts=1, dims=0)
        corrupt_energy = self.data_energy_head(
            hidden.detach(),
            corrupt_identity.detach(),
        )
        aux.data_energy = clean_energy
        aux.losses["data_energy"] = F.relu(
            1.0 + clean_energy - corrupt_energy
        ).mean()
        aux.metrics["data_energy_clean"] = clean_energy.mean()
        aux.metrics["data_energy_corrupt"] = corrupt_energy.mean()

    def _update_content_token_memory(
        self,
        state: IdentityState,
        previous_state: Optional[IdentityState],
        input_ids: Tensor,
        *,
        update_content_memory: bool,
        content_write_mask: Optional[Tensor],
    ) -> IdentityState:
        if self.config.memory_read_type != "content_addressed" or input_ids.shape[1] < 2:
            return state
        batch_size = input_ids.shape[0]
        device = input_ids.device
        previous_cues = (
            previous_state.content_cue_token_ids.to(device)
            if previous_state is not None
            and previous_state.content_cue_token_ids is not None
            else torch.full(
                (batch_size, self.config.content_store_size),
                -1,
                dtype=torch.long,
                device=device,
            )
        )
        previous_values = (
            previous_state.content_value_token_ids.to(device)
            if previous_state is not None
            and previous_state.content_value_token_ids is not None
            else torch.full(
                (batch_size, self.config.content_store_size),
                -1,
                dtype=torch.long,
                device=device,
            )
        )
        if not update_content_memory:
            return replace(
                state,
                content_cue_token_ids=previous_cues,
                content_value_token_ids=previous_values,
            )

        cue_tokens = input_ids[:, :-1]
        value_tokens = input_ids[:, 1:]
        pair_slots = cue_tokens.shape[1]
        stored_cues = previous_cues.clone()
        stored_values = previous_values.clone()
        if content_write_mask is None:
            pair_count = min(pair_slots, self.config.content_store_size)
            stored_cues = torch.roll(stored_cues, shifts=-pair_count, dims=1)
            stored_values = torch.roll(stored_values, shifts=-pair_count, dims=1)
            stored_cues[:, -pair_count:] = cue_tokens[:, :pair_count]
            stored_values[:, -pair_count:] = value_tokens[:, :pair_count]
        else:
            if content_write_mask.shape != cue_tokens.shape:
                raise ValueError("content_write_mask must have shape (batch, seq_len - 1)")
            write_mask = content_write_mask.to(device=device, dtype=torch.bool)
            for batch_index in range(batch_size):
                selected = torch.nonzero(
                    write_mask[batch_index],
                    as_tuple=False,
                ).flatten()
                if selected.numel() == 0:
                    continue
                selected = selected[: self.config.content_store_size]
                count = int(selected.numel())
                stored_cues[batch_index] = torch.roll(
                    stored_cues[batch_index],
                    shifts=-count,
                    dims=0,
                )
                stored_values[batch_index] = torch.roll(
                    stored_values[batch_index],
                    shifts=-count,
                    dims=0,
                )
                stored_cues[batch_index, -count:] = cue_tokens[batch_index, selected]
                stored_values[batch_index, -count:] = value_tokens[batch_index, selected]
        return replace(
            state,
            content_cue_token_ids=stored_cues,
            content_value_token_ids=stored_values,
        )

    def _add_authority_supervision_losses(
        self,
        aux: TACAuxiliaryOutput,
        *,
        authority_mode_targets: Optional[Tensor],
        verifier_required_targets: Optional[Tensor],
        authority_halt_targets: Optional[Tensor],
    ) -> None:
        if (
            authority_mode_targets is None
            and verifier_required_targets is None
            and authority_halt_targets is None
        ):
            return
        if (
            aux.authority_logits is None
            or aux.authority_probs is None
            or aux.halt_probability is None
        ):
            raise ValueError(
                "authority supervision requires routing_type='authority_gated'"
            )

        if authority_mode_targets is not None:
            targets = authority_mode_targets.to(
                device=aux.authority_logits.device,
                dtype=torch.long,
            )
            if targets.shape != aux.authority_logits.shape[:-1]:
                raise ValueError(
                    "authority_mode_targets must match authority_logits batch shape"
                )
            aux.losses["authority_mode"] = F.cross_entropy(
                aux.authority_logits.reshape(-1, len(AUTHORITY_MODE_NAMES)),
                targets.reshape(-1),
            )
            aux.metrics["authority_mode_supervised_accuracy"] = (
                aux.authority_logits.argmax(dim=-1) == targets
            ).float().mean()

        if verifier_required_targets is not None:
            targets = verifier_required_targets.to(
                device=aux.authority_probs.device,
                dtype=aux.authority_probs.dtype,
            )
            verifier_required_prob = self._authority_verifier_required_probability(
                aux.authority_probs,
            )
            if targets.shape != verifier_required_prob.shape:
                raise ValueError(
                    "verifier_required_targets must match verifier probability batch shape"
                )
            aux.losses["authority_verifier_required"] = F.binary_cross_entropy(
                verifier_required_prob.clamp(1e-6, 1.0 - 1e-6),
                targets,
            )

        if authority_halt_targets is not None:
            targets = authority_halt_targets.to(
                device=aux.halt_probability.device,
                dtype=aux.halt_probability.dtype,
            )
            if targets.shape != aux.halt_probability.shape:
                raise ValueError(
                    "authority_halt_targets must match halt_probability batch shape"
                )
            aux.losses["authority_halt"] = F.binary_cross_entropy(
                aux.halt_probability.clamp(1e-6, 1.0 - 1e-6),
                targets,
            )

    def _authority_verifier_required_probability(self, authority_probs: Tensor) -> Tensor:
        trusted_probability = (
            authority_probs[..., AUTHORITY_EXACT_MEMORY_INDEX]
            + authority_probs[..., AUTHORITY_CALIBRATED_FAST_PATH_INDEX]
        ).clamp(0.0, 1.0)
        return 1.0 - trusted_probability

    def _multi_token_loss(
        self,
        multi_token_logits: list[Tensor],
        labels: Optional[Tensor],
        reference_logits: Tensor,
    ) -> Tensor:
        if not multi_token_logits or labels is None:
            return reference_logits.new_zeros(())
        losses = []
        for head_index, logits in enumerate(multi_token_logits):
            future_offset = head_index + 1
            if labels.shape[1] <= future_offset:
                continue
            prediction = logits[:, :-future_offset, :]
            target = labels[:, future_offset:]
            losses.append(
                F.cross_entropy(
                    prediction.reshape(-1, self.config.vocab_size),
                    target.reshape(-1),
                )
            )
        if not losses:
            return reference_logits.new_zeros(())
        return torch.stack(losses).mean()

    def _merge_auxiliary(
        self,
        identity_outputs: list[IdentityFieldOutput],
        attention_outputs: list[AttentionOutput],
    ) -> TACAuxiliaryOutput:
        last_identity = identity_outputs[-1]
        last_attention = attention_outputs[-1]
        loss_names = last_identity.losses.keys()
        losses = {
            name: torch.stack([output.losses[name] for output in identity_outputs]).mean()
            for name in loss_names
        }
        metric_names = last_identity.metrics.keys()
        metrics = {
            name: torch.stack(
                [
                    output.metrics.get(
                        name,
                        last_identity.metrics[name].new_zeros(()),
                    )
                    for output in identity_outputs
                ]
            ).mean()
            for name in metric_names
        }
        return TACAuxiliaryOutput(
            coherence=last_identity.coherence,
            program_activations=last_identity.activations,
            selected_program_mask=last_identity.selected_program_mask,
            used_energy=last_identity.used_energy,
            attention_probs=last_attention.attention_probs,
            losses=losses,
            metrics=metrics,
            token_program_activations=last_identity.token_activations,
            token_selected_program_mask=last_identity.token_selected_program_mask,
            authority_logits=last_identity.authority_logits,
            authority_probs=last_identity.authority_probs,
            authority_indices=last_identity.authority_indices,
            verifier_required=last_identity.verifier_required,
            halt_probability=last_identity.halt_probability,
            token_authority_logits=last_identity.token_authority_logits,
            token_authority_probs=last_identity.token_authority_probs,
            token_verifier_required=last_identity.token_verifier_required,
        )

    def memory_read_logits(
        self,
        key_ids: Tensor,
        identity_states: list[IdentityState],
        *,
        layer_index: int = -1,
    ) -> Tensor:
        if self.config.memory_read_type == "content_addressed":
            if not identity_states:
                raise ValueError("identity_states are required for memory readout")
            state = identity_states[layer_index]
            if (
                state.content_cues is None
                or state.content_values is None
                or state.content_mask is None
            ):
                raise ValueError("content-addressed state is required for memory readout")
            key_embedding = self.token_embedding(key_ids)
            field = self.blocks[layer_index].identity_field
            if (
                self.config.content_read_steps > 1
                and self.config.content_read_gate_type
                in {"confidence_margin", "cue_match"}
            ):
                if (
                    self.config.content_read_gate_type == "cue_match"
                    and state.content_cue_token_ids is not None
                    and state.content_value_token_ids is not None
                ):
                    fallback_logits = self._content_embedding_memory_logits(
                        key_embedding,
                        state,
                        layer_index=layer_index,
                    )
                    exact_logits = self._content_token_chain_logits(
                        key_ids,
                        state,
                        fallback_logits,
                    )
                    if exact_logits is not None:
                        return exact_logits
                first_read, _, first_token_hit = field._content_addressed_read_with_token_hit(
                    key_embedding,
                    state.content_cues,
                    state.content_values,
                    state.content_mask,
                )
                second_read, _, second_token_hit = field._content_addressed_read_with_token_hit(
                    first_read,
                    state.content_cues,
                    state.content_values,
                    state.content_mask,
                )
                first_logits = torch.matmul(
                    self.final_norm(first_read),
                    self.token_embedding.weight.T,
                )
                second_logits = torch.matmul(
                    self.final_norm(second_read),
                    self.token_embedding.weight.T,
                )
                if self.config.content_read_gate_type == "cue_match":
                    threshold = first_read.new_tensor(
                        float(self.config.content_read_cue_match_threshold)
                    )
                    continue_gate = torch.sigmoid(
                        64.0 * (second_token_hit - threshold)
                    )
                else:
                    confidence_gap = (first_token_hit - second_token_hit).abs()
                    margin = first_read.new_tensor(
                        float(self.config.content_read_confidence_margin)
                    )
                    continue_gate = torch.sigmoid(512.0 * (margin - confidence_gap))
                return (
                    (1.0 - continue_gate[:, None]) * first_logits
                    + continue_gate[:, None] * second_logits
                )
        read_vector = self.memory_read_vector(
            key_ids,
            identity_states,
            layer_index=layer_index,
        )
        if self.config.memory_read_type == "content_addressed":
            read_vector = self.final_norm(read_vector)
            return torch.matmul(read_vector, self.token_embedding.weight.T)
        return self.lm_head(read_vector)

    def _content_embedding_memory_logits(
        self,
        key_embedding: Tensor,
        state: IdentityState,
        *,
        layer_index: int = -1,
    ) -> Tensor:
        field = self.blocks[layer_index].identity_field
        read_vector, _ = field._content_addressed_iterative_read(
            key_embedding,
            state.content_cues,
            state.content_values,
            state.content_mask,
        )
        return torch.matmul(self.final_norm(read_vector), self.token_embedding.weight.T)

    def _content_token_chain_logits(
        self,
        key_ids: Tensor,
        state: IdentityState,
        fallback_logits: Tensor,
    ) -> Optional[Tensor]:
        if (
            state.content_cue_token_ids is None
            or state.content_value_token_ids is None
            or state.content_mask is None
        ):
            return None
        cue_ids = state.content_cue_token_ids.to(key_ids.device)
        value_ids = state.content_value_token_ids.to(key_ids.device)
        valid = (state.content_mask.to(key_ids.device) > 0) & (cue_ids >= 0)
        first_matches = valid & (cue_ids == key_ids[:, None])
        has_first = first_matches.any(dim=1)
        first_positions = (
            first_matches.long()
            * torch.arange(cue_ids.shape[1], device=key_ids.device)[None, :]
        ).argmax(dim=1)
        first_values = value_ids.gather(1, first_positions[:, None]).squeeze(1)

        second_matches = valid & (cue_ids == first_values[:, None])
        has_second = second_matches.any(dim=1)
        second_positions = (
            second_matches.long()
            * torch.arange(cue_ids.shape[1], device=key_ids.device)[None, :]
        ).argmax(dim=1)
        second_values = value_ids.gather(1, second_positions[:, None]).squeeze(1)
        output_ids = torch.where(has_second, second_values, first_values)
        output_ids = output_ids.clamp(0, self.config.vocab_size - 1)

        exact_logits = torch.zeros_like(fallback_logits)
        exact_logits.scatter_(1, output_ids[:, None], 20.0)
        return torch.where(has_first[:, None], exact_logits, fallback_logits)

    def memory_read_vector(
        self,
        key_ids: Tensor,
        identity_states: list[IdentityState],
        *,
        layer_index: int = -1,
    ) -> Tensor:
        if self.config.memory_read_type not in {
            "program_memory",
            "pattern_completion",
            "content_addressed",
        }:
            raise ValueError(
                "memory_read_type must be 'program_memory', 'pattern_completion', or 'content_addressed' to read memory"
            )
        if not identity_states:
            raise ValueError("identity_states are required for memory readout")
        state = identity_states[layer_index]
        if self.config.memory_read_type == "content_addressed":
            if state.content_cues is None or state.content_values is None or state.content_mask is None:
                raise ValueError("content-addressed state is required for memory readout")
            key_embedding = self.token_embedding(key_ids)
            read_vector, _ = self.blocks[layer_index].identity_field._content_addressed_iterative_read(
                key_embedding,
                state.content_cues,
                state.content_values,
                state.content_mask,
            )
            return self.final_norm(read_vector)
        if self.config.memory_read_type == "pattern_completion":
            if state.engram_values is None or state.engram_mask is None:
                raise ValueError("engram state is required for pattern-completion readout")
            values = state.engram_values
            mask = state.engram_mask
            key_embedding = self.token_embedding(key_ids)
            scores = torch.einsum("bd,bkd->bk", key_embedding, values) / sqrt(
                self.config.d_model
            )
            scores = scores.masked_fill(mask <= 0, -1e4)
            weights = F.softmax(scores, dim=-1) * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            read_vector = torch.einsum("bk,bkd->bd", weights, values)
            return self.final_norm(read_vector)
        memory = state.program_memory
        if self.config.memory_tier_type == "hierarchical":
            stable_memory = (
                state.stable_program_memory
                if state.stable_program_memory is not None
                else torch.zeros_like(memory)
            )
            archival_memory = (
                state.archival_program_memory
                if state.archival_program_memory is not None
                else torch.zeros_like(memory)
            )
            memory = 0.6 * memory + 0.3 * stable_memory + 0.1 * archival_memory
        key_embedding = self.token_embedding(key_ids)
        scores = torch.einsum(
            "bd,bpd->bp",
            key_embedding,
            memory,
        ) / sqrt(self.config.d_model)
        weights = F.softmax(scores, dim=-1)
        read_vector = torch.einsum("bp,bpd->bd", weights, memory)
        return self.final_norm(read_vector)

    def memory_adapted_logits(
        self,
        hidden_states: Tensor,
        memory_vector: Tensor,
        *,
        value_label_index: int,
        weight: float = 1.0,
    ) -> Tensor:
        if self.memory_adapter is None:
            raise ValueError("memory_adapter_type must be 'residual'")
        adapted = hidden_states.clone()
        memory_update = self.memory_adapter(memory_vector)
        if self.memory_adapter_gate is not None:
            target_hidden = hidden_states[:, value_label_index, :]
            gate = torch.sigmoid(
                self.memory_adapter_gate(torch.cat([target_hidden, memory_vector], dim=-1))
            )
            memory_update = gate * memory_update
        adapted[:, value_label_index, :] = (
            adapted[:, value_label_index, :]
            + weight * memory_update
        )
        return self.lm_head(adapted)


class VanillaTransformerLM(nn.Module):
    """Causal transformer baseline with the same backbone but no identity field."""

    def __init__(self, config: TACConfig):
        super().__init__()
        _validate_program_embed_dim(config)
        if config.n_prediction_heads < 1:
            raise ValueError("n_prediction_heads must be at least 1")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = _build_position_embedding(config)
        self.blocks = nn.ModuleList(
            [
                VanillaTransformerBlock(config, layer_index=index)
                for index in range(config.n_layers)
            ]
        )
        self.final_norm = _build_norm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.multi_token_heads = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.vocab_size, bias=False)
                for _ in range(config.n_prediction_heads - 1)
            ]
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        identity_states: Optional[list[IdentityState]] = None,
        labels: Optional[Tensor] = None,
        collect_auxiliary: bool = True,
        collect_metrics: bool = True,
        update_content_memory: bool = True,
        content_write_mask: Optional[Tensor] = None,
        write_policy: ContentWritePolicy | str | None = None,
    ) -> TACOutput:
        _resolve_content_write_policy(write_policy)
        del identity_states, collect_auxiliary, collect_metrics, update_content_memory, content_write_mask
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")

        hidden = _apply_token_positions(
            self.token_embedding(input_ids),
            self.position_embedding,
        )
        attention_outputs = []
        global_token_mask = self._global_token_mask(input_ids)

        for block in self.blocks:
            hidden, attention = block(hidden, global_token_mask=global_token_mask)
            attention_outputs.append(attention)

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        multi_token_logits = [head(hidden) for head in self.multi_token_heads]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, self.config.vocab_size),
                labels.reshape(batch_size * seq_len),
            )

        zero = logits.new_zeros(())
        multi_token_loss = self._multi_token_loss(multi_token_logits, labels, logits)
        last_attention = attention_outputs[-1]
        aux = TACAuxiliaryOutput(
            coherence=logits.new_zeros(batch_size, seq_len, seq_len),
            program_activations=logits.new_zeros(batch_size, 0),
            selected_program_mask=logits.new_zeros(batch_size, 0),
            used_energy=logits.new_zeros(batch_size),
            attention_probs=last_attention.attention_probs,
            losses={
                "coherence": zero,
                "program_reuse": zero,
                "energy": zero,
                "decision_continuity": zero,
                "multi_token": multi_token_loss,
            },
            metrics={
                "active_expert_parameters": zero,
                "total_expert_parameters": zero,
                "active_expert_fraction": zero,
                "sink_programs": zero,
                "memory_tiers": zero,
                "routing_type": zero,
                "routing_load_std": zero,
                "memory_lookup_slots": zero,
                "residual_streams": zero,
                "sequence_mixer_type": logits.new_tensor(
                    float(_sequence_mixer_type_id(self.config.sequence_mixer_type))
                ),
                "decision_continuity_agreement": zero,
                "decision_continuity_memory_mass": zero,
            },
            token_program_activations=logits.new_zeros(batch_size, seq_len, 0),
            token_selected_program_mask=logits.new_zeros(batch_size, seq_len, 0),
        )

        return TACOutput(
            logits=logits,
            identity_states=[],
            aux=aux,
            loss=loss,
            hidden_states=hidden,
            multi_token_logits=multi_token_logits,
        )

    def _global_token_mask(self, input_ids: Tensor) -> Optional[Tensor]:
        if not self.config.global_attention_token_ids:
            return None
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.config.global_attention_token_ids:
            mask = mask | (input_ids == int(token_id))
        return mask

    def _multi_token_loss(
        self,
        multi_token_logits: list[Tensor],
        labels: Optional[Tensor],
        reference_logits: Tensor,
    ) -> Tensor:
        if not multi_token_logits or labels is None:
            return reference_logits.new_zeros(())
        losses = []
        for head_index, logits in enumerate(multi_token_logits):
            future_offset = head_index + 1
            if labels.shape[1] <= future_offset:
                continue
            prediction = logits[:, :-future_offset, :]
            target = labels[:, future_offset:]
            losses.append(
                F.cross_entropy(
                    prediction.reshape(-1, self.config.vocab_size),
                    target.reshape(-1),
                )
            )
        if not losses:
            return reference_logits.new_zeros(())
        return torch.stack(losses).mean()


def _inverse_softplus(value: Tensor) -> Tensor:
    return torch.log(torch.expm1(value).clamp_min(1e-6))
