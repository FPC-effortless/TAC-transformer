from __future__ import annotations

import random
import time
import math
from dataclasses import asdict, dataclass, replace
import os
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import (
    AUTHORITY_FEATURE_DIM,
    AUTHORITY_MODE_NAMES,
    IdentityState,
    TACConfig,
    TACOutput,
    TACTransformerLM,
    VanillaTransformerLM,
)
from .optimization import TACOptimizerConfig, build_tac_optimizer

try:
    import numpy as np
except ImportError:  # pragma: no cover - tests run with numpy via torch installs.
    np = None


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    identity_field = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".identity_field." in name
    )
    return {
        "total": total,
        "trainable": trainable,
        "identity_field": identity_field,
    }


def estimate_vanilla_parameter_count(config: TACConfig) -> int:
    d_model = config.d_model
    mlp_ratio = config.mlp_ratio
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
    if config.state_mixer_kernel_size < 1:
        raise ValueError("state_mixer_kernel_size must be at least 1")
    if config.attention_window_size is not None and config.attention_window_size < 1:
        raise ValueError("attention_window_size must be positive when set")
    n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
    head_dim = d_model // config.n_heads
    kv_dim = n_kv_heads * head_dim

    attention = 2 * d_model * d_model + 2 * d_model * kv_dim
    attention += 2 * d_model + 2 * kv_dim

    if config.mlp_type == "gelu":
        mlp = 2 * mlp_ratio * d_model * d_model
        mlp += (mlp_ratio + 1) * d_model
    elif config.mlp_type == "swiglu":
        mlp = 3 * mlp_ratio * d_model * d_model
        mlp += (2 * mlp_ratio + 1) * d_model
    else:
        raise ValueError("mlp_type must be 'gelu' or 'swiglu'")

    if config.norm_type == "layernorm":
        block_norms = 4 * d_model
        final_norm = 2 * d_model
    elif config.norm_type == "rmsnorm":
        block_norms = 2 * d_model
        final_norm = d_model
    else:
        raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")

    state_mixer = _state_mixer_parameter_count(config)
    block_total = 0
    for layer_index in range(config.n_layers):
        block = mlp + block_norms
        if _uses_attention_mixer(config.sequence_mixer_type, layer_index):
            block += attention
        if _uses_state_mixer(config.sequence_mixer_type, layer_index):
            block += state_mixer
        block_total += block
    position_parameters = config.max_seq_len * d_model if config.position_type == "learned" else 0
    embedding_and_head = 2 * config.vocab_size * d_model + position_parameters + final_norm
    if config.n_prediction_heads < 1:
        raise ValueError("n_prediction_heads must be at least 1")
    embedding_and_head += (config.n_prediction_heads - 1) * config.vocab_size * d_model
    return block_total + embedding_and_head


def estimate_tac_parameter_count(config: TACConfig) -> int:
    if config.n_sink_programs < 0 or config.n_sink_programs > config.n_programs:
        raise ValueError("n_sink_programs must be between 0 and n_programs")
    if config.program_compute_type not in {
        "embedding",
        "linear_expert",
        "sparse_linear_expert",
    }:
        raise ValueError(
            "program_compute_type must be 'embedding', 'linear_expert', or 'sparse_linear_expert'"
        )
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
    if config.memory_write_type not in {"standard", "novelty_gated"}:
        raise ValueError("memory_write_type must be 'standard' or 'novelty_gated'")
    if config.memory_tier_type not in {"flat", "hierarchical"}:
        raise ValueError("memory_tier_type must be 'flat' or 'hierarchical'")
    if config.memory_lookup_type not in {"none", "product_key"}:
        raise ValueError("memory_lookup_type must be 'none' or 'product_key'")
    if config.memory_lookup_slots < 1:
        raise ValueError("memory_lookup_slots must be at least 1")
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
    if config.memory_allocation_type not in {"stability", "creb"}:
        raise ValueError("memory_allocation_type must be 'stability' or 'creb'")
    if config.memory_allocation_k < 1:
        raise ValueError("memory_allocation_k must be at least 1")
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
    if not 0.0 <= config.authority_trusted_threshold <= 1.0:
        raise ValueError("authority_trusted_threshold must be between 0 and 1")
    if not 0.0 <= config.content_reconsolidate_rate <= 1.0:
        raise ValueError("content_reconsolidate_rate must be between 0 and 1")
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
    if config.state_mixer_kernel_size < 1:
        raise ValueError("state_mixer_kernel_size must be at least 1")
    identity_per_block = (
        2 * config.d_model * config.d_model
        + (config.n_programs + 2) * config.d_model
        + config.n_programs
    )
    if config.state_update_type == "gated":
        identity_per_block += 2 * (
            config.d_model * config.n_programs
            + config.n_programs
        )
    elif config.state_update_type != "fixed":
        raise ValueError("state_update_type must be 'fixed' or 'gated'")
    if config.memory_write_type == "novelty_gated":
        identity_per_block += 2 * config.d_model + 1
    if config.program_memory_update_type == "program_conditioned":
        identity_per_block += 2 * config.d_model * config.d_model + config.d_model
    if config.coalition_context_type == "program_memory":
        identity_per_block += config.d_model * config.d_model + config.d_model
    if config.coalition_context_type == "program_memory_graph":
        identity_per_block += 4 * (
            config.d_model * config.d_model
            + config.d_model
        )
    if config.coalition_context_type == "program_memory_task_graph":
        identity_per_block += 5 * (
            config.d_model * config.d_model
            + config.d_model
        )
    if config.routing_type == "authority_gated":
        authority_hidden_dim = max(config.d_model, config.n_programs * 2)
        authority_input_dim = config.n_programs * 2 + AUTHORITY_FEATURE_DIM
        identity_per_block += authority_input_dim * authority_hidden_dim + authority_hidden_dim
        identity_per_block += authority_hidden_dim * authority_hidden_dim + authority_hidden_dim
        identity_per_block += authority_hidden_dim * config.n_programs + config.n_programs
        identity_per_block += (
            authority_hidden_dim * len(AUTHORITY_MODE_NAMES)
            + len(AUTHORITY_MODE_NAMES)
        )
        identity_per_block += authority_hidden_dim + 1
    if config.program_compute_type in {"linear_expert", "sparse_linear_expert"}:
        identity_per_block += (
            config.n_programs * config.d_model * config.d_model
            + config.n_programs * config.d_model
        )
    if config.memory_lookup_type == "product_key":
        identity_per_block += (
            config.d_model * config.d_model
            + config.d_model
            + 2 * config.memory_lookup_slots * config.d_model
        )
    if config.memory_reconsolidate:
        if config.reconsolidate_gate_type == "mlp":
            identity_per_block += (
                3 * config.d_model * config.d_model
                + config.d_model
                + config.d_model
                + 1
            )
        else:
            identity_per_block += 3 * config.d_model + 1
    if config.content_read_steps > 1 and config.content_read_gate_type == "learned":
        identity_per_block += 2 * config.d_model + 2
    if config.content_read_steps > 1 and config.content_read_gate_type == "synthesis":
        identity_per_block += 5 * config.d_model * config.d_model + config.d_model
        identity_per_block += 5 * config.d_model + 1
    if config.residual_stream_type == "dual_stream":
        identity_per_block += 2 * (2 * config.d_model * config.d_model + config.d_model)
    model_level = 0
    if config.memory_adapter_type == "residual":
        model_level += config.d_model * config.d_model + config.d_model
    elif config.memory_adapter_type == "gated_residual":
        adapter_hidden_dim = config.d_model * config.mlp_ratio
        model_level += config.d_model * adapter_hidden_dim + adapter_hidden_dim
        model_level += adapter_hidden_dim * config.d_model + config.d_model
        model_level += 2 * config.d_model * config.d_model + config.d_model
    elif config.memory_adapter_type != "none":
        raise ValueError(
            "memory_adapter_type must be 'none', 'residual', or 'gated_residual'"
        )
    identity_attention_extra = 0
    if config.identity_attention_type == "identity_first":
        n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        head_dim = config.d_model // config.n_heads
        kv_dim = n_kv_heads * head_dim
        identity_kv_projection = 2 * config.d_model * (2 * kv_dim) + (2 * kv_dim)
        identity_attention_extra = sum(
            identity_kv_projection
            for layer_index in range(config.n_layers)
            if _uses_attention_mixer(config.sequence_mixer_type, layer_index)
        )
    return (
        estimate_vanilla_parameter_count(config)
        + config.n_layers * identity_per_block
        + identity_attention_extra
        + model_level
    )


def _uses_attention_mixer(sequence_mixer_type: str, layer_index: int) -> bool:
    if sequence_mixer_type in {"attention", "hybrid"}:
        return True
    if sequence_mixer_type == "alternating":
        return layer_index % 2 == 0
    return False


def _state_mixer_parameter_count(config: TACConfig) -> int:
    d_model = config.d_model
    if config.sequence_mixer_type in {"state", "hybrid", "alternating"}:
        return 3 * d_model * d_model + (config.state_mixer_kernel_size + 5) * d_model
    if config.sequence_mixer_type == "selective_state":
        return 4 * d_model * d_model + 5 * d_model
    if config.sequence_mixer_type == "rwkv":
        return 4 * d_model * d_model + 8 * d_model
    if config.sequence_mixer_type == "xlstm":
        return 9 * d_model * d_model + 5 * d_model
    return 0


def _uses_state_mixer(sequence_mixer_type: str, layer_index: int) -> bool:
    if sequence_mixer_type in {
        "state",
        "hybrid",
        "selective_state",
        "rwkv",
        "xlstm",
    }:
        return True
    if sequence_mixer_type == "alternating":
        return layer_index % 2 == 1
    return False


def parameter_matched_baseline_config(config: TACConfig) -> TACConfig:
    target = estimate_tac_parameter_count(config)
    best = config
    best_gap = abs(target - estimate_vanilla_parameter_count(best))
    max_d_model = max(config.d_model * 4, config.d_model + 512)

    for d_model in range(config.n_heads, max_d_model + 1, config.n_heads):
        if not _is_valid_baseline_width(config, d_model):
            continue
        candidate = replace(config, d_model=d_model)
        gap = abs(target - estimate_vanilla_parameter_count(candidate))
        if gap < best_gap:
            best = candidate
            best_gap = gap

    return best


def _is_valid_baseline_width(config: TACConfig, d_model: int) -> bool:
    if d_model % config.n_heads != 0:
        return False
    if config.n_kv_heads is not None and config.n_heads % config.n_kv_heads != 0:
        return False
    head_dim = d_model // config.n_heads
    return config.position_type != "rope" or head_dim % 2 == 0


class SyntheticProgramBatcher:
    """Generates tiny executable-pattern sequences for TAC smoke training."""

    def __init__(self, vocab_size: int, seq_len: int, seed: int = 0):
        if vocab_size < 16:
            raise ValueError("vocab_size must be at least 16")
        if seq_len < 6:
            raise ValueError("seq_len must be at least 6")

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data_floor = 4
        self.rng = random.Random(seed)

    def next_batch(self, batch_size: int, device: str | torch.device = "cpu") -> tuple[Tensor, Tensor]:
        sequences = [self._make_sequence() for _ in range(batch_size)]
        batch = torch.tensor(sequences, dtype=torch.long, device=device)
        return batch[:, :-1], batch[:, 1:]

    def _make_sequence(self) -> list[int]:
        program_id = self.rng.randrange(4)
        data_len = self.seq_len

        if program_id == 0:
            sequence = self._increment(data_len)
        elif program_id == 1:
            sequence = self._repeat_block(data_len)
        elif program_id == 2:
            sequence = self._mirror(data_len)
        else:
            sequence = self._alternate(data_len)

        return [program_id] + sequence

    def _increment(self, length: int) -> list[int]:
        span = self.vocab_size - self.data_floor
        start = self.rng.randrange(span)
        step = self.rng.choice([1, 2, 3])
        return [self.data_floor + ((start + step * index) % span) for index in range(length)]

    def _repeat_block(self, length: int) -> list[int]:
        block = [self._data_token() for _ in range(4)]
        return [block[index % len(block)] for index in range(length)]

    def _mirror(self, length: int) -> list[int]:
        half = [self._data_token() for _ in range((length + 1) // 2)]
        mirrored = half + list(reversed(half))
        return mirrored[:length]

    def _alternate(self, length: int) -> list[int]:
        left = self._data_token()
        right = self._data_token()
        if left == right:
            right = self.data_floor + ((right - self.data_floor + 1) % (self.vocab_size - self.data_floor))
        return [left if index % 2 == 0 else right for index in range(length)]

    def _data_token(self) -> int:
        return self.rng.randrange(self.data_floor, self.vocab_size)


class JsonlTextBatcher:
    """Byte-level language-model batches from prepared JSONL rows with a text field.

    The batcher stores only line offsets, then seeks to sampled records on demand.
    That keeps full prepared corpora trainable without loading every byte token into
    Python memory.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        vocab_size: int,
        seed: int = 0,
        text_field: str = "text",
    ):
        if vocab_size < 260:
            raise ValueError("vocab_size must be at least 260 for byte-level text batches")
        self.path = Path(path)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.text_field = text_field
        self.rng = random.Random(seed)
        self.offsets = self._build_offsets()

    def next_batch(self, batch_size: int, device: str | torch.device = "cpu") -> tuple[Tensor, Tensor]:
        windows = []
        for _ in range(batch_size):
            windows.append(self._sample_window())
        batch = torch.tensor(windows, dtype=torch.long, device=device)
        return batch[:, :-1], batch[:, 1:]

    def _build_offsets(self) -> list[int]:
        offsets = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        if not offsets:
            raise ValueError(f"no JSONL records found in {self.path}")
        return offsets

    def _sample_window(self) -> list[int]:
        for _ in range(32):
            token_ids = self._read_random_record_tokens()
            if len(token_ids) >= self.seq_len + 1:
                start = self.rng.randrange(len(token_ids) - self.seq_len)
                return token_ids[start : start + self.seq_len + 1]

        token_ids = self._read_random_record_tokens()
        padded = token_ids[: self.seq_len + 1]
        padded.extend([3] * (self.seq_len + 1 - len(padded)))
        return padded

    def _read_random_record_tokens(self) -> list[int]:
        offset = self.rng.choice(self.offsets)
        with self.path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline().decode("utf-8")
        row = json.loads(line)
        text = str(row.get(self.text_field, ""))
        return _byte_tokens(text) + [3]


class JsonlLabeledTextBatcher:
    """Byte-level JSONL batches with a categorical label from each row.

    Categories are sampled uniformly first, then a row is sampled within the
    chosen category. That keeps a category-route objective from being dominated
    by the largest corpus domain.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        seq_len: int,
        vocab_size: int,
        seed: int = 0,
        text_field: str = "text",
        label_field: str = "domain",
    ):
        if vocab_size < 260:
            raise ValueError("vocab_size must be at least 260 for byte-level text batches")
        self.path = Path(path)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.text_field = text_field
        self.label_field = label_field
        self.rng = random.Random(seed)
        self.category_offsets = self._build_category_offsets()
        self.categories = sorted(self.category_offsets)
        self.category_to_id = {
            category: index for index, category in enumerate(self.categories)
        }
        self.offsets = [
            offset
            for category in self.categories
            for offset in self.category_offsets[category]
        ]

    def next_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor, Tensor]:
        windows = []
        category_ids = []
        for _ in range(batch_size):
            category = self.rng.choice(self.categories)
            windows.append(self._sample_window(category))
            category_ids.append(self.category_to_id[category])
        batch = torch.tensor(windows, dtype=torch.long, device=device)
        categories = torch.tensor(category_ids, dtype=torch.long, device=device)
        return batch[:, :-1], batch[:, 1:], categories

    def _build_category_offsets(self) -> dict[str, list[int]]:
        offsets: dict[str, list[int]] = {}
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line.decode("utf-8"))
                category = str(row.get(self.label_field, ""))
                text = str(row.get(self.text_field, ""))
                if not category or not text:
                    continue
                offsets.setdefault(category, []).append(offset)
        if not offsets:
            raise ValueError(
                f"no JSONL records with {self.label_field!r} and {self.text_field!r} found in {self.path}"
            )
        return offsets

    def _sample_window(self, category: str) -> list[int]:
        for _ in range(32):
            token_ids = self._read_random_record_tokens(category)
            if len(token_ids) >= self.seq_len + 1:
                start = self.rng.randrange(len(token_ids) - self.seq_len)
                return token_ids[start : start + self.seq_len + 1]

        token_ids = self._read_random_record_tokens(category)
        padded = token_ids[: self.seq_len + 1]
        padded.extend([3] * (self.seq_len + 1 - len(padded)))
        return padded

    def _read_random_record_tokens(self, category: str) -> list[int]:
        offset = self.rng.choice(self.category_offsets[category])
        with self.path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline().decode("utf-8")
        row = json.loads(line)
        text = str(row.get(self.text_field, ""))
        return _byte_tokens(text) + [3]


def build_tokenized_memmap_from_jsonl(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    vocab_size: int,
    text_field: str = "text",
    label_field: str = "domain",
    dtype: str | None = None,
    eos_token_id: int = 3,
) -> dict[str, object]:
    """Persist prepared JSONL text as token arrays plus record metadata.

    This is intentionally tokenizer-agnostic. Today it stores the existing
    byte-level IDs, but the file contract also works for future BPE/subword IDs.
    """

    if np is None:
        raise RuntimeError("numpy is required for tokenized memmap files")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_dtype = _token_memmap_dtype(vocab_size, dtype)
    tokens: list[int] = []
    offsets: list[int] = []
    lengths: list[int] = []
    category_ids: list[int] = []
    category_to_id: dict[str, int] = {}
    records = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get(text_field, ""))
            if not text:
                continue
            record_tokens = _byte_tokens(text) + [eos_token_id]
            if max(record_tokens, default=0) >= vocab_size:
                raise ValueError("vocab_size is too small for encoded token IDs")
            offsets.append(len(tokens))
            lengths.append(len(record_tokens))
            tokens.extend(record_tokens)
            category = str(row.get(label_field, ""))
            if category:
                category_id = category_to_id.setdefault(category, len(category_to_id))
            else:
                category_id = -1
            category_ids.append(category_id)
            records += 1

    if records == 0:
        raise ValueError(f"no tokenizable records found in {input_path}")

    tokens_path = output_dir / f"tokens.{token_dtype}.bin"
    np.asarray(tokens, dtype=token_dtype).tofile(tokens_path)
    offsets_path = output_dir / "record_offsets.int64.npy"
    lengths_path = output_dir / "record_lengths.int32.npy"
    category_ids_path = output_dir / "category_ids.int16.npy"
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64))
    np.save(lengths_path, np.asarray(lengths, dtype=np.int32))
    np.save(category_ids_path, np.asarray(category_ids, dtype=np.int16))
    categories_path = output_dir / "categories.json"
    categories_path.write_text(
        json.dumps(
            {category: index for category, index in sorted(category_to_id.items(), key=lambda item: item[1])},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "input_path": str(input_path),
        "records": records,
        "tokens": len(tokens),
        "vocab_size": vocab_size,
        "dtype": token_dtype,
        "text_field": text_field,
        "label_field": label_field,
        "tokens_path": str(tokens_path),
        "record_offsets_path": str(offsets_path),
        "record_lengths_path": str(lengths_path),
        "category_ids_path": str(category_ids_path),
        "categories_path": str(categories_path),
        "categories": {category: index for category, index in sorted(category_to_id.items(), key=lambda item: item[1])},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


class TokenizedMemmapBatcher:
    """Window sampler for pretokenized arrays with record offsets.

    The batcher avoids JSON parsing and online byte encoding inside the training
    loop. It supports the same next-token batch shape as JsonlTextBatcher.
    """

    def __init__(
        self,
        tokens_path: str | Path,
        record_offsets_path: str | Path,
        record_lengths_path: str | Path,
        *,
        seq_len: int,
        vocab_size: int,
        dtype: str = "uint16",
        category_ids_path: str | Path | None = None,
        seed: int = 0,
        pad_token_id: int = 3,
    ):
        if np is None:
            raise RuntimeError("numpy is required for tokenized memmap files")
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        self.tokens_path = Path(tokens_path)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dtype = np.dtype(dtype)
        self.pad_token_id = pad_token_id
        self.rng = random.Random(seed)
        self.tokens = np.memmap(self.tokens_path, mode="r", dtype=self.dtype)
        self.record_offsets = np.load(record_offsets_path).astype(np.int64, copy=False)
        self.record_lengths = np.load(record_lengths_path).astype(np.int64, copy=False)
        if self.record_offsets.shape != self.record_lengths.shape:
            raise ValueError("record offsets and lengths must have the same shape")
        if self.record_offsets.size == 0:
            raise ValueError("no tokenized records available")
        self.category_ids = None
        if category_ids_path is not None:
            category_ids = np.load(category_ids_path).astype(np.int64, copy=False)
            if category_ids.shape[0] != self.record_offsets.shape[0]:
                raise ValueError("category IDs must match record count")
            self.category_ids = category_ids

    def close(self) -> None:
        mmap = getattr(self.tokens, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        seq_len: int,
        seed: int = 0,
        include_categories: bool = False,
    ) -> "TokenizedMemmapBatcher":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        category_ids_path = manifest.get("category_ids_path") if include_categories else None
        return cls(
            manifest["tokens_path"],
            manifest["record_offsets_path"],
            manifest["record_lengths_path"],
            seq_len=seq_len,
            vocab_size=int(manifest["vocab_size"]),
            dtype=str(manifest["dtype"]),
            category_ids_path=category_ids_path,
            seed=seed,
        )

    def next_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        batch = self._sample_window_array(batch_size)
        batch = torch.from_numpy(batch)
        if str(device) != "cpu":
            batch = batch.to(device)
        return batch[:, :-1], batch[:, 1:]

    def next_labeled_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.category_ids is None:
            raise ValueError("category_ids_path is required for labeled batches")
        batch = np.empty((batch_size, self.seq_len + 1), dtype=np.int64)
        categories = np.empty((batch_size,), dtype=np.int64)
        for row_index in range(batch_size):
            record_index = self.rng.randrange(self.record_offsets.shape[0])
            self._fill_window(batch[row_index], record_index)
            categories[row_index] = int(self.category_ids[record_index])
        batch_tensor = torch.from_numpy(batch)
        category_tensor = torch.from_numpy(categories)
        if str(device) != "cpu":
            batch_tensor = batch_tensor.to(device)
            category_tensor = category_tensor.to(device)
        return batch_tensor[:, :-1], batch_tensor[:, 1:], category_tensor

    def _sample_window(self, record_index: int | None = None) -> list[int]:
        if record_index is None:
            record_index = self.rng.randrange(self.record_offsets.shape[0])
        offset = int(self.record_offsets[record_index])
        length = int(self.record_lengths[record_index])
        if length <= 0:
            return [self.pad_token_id] * (self.seq_len + 1)
        if length >= self.seq_len + 1:
            start = offset + self.rng.randrange(length - self.seq_len)
            window = self.tokens[start : start + self.seq_len + 1]
            return [int(token) for token in window]
        record = [int(token) for token in self.tokens[offset : offset + length]]
        record.extend([self.pad_token_id] * (self.seq_len + 1 - len(record)))
        return record

    def _sample_window_array(self, batch_size: int) -> "np.ndarray":
        batch = np.empty((batch_size, self.seq_len + 1), dtype=np.int64)
        for row_index in range(batch_size):
            self._fill_window(batch[row_index])
        return batch

    def _fill_window(self, row: "np.ndarray", record_index: int | None = None) -> None:
        if record_index is None:
            record_index = self.rng.randrange(self.record_offsets.shape[0])
        offset = int(self.record_offsets[record_index])
        length = int(self.record_lengths[record_index])
        if length <= 0:
            row.fill(self.pad_token_id)
            return
        if length >= self.seq_len + 1:
            start = offset + self.rng.randrange(length - self.seq_len)
            row[:] = self.tokens[start : start + self.seq_len + 1]
            return
        row[:length] = self.tokens[offset : offset + length]
        row[length:] = self.pad_token_id


def _token_memmap_dtype(vocab_size: int, requested: str | None) -> str:
    if requested is not None:
        if requested not in {"uint16", "uint32"}:
            raise ValueError("dtype must be 'uint16' or 'uint32'")
        if requested == "uint16" and vocab_size > 65536:
            raise ValueError("uint16 cannot store vocab_size above 65536")
        return requested
    return "uint16" if vocab_size <= 65536 else "uint32"


def category_route_loss(
    token_program_activations: Tensor | None,
    category_ids: Tensor,
    *,
    n_categories: int | None = None,
) -> Tensor:
    """Encourage category-specific examples to activate stable program IDs."""

    if token_program_activations is None or token_program_activations.numel() == 0:
        return category_ids.new_zeros((), dtype=torch.float32)
    activations = token_program_activations.clamp_min(0.0)
    route_probs = activations / activations.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    mean_probs = route_probs.mean(dim=1)
    target_programs = category_ids % mean_probs.shape[-1]
    return F.nll_loss(mean_probs.clamp_min(1e-8).log(), target_programs)


def _category_program_mi_from_probs(
    sample_program_probs: Tensor,
    category_ids: Tensor,
    *,
    n_categories: int,
) -> Tensor:
    if sample_program_probs.numel() == 0 or n_categories < 1:
        return sample_program_probs.new_zeros(())
    category_one_hot = F.one_hot(
        category_ids.clamp_min(0) % n_categories,
        num_classes=n_categories,
    ).to(sample_program_probs.dtype)
    joint = category_one_hot.transpose(0, 1) @ sample_program_probs
    joint = joint / max(category_ids.numel(), 1)
    category_marginal = joint.sum(dim=1, keepdim=True)
    program_marginal = joint.sum(dim=0, keepdim=True)
    independent = category_marginal @ program_marginal
    positive = joint > 0
    mi = torch.where(
        positive,
        joint * (joint.clamp_min(1e-8) / independent.clamp_min(1e-8)).log(),
        joint.new_zeros(joint.shape),
    ).sum()
    return -mi


def category_program_mi_loss(
    token_program_activations: Tensor | None,
    category_ids: Tensor,
    *,
    n_categories: int,
) -> Tensor:
    """Negative differentiable MI between category labels and program activations."""

    if token_program_activations is None or token_program_activations.numel() == 0:
        return category_ids.new_zeros((), dtype=torch.float32)
    if n_categories < 1:
        return category_ids.new_zeros((), dtype=torch.float32)

    activations = token_program_activations.clamp_min(0.0)
    route_probs = activations / activations.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sample_program_probs = route_probs.mean(dim=1)
    return _category_program_mi_from_probs(
        sample_program_probs,
        category_ids,
        n_categories=n_categories,
    )


def selected_program_mi_loss(
    program_activations: Tensor | None,
    selected_program_mask: Tensor | None,
    category_ids: Tensor,
    *,
    n_categories: int,
) -> Tensor:
    """Negative MI for the record-level selected programs used by Phase B gates."""

    if program_activations is None or program_activations.numel() == 0:
        return category_ids.new_zeros((), dtype=torch.float32)
    if selected_program_mask is None or selected_program_mask.numel() == 0:
        return category_ids.new_zeros((), dtype=torch.float32)
    if n_categories < 1:
        return category_ids.new_zeros((), dtype=torch.float32)
    if program_activations.shape != selected_program_mask.shape:
        raise ValueError("program_activations and selected_program_mask must have matching shapes")

    activations = program_activations.clamp_min(0.0)
    selected = selected_program_mask.to(dtype=activations.dtype).clamp_min(0.0)
    scores = activations * selected
    fallback = scores.sum(dim=-1, keepdim=True) <= 0
    scores = torch.where(fallback, activations, scores)
    sample_program_probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return _category_program_mi_from_probs(
        sample_program_probs,
        category_ids,
        n_categories=n_categories,
    )


@dataclass(frozen=True)
class ChunkedRecallBatch:
    context_inputs: Tensor
    context_labels: Tensor
    query_inputs: Tensor
    query_labels: Tensor
    value_targets: Tensor
    context_write_mask: Optional[Tensor] = None
    value_label_index: int = 2


class ChunkedRecallBatcher:
    """Paired chunks where the query target is a value introduced in context."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        seed: int = 0,
        task_variant: str = "single_key",
    ):
        if vocab_size < 16:
            raise ValueError("vocab_size must be at least 16")
        if seq_len < 6:
            raise ValueError("seq_len must be at least 6")
        if task_variant not in {
            "single_key",
            "multi_key",
            "delayed_query",
            "noisy_key",
            "multi_hop",
        }:
            raise ValueError(
                "task_variant must be 'single_key', 'multi_key', 'delayed_query', 'noisy_key', or 'multi_hop'"
            )
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.task_variant = task_variant
        self.context_token = 0
        self.query_token = 1
        self.noise_token = 2
        self.recall_token = 3
        self.data_floor = 4
        self.rng = random.Random(seed)

    def next_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> ChunkedRecallBatch:
        context_rows = []
        query_input_rows = []
        query_label_rows = []
        context_write_masks = []
        value_targets = []
        for _ in range(batch_size):
            (
                context,
                query_inputs,
                query_labels,
                value,
                value_label_index,
                context_write_mask,
            ) = self._make_pair()
            context_rows.append(context)
            query_input_rows.append(query_inputs)
            query_label_rows.append(query_labels)
            context_write_masks.append(context_write_mask)
            value_targets.append(value)

        context = torch.tensor(context_rows, dtype=torch.long, device=device)
        return ChunkedRecallBatch(
            context_inputs=context[:, :-1],
            context_labels=context[:, 1:],
            query_inputs=torch.tensor(query_input_rows, dtype=torch.long, device=device),
            query_labels=torch.tensor(query_label_rows, dtype=torch.long, device=device),
            value_targets=torch.tensor(value_targets, dtype=torch.long, device=device),
            context_write_mask=torch.tensor(
                context_write_masks,
                dtype=torch.bool,
                device=device,
            ),
            value_label_index=value_label_index,
        )

    def _make_pair(self) -> tuple[list[int], list[int], list[int], int, int, list[bool]]:
        if self.task_variant == "multi_key":
            return self._make_multi_key_pair()
        if self.task_variant == "delayed_query":
            return self._make_delayed_query_pair()
        if self.task_variant == "noisy_key":
            return self._make_noisy_key_pair()
        if self.task_variant == "multi_hop":
            return self._make_multi_hop_pair()
        return self._make_single_key_pair()

    def _make_single_key_pair(self) -> tuple[list[int], list[int], list[int], int, int, list[bool]]:
        key = self._data_token()
        value = self._data_token(exclude=key)
        context = [self.context_token, key, value]
        query_inputs = [self.query_token, key, self.recall_token]
        value_label_index = 2
        query_labels = self._query_labels(value_label_index, value)

        return (
            self._pad_context(context),
            self._pad_query_inputs(query_inputs, value),
            query_labels,
            value,
            value_label_index,
            self._context_write_mask([1]),
        )

    def _make_multi_key_pair(self) -> tuple[list[int], list[int], list[int], int, int, list[bool]]:
        pair_count = min(4, max(2, (self.seq_len - 1) // 2))
        used: set[int] = set()
        pairs = []
        for _ in range(pair_count):
            key = self._unique_data_token(used)
            value = self._unique_data_token(used)
            pairs.append((key, value))
        target_index = self.rng.randrange(len(pairs))
        key, value = pairs[target_index]
        context = [self.context_token]
        for pair_key, pair_value in pairs:
            context.extend([pair_key, pair_value])
        query_inputs = [self.query_token, key, self.recall_token]
        value_label_index = 2
        return (
            self._pad_context(context),
            self._pad_query_inputs(query_inputs, value),
            self._query_labels(value_label_index, value),
            value,
            value_label_index,
            self._context_write_mask(range(1, len(context), 2)),
        )

    def _make_delayed_query_pair(self) -> tuple[list[int], list[int], list[int], int, int, list[bool]]:
        key = self._data_token()
        value = self._data_token(exclude=key)
        delay = min(4, self.seq_len - 3)
        context = [self.context_token, key, value]
        query_inputs = [self.query_token, key]
        for _ in range(delay):
            query_inputs.append(self._data_token(exclude=value))
        query_inputs.append(self.recall_token)
        value_label_index = len(query_inputs) - 1
        return (
            self._pad_context(context),
            self._pad_query_inputs(query_inputs, value),
            self._query_labels(value_label_index, value),
            value,
            value_label_index,
            self._context_write_mask([1]),
        )

    def _make_noisy_key_pair(self) -> tuple[list[int], list[int], list[int], int, int, list[bool]]:
        key = self._data_token()
        value = self._data_token(exclude=key)
        noisy_key = self.data_floor + (
            (key - self.data_floor + 1) % (self.vocab_size - self.data_floor)
        )
        if noisy_key == value:
            noisy_key = self.data_floor + (
                (noisy_key - self.data_floor + 1) % (self.vocab_size - self.data_floor)
            )
        context = [self.context_token, key, value]
        query_inputs = [self.query_token, noisy_key, self.recall_token]
        value_label_index = 2
        return (
            self._pad_context(context),
            self._pad_query_inputs(query_inputs, value),
            self._query_labels(value_label_index, value),
            value,
            value_label_index,
            self._context_write_mask([1]),
        )

    def _make_multi_hop_pair(self) -> tuple[list[int], list[int], list[int], int, int, list[bool]]:
        used: set[int] = set()
        first_key = self._unique_data_token(used)
        bridge_key = self._unique_data_token(used)
        value = self._unique_data_token(used)
        context = [self.context_token, first_key, bridge_key, bridge_key, value]
        query_inputs = [self.query_token, first_key, self.recall_token]
        value_label_index = 2
        return (
            self._pad_context(context),
            self._pad_query_inputs(query_inputs, value),
            self._query_labels(value_label_index, value),
            value,
            value_label_index,
            self._context_write_mask([1, 3]),
        )

    def _context_write_mask(self, pair_indices: Iterable[int]) -> list[bool]:
        mask = [False] * max(self.seq_len - 1, 0)
        for index in pair_indices:
            if 0 <= int(index) < len(mask):
                mask[int(index)] = True
        return mask

    def _pad_context(self, context: list[int]) -> list[int]:
        while len(context) < self.seq_len + 1:
            context.append(self._data_token())
        return context[: self.seq_len + 1]

    def _pad_query_inputs(self, query_inputs: list[int], value: int) -> list[int]:
        while len(query_inputs) < self.seq_len:
            query_inputs.append(self._data_token(exclude=value))
        return query_inputs[: self.seq_len]

    def _query_labels(self, value_label_index: int, value: int) -> list[int]:
        labels = [self.noise_token] * self.seq_len
        labels[value_label_index] = value
        for index in range(self.seq_len):
            if index != value_label_index:
                labels[index] = self._data_token(exclude=value)
        return labels

    def _unique_data_token(self, used: set[int]) -> int:
        for _ in range(64):
            token = self._data_token()
            if token not in used:
                used.add(token)
                return token
        token = self.data_floor + (len(used) % (self.vocab_size - self.data_floor))
        used.add(token)
        return token

    def _data_token(self, exclude: int | None = None) -> int:
        token = self.rng.randrange(self.data_floor, self.vocab_size)
        if exclude is not None and token == exclude:
            token = self.data_floor + ((token - self.data_floor + 1) % (self.vocab_size - self.data_floor))
        return token


def train_synthetic(
    model: TACTransformerLM,
    batcher: SyntheticProgramBatcher,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    aux_weights: Optional[dict[str, float]] = None,
    device: str | torch.device = "cpu",
    log_every: int = 0,
    checkpoint_path: Optional[str | Path] = None,
    optimizer_config: Optional[TACOptimizerConfig] = None,
) -> dict[str, float]:
    model.to(device)
    model.train()
    optimizer = build_tac_optimizer(
        model,
        optimizer_config or TACOptimizerConfig(learning_rate=learning_rate),
    )
    aux_weights = aux_weights or {
        "coherence": 0.05,
        "program_reuse": 0.05,
        "energy": 0.01,
        "multi_token": getattr(model.config, "multi_token_loss_weight", 0.0),
        "separation": getattr(model.config, "memory_separation_weight", 0.0),
        "content_cue_separation": getattr(
            model.config,
            "content_cue_separation_weight",
            0.0,
        ),
        "content_gate_entropy": getattr(
            model.config,
            "content_gate_entropy_weight",
            0.0,
        ),
        "routing_load_balance": getattr(
            model.config,
            "routing_load_balance_weight",
            0.0,
        ),
    }
    started = time.perf_counter()
    identity_states = None
    latest_metrics = {
        "loss": 0.0,
        "next_token_loss": 0.0,
        "coherence_loss": 0.0,
        "program_reuse_loss": 0.0,
        "energy_loss": 0.0,
        "used_energy": 0.0,
    }

    for step in range(1, steps + 1):
        input_ids, labels = batcher.next_batch(batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels, identity_states=identity_states)
        next_token_loss = output.loss
        if next_token_loss is None:
            next_token_loss = F.cross_entropy(
                output.logits.reshape(-1, model.config.vocab_size),
                labels.reshape(-1),
            )

        aux_loss = sum(
            aux_weights.get(name, 0.0) * loss
            for name, loss in output.aux.losses.items()
        )
        loss = next_token_loss + aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        identity_states = output.identity_states

        latest_metrics = {
            "loss": float(loss.detach()),
            "next_token_loss": float(next_token_loss.detach()),
            "coherence_loss": float(output.aux.losses["coherence"].detach()),
            "program_reuse_loss": float(output.aux.losses["program_reuse"].detach()),
            "energy_loss": float(output.aux.losses["energy"].detach()),
            "used_energy": float(output.aux.used_energy.mean().detach()),
        }

        if log_every and step % log_every == 0:
            print(
                "step={step} loss={loss:.4f} next={next:.4f} energy={energy:.3f}".format(
                    step=step,
                    loss=latest_metrics["loss"],
                    next=latest_metrics["next_token_loss"],
                    energy=latest_metrics["used_energy"],
                )
            )

    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens_per_second = steps * batch_size * (batcher.seq_len - 1) / elapsed
    metrics = {
        **latest_metrics,
        "steps": steps,
        "tokens_per_second": tokens_per_second,
    }

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(model.config),
                "metrics": metrics,
                "parameter_counts": count_parameters(model),
            },
            path,
        )

    return metrics


def _scalar_metric_value(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        result = float(value.detach())
    elif isinstance(value, (int, float)):
        result = float(value)
    else:
        return None
    return result if math.isfinite(result) else None


def _accumulate_prefixed_scalars(
    totals: dict[str, float],
    values: dict[str, Any],
    *,
    prefix: str,
) -> None:
    for name, value in values.items():
        scalar = _scalar_metric_value(value)
        if scalar is None:
            continue
        key = f"{prefix}{name}"
        totals[key] = totals.get(key, 0.0) + scalar


def _average_scalar_totals(
    totals: dict[str, float],
    count: int,
) -> dict[str, float]:
    denominator = max(count, 1)
    return {name: value / denominator for name, value in sorted(totals.items())}


def evaluate_language_model(
    model: nn.Module,
    batcher: SyntheticProgramBatcher | JsonlTextBatcher,
    *,
    batches: int,
    batch_size: int,
    device: str | torch.device = "cpu",
    carry_state_across_batches: bool = False,
    chunked_state_within_batch: bool = False,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    losses = []
    correct = 0.0
    total = 0
    used_energy = []
    content_addressed_hit = []
    content_synthesis_gate = []
    content_gate_entropy = []
    content_cue_cosine = []
    content_reconsolidation_gate = []
    program_memory_cosine = []
    aux_loss_components: dict[str, float] = {}
    aux_metrics: dict[str, float] = {}
    identity_states = None
    started = time.perf_counter()

    with torch.no_grad():
        for _ in range(batches):
            input_ids, labels = batcher.next_batch(batch_size, device=device)
            output, loss, logits = forward_language_model_window(
                model,
                input_ids,
                labels,
                identity_states=identity_states,
                chunked_state_within_batch=chunked_state_within_batch,
            )
            losses.append(float(loss.detach()))
            predictions = logits.argmax(dim=-1)
            correct += float((predictions == labels).sum().detach())
            total += labels.numel()
            used_energy.append(float(output.aux.used_energy.mean().detach()))
            _accumulate_prefixed_scalars(
                aux_loss_components,
                output.aux.losses,
                prefix="aux_loss_",
            )
            _accumulate_prefixed_scalars(
                aux_metrics,
                output.aux.metrics,
                prefix="metric_",
            )
            content_addressed_hit.append(
                float(
                    output.aux.metrics.get(
                        "content_addressed_hit",
                        output.logits.new_zeros(()),
                    ).detach()
                )
            )
            content_synthesis_gate.append(
                float(
                    output.aux.metrics.get(
                        "content_synthesis_gate",
                        output.logits.new_zeros(()),
                    ).detach()
                )
            )
            content_gate_entropy.append(
                float(
                    output.aux.metrics.get(
                        "content_gate_entropy",
                        output.logits.new_zeros(()),
                    ).detach()
                )
            )
            content_cue_cosine.append(
                float(
                    output.aux.metrics.get(
                        "content_cue_cosine",
                        output.logits.new_zeros(()),
                    ).detach()
                )
            )
            content_reconsolidation_gate.append(
                float(
                    output.aux.metrics.get(
                        "content_reconsolidation_gate",
                        output.logits.new_zeros(()),
                    ).detach()
                )
            )
            program_memory_cosine.append(
                float(
                    output.aux.metrics.get(
                        "program_memory_cosine",
                        output.logits.new_zeros(()),
                    ).detach()
                )
            )
            identity_states = (
                output.identity_states or None
                if carry_state_across_batches
                else None
            )

    elapsed = max(time.perf_counter() - started, 1e-9)
    mean_loss = sum(losses) / max(len(losses), 1)
    tokens = batches * batch_size * (batcher.seq_len - 1)
    metrics = {
        "loss": mean_loss,
        "perplexity": float(torch.exp(torch.tensor(mean_loss))),
        "accuracy": correct / max(total, 1),
        "used_energy": sum(used_energy) / max(len(used_energy), 1),
        "content_addressed_hit": sum(content_addressed_hit)
        / max(len(content_addressed_hit), 1),
        "content_synthesis_gate": sum(content_synthesis_gate)
        / max(len(content_synthesis_gate), 1),
        "content_gate_entropy": sum(content_gate_entropy)
        / max(len(content_gate_entropy), 1),
        "content_cue_cosine": sum(content_cue_cosine)
        / max(len(content_cue_cosine), 1),
        "content_reconsolidation_gate": sum(content_reconsolidation_gate)
        / max(len(content_reconsolidation_gate), 1),
        "program_memory_cosine": sum(program_memory_cosine)
        / max(len(program_memory_cosine), 1),
        "tokens_per_second": tokens / elapsed,
    }
    metrics.update(_average_scalar_totals(aux_loss_components, batches))
    metrics.update(_average_scalar_totals(aux_metrics, batches))
    return metrics


def forward_language_model_window(
    model: nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    *,
    identity_states: Optional[list[IdentityState]] = None,
    chunked_state_within_batch: bool = False,
    collect_auxiliary: bool = True,
    collect_metrics: bool = True,
) -> tuple[TACOutput, Tensor, Tensor]:
    if not chunked_state_within_batch or input_ids.shape[1] < 2:
        output = model(
            input_ids,
            labels=labels,
            identity_states=identity_states,
            collect_auxiliary=collect_auxiliary,
            collect_metrics=collect_metrics,
        )
        loss = output.loss
        if loss is None:
            loss = F.cross_entropy(
                output.logits.reshape(-1, model.config.vocab_size),
                labels.reshape(-1),
            )
        loss = loss + _zero_auxiliary_gradient_safety(output)
        return output, loss, output.logits

    split = max(1, input_ids.shape[1] // 2)
    context = model(
        input_ids[:, :split],
        labels=labels[:, :split],
        identity_states=identity_states,
        collect_auxiliary=False,
        collect_metrics=False,
    )
    query = model(
        input_ids[:, split:],
        labels=labels[:, split:],
        identity_states=context.identity_states,
        collect_auxiliary=collect_auxiliary,
        collect_metrics=collect_metrics,
    )
    context_loss = context.loss
    if context_loss is None:
        context_loss = F.cross_entropy(
            context.logits.reshape(-1, model.config.vocab_size),
            labels[:, :split].reshape(-1),
        )
    query_loss = query.loss
    if query_loss is None:
        query_loss = F.cross_entropy(
            query.logits.reshape(-1, model.config.vocab_size),
            labels[:, split:].reshape(-1),
        )
    total_tokens = labels.numel()
    loss = (
        context_loss * labels[:, :split].numel()
        + query_loss * labels[:, split:].numel()
    ) / max(total_tokens, 1)
    loss = (
        loss
        + _zero_auxiliary_gradient_safety(context)
        + _zero_auxiliary_gradient_safety(query)
    )
    logits = torch.cat([context.logits, query.logits], dim=1)
    return query, loss, logits


def _zero_auxiliary_gradient_safety(output: TACOutput) -> Tensor:
    return output.aux.used_energy.sum() * 0.0


def train_language_model(
    model: nn.Module,
    batcher: SyntheticProgramBatcher | JsonlTextBatcher,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    aux_weights: Optional[dict[str, float]] = None,
    device: str | torch.device = "cpu",
    optimizer_config: Optional[TACOptimizerConfig] = None,
) -> dict[str, float]:
    model.to(device)
    model.train()
    optimizer = build_tac_optimizer(
        model,
        optimizer_config or TACOptimizerConfig(learning_rate=learning_rate),
    )
    aux_weights = aux_weights or {
        "coherence": 0.05,
        "program_reuse": 0.05,
        "energy": 0.01,
        "multi_token": getattr(model.config, "multi_token_loss_weight", 0.0),
        "separation": getattr(model.config, "memory_separation_weight", 0.0),
        "content_cue_separation": getattr(
            model.config,
            "content_cue_separation_weight",
            0.0,
        ),
        "content_gate_entropy": getattr(
            model.config,
            "content_gate_entropy_weight",
            0.0,
        ),
        "routing_load_balance": getattr(
            model.config,
            "routing_load_balance_weight",
            0.0,
        ),
    }
    identity_states = None
    latest_loss = 0.0
    latest_next_token_loss = 0.0
    started = time.perf_counter()

    for _ in range(steps):
        input_ids, labels = batcher.next_batch(batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels, identity_states=identity_states)
        next_token_loss = output.loss
        if next_token_loss is None:
            next_token_loss = F.cross_entropy(
                output.logits.reshape(-1, model.config.vocab_size),
                labels.reshape(-1),
            )
        aux_loss = sum(
            aux_weights.get(name, 0.0) * loss
            for name, loss in output.aux.losses.items()
        )
        loss = next_token_loss + aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        identity_states = output.identity_states or None
        latest_loss = float(loss.detach())
        latest_next_token_loss = float(next_token_loss.detach())

    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = steps * batch_size * (batcher.seq_len - 1)
    return {
        "loss": latest_loss,
        "next_token_loss": latest_next_token_loss,
        "steps": steps,
        "tokens_per_second": tokens / elapsed,
    }


def train_chunked_memory(
    model: nn.Module,
    batcher: ChunkedRecallBatcher,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    aux_weights: Optional[dict[str, float]] = None,
    value_loss_weight: float = 0.0,
    memory_read_loss_weight: float = 0.0,
    memory_injection_weight: float = 0.0,
    memory_adapter_weight: float = 0.0,
    device: str | torch.device = "cpu",
    optimizer_config: Optional[TACOptimizerConfig] = None,
) -> dict[str, float]:
    model.to(device)
    model.train()
    optimizer = build_tac_optimizer(
        model,
        optimizer_config or TACOptimizerConfig(learning_rate=learning_rate),
    )
    aux_weights = aux_weights or {
        "coherence": 0.05,
        "program_reuse": 0.05,
        "energy": 0.01,
        "multi_token": getattr(model.config, "multi_token_loss_weight", 0.0),
        "separation": getattr(model.config, "memory_separation_weight", 0.0),
        "content_cue_separation": getattr(
            model.config,
            "content_cue_separation_weight",
            0.0,
        ),
        "content_gate_entropy": getattr(
            model.config,
            "content_gate_entropy_weight",
            0.0,
        ),
        "routing_load_balance": getattr(
            model.config,
            "routing_load_balance_weight",
            0.0,
        ),
    }
    latest_loss = 0.0
    latest_value_loss = 0.0
    latest_memory_read_loss = 0.0
    latest_memory_read_accuracy = 0.0
    latest_value_accuracy = 0.0
    latest_separation_loss = 0.0
    latest_program_memory_cosine = 0.0
    latest_memory_reconsolidation_gate = 0.0
    latest_memory_allocation_dead_rate = 0.0
    latest_memory_allocation_age = 0.0
    latest_memory_allocation_write_frequency = 0.0
    started = time.perf_counter()

    for _ in range(steps):
        batch = batcher.next_batch(batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)

        context = model(
            batch.context_inputs,
            labels=batch.context_labels,
            content_write_mask=batch.context_write_mask,
        )
        query = model(
            batch.query_inputs,
            labels=batch.query_labels,
            identity_states=context.identity_states or None,
        )
        next_token_loss = _output_loss(context, batch.context_labels) + _output_loss(
            query,
            batch.query_labels,
        )
        query_logits = query.logits
        value_loss = _value_token_loss(query.logits, batch)
        memory_read_loss = query.logits.new_zeros(())
        needs_memory_read = (
            memory_read_loss_weight
            or memory_injection_weight
            or memory_adapter_weight
        )
        if needs_memory_read:
            memory_vector = model.memory_read_vector(
                batch.query_inputs[:, 1],
                context.identity_states,
            )
            memory_logits = model.memory_read_logits(
                batch.query_inputs[:, 1],
                context.identity_states,
            )
            if memory_adapter_weight:
                query_logits = model.memory_adapted_logits(
                    query.hidden_states,
                    memory_vector,
                    value_label_index=batch.value_label_index,
                    weight=memory_adapter_weight,
                )
            query_logits = apply_memory_read_logits(
                query_logits,
                memory_logits,
                value_label_index=batch.value_label_index,
                weight=memory_injection_weight,
            )
            value_loss = _value_token_loss(query_logits, batch)
            if memory_read_loss_weight:
                memory_read_loss = F.cross_entropy(memory_logits, batch.value_targets)
            latest_memory_read_accuracy = _classification_accuracy(
                memory_logits,
                batch.value_targets,
            )
        aux_loss = _weighted_auxiliary_loss(context, aux_weights) + _weighted_auxiliary_loss(
            query,
            aux_weights,
        )
        loss = (
            next_token_loss
            + aux_loss
            + value_loss_weight * value_loss
            + memory_read_loss_weight * memory_read_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        latest_loss = float(loss.detach())
        latest_value_loss = float(value_loss.detach())
        latest_memory_read_loss = float(memory_read_loss.detach())
        latest_value_accuracy = _value_accuracy(query_logits, batch)
        latest_separation_loss = float(
            query.aux.losses.get(
                "separation",
                query_logits.new_zeros(()),
            ).detach()
        )
        latest_program_memory_cosine = float(
            query.aux.metrics.get(
                "program_memory_cosine",
                query_logits.new_zeros(()),
            ).detach()
        )
        latest_memory_reconsolidation_gate = float(
            query.aux.metrics.get(
                "memory_reconsolidation_gate",
                query_logits.new_zeros(()),
            ).detach()
        )
        latest_memory_allocation_dead_rate = float(
            query.aux.metrics.get(
                "memory_allocation_dead_rate",
                query_logits.new_zeros(()),
            ).detach()
        )
        latest_memory_allocation_age = float(
            query.aux.metrics.get(
                "memory_allocation_age",
                query_logits.new_zeros(()),
            ).detach()
        )
        latest_memory_allocation_write_frequency = float(
            query.aux.metrics.get(
                "memory_allocation_write_frequency",
                query_logits.new_zeros(()),
            ).detach()
        )

    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = steps * batch_size * batcher.seq_len * 2
    return {
        "loss": latest_loss,
        "value_loss": latest_value_loss,
        "memory_read_loss": latest_memory_read_loss,
        "memory_read_accuracy": latest_memory_read_accuracy,
        "value_accuracy": latest_value_accuracy,
        "separation_loss": latest_separation_loss,
        "program_memory_cosine": latest_program_memory_cosine,
        "memory_reconsolidation_gate": latest_memory_reconsolidation_gate,
        "memory_allocation_dead_rate": latest_memory_allocation_dead_rate,
        "memory_allocation_age": latest_memory_allocation_age,
        "memory_allocation_write_frequency": latest_memory_allocation_write_frequency,
        "steps": steps,
        "tokens_per_second": tokens / elapsed,
    }


def evaluate_chunked_memory(
    model: nn.Module,
    batcher: ChunkedRecallBatcher,
    *,
    batches: int,
    batch_size: int,
    mode: str = "carry",
    memory_injection_weight: float = 0.0,
    memory_adapter_weight: float = 0.0,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    if mode not in {"carry", "reset", "shuffled"}:
        raise ValueError("mode must be 'carry', 'reset', or 'shuffled'")
    model.to(device)
    model.eval()
    losses = []
    correct_values = 0.0
    total_values = 0
    used_energy = []
    active_programs = []
    active_expert_parameters = []
    total_expert_parameters = []
    active_expert_fraction = []
    program_memory_cosine = []
    memory_reconsolidation_gate = []
    memory_allocation_dead_rate = []
    memory_allocation_age = []
    memory_allocation_load_std = []
    memory_allocation_write_frequency = []
    content_read_queries = []
    content_read_query_fraction = []
    content_read_skipped_fraction = []
    coalition_context_norm = []
    started = time.perf_counter()

    with torch.no_grad():
        for _ in range(batches):
            batch = batcher.next_batch(batch_size, device=device)
            context = model(
                batch.context_inputs,
                labels=batch.context_labels,
                content_write_mask=batch.context_write_mask,
            )
            if mode == "carry":
                states = context.identity_states or None
            elif mode == "shuffled":
                states = _shuffle_identity_states(context.identity_states)
            else:
                states = None

            query = model(
                batch.query_inputs,
                labels=batch.query_labels,
                identity_states=states,
            )
            query_logits = query.logits
            if (
                (memory_injection_weight or memory_adapter_weight)
                and hasattr(model, "memory_read_logits")
                and model.config.memory_read_type
                in {"program_memory", "pattern_completion", "content_addressed"}
                and states is not None
            ):
                memory_vector = model.memory_read_vector(batch.query_inputs[:, 1], states)
                memory_logits = model.memory_read_logits(batch.query_inputs[:, 1], states)
                if memory_adapter_weight:
                    query_logits = model.memory_adapted_logits(
                        query.hidden_states,
                        memory_vector,
                        value_label_index=batch.value_label_index,
                        weight=memory_adapter_weight,
                    )
                query_logits = apply_memory_read_logits(
                    query_logits,
                    memory_logits,
                    value_label_index=batch.value_label_index,
                    weight=memory_injection_weight,
                )
            losses.append(float(_query_loss(query_logits, batch.query_labels).detach()))
            value_logits = query_logits[:, batch.value_label_index, :]
            predictions = value_logits.argmax(dim=-1)
            correct_values += float((predictions == batch.value_targets).sum().detach())
            total_values += batch.value_targets.numel()
            used_energy.append(float(query.aux.used_energy.mean().detach()))
            active_programs.append(float(query.aux.selected_program_mask.sum(dim=-1).mean().detach()))
            active_expert_parameters.append(
                float(query.aux.metrics["active_expert_parameters"].detach())
            )
            total_expert_parameters.append(
                float(query.aux.metrics["total_expert_parameters"].detach())
            )
            active_expert_fraction.append(
                float(query.aux.metrics["active_expert_fraction"].detach())
            )
            program_memory_cosine.append(
                float(
                    query.aux.metrics.get(
                        "program_memory_cosine",
                        query.logits.new_zeros(()),
                    ).detach()
                )
            )
            memory_reconsolidation_gate.append(
                float(
                    query.aux.metrics.get(
                        "memory_reconsolidation_gate",
                        query.logits.new_zeros(()),
                    ).detach()
                )
            )
            memory_allocation_dead_rate.append(
                float(
                    query.aux.metrics.get(
                        "memory_allocation_dead_rate",
                        query.logits.new_zeros(()),
                    ).detach()
                )
            )
            memory_allocation_age.append(
                float(
                    query.aux.metrics.get(
                        "memory_allocation_age",
                        query.logits.new_zeros(()),
                    ).detach()
                )
            )
            memory_allocation_load_std.append(
                float(
                    query.aux.metrics.get(
                        "memory_allocation_load_std",
                        query.logits.new_zeros(()),
                    ).detach()
                )
            )
            memory_allocation_write_frequency.append(
                float(
                    query.aux.metrics.get(
                        "memory_allocation_write_frequency",
                        query.logits.new_zeros(()),
                    ).detach()
                )
            )
            content_read_queries.append(
                float(
                    query.aux.metrics.get(
                        "content_read_queries",
                        query.logits.new_zeros(()),
                    ).detach()
                )
            )
            content_read_query_fraction.append(
                float(
                    query.aux.metrics.get(
                        "content_read_query_fraction",
                        query.logits.new_zeros(()),
                    ).detach()
                )
            )
            content_read_skipped_fraction.append(
                float(
                    query.aux.metrics.get(
                        "content_read_skipped_fraction",
                        query.logits.new_zeros(()),
                    ).detach()
                )
            )
            coalition_context_norm.append(
                float(
                    query.aux.metrics.get(
                        "coalition_context_norm",
                        query.logits.new_zeros(()),
                    ).detach()
                )
            )

    elapsed = max(time.perf_counter() - started, 1e-9)
    mean_loss = sum(losses) / max(len(losses), 1)
    tokens = batches * batch_size * batcher.seq_len * 2
    return {
        "loss": mean_loss,
        "perplexity": float(torch.exp(torch.tensor(mean_loss))),
        "value_accuracy": correct_values / max(total_values, 1),
        "used_energy": sum(used_energy) / max(len(used_energy), 1),
        "active_programs": sum(active_programs) / max(len(active_programs), 1),
        "active_expert_parameters": sum(active_expert_parameters) / max(len(active_expert_parameters), 1),
        "total_expert_parameters": sum(total_expert_parameters) / max(len(total_expert_parameters), 1),
        "active_expert_fraction": sum(active_expert_fraction) / max(len(active_expert_fraction), 1),
        "program_memory_cosine": sum(program_memory_cosine) / max(len(program_memory_cosine), 1),
        "memory_reconsolidation_gate": sum(memory_reconsolidation_gate)
        / max(len(memory_reconsolidation_gate), 1),
        "memory_allocation_dead_rate": sum(memory_allocation_dead_rate)
        / max(len(memory_allocation_dead_rate), 1),
        "memory_allocation_age": sum(memory_allocation_age)
        / max(len(memory_allocation_age), 1),
        "memory_allocation_load_std": sum(memory_allocation_load_std)
        / max(len(memory_allocation_load_std), 1),
        "memory_allocation_write_frequency": sum(memory_allocation_write_frequency)
        / max(len(memory_allocation_write_frequency), 1),
        "content_read_queries": sum(content_read_queries)
        / max(len(content_read_queries), 1),
        "content_read_query_fraction": sum(content_read_query_fraction)
        / max(len(content_read_query_fraction), 1),
        "content_read_skipped_fraction": sum(content_read_skipped_fraction)
        / max(len(content_read_skipped_fraction), 1),
        "coalition_context_norm": sum(coalition_context_norm)
        / max(len(coalition_context_norm), 1),
        "tokens_per_second": tokens / elapsed,
    }


def benchmark_chunked_memory(
    config: TACConfig,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    eval_batches: int = 8,
    eval_batch_size: Optional[int] = None,
    seed: int = 7,
    device: str | torch.device = "cpu",
    aux_weights: Optional[dict[str, float]] = None,
    match_baseline_parameters: bool = False,
    min_value_accuracy_delta: float = 0.0,
    value_loss_weight: float = 0.0,
    memory_read_loss_weight: float = 0.0,
    memory_injection_weight: float = 0.0,
    memory_adapter_weight: float = 0.0,
    task_variant: str = "single_key",
) -> dict[str, object]:
    eval_batch_size = eval_batch_size or batch_size
    baseline_config = (
        parameter_matched_baseline_config(config)
        if match_baseline_parameters
        else config
    )
    torch.manual_seed(seed)
    tac_model = TACTransformerLM(config)
    torch.manual_seed(seed)
    baseline_model = VanillaTransformerLM(baseline_config)

    tac_train = train_chunked_memory(
        tac_model,
        ChunkedRecallBatcher(
            config.vocab_size,
            config.max_seq_len,
            seed=seed + 100,
            task_variant=task_variant,
        ),
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        aux_weights=aux_weights,
        value_loss_weight=value_loss_weight,
        memory_read_loss_weight=memory_read_loss_weight,
        memory_injection_weight=memory_injection_weight,
        memory_adapter_weight=memory_adapter_weight,
        device=device,
    )
    baseline_train = train_chunked_memory(
        baseline_model,
        ChunkedRecallBatcher(
            config.vocab_size,
            config.max_seq_len,
            seed=seed + 100,
            task_variant=task_variant,
        ),
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        aux_weights=aux_weights,
        value_loss_weight=value_loss_weight,
        memory_read_loss_weight=0.0,
        memory_injection_weight=0.0,
        device=device,
    )

    def probe(model: nn.Module, offset: int) -> dict[str, object]:
        carry = evaluate_chunked_memory(
            model,
            ChunkedRecallBatcher(
                config.vocab_size,
                config.max_seq_len,
                seed=seed + offset,
                task_variant=task_variant,
            ),
            batches=eval_batches,
            batch_size=eval_batch_size,
            mode="carry",
            memory_injection_weight=memory_injection_weight,
            memory_adapter_weight=memory_adapter_weight,
            device=device,
        )
        reset = evaluate_chunked_memory(
            model,
            ChunkedRecallBatcher(
                config.vocab_size,
                config.max_seq_len,
                seed=seed + offset,
                task_variant=task_variant,
            ),
            batches=eval_batches,
            batch_size=eval_batch_size,
            mode="reset",
            memory_injection_weight=memory_injection_weight,
            memory_adapter_weight=memory_adapter_weight,
            device=device,
        )
        shuffled = evaluate_chunked_memory(
            model,
            ChunkedRecallBatcher(
                config.vocab_size,
                config.max_seq_len,
                seed=seed + offset,
                task_variant=task_variant,
            ),
            batches=eval_batches,
            batch_size=eval_batch_size,
            mode="shuffled",
            memory_injection_weight=memory_injection_weight,
            memory_adapter_weight=memory_adapter_weight,
            device=device,
        )
        return {
            "carry": carry,
            "reset": reset,
            "shuffled": shuffled,
            "loss_carry_delta": reset["loss"] - carry["loss"],
            "value_accuracy_delta": carry["value_accuracy"] - reset["value_accuracy"],
            "shuffled_value_penalty": carry["value_accuracy"] - shuffled["value_accuracy"],
        }

    tac_probe = probe(tac_model, 200)
    baseline_probe = probe(baseline_model, 200)
    decision = _chunked_memory_decision(
        tac_probe,
        baseline_probe,
        min_value_accuracy_delta=min_value_accuracy_delta,
    )

    return {
        "config": asdict(config),
        "baseline_config": asdict(baseline_config),
        "match_baseline_parameters": match_baseline_parameters,
        "steps": steps,
        "task_variant": task_variant,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "value_loss_weight": value_loss_weight,
        "memory_read_loss_weight": memory_read_loss_weight,
        "memory_injection_weight": memory_injection_weight,
        "memory_adapter_weight": memory_adapter_weight,
        "decision": decision,
        "tac": {
            "parameter_counts": count_parameters(tac_model),
            "train": tac_train,
            "chunked_probe": tac_probe,
        },
        "baseline": {
            "parameter_counts": count_parameters(baseline_model),
            "train": baseline_train,
            "chunked_probe": baseline_probe,
        },
    }


def benchmark_data_energy_efficiency(
    config: TACConfig,
    *,
    budgets: list[int],
    batch_size: int,
    learning_rate: float,
    eval_batches: int = 8,
    eval_batch_size: Optional[int] = None,
    seed: int = 7,
    device: str | torch.device = "cpu",
    aux_weights: Optional[dict[str, float]] = None,
    match_baseline_parameters: bool = True,
    min_value_accuracy_delta: float = 0.0,
    value_loss_weight: float = 0.0,
    memory_read_loss_weight: float = 0.0,
    memory_injection_weight: float = 0.0,
    memory_adapter_weight: float = 0.0,
) -> dict[str, object]:
    """Run chunked-memory benchmarks at several training budgets.

    The scorecard uses train tokens as the data-efficiency axis and observed
    throughput/parameter counts as practical compute proxies. It does not claim
    hardware energy in joules; that needs a profiler outside this Python harness.
    """
    if not budgets:
        raise ValueError("budgets must contain at least one step count")
    if any(step < 1 for step in budgets):
        raise ValueError("budgets must be positive step counts")

    results = []
    for index, steps in enumerate(budgets):
        run = benchmark_chunked_memory(
            config,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            eval_batches=eval_batches,
            eval_batch_size=eval_batch_size,
            seed=seed + index * 1000,
            device=device,
            aux_weights=aux_weights,
            match_baseline_parameters=match_baseline_parameters,
            min_value_accuracy_delta=min_value_accuracy_delta,
            value_loss_weight=value_loss_weight,
            memory_read_loss_weight=memory_read_loss_weight,
            memory_injection_weight=memory_injection_weight,
            memory_adapter_weight=memory_adapter_weight,
        )
        results.append(_efficiency_budget_result(run))

    return {
        "config": asdict(config),
        "budgets": budgets,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "metric_notes": {
            "data_efficiency": "quality per observed training token on the no-leak chunked recall task",
            "energy_efficiency": "throughput, parameter counts, and TAC routing energy proxies; not hardware joules",
        },
        "budget_results": results,
        "summary": _efficiency_summary(results),
    }


def _efficiency_budget_result(run: dict[str, object]) -> dict[str, object]:
    steps = int(run["steps"])
    batch_size = int(run["batch_size"])
    seq_len = int(run["config"]["max_seq_len"])
    train_tokens = steps * batch_size * seq_len * 2
    tac_probe = run["tac"]["chunked_probe"]
    baseline_probe = run["baseline"]["chunked_probe"]
    tac_carry = tac_probe["carry"]
    baseline_carry = baseline_probe["carry"]
    tac_accuracy = float(tac_carry["value_accuracy"])
    baseline_accuracy = float(baseline_carry["value_accuracy"])
    tac_train_tps = float(run["tac"]["train"]["tokens_per_second"])
    baseline_train_tps = float(run["baseline"]["train"]["tokens_per_second"])
    tac_eval_tps = float(tac_carry["tokens_per_second"])
    baseline_eval_tps = float(baseline_carry["tokens_per_second"])
    tac_params = int(run["tac"]["parameter_counts"]["total"])
    baseline_params = int(run["baseline"]["parameter_counts"]["total"])

    return {
        "steps": steps,
        "decision": run["decision"],
        "data_efficiency": {
            "train_tokens": train_tokens,
            "tac_carry_value_accuracy": tac_accuracy,
            "baseline_value_accuracy": baseline_accuracy,
            "accuracy_gain": tac_accuracy - baseline_accuracy,
            "tac_accuracy_per_1k_train_tokens": tac_accuracy / max(train_tokens / 1000.0, 1e-9),
            "baseline_accuracy_per_1k_train_tokens": baseline_accuracy / max(train_tokens / 1000.0, 1e-9),
        },
        "energy_efficiency": {
            "tac_train_tokens_per_second": tac_train_tps,
            "baseline_train_tokens_per_second": baseline_train_tps,
            "tokens_per_second_ratio": tac_train_tps / max(baseline_train_tps, 1e-9),
            "tac_eval_tokens_per_second": tac_eval_tps,
            "baseline_eval_tokens_per_second": baseline_eval_tps,
            "eval_tokens_per_second_ratio": tac_eval_tps / max(baseline_eval_tps, 1e-9),
            "tac_parameters": tac_params,
            "baseline_parameters": baseline_params,
            "parameter_ratio": tac_params / max(baseline_params, 1),
            "tac_used_energy": float(tac_carry["used_energy"]),
            "tac_active_programs": float(tac_carry["active_programs"]),
            "tac_active_expert_parameters": float(tac_carry["active_expert_parameters"]),
            "tac_total_expert_parameters": float(tac_carry["total_expert_parameters"]),
            "tac_active_expert_fraction": float(tac_carry["active_expert_fraction"]),
            "baseline_used_energy": float(baseline_carry["used_energy"]),
            "baseline_active_programs": float(baseline_carry["active_programs"]),
            "baseline_active_expert_parameters": float(baseline_carry["active_expert_parameters"]),
            "baseline_total_expert_parameters": float(baseline_carry["total_expert_parameters"]),
            "baseline_active_expert_fraction": float(baseline_carry["active_expert_fraction"]),
        },
        "raw_result": run,
    }


def _efficiency_summary(results: list[dict[str, object]]) -> dict[str, object]:
    best_by_accuracy = max(
        results,
        key=lambda item: item["data_efficiency"]["tac_carry_value_accuracy"],
    )
    best_by_gain = max(
        results,
        key=lambda item: item["data_efficiency"]["accuracy_gain"],
    )
    final = results[-1]
    return {
        "best_accuracy_steps": best_by_accuracy["steps"],
        "best_accuracy": best_by_accuracy["data_efficiency"]["tac_carry_value_accuracy"],
        "best_gain_steps": best_by_gain["steps"],
        "best_gain": best_by_gain["data_efficiency"]["accuracy_gain"],
        "final_tac_accuracy": final["data_efficiency"]["tac_carry_value_accuracy"],
        "final_baseline_accuracy": final["data_efficiency"]["baseline_value_accuracy"],
        "final_accuracy_gain": final["data_efficiency"]["accuracy_gain"],
        "final_tokens_per_second_ratio": final["energy_efficiency"]["tokens_per_second_ratio"],
        "final_parameter_ratio": final["energy_efficiency"]["parameter_ratio"],
    }


def benchmark_synthetic(
    config: TACConfig,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    eval_batches: int = 8,
    eval_batch_size: Optional[int] = None,
    seed: int = 7,
    device: str | torch.device = "cpu",
    aux_weights: Optional[dict[str, float]] = None,
    match_baseline_parameters: bool = False,
) -> dict[str, object]:
    eval_batch_size = eval_batch_size or batch_size
    baseline_config = (
        parameter_matched_baseline_config(config)
        if match_baseline_parameters
        else config
    )
    torch.manual_seed(seed)
    tac_model = TACTransformerLM(config)
    torch.manual_seed(seed)
    baseline_model = VanillaTransformerLM(baseline_config)

    def make_batcher(offset: int) -> SyntheticProgramBatcher:
        return SyntheticProgramBatcher(
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            seed=seed + offset,
        )

    tac_initial = evaluate_language_model(
        tac_model,
        make_batcher(1000),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
    )
    baseline_initial = evaluate_language_model(
        baseline_model,
        make_batcher(1000),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
    )

    tac_train = train_language_model(
        tac_model,
        make_batcher(2000),
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        aux_weights=aux_weights,
        device=device,
    )
    baseline_train = train_language_model(
        baseline_model,
        make_batcher(2000),
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        aux_weights=aux_weights,
        device=device,
    )

    tac_final = evaluate_language_model(
        tac_model,
        make_batcher(1000),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
    )
    baseline_final = evaluate_language_model(
        baseline_model,
        make_batcher(1000),
        batches=eval_batches,
        batch_size=eval_batch_size,
        device=device,
    )

    return {
        "config": asdict(config),
        "baseline_config": asdict(baseline_config),
        "match_baseline_parameters": match_baseline_parameters,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "tac": {
            "parameter_counts": count_parameters(tac_model),
            "initial_eval": tac_initial,
            "train": tac_train,
            "final_eval": tac_final,
        },
        "baseline": {
            "parameter_counts": count_parameters(baseline_model),
            "initial_eval": baseline_initial,
            "train": baseline_train,
            "final_eval": baseline_final,
        },
    }


def _output_loss(output, labels: Tensor) -> Tensor:
    if output.loss is not None:
        return output.loss
    return F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
    )


def _weighted_auxiliary_loss(output, aux_weights: dict[str, float]) -> Tensor:
    return sum(
        aux_weights.get(name, 0.0) * loss
        for name, loss in output.aux.losses.items()
    )


def _value_accuracy(logits: Tensor, batch: ChunkedRecallBatch) -> float:
    predictions = logits[:, batch.value_label_index, :].argmax(dim=-1)
    return float((predictions == batch.value_targets).float().mean().detach())


def _value_token_loss(logits: Tensor, batch: ChunkedRecallBatch) -> Tensor:
    return F.cross_entropy(
        logits[:, batch.value_label_index, :],
        batch.value_targets,
    )


def _query_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
    )


def apply_memory_read_logits(
    query_logits: Tensor,
    memory_logits: Tensor,
    *,
    value_label_index: int,
    weight: float,
) -> Tensor:
    if weight == 0:
        return query_logits
    updated = query_logits.clone()
    updated[:, value_label_index, :] = (
        updated[:, value_label_index, :] + weight * memory_logits
    )
    return updated


def _classification_accuracy(logits: Tensor, targets: Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().detach())


def _shuffle_identity_states(states: list[IdentityState]) -> list[IdentityState] | None:
    if not states:
        return None
    shuffled = []
    for state in states:
        if state.stability.shape[0] < 2:
            shuffled.append(state)
            continue
        shuffled.append(
            IdentityState(
                stability=state.stability.roll(shifts=1, dims=0),
                program_memory=state.program_memory.roll(shifts=1, dims=0),
                stable_program_memory=(
                    state.stable_program_memory.roll(shifts=1, dims=0)
                    if state.stable_program_memory is not None
                    else None
                ),
                archival_program_memory=(
                    state.archival_program_memory.roll(shifts=1, dims=0)
                    if state.archival_program_memory is not None
                    else None
                ),
                program_age=(
                    state.program_age.roll(shifts=1, dims=0)
                    if state.program_age is not None
                    else None
                ),
                program_write_frequency=(
                    state.program_write_frequency.roll(shifts=1, dims=0)
                    if state.program_write_frequency is not None
                    else None
                ),
                engram_patterns=(
                    state.engram_patterns.roll(shifts=1, dims=0)
                    if state.engram_patterns is not None
                    else None
                ),
                engram_values=(
                    state.engram_values.roll(shifts=1, dims=0)
                    if state.engram_values is not None
                    else None
                ),
                engram_mask=(
                    state.engram_mask.roll(shifts=1, dims=0)
                    if state.engram_mask is not None
                    else None
                ),
                content_cues=(
                    state.content_cues.roll(shifts=1, dims=0)
                    if state.content_cues is not None
                    else None
                ),
                content_values=(
                    state.content_values.roll(shifts=1, dims=0)
                    if state.content_values is not None
                    else None
                ),
                content_mask=(
                    state.content_mask.roll(shifts=1, dims=0)
                    if state.content_mask is not None
                    else None
                ),
            )
        )
    return shuffled


def _chunked_memory_decision(
    tac_probe: dict[str, object],
    baseline_probe: dict[str, object],
    *,
    min_value_accuracy_delta: float,
) -> dict[str, object]:
    tac_carry = tac_probe["carry"]
    baseline_carry = baseline_probe["carry"]
    value_accuracy_delta = float(tac_probe["value_accuracy_delta"])
    shuffled_value_penalty = float(tac_probe["shuffled_value_penalty"])
    baseline_gap = tac_carry["value_accuracy"] - baseline_carry["value_accuracy"]
    checks = {
        "carry_beats_reset": value_accuracy_delta > min_value_accuracy_delta,
        "carry_beats_shuffled": shuffled_value_penalty > min_value_accuracy_delta,
        "tac_matches_or_beats_baseline_value_accuracy": baseline_gap >= 0.0,
    }
    return {
        "status": "effective" if all(checks.values()) else "inconclusive",
        "checks": checks,
        "value_accuracy_delta": value_accuracy_delta,
        "shuffled_value_penalty": shuffled_value_penalty,
        "baseline_value_accuracy_gap": baseline_gap,
        "thresholds": {
            "min_value_accuracy_delta": min_value_accuracy_delta,
        },
    }


def default_kaggle_output_path() -> Path:
    kaggle_working = Path("/kaggle/working")
    is_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()
    if is_kaggle and kaggle_working.exists():
        return kaggle_working / "tac_transformer.pt"
    return Path("runs") / "tac_transformer.pt"


def _byte_tokens(text: str) -> list[int]:
    return [byte + 4 for byte in text.encode("utf-8", errors="replace")]
