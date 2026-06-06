from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch import Tensor

from .model import TACConfig, TACTransformerLM, VanillaTransformerLM


TAC_EOS_TOKEN_ID = 3
TAC_BYTE_TOKEN_OFFSET = 4
TAC_MIN_BYTE_VOCAB_SIZE = TAC_BYTE_TOKEN_OFFSET + 256


def encode_tac_byte_tokens(
    text: str,
    *,
    vocab_size: int,
    append_eos: bool = False,
) -> list[int]:
    """Encode text with the byte-level TAC training contract."""

    if int(vocab_size) < TAC_MIN_BYTE_VOCAB_SIZE:
        raise ValueError(
            f"vocab_size must be at least {TAC_MIN_BYTE_VOCAB_SIZE} "
            "for TAC byte-token generation"
        )
    token_ids = [
        byte + TAC_BYTE_TOKEN_OFFSET
        for byte in str(text).encode("utf-8", errors="replace")
    ]
    if append_eos:
        token_ids.append(TAC_EOS_TOKEN_ID)
    return token_ids


def decode_tac_byte_tokens(
    token_ids: Iterable[int],
    *,
    stop_at_eos: bool = True,
) -> str:
    bytes_out = bytearray()
    for token_id in token_ids:
        token = int(token_id)
        if token == TAC_EOS_TOKEN_ID and stop_at_eos:
            break
        byte_value = token - TAC_BYTE_TOKEN_OFFSET
        if 0 <= byte_value <= 255:
            bytes_out.append(byte_value)
    return bytes(bytes_out).decode("utf-8", errors="replace")


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 0.7,
    top_k: int | None = 50,
    top_p: float = 0.9,
    generator: torch.Generator | None = None,
) -> int:
    """Sample from one token-logit vector with common LLM controls."""

    if logits.dim() != 1:
        raise ValueError("logits must be a 1D token-logit tensor")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and int(top_k) < 1:
        raise ValueError("top_k must be positive when set")
    if not 0.0 < float(top_p) <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return int(torch.argmax(logits).detach().cpu())

    filtered = logits.float() / max(float(temperature), 1e-8)
    if top_k is not None and int(top_k) < filtered.numel():
        values, _ = torch.topk(filtered, int(top_k))
        filtered = filtered.masked_fill(filtered < values[-1], float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumulative > float(top_p)
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        remove_indices = sorted_indices[remove_sorted]
        filtered = filtered.clone()
        filtered[remove_indices] = float("-inf")

    probabilities = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0.0:
        return int(torch.argmax(logits).detach().cpu())
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def generate_tac_completion(
    model: torch.nn.Module,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int | None = 50,
    top_p: float = 0.9,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a byte-level completion from a TAC or vanilla checkpoint model."""

    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be at least 1")
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model must expose a TACConfig-compatible config")
    resolved_device = torch.device(device)
    if hasattr(model, "to"):
        model.to(resolved_device)
    if hasattr(model, "eval"):
        model.eval()

    prompt_tokens = encode_tac_byte_tokens(
        prompt,
        vocab_size=int(config.vocab_size),
        append_eos=False,
    )
    if not prompt_tokens:
        prompt_tokens = [TAC_EOS_TOKEN_ID]
    context_window = max(1, int(config.max_seq_len))
    all_tokens = list(prompt_tokens)
    generated_tokens: list[int] = []
    rng = None
    if seed is not None:
        rng = torch.Generator(device="cpu").manual_seed(int(seed))

    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
            window = all_tokens[-context_window:]
            input_ids = torch.tensor([window], dtype=torch.long, device=resolved_device)
            with _autocast_context(resolved_device, precision):
                output = model(input_ids, collect_auxiliary=False)
            next_token = sample_next_token(
                output.logits[0, -1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=rng,
            )
            if next_token == TAC_EOS_TOKEN_ID:
                break
            generated_tokens.append(next_token)
            all_tokens.append(next_token)

    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "prompt": str(prompt),
        "completion": decode_tac_byte_tokens(generated_tokens),
        "generated_token_ids": generated_tokens,
        "generated_token_count": len(generated_tokens),
        "prompt_token_count": len(prompt_tokens),
        "truncated_prompt_token_count": max(0, len(prompt_tokens) - context_window),
        "context_window": context_window,
        "temperature": float(temperature),
        "top_k": None if top_k is None else int(top_k),
        "top_p": float(top_p),
        "tokenizer": "tac_byte",
        "wall_clock_seconds": elapsed,
        "tokens_per_second": len(generated_tokens) / elapsed,
    }


def stream_tac_completion(
    model: torch.nn.Module,
    prompt: str,
    **kwargs: Any,
) -> Iterator[str]:
    """Yield cumulative text for simple GUI streaming."""

    max_new_tokens = int(kwargs.pop("max_new_tokens", 128))
    partial_tokens: list[int] = []
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model must expose a TACConfig-compatible config")
    device = torch.device(kwargs.pop("device", "cpu"))
    precision = str(kwargs.pop("precision", "fp32"))
    temperature = float(kwargs.pop("temperature", 0.7))
    top_k = kwargs.pop("top_k", 50)
    top_p = float(kwargs.pop("top_p", 0.9))
    seed = kwargs.pop("seed", None)
    if kwargs:
        raise ValueError(f"unsupported generation kwargs: {sorted(kwargs)}")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    all_tokens = encode_tac_byte_tokens(
        prompt,
        vocab_size=int(config.vocab_size),
        append_eos=False,
    ) or [TAC_EOS_TOKEN_ID]
    context_window = max(1, int(config.max_seq_len))
    rng = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor(
                [all_tokens[-context_window:]],
                dtype=torch.long,
                device=device,
            )
            with _autocast_context(device, precision):
                output = model(input_ids, collect_auxiliary=False)
            token_id = sample_next_token(
                output.logits[0, -1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=rng,
            )
            if token_id == TAC_EOS_TOKEN_ID:
                break
            partial_tokens.append(token_id)
            all_tokens.append(token_id)
            yield decode_tac_byte_tokens(partial_tokens)


def load_tac_checkpoint_for_generation(
    checkpoint_path: str | Path,
    *,
    model_type: str = "auto",
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if model_type not in {"auto", "tac", "vanilla"}:
        raise ValueError("model_type must be auto, tac, or vanilla")
    resolved_device = torch.device(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=resolved_device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint is missing model_state_dict: {checkpoint_path}")
    config = _coerce_config(checkpoint.get("config"))
    state_dict = checkpoint["model_state_dict"]
    resolved_type = _resolve_model_type(model_type, state_dict)
    model: torch.nn.Module
    if resolved_type == "tac":
        model = TACTransformerLM(config)
    else:
        model = VanillaTransformerLM(config)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    return model, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": _optional_int(checkpoint.get("step")),
        "best_eval_loss": _optional_float(checkpoint.get("best_eval_loss")),
        "model_type": resolved_type,
        "config": _config_to_dict(config),
        "parameter_counts": checkpoint.get("parameter_counts"),
        "tokenizer": "tac_byte",
    }


def _resolve_model_type(requested: str, state_dict: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    if any(".identity_field." in str(key) for key in state_dict):
        return "tac"
    return "vanilla"


def _coerce_config(raw_config: Any) -> TACConfig:
    if isinstance(raw_config, TACConfig):
        return raw_config
    if isinstance(raw_config, dict):
        valid = {field.name for field in TACConfig.__dataclass_fields__.values()}
        return TACConfig(**{key: value for key, value in raw_config.items() if key in valid})
    raise ValueError("checkpoint config must be a TACConfig or dict")


def _config_to_dict(config: TACConfig) -> dict[str, Any]:
    return {
        field_name: getattr(config, field_name)
        for field_name in TACConfig.__dataclass_fields__
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()
