from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

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


def _rerank_candidate_token_ids(
    logits: Tensor,
    *,
    candidate_count: int,
    top_k: int | None,
    top_p: float,
) -> list[int]:
    if candidate_count < 1:
        return []
    filtered = logits.float().clone()
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
        filtered[remove_indices] = float("-inf")
    finite_indices = torch.nonzero(torch.isfinite(filtered), as_tuple=False).flatten()
    if finite_indices.numel() == 0:
        return [int(torch.argmax(logits).detach().cpu())]
    pool_size = min(int(candidate_count), int(finite_indices.numel()))
    _, order = torch.topk(filtered[finite_indices], pool_size)
    return [int(finite_indices[index].detach().cpu()) for index in order]


def _last_data_energy(output: Any) -> float | None:
    aux = getattr(output, "aux", None)
    data_energy = getattr(aux, "data_energy", None)
    if data_energy is None:
        return None
    if not isinstance(data_energy, Tensor) or data_energy.numel() == 0:
        return None
    return float(data_energy.reshape(-1)[-1].detach().cpu())


def _energy_rerank_next_token(
    model: torch.nn.Module,
    *,
    logits: Tensor,
    window: list[int],
    context_window: int,
    device: torch.device,
    precision: str,
    top_k: int | None,
    top_p: float,
    candidate_count: int,
    data_energy_weight: float,
    verifier_threshold: float | None,
) -> tuple[int, float | None, bool, bool, list[dict[str, float | int]]]:
    candidates = _rerank_candidate_token_ids(
        logits,
        candidate_count=candidate_count,
        top_k=top_k,
        top_p=top_p,
    )
    if not candidates:
        return int(torch.argmax(logits).detach().cpu()), None, False, False, []

    best_token = candidates[0]
    best_energy: float | None = None
    best_score: float | None = None
    saw_data_energy = False
    candidate_records: list[dict[str, float | int]] = []
    for token_id in candidates:
        candidate_window = (window + [int(token_id)])[-context_window:]
        input_ids = torch.tensor([candidate_window], dtype=torch.long, device=device)
        with _autocast_context(device, precision):
            output = model(input_ids, collect_auxiliary=True)
        energy = _last_data_energy(output)
        if energy is None:
            continue
        saw_data_energy = True
        score = float(logits[int(token_id)].detach().cpu()) - float(data_energy_weight) * energy
        candidate_records.append(
            {
                "token_id": int(token_id),
                "lm_logit": float(logits[int(token_id)].detach().cpu()),
                "data_energy": energy,
                "combined_score": score,
            }
        )
        if best_score is None or score > best_score:
            best_score = score
            best_token = int(token_id)
            best_energy = energy

    if not saw_data_energy:
        return candidates[0], None, False, False, []
    verifier_required = (
        verifier_threshold is not None
        and best_energy is not None
        and best_energy >= float(verifier_threshold)
    )
    return best_token, best_energy, verifier_required, True, candidate_records


def _apply_data_energy_verifier(
    verifier: Callable[[dict[str, Any]], Any],
    *,
    prompt: str,
    window: list[int],
    selected_token: int,
    selected_data_energy: float | None,
    candidates: list[dict[str, float | int]],
) -> tuple[int, dict[str, Any]]:
    payload = {
        "prompt": prompt,
        "window_token_ids": list(window),
        "selected_token_id": int(selected_token),
        "selected_data_energy": selected_data_energy,
        "candidates": candidates,
    }
    result = verifier(payload)
    replacement_token = _coerce_verifier_token_id(result)
    allowed_tokens = {int(candidate["token_id"]) for candidate in candidates}
    accepted = replacement_token is not None and replacement_token in allowed_tokens
    action = {
        "called": True,
        "accepted": accepted,
        "selected_token_id": int(selected_token),
        "replacement_token_id": replacement_token if accepted else None,
        "reason": _coerce_verifier_reason(result),
    }
    if not accepted:
        return int(selected_token), action
    return int(replacement_token), action


def _coerce_verifier_token_id(result: Any) -> int | None:
    if result is None:
        return None
    if isinstance(result, int):
        return int(result)
    if isinstance(result, dict):
        for key in ("token_id", "replacement_token_id"):
            if key in result and result[key] is not None:
                return int(result[key])
    return None


def _coerce_verifier_reason(result: Any) -> str | None:
    if isinstance(result, dict) and result.get("reason") is not None:
        return str(result["reason"])
    return None


def generate_tac_completion(
    model: torch.nn.Module,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    context_window: int | None = None,
    temperature: float = 0.7,
    top_k: int | None = 50,
    top_p: float = 0.9,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
    seed: int | None = None,
    energy_rerank_top_k: int = 0,
    data_energy_weight: float = 1.0,
    data_energy_verifier_threshold: float | None = None,
    data_energy_verifier: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Generate a byte-level completion from a TAC or vanilla checkpoint model."""

    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if int(energy_rerank_top_k) < 0:
        raise ValueError("energy_rerank_top_k must be non-negative")
    if float(data_energy_weight) < 0.0:
        raise ValueError("data_energy_weight must be non-negative")
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
    resolved_context_window = _resolve_context_window(config, context_window)
    all_tokens = list(prompt_tokens)
    generated_tokens: list[int] = []
    rng = None
    if seed is not None:
        rng = torch.Generator(device="cpu").manual_seed(int(seed))
    use_energy_rerank = int(energy_rerank_top_k) > 0
    data_energy_trace: list[float | None] = []
    verifier_required_trace: list[bool] = []
    verifier_actions: list[dict[str, Any]] = []
    reranked_token_count = 0

    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
            window = all_tokens[-resolved_context_window:]
            input_ids = torch.tensor([window], dtype=torch.long, device=resolved_device)
            with _autocast_context(resolved_device, precision):
                output = model(input_ids, collect_auxiliary=False)
            step_logits = output.logits[0, -1]
            if use_energy_rerank:
                (
                    next_token,
                    data_energy,
                    verifier_required,
                    reranked,
                    candidates,
                ) = _energy_rerank_next_token(
                    model,
                    logits=step_logits,
                    window=window,
                    context_window=resolved_context_window,
                    device=resolved_device,
                    precision=precision,
                    top_k=top_k,
                    top_p=top_p,
                    candidate_count=int(energy_rerank_top_k),
                    data_energy_weight=float(data_energy_weight),
                    verifier_threshold=data_energy_verifier_threshold,
                )
                data_energy_trace.append(data_energy)
                verifier_required_trace.append(verifier_required)
                reranked_token_count += int(reranked)
                if verifier_required and data_energy_verifier is not None:
                    next_token, action = _apply_data_energy_verifier(
                        data_energy_verifier,
                        prompt=str(prompt),
                        window=window,
                        selected_token=next_token,
                        selected_data_energy=data_energy,
                        candidates=candidates,
                    )
                    verifier_actions.append(action)
                if not reranked:
                    next_token = sample_next_token(
                        step_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=rng,
                    )
            else:
                next_token = sample_next_token(
                    step_logits,
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
        "truncated_prompt_token_count": max(0, len(prompt_tokens) - resolved_context_window),
        "context_window": resolved_context_window,
        "checkpoint_max_seq_len": int(config.max_seq_len),
        "temperature": float(temperature),
        "top_k": None if top_k is None else int(top_k),
        "top_p": float(top_p),
        "energy_rerank_top_k": int(energy_rerank_top_k),
        "data_energy_weight": float(data_energy_weight),
        "data_energy_trace": data_energy_trace,
        "data_energy_reranked_token_count": reranked_token_count,
        "data_energy_verifier_required": verifier_required_trace,
        "data_energy_verifier_required_count": sum(verifier_required_trace),
        "data_energy_verifier_called_count": len(verifier_actions),
        "data_energy_verifier_actions": verifier_actions,
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
    context_window = kwargs.pop("context_window", None)
    temperature = float(kwargs.pop("temperature", 0.7))
    top_k = kwargs.pop("top_k", 50)
    top_p = float(kwargs.pop("top_p", 0.9))
    seed = kwargs.pop("seed", None)
    energy_rerank_top_k = int(kwargs.pop("energy_rerank_top_k", 0))
    data_energy_weight = float(kwargs.pop("data_energy_weight", 1.0))
    data_energy_verifier_threshold = kwargs.pop(
        "data_energy_verifier_threshold",
        None,
    )
    data_energy_verifier = kwargs.pop("data_energy_verifier", None)
    if kwargs:
        raise ValueError(f"unsupported generation kwargs: {sorted(kwargs)}")
    if energy_rerank_top_k < 0:
        raise ValueError("energy_rerank_top_k must be non-negative")
    if data_energy_weight < 0.0:
        raise ValueError("data_energy_weight must be non-negative")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    all_tokens = encode_tac_byte_tokens(
        prompt,
        vocab_size=int(config.vocab_size),
        append_eos=False,
    ) or [TAC_EOS_TOKEN_ID]
    resolved_context_window = _resolve_context_window(config, context_window)
    rng = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor(
                [all_tokens[-resolved_context_window:]],
                dtype=torch.long,
                device=device,
            )
            with _autocast_context(device, precision):
                output = model(input_ids, collect_auxiliary=False)
            step_logits = output.logits[0, -1]
            if energy_rerank_top_k > 0:
                (
                    token_id,
                    data_energy,
                    verifier_required,
                    reranked,
                    candidates,
                ) = _energy_rerank_next_token(
                    model,
                    logits=step_logits,
                    window=all_tokens[-resolved_context_window:],
                    context_window=resolved_context_window,
                    device=device,
                    precision=precision,
                    top_k=top_k,
                    top_p=top_p,
                    candidate_count=energy_rerank_top_k,
                    data_energy_weight=data_energy_weight,
                    verifier_threshold=data_energy_verifier_threshold,
                )
                if verifier_required and data_energy_verifier is not None:
                    token_id, _ = _apply_data_energy_verifier(
                        data_energy_verifier,
                        prompt=str(prompt),
                        window=all_tokens[-resolved_context_window:],
                        selected_token=token_id,
                        selected_data_energy=data_energy,
                        candidates=candidates,
                    )
                if not reranked:
                    token_id = sample_next_token(
                        step_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=rng,
                    )
            else:
                token_id = sample_next_token(
                    step_logits,
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


def _resolve_context_window(config: Any, context_window: int | None) -> int:
    max_seq_len = max(1, int(config.max_seq_len))
    if context_window is None:
        return max_seq_len
    resolved = int(context_window)
    if resolved < 1:
        raise ValueError("context_window must be at least 1")
    if resolved > max_seq_len:
        raise ValueError(
            f"context_window must be <= checkpoint max_seq_len ({max_seq_len})"
        )
    return resolved


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
