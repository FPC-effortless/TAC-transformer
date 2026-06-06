from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM


def inspect_checkpoint_memory(
    checkpoint: str | Path,
    *,
    prompt: str,
    max_slots: int = 8,
    top_k: int = 5,
    max_programs: int = 8,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device = _select_device(device)
    model, config, checkpoint_data = _load_checkpoint_model(Path(checkpoint), device)
    token_ids = _encode_text(prompt, config.vocab_size)
    token_ids = token_ids[: config.max_seq_len]
    if not token_ids:
        token_ids = [3]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_ids)

    hidden = output.hidden_states
    if hidden is None:
        raise RuntimeError("TAC output did not include hidden states")
    hidden = hidden[0].detach()
    prompt_tokens = [
        {
            "index": index,
            "token_id": int(token_id),
            "text": _decode_token(int(token_id)),
        }
        for index, token_id in enumerate(token_ids)
    ]

    layers = []
    for layer_index, state in enumerate(output.identity_states):
        stability = state.stability[0].detach()
        program_count = min(max_programs, stability.numel())
        program_scores, program_indices = torch.topk(stability, k=program_count)
        layer = {
            "layer": layer_index,
            "programs": [
                {
                    "program": int(program_indices[i]),
                    "stability": float(program_scores[i]),
                    "memory_norm": float(state.program_memory[0, program_indices[i]].norm()),
                }
                for i in range(program_count)
            ],
            "content_slots": _inspect_content_slots(
                model,
                state,
                hidden,
                prompt_tokens,
                max_slots=max_slots,
                top_k=top_k,
            ),
        }
        layers.append(layer)

    return {
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_step": int(checkpoint_data.get("step", 0)),
        "best_eval_loss": _optional_float(checkpoint_data.get("best_eval_loss")),
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "config": {
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "n_layers": config.n_layers,
            "n_programs": config.n_programs,
            "memory_read_type": config.memory_read_type,
            "content_store_size": config.content_store_size,
            "content_read_steps": config.content_read_steps,
            "content_read_gate_type": config.content_read_gate_type,
            "identity_attention_type": config.identity_attention_type,
            "memory_adapter_type": config.memory_adapter_type,
        },
        "metrics": _tensor_dict_to_floats(output.aux.metrics),
        "layers": layers,
    }


def _inspect_content_slots(
    model: TACTransformerLM,
    state,
    hidden: torch.Tensor,
    prompt_tokens: list[dict[str, Any]],
    *,
    max_slots: int,
    top_k: int,
) -> list[dict[str, Any]]:
    if (
        state.content_cues is None
        or state.content_values is None
        or state.content_mask is None
    ):
        return []
    mask = state.content_mask[0].detach().bool()
    active_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    slots = []
    for slot_index in active_indices[:max_slots]:
        cue = state.content_cues[0, slot_index].detach()
        value = state.content_values[0, slot_index].detach()
        cue_top = _top_tokens(model, cue, top_k=top_k)
        value_top = _top_tokens(model, value, top_k=top_k)
        nearest_cue = _nearest_prompt_token(cue, hidden, prompt_tokens)
        nearest_value = _nearest_prompt_token(value, hidden, prompt_tokens)
        slots.append(
            {
                "slot": int(slot_index),
                "cue_norm": float(cue.norm()),
                "value_norm": float(value.norm()),
                "nearest_cue_prompt_token": nearest_cue,
                "nearest_value_prompt_token": nearest_value,
                "cue_top_tokens": cue_top,
                "value_top_tokens": value_top,
                "decoded_cue": "".join(item["text"] for item in cue_top[:1]),
                "decoded_value": "".join(item["text"] for item in value_top[:1]),
            }
        )
    return slots


def _top_tokens(
    model: TACTransformerLM,
    vector: torch.Tensor,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    logits = model.lm_head(vector)
    probs = torch.softmax(logits, dim=-1)
    count = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k=count)
    return [
        {
            "token_id": int(indices[i]),
            "text": _decode_token(int(indices[i])),
            "probability": float(values[i].detach()),
        }
        for i in range(count)
    ]


def _nearest_prompt_token(
    vector: torch.Tensor,
    hidden: torch.Tensor,
    prompt_tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    if hidden.numel() == 0:
        return {"index": None, "text": "", "cosine": 0.0}
    vector_norm = F.normalize(vector[None, :], dim=-1)
    hidden_norm = F.normalize(hidden, dim=-1)
    similarities = torch.matmul(hidden_norm, vector_norm.transpose(0, 1)).squeeze(-1)
    index = int(similarities.argmax())
    token = dict(prompt_tokens[index])
    token["cosine"] = float(similarities[index])
    return token


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TACTransformerLM, TACConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TACConfig(**checkpoint["config"])
    model = TACTransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def _encode_text(text: str, vocab_size: int) -> list[int]:
    tokens = [byte + 4 for byte in text.encode("utf-8", errors="replace")]
    return [token for token in tokens if token < vocab_size]


def _decode_token(token_id: int) -> str:
    special = {
        0: "<pad>",
        1: "<bos>",
        2: "<query>",
        3: "<eos>",
    }
    if token_id in special:
        return special[token_id]
    if 4 <= token_id < 260:
        return bytes([token_id - 4]).decode("utf-8", errors="replace")
    return f"<tok_{token_id}>"


def _tensor_dict_to_floats(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(value.detach())
        for name, value in sorted(values.items())
        if torch.is_tensor(value) and value.numel() == 1
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _select_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect TAC IdentityState memory slots from a checkpoint and prompt."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-slots", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-programs", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = inspect_checkpoint_memory(
        args.checkpoint,
        prompt=args.prompt,
        max_slots=args.max_slots,
        top_k=args.top_k,
        max_programs=args.max_programs,
        device=args.device,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
