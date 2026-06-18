from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_v02_lm import build_model, select_device
from tac_transformer import TACTransformerLM
from tac_transformer.training import count_parameters


EOS_TOKEN_ID = 3


@dataclass(frozen=True)
class ProbeCase:
    family: str
    name: str
    prefix: str
    prompt: str
    completion: str


def byte_tokens(text: str) -> list[int]:
    return [byte + 4 for byte in text.encode("utf-8", errors="replace")]


def completion_tensors(
    *,
    prompt: str,
    completion: str,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = byte_tokens(prompt)
    completion_ids = byte_tokens(completion)
    ids = prompt_ids + completion_ids
    if len(ids) > max_seq_len:
        overflow = len(ids) - max_seq_len
        prompt_ids = prompt_ids[overflow:]
        ids = prompt_ids + completion_ids
    labels = [-100] * len(ids)
    for offset, token_id in enumerate(completion_ids):
        pos = len(prompt_ids) - 1 + offset
        if 0 <= pos < len(labels):
            labels[pos] = token_id
    if len(ids) < 2:
        ids.append(EOS_TOKEN_ID)
        labels.append(-100)
    return (
        torch.tensor([ids], dtype=torch.long, device=device),
        torch.tensor([labels], dtype=torch.long, device=device),
    )


def prefix_chunks(prefix: str, *, max_seq_len: int, device: torch.device) -> list[torch.Tensor]:
    tokens = byte_tokens(prefix)
    if not tokens:
        return []
    chunks = []
    for start in range(0, len(tokens), max_seq_len):
        chunk = tokens[start : start + max_seq_len]
        if len(chunk) < 2:
            chunk = chunk + [EOS_TOKEN_ID]
        chunks.append(torch.tensor([chunk], dtype=torch.long, device=device))
    return chunks


def fit_full_context(
    *,
    prefix: str,
    prompt: str,
    completion: str,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_ids = byte_tokens(prefix)
    prompt_ids = byte_tokens(prompt)
    completion_ids = byte_tokens(completion)
    available_prefix = max_seq_len - len(prompt_ids) - len(completion_ids)
    if available_prefix < 0:
        return completion_tensors(
            prompt=prompt,
            completion=completion,
            max_seq_len=max_seq_len,
            device=device,
        )
    prefix_ids = prefix_ids[-available_prefix:] if available_prefix else []
    ids = prefix_ids + prompt_ids + completion_ids
    labels = [-100] * len(ids)
    completion_start = len(prefix_ids) + len(prompt_ids)
    for offset, token_id in enumerate(completion_ids):
        pos = completion_start - 1 + offset
        if 0 <= pos < len(labels):
            labels[pos] = token_id
    return (
        torch.tensor([ids], dtype=torch.long, device=device),
        torch.tensor([labels], dtype=torch.long, device=device),
    )


@torch.no_grad()
def masked_loss(model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, identity_states=None) -> float:
    output = model(
        input_ids,
        labels=labels,
        identity_states=identity_states,
        collect_auxiliary=True,
        collect_metrics=True,
    )
    loss = F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return float(loss.detach().cpu())


@torch.no_grad()
def tac_carried_states(model: TACTransformerLM, case: ProbeCase, device: torch.device):
    states = None
    for chunk in prefix_chunks(case.prefix, max_seq_len=model.config.max_seq_len, device=device):
        output = model(
            chunk,
            identity_states=states,
            collect_auxiliary=False,
            collect_metrics=False,
        )
        states = output.identity_states
    return states


def load_checkpoint_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model, _, _, _ = build_model(model_name)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def zero_tac_program_bottleneck(model: TACTransformerLM) -> None:
    with torch.no_grad():
        for block in model.blocks:
            field = block.identity_field
            if field.program_expert_down is not None:
                field.program_expert_down.zero_()
            if field.program_expert_up is not None:
                field.program_expert_up.zero_()
            if field.program_expert_weight is not None:
                field.program_expert_weight.zero_()
            if field.program_expert_bias is not None:
                field.program_expert_bias.zero_()


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def evaluate_transformer(
    model: torch.nn.Module,
    cases: list[ProbeCase],
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        full_ids, full_labels = fit_full_context(
            prefix=case.prefix,
            prompt=case.prompt,
            completion=case.completion,
            max_seq_len=model.config.max_seq_len,
            device=device,
        )
        reset_ids, reset_labels = completion_tensors(
            prompt=case.prompt,
            completion=case.completion,
            max_seq_len=model.config.max_seq_len,
            device=device,
        )
        rows.append(
            {
                "family": case.family,
                "name": case.name,
                "full_context_loss": masked_loss(model, full_ids, full_labels),
                "reset_loss": masked_loss(model, reset_ids, reset_labels),
            }
        )
    return summarize_rows(rows, family_metric_keys=("full_context_loss", "reset_loss"))


def evaluate_tac(
    model: TACTransformerLM,
    knockout_model: TACTransformerLM,
    cases: list[ProbeCase],
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        query_ids, query_labels = completion_tensors(
            prompt=case.prompt,
            completion=case.completion,
            max_seq_len=model.config.max_seq_len,
            device=device,
        )
        states = tac_carried_states(model, case, device)
        knockout_states = tac_carried_states(knockout_model, case, device)
        rows.append(
            {
                "family": case.family,
                "name": case.name,
                "carry_loss": masked_loss(model, query_ids, query_labels, identity_states=states),
                "reset_loss": masked_loss(model, query_ids, query_labels, identity_states=None),
                "knockout_loss": masked_loss(
                    knockout_model,
                    query_ids,
                    query_labels,
                    identity_states=knockout_states,
                ),
            }
        )
    return summarize_rows(rows, family_metric_keys=("carry_loss", "reset_loss", "knockout_loss"))


def summarize_rows(rows: list[dict[str, Any]], *, family_metric_keys: tuple[str, ...]) -> dict[str, Any]:
    families = sorted({row["family"] for row in rows})
    by_family = {}
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        by_family[family] = {
            key: mean([float(row[key]) for row in family_rows])
            for key in family_metric_keys
        }
        by_family[family]["cases"] = len(family_rows)
    overall = {
        key: mean([float(row[key]) for row in rows])
        for key in family_metric_keys
    }
    return {"overall": overall, "by_family": by_family, "rows": rows}


def compare_results(transformer: dict[str, Any], tac: dict[str, Any]) -> dict[str, Any]:
    by_family = {}
    tac_wins = 0
    carry_positive = 0
    for family, tac_metrics in tac["by_family"].items():
        tr_metrics = transformer["by_family"][family]
        row = {
            "transformer_full_context_loss": tr_metrics["full_context_loss"],
            "transformer_reset_loss": tr_metrics["reset_loss"],
            "tac_carry_loss": tac_metrics["carry_loss"],
            "tac_reset_loss": tac_metrics["reset_loss"],
            "tac_knockout_loss": tac_metrics["knockout_loss"],
        }
        row["tac_vs_transformer_full_delta"] = (
            row["transformer_full_context_loss"] - row["tac_carry_loss"]
        )
        row["tac_carry_advantage"] = row["tac_reset_loss"] - row["tac_carry_loss"]
        row["transformer_context_advantage"] = (
            row["transformer_reset_loss"] - row["transformer_full_context_loss"]
        )
        row["bottleneck_knockout_delta"] = row["tac_knockout_loss"] - row["tac_carry_loss"]
        if row["tac_vs_transformer_full_delta"] > 0:
            tac_wins += 1
        if row["tac_carry_advantage"] > 0:
            carry_positive += 1
        by_family[family] = row

    overall = {
        "transformer_full_context_loss": transformer["overall"]["full_context_loss"],
        "transformer_reset_loss": transformer["overall"]["reset_loss"],
        "tac_carry_loss": tac["overall"]["carry_loss"],
        "tac_reset_loss": tac["overall"]["reset_loss"],
        "tac_knockout_loss": tac["overall"]["knockout_loss"],
    }
    overall["tac_vs_transformer_full_delta"] = (
        overall["transformer_full_context_loss"] - overall["tac_carry_loss"]
    )
    overall["tac_carry_advantage"] = overall["tac_reset_loss"] - overall["tac_carry_loss"]
    overall["transformer_context_advantage"] = (
        overall["transformer_reset_loss"] - overall["transformer_full_context_loss"]
    )
    overall["bottleneck_knockout_delta"] = (
        overall["tac_knockout_loss"] - overall["tac_carry_loss"]
    )
    decision = {
        "families": len(by_family),
        "tac_win_families": tac_wins,
        "carry_positive_families": carry_positive,
        "carry_state_survives": overall["tac_carry_advantage"] > 0.01,
        "bottleneck_matters": overall["bottleneck_knockout_delta"] > 0.01,
        "tac_beats_transformer": tac_wins >= math.ceil(len(by_family) / 2),
    }
    if (
        decision["carry_state_survives"]
        and decision["bottleneck_matters"]
        and decision["tac_beats_transformer"]
    ):
        status = "mechanism_advantage"
    elif decision["carry_state_survives"] or decision["bottleneck_matters"]:
        status = "mixed"
    else:
        status = "not_validated"
    decision["status"] = status
    return {"overall": overall, "by_family": by_family, "decision": decision}


def build_probe_cases(cases_per_family: int) -> list[ProbeCase]:
    cases: list[ProbeCase] = []
    for i in range(cases_per_family):
        key = f"k{i:02d}"
        value = f"v{(i * 7 + 3) % 97:02d}"
        prefix = "\n".join(
            f"memory item {j}: key k{j:02d} maps to value v{(j * 7 + 3) % 97:02d}."
            for j in range(32)
        )
        cases.append(
            ProbeCase(
                family="persistent_state_carry",
                name=f"kv_{i}",
                prefix=prefix,
                prompt=f"\nquestion: key {key} maps to value ",
                completion=value + "\n",
            )
        )

    for i in range(cases_per_family):
        bug = ["off_by_one", "missing_guard", "wrong_field", "stale_cache"][i % 4]
        fix = {
            "off_by_one": "return count + 1",
            "missing_guard": "if item is None: return fallback",
            "wrong_field": "return record.expected",
            "stale_cache": "cache.invalidate(key)",
        }[bug]
        prefix = "\n".join(
            f"repair trace {j}: bug={bug}; failing test observed; patch={fix}; verify=pass."
            for j in range(18)
        )
        cases.append(
            ProbeCase(
                family="repair_trace_reuse",
                name=f"repair_{bug}_{i}",
                prefix=prefix,
                prompt=f"\nnew repair: bug={bug}; patch=",
                completion=fix + "\n",
            )
        )

    for i in range(cases_per_family):
        structure = ["parser", "router", "planner", "compressor"][i % 4]
        action = {
            "parser": "normalize tokens then validate schema",
            "router": "select family route then specialist route",
            "planner": "expand plan then verify each step",
            "compressor": "keep keys and drop filler spans",
        }[structure]
        noisy_lines = [
            f"trace {j}: structure={structure}; reusable_action={action}; filler={'x' * 70}"
            for j in range(44)
        ]
        cases.append(
            ProbeCase(
                family="compression_structure_reuse",
                name=f"structure_{structure}_{i}",
                prefix="\n".join(noisy_lines),
                prompt=f"\ncompressed structure {structure}: reusable_action=",
                completion=action + "\n",
            )
        )

    for i in range(cases_per_family):
        key = f"noisy-{i:02d}"
        value = f"target-{(i * 11 + 5) % 89:02d}"
        noise = " ".join(f"distractor_{j}=z{(j * 13) % 101:02d}" for j in range(80))
        prefix = (
            f"anchor {key} has retrieval value {value}.\n"
            f"noise block: {noise}\n"
            f"repeat anchor {key} -> {value}."
        )
        cases.append(
            ProbeCase(
                family="noisy_key_retrieval",
                name=f"noisy_{i}",
                prefix=prefix,
                prompt=f"\nretrieve anchor {key}: ",
                completion=value + "\n",
            )
        )
    return cases


def run_retests(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    cases = build_probe_cases(args.cases_per_family)

    transformer = load_checkpoint_model(args.transformer_model, args.transformer_checkpoint, device)
    transformer_eval = evaluate_transformer(transformer, cases, device)
    del transformer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tac = load_checkpoint_model(args.tac_model, args.tac_checkpoint, device)
    tac_knockout = load_checkpoint_model(args.tac_model, args.tac_checkpoint, device)
    if not isinstance(tac, TACTransformerLM) or not isinstance(tac_knockout, TACTransformerLM):
        raise TypeError(f"{args.tac_model} did not build a TACTransformerLM")
    zero_tac_program_bottleneck(tac_knockout)
    tac_eval = evaluate_tac(tac, tac_knockout, cases, device)
    comparison = compare_results(transformer_eval, tac_eval)

    result = {
        "schema": "tac_v02_checkpoint_mechanism_retests.v1",
        "checkpoint_source": {
            "transformer_model": args.transformer_model,
            "transformer_checkpoint": str(args.transformer_checkpoint),
            "tac_model": args.tac_model,
            "tac_checkpoint": str(args.tac_checkpoint),
        },
        "device": str(device),
        "cases_per_family": args.cases_per_family,
        "case_count": len(cases),
        "parameter_counts": {
            "tac": count_parameters(tac),
            "tac_knockout": count_parameters(tac_knockout),
        },
        "transformer": transformer_eval,
        "tac": tac_eval,
        "comparison": comparison,
        "boundary": (
            "Post-training probe suite over deterministic text mechanisms. "
            "Positive loss deltas show checkpoint sensitivity, not product-grade agent performance."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "tac280_checkpoint_mechanism_retests.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-model", default="transformer_50m")
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--tac-model", default="tac_50m")
    parser.add_argument("--tac-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=280)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = run_retests(parse_args(argv))
    print(json.dumps({"decision": result["comparison"]["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
