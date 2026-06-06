from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    ContentWritePolicy,
    TACConfig,
    TACTransformerLM,
    VanillaTransformerLM,
    best_tac_config,
)
from tac_transformer.training import count_parameters, parameter_matched_baseline_config


VARIANTS: dict[str, dict[str, object]] = {
    "vanilla_matched": {"kind": "vanilla"},
    "current_best": {"kind": "tac"},
    "base_program_memory": {
        "kind": "tac",
        "memory_read_type": "program_memory",
        "routing_type": "base",
        "routing_top_k": 1,
    },
    "content_addressed_k1": {
        "kind": "tac",
        "memory_read_type": "content_addressed",
        "routing_type": "base",
        "routing_top_k": 1,
    },
    "content_addressed_k2": {
        "kind": "tac",
        "memory_read_type": "content_addressed",
        "routing_type": "sparse_ensemble",
        "routing_top_k": 2,
    },
    "current_best_local": {
        "kind": "tac",
        "attention_window_size": 128,
    },
    "tac_no_memory": {
        "kind": "tac",
        "memory_read_type": "none",
        "memory_adapter_type": "none",
        "identity_attention_type": "none",
        "content_read_steps": 1,
    },
    "tac_no_memory_local": {
        "kind": "tac",
        "memory_read_type": "none",
        "memory_adapter_type": "none",
        "identity_attention_type": "none",
        "content_read_steps": 1,
        "attention_window_size": 128,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile TAC inference prefill, carried-query, and decode overhead."
    )
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=16)
    parser.add_argument("--energy-budget", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--attention-window-size", type=int, default=128)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--rope-scale", type=float, default=1.0)
    parser.add_argument("--rope-scaling-type", choices=["none", "linear", "yarn"], default="none")
    parser.add_argument("--original-context-length", type=int, default=None)
    parser.add_argument("--target-context-length", type=int, default=None)
    parser.add_argument("--content-store-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--content-read-steps", type=int, default=1)
    parser.add_argument(
        "--content-read-gate-type",
        choices=["learned", "confidence", "synthesis"],
        default="learned",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=["vanilla_matched", "base_program_memory", "content_addressed_k1", "content_addressed_k2"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument(
        "--decode-policies",
        default=ContentWritePolicy.QUERY_SKIP.value,
        help="Comma-separated TAC decode write policies to profile.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/inference_profile_2026_05_30"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for seq_len in args.seq_lens:
        for variant_name in args.variants:
            decode_policies = (
                [None]
                if VARIANTS[variant_name]["kind"] == "vanilla"
                else parse_decode_policies(args.decode_policies)
            )
            store_sizes = (
                args.content_store_sizes
                if variant_name.startswith("content_addressed")
                else [args.content_store_sizes[0]]
            )
            for store_size in store_sizes:
                for decode_policy in decode_policies:
                    row = profile_variant(
                        args=args,
                        device=device,
                        variant_name=variant_name,
                        seq_len=seq_len,
                        content_store_size=store_size,
                        decode_policy=decode_policy,
                    )
                    rows.append(row)
                    print(one_line(row), flush=True)

    aggregate = aggregate_rows(rows)
    (args.output_dir / "inference_profile.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(aggregate),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2), flush=True)


def profile_variant(
    *,
    args: argparse.Namespace,
    device: torch.device,
    variant_name: str,
    seq_len: int,
    content_store_size: int,
    decode_policy: ContentWritePolicy | None,
) -> dict[str, Any]:
    torch.manual_seed(args.seed + seq_len + content_store_size)
    config, model_kind = build_config(args, variant_name, seq_len, content_store_size)
    if model_kind == "vanilla":
        model = VanillaTransformerLM(parameter_matched_baseline_config(config))
    else:
        model = TACTransformerLM(config)
    model.to(device)
    model.eval()

    context = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)
    query = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)
    decode_tokens = torch.randint(
        0,
        config.vocab_size,
        (args.decode_steps, args.batch_size, 1),
        device=device,
    )

    with torch.inference_mode():
        context_output = model(context, collect_auxiliary=False)
        states = context_output.identity_states
        prefill = time_call(
            lambda: model(context, collect_auxiliary=False),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        prefill_peak_memory = measure_peak_memory_bytes(
            lambda: model(context, collect_auxiliary=False),
            device=device,
        )
        carried_query = time_call(
            lambda: model(query, identity_states=states, collect_auxiliary=False),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        carried_query_peak_memory = measure_peak_memory_bytes(
            lambda: model(query, identity_states=states, collect_auxiliary=False),
            device=device,
        )
        decode = time_decode(
            model=model,
            tokens=decode_tokens,
            states=states,
            warmup=max(1, args.warmup),
            iters=args.iters,
            device=device,
            write_policy=decode_policy,
        )
        decode_peak_memory = measure_peak_memory_bytes(
            lambda: run_decode_once(
                model=model,
                tokens=decode_tokens,
                states=states,
                write_policy=decode_policy,
            ),
            device=device,
        )

    parameter_counts = count_parameters(model)
    return {
        "variant": variant_name,
        "profile": profile_label(variant_name, decode_policy),
        "model_kind": model_kind,
        "decode_policy": decode_policy.value if decode_policy is not None else "n/a",
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "content_store_size": content_store_size if model_kind == "tac" else 0,
        "decode_steps": args.decode_steps,
        "parameter_counts": parameter_counts,
        "prefill_seconds": prefill,
        "prefill_tokens_per_second": args.batch_size * seq_len / prefill,
        "prefill_peak_memory_bytes": prefill_peak_memory,
        "carried_query_seconds": carried_query,
        "carried_query_tokens_per_second": args.batch_size * seq_len / carried_query,
        "carried_query_peak_memory_bytes": carried_query_peak_memory,
        "decode_seconds": decode,
        "decode_tokens_per_second": args.batch_size * args.decode_steps / decode,
        "decode_peak_memory_bytes": decode_peak_memory,
    }


def build_config(
    args: argparse.Namespace,
    variant_name: str,
    seq_len: int,
    content_store_size: int,
) -> tuple[TACConfig, str]:
    variant = VARIANTS[variant_name]
    model_kind = str(variant["kind"])
    overrides = {key: value for key, value in variant.items() if key != "kind"}
    if "attention_window_size" in overrides:
        overrides["attention_window_size"] = min(int(overrides["attention_window_size"]), seq_len)
    overrides["content_store_size"] = content_store_size
    overrides["content_read_steps"] = args.content_read_steps
    overrides["content_read_gate_type"] = args.content_read_gate_type
    if variant_name.endswith("_local") and "attention_window_size" not in overrides:
        overrides["attention_window_size"] = min(args.attention_window_size, seq_len)
    config = best_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=seq_len,
        beta=args.beta,
        energy_budget=args.energy_budget,
        rope_base=args.rope_base,
        rope_scale=args.rope_scale,
        rope_scaling_type=args.rope_scaling_type,
        original_context_length=args.original_context_length,
        target_context_length=args.target_context_length,
        **overrides,
    )
    return config, model_kind


def time_call(
    fn: Any,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - start) / max(iters, 1)


def time_decode(
    *,
    model: torch.nn.Module,
    tokens: torch.Tensor,
    states: Any,
    warmup: int,
    iters: int,
    device: torch.device,
    write_policy: ContentWritePolicy | None,
) -> float:
    token_count = tokens.shape[0]
    for _ in range(warmup):
        run_decode_once(model=model, tokens=tokens, states=states, write_policy=write_policy)
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        run_decode_once(model=model, tokens=tokens, states=states, write_policy=write_policy)
    sync(device)
    return (time.perf_counter() - start) / max(iters, 1)


def run_decode_once(
    *,
    model: torch.nn.Module,
    tokens: torch.Tensor,
    states: Any,
    write_policy: ContentWritePolicy | None,
) -> Any:
    current_states = states
    for index in range(tokens.shape[0]):
        output = model(
            tokens[index],
            identity_states=current_states,
            collect_auxiliary=False,
            write_policy=write_policy,
        )
        current_states = output.identity_states
    return current_states


def parse_decode_policies(raw: str) -> list[ContentWritePolicy]:
    policies = []
    for name in raw.split(","):
        cleaned = name.strip()
        if cleaned:
            policies.append(ContentWritePolicy(cleaned))
    if not policies:
        raise ValueError("--decode-policies must include at least one TAC write policy")
    return policies


def measure_peak_memory_bytes(fn: Any, *, device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        fn()
        sync(device)
        return int(torch.cuda.max_memory_allocated(device))
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
        return int(peak)
    finally:
        tracemalloc.stop()


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seq: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_seq.setdefault(str(row["seq_len"]), []).append(row)

    ratios: dict[str, list[dict[str, float]]] = {}
    for seq_len, seq_rows in by_seq.items():
        vanilla = next(row for row in seq_rows if row["variant"] == "vanilla_matched")
        for row in seq_rows:
            base = next(
                (
                    candidate
                    for candidate in seq_rows
                    if candidate["variant"] == "base_program_memory"
                    and candidate["decode_policy"] == row["decode_policy"]
                ),
                None,
            )
            row["prefill_vs_vanilla"] = row["prefill_tokens_per_second"] / max(
                vanilla["prefill_tokens_per_second"],
                1e-9,
            )
            row["carried_query_vs_vanilla"] = row["carried_query_tokens_per_second"] / max(
                vanilla["carried_query_tokens_per_second"],
                1e-9,
            )
            row["decode_vs_vanilla"] = row["decode_tokens_per_second"] / max(
                vanilla["decode_tokens_per_second"],
                1e-9,
            )
            if base is None:
                row["carried_query_vs_base"] = None
                row["decode_vs_base"] = None
            else:
                row["carried_query_vs_base"] = row["carried_query_tokens_per_second"] / max(
                    base["carried_query_tokens_per_second"],
                    1e-9,
                )
                row["decode_vs_base"] = row["decode_tokens_per_second"] / max(
                    base["decode_tokens_per_second"],
                    1e-9,
                )
        ratios[seq_len] = [
            {
                "profile": row["profile"],
                "variant": row["variant"],
                "decode_policy": row["decode_policy"],
                "content_store_size": row["content_store_size"],
                "carried_query_vs_base": row["carried_query_vs_base"],
                "decode_vs_base": row["decode_vs_base"],
            }
            for row in seq_rows
        ]

    return {
        "rows": rows,
        "ratios_by_seq_len": ratios,
        "summary": summarize(rows),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for profile in sorted({str(row["profile"]) for row in rows}):
        selected = [row for row in rows if row["profile"] == profile]
        summary[profile] = {
            "mean_prefill_vs_vanilla": mean(row["prefill_vs_vanilla"] for row in selected),
            "mean_carried_query_vs_vanilla": mean(row["carried_query_vs_vanilla"] for row in selected),
            "mean_decode_vs_vanilla": mean(row["decode_vs_vanilla"] for row in selected),
            "mean_prefill_peak_memory_bytes": mean(
                row["prefill_peak_memory_bytes"] for row in selected
            ),
            "mean_decode_peak_memory_bytes": mean(
                row["decode_peak_memory_bytes"] for row in selected
            ),
            "mean_carried_query_vs_base": _mean_optional(
                row["carried_query_vs_base"] for row in selected
            ),
            "mean_decode_vs_base": _mean_optional(row["decode_vs_base"] for row in selected),
        }
    return summary


def profile_label(variant: str, decode_policy: ContentWritePolicy | None) -> str:
    if decode_policy is None:
        return variant
    return f"{variant}:{decode_policy.value}"


def _mean_optional(values: Any) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return mean(present)


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# TAC Inference Profile",
        "",
        "Profiles inference-only prefill, carried-query, and one-token decode with carried identity state.",
        "",
        "## Summary",
        "",
        "| Profile | Mean carried-query vs vanilla | Mean decode vs vanilla | Mean prefill peak | Mean decode peak | Mean carried-query vs BASE | Mean decode vs BASE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile, values in result["summary"].items():
        lines.append(
            f"| `{profile}` | {values['mean_carried_query_vs_vanilla']:.4f} | "
            f"{values['mean_decode_vs_vanilla']:.4f} | "
            f"{_format_bytes(values['mean_prefill_peak_memory_bytes'])} | "
            f"{_format_bytes(values['mean_decode_peak_memory_bytes'])} | "
            f"{_format_optional_ratio(values['mean_carried_query_vs_base'])} | "
            f"{_format_optional_ratio(values['mean_decode_vs_base'])} |"
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Seq len | Variant | Decode policy | Store | Prefill tok/s | Query tok/s | Decode tok/s | Prefill peak | Decode peak | Query vs BASE | Decode vs BASE |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["rows"]:
        lines.append(
            f"| {row['seq_len']} | `{row['variant']}` | `{row['decode_policy']}` | "
            f"{row['content_store_size']} | "
            f"{row['prefill_tokens_per_second']:.2f} | "
            f"{row['carried_query_tokens_per_second']:.2f} | "
            f"{row['decode_tokens_per_second']:.2f} | "
            f"{_format_bytes(row['prefill_peak_memory_bytes'])} | "
            f"{_format_bytes(row['decode_peak_memory_bytes'])} | "
            f"{_format_optional_ratio(row['carried_query_vs_base'])} | "
            f"{_format_optional_ratio(row['decode_vs_base'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _format_optional_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_bytes(value: float | int) -> str:
    value = float(value)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"


def one_line(row: dict[str, Any]) -> str:
    return (
        f"seq={row['seq_len']} variant={row['variant']} store={row['content_store_size']} "
        f"decode_policy={row['decode_policy']} "
        f"prefill={row['prefill_tokens_per_second']:.2f}tok/s "
        f"query={row['carried_query_tokens_per_second']:.2f}tok/s "
        f"decode={row['decode_tokens_per_second']:.2f}tok/s "
        f"prefill_peak={_format_bytes(row['prefill_peak_memory_bytes'])} "
        f"decode_peak={_format_bytes(row['decode_peak_memory_bytes'])}"
    )


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
