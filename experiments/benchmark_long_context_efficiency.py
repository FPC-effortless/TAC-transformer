from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM, VanillaTransformerLM, best_tac_config
from tac_transformer.optimization import TACOptimizerConfig, build_tac_optimizer
from tac_transformer.training import (
    JsonlTextBatcher,
    TokenizedMemmapBatcher,
    benchmark_chunked_memory,
    build_tokenized_memmap_from_jsonl,
    count_parameters,
    parameter_matched_baseline_config,
)


MODEL_VARIANTS: dict[str, dict[str, object]] = {
    "vanilla_dense": {"kind": "vanilla"},
    "vanilla_local": {"kind": "vanilla", "attention_window": True},
    "tac_dense_no_memory": {
        "kind": "tac",
        "memory_read_type": "none",
        "memory_adapter_type": "none",
        "identity_attention_type": "none",
        "content_read_steps": 1,
    },
    "tac_local_no_memory": {
        "kind": "tac",
        "attention_window": True,
        "memory_read_type": "none",
        "memory_adapter_type": "none",
        "identity_attention_type": "none",
        "content_read_steps": 1,
    },
    "tac_dense_identity_first": {
        "kind": "tac",
        "memory_read_type": "none",
        "memory_adapter_type": "none",
        "identity_attention_type": "identity_first",
        "content_read_steps": 1,
    },
    "tac_local_identity_first": {
        "kind": "tac",
        "attention_window": True,
        "memory_read_type": "none",
        "memory_adapter_type": "none",
        "identity_attention_type": "identity_first",
        "content_read_steps": 1,
    },
    "tac_local_content_memory": {
        "kind": "tac",
        "attention_window": True,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local TAC long-context efficiency screens without Kaggle APIs."
    )
    parser.add_argument(
        "--prepared-jsonl",
        type=Path,
        default=Path("runs/prepared_corpus_agentic_hard/eval.prepared.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/long_context_efficiency_local_2026_06_02"),
    )
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--attention-window-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--profile-batch-size", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--batcher-batches", type=int, default=32)
    parser.add_argument("--batcher-batch-size", type=int, default=8)
    parser.add_argument("--segment-seq-lens", type=int, nargs="+", default=[64])
    parser.add_argument("--segment-steps", type=int, default=8)
    parser.add_argument("--segment-batch-size", type=int, default=8)
    parser.add_argument("--segment-eval-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--skip-segment-screen", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.prepared_jsonl.exists():
        raise FileNotFoundError(f"prepared JSONL not found: {args.prepared_jsonl}")

    tokenized_manifest = build_tokenized_memmap_from_jsonl(
        args.prepared_jsonl,
        args.output_dir / "tokenized",
        vocab_size=args.vocab_size,
    )
    batcher_results = benchmark_batchers(args, tokenized_manifest)
    profile_results = profile_model_variants(args, device)
    segment_results = [] if args.skip_segment_screen else run_segment_screen(args, device)

    aggregate = {
        "date": "2026-06-02",
        "device": str(device),
        "prepared_jsonl": str(args.prepared_jsonl),
        "tokenized_manifest": tokenized_manifest,
        "batcher_results": batcher_results,
        "model_profiles": profile_results,
        "segment_carry_screen": segment_results,
        "interpretation": interpret_results(batcher_results, profile_results, segment_results),
    }
    (args.output_dir / "long_context_efficiency_local.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(format_markdown(aggregate), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


def benchmark_batchers(args: argparse.Namespace, manifest: dict[str, object]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq_len in args.seq_lens:
        jsonl = JsonlTextBatcher(
            args.prepared_jsonl,
            seq_len=seq_len,
            vocab_size=args.vocab_size,
            seed=args.seed + seq_len,
        )
        memmap = TokenizedMemmapBatcher.from_manifest(
            Path(manifest["tokens_path"]).parent / "manifest.json",
            seq_len=seq_len,
            seed=args.seed + seq_len,
        )
        rows.append(
            time_batcher(
                "jsonl_byte_online",
                jsonl,
                seq_len=seq_len,
                batches=args.batcher_batches,
                batch_size=args.batcher_batch_size,
            )
        )
        rows.append(
            time_batcher(
                "tokenized_memmap",
                memmap,
                seq_len=seq_len,
                batches=args.batcher_batches,
                batch_size=args.batcher_batch_size,
            )
        )
    return rows


def time_batcher(
    name: str,
    batcher: Any,
    *,
    seq_len: int,
    batches: int,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    checksum = 0
    for _ in range(batches):
        inputs, labels = batcher.next_batch(batch_size)
        checksum += int(inputs[0, 0]) + int(labels[-1, -1])
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = batches * batch_size * seq_len
    return {
        "batcher": name,
        "seq_len": seq_len,
        "batches": batches,
        "batch_size": batch_size,
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "checksum": checksum,
    }


def profile_model_variants(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq_len in args.seq_lens:
        for variant_name in MODEL_VARIANTS:
            row = profile_model_variant(args, device, seq_len, variant_name)
            rows.append(row)
            print(
                f"profile seq={seq_len} variant={variant_name} "
                f"train_tps={row['train_tokens_per_second']:.2f} "
                f"prefill_tps={row['prefill_tokens_per_second']:.2f}",
                flush=True,
            )
    return rows


def profile_model_variant(
    args: argparse.Namespace,
    device: torch.device,
    seq_len: int,
    variant_name: str,
) -> dict[str, Any]:
    torch.manual_seed(args.seed + seq_len + len(variant_name))
    config, kind = build_profile_config(args, seq_len, variant_name)
    if kind == "vanilla":
        model = VanillaTransformerLM(parameter_matched_baseline_config(config))
    else:
        model = TACTransformerLM(config)
    model.to(device)
    inputs = torch.randint(0, config.vocab_size, (args.profile_batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (args.profile_batch_size, seq_len), device=device)
    query = torch.randint(0, config.vocab_size, (args.profile_batch_size, seq_len), device=device)
    decode_tokens = torch.randint(
        0,
        config.vocab_size,
        (args.decode_steps, args.profile_batch_size, 1),
        device=device,
    )

    model.eval()
    with torch.inference_mode():
        context_output = model(inputs)
        states = context_output.identity_states
        prefill_seconds = time_call(
            lambda: model(inputs),
            warmup=args.profile_warmup,
            iters=args.profile_iters,
            device=device,
        )
        carried_seconds = time_call(
            lambda: model(query, identity_states=states),
            warmup=args.profile_warmup,
            iters=args.profile_iters,
            device=device,
        )
        decode_seconds = time_decode(
            model,
            decode_tokens,
            states=states,
            warmup=max(args.profile_warmup, 1),
            iters=args.profile_iters,
            device=device,
        )

    train_seconds = time_train_step(
        model,
        inputs,
        labels,
        warmup=args.profile_warmup,
        iters=args.profile_iters,
        device=device,
    )
    train_tokens = args.profile_batch_size * seq_len
    attention_proxy = attention_entry_proxy(
        config,
        batch_size=args.profile_batch_size,
        seq_len=seq_len,
    )
    return {
        "variant": variant_name,
        "kind": kind,
        "seq_len": seq_len,
        "attention_window_size": config.attention_window_size,
        "rope_scaling_type": config.rope_scaling_type,
        "parameter_counts": count_parameters(model),
        "attention_proxy": attention_proxy,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": train_tokens / prefill_seconds,
        "carried_query_seconds": carried_seconds,
        "carried_query_tokens_per_second": train_tokens / carried_seconds,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": args.profile_batch_size * args.decode_steps / decode_seconds,
        "train_step_seconds": train_seconds,
        "train_tokens_per_second": train_tokens / train_seconds,
    }


def build_profile_config(
    args: argparse.Namespace,
    seq_len: int,
    variant_name: str,
) -> tuple[TACConfig, str]:
    variant = MODEL_VARIANTS[variant_name]
    kind = str(variant["kind"])
    overrides = {
        key: value
        for key, value in variant.items()
        if key not in {"kind", "attention_window"}
    }
    if variant.get("attention_window"):
        overrides["attention_window_size"] = min(args.attention_window_size, seq_len)
    config = best_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=seq_len,
        rope_scaling_type="linear" if seq_len > 256 else "none",
        original_context_length=256,
        target_context_length=seq_len,
        **overrides,
    )
    return config, kind


def time_call(fn: Any, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return max(time.perf_counter() - started, 1e-9) / max(iters, 1)


def time_train_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    optimizer = build_tac_optimizer(
        model,
        TACOptimizerConfig(learning_rate=1e-4),
    )
    model.train()
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, labels=labels)
        loss = output.loss if output.loss is not None else output.logits.mean()
        loss.backward()
        optimizer.step()
    sync(device)
    started = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, labels=labels)
        loss = output.loss if output.loss is not None else output.logits.mean()
        loss.backward()
        optimizer.step()
    sync(device)
    return max(time.perf_counter() - started, 1e-9) / max(iters, 1)


def time_decode(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    states: Any,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        current_states = states
        for index in range(tokens.shape[0]):
            output = model(tokens[index], identity_states=current_states)
            current_states = output.identity_states
    sync(device)
    started = time.perf_counter()
    for _ in range(iters):
        current_states = states
        for index in range(tokens.shape[0]):
            output = model(tokens[index], identity_states=current_states)
            current_states = output.identity_states
    sync(device)
    return max(time.perf_counter() - started, 1e-9) / max(iters, 1)


def attention_entry_proxy(
    config: TACConfig,
    *,
    batch_size: int,
    seq_len: int,
) -> dict[str, int | float]:
    dense = batch_size * config.n_layers * config.n_heads * seq_len * seq_len
    if config.attention_window_size is None:
        useful = dense
        materialized = dense
    else:
        useful = batch_size * config.n_layers * config.n_heads * seq_len * min(
            seq_len,
            config.attention_window_size,
        )
        materialized = useful
    return {
        "dense_entries": dense,
        "useful_local_entries": useful,
        "useful_to_dense_ratio": useful / max(dense, 1),
        "materialized_entries_current_impl": materialized,
    }


def run_segment_screen(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq_len in args.segment_seq_lens:
        for variant_name, window in [
            ("dense_identity_first", None),
            ("local_identity_first", min(args.attention_window_size, seq_len)),
        ]:
            config = best_tac_config(
                vocab_size=64,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                n_programs=args.n_programs,
                max_seq_len=seq_len,
                attention_window_size=window,
            )
            result = benchmark_chunked_memory(
                config,
                steps=args.segment_steps,
                batch_size=args.segment_batch_size,
                learning_rate=3e-4,
                eval_batches=args.segment_eval_batches,
                eval_batch_size=args.segment_batch_size,
                seed=args.seed + seq_len,
                device=device,
                match_baseline_parameters=True,
                value_loss_weight=3.0,
                memory_read_loss_weight=3.0,
                memory_adapter_weight=6.0,
                task_variant="delayed_query",
            )
            rows.append(
                {
                    "variant": variant_name,
                    "seq_len": seq_len,
                    "attention_window_size": window,
                    "decision": result["decision"],
                    "tac": result["tac"]["chunked_probe"],
                    "baseline": result["baseline"]["chunked_probe"],
                    "tac_train": result["tac"]["train"],
                    "baseline_train": result["baseline"]["train"],
                }
            )
            print(
                f"segment seq={seq_len} variant={variant_name} "
                f"carry={result['tac']['chunked_probe']['carry']['value_accuracy']:.4f}",
                flush=True,
            )
    return rows


def interpret_results(
    batcher_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    batcher_by_seq: dict[int, dict[str, float]] = {}
    for row in batcher_rows:
        batcher_by_seq.setdefault(int(row["seq_len"]), {})[str(row["batcher"])] = float(
            row["tokens_per_second"]
        )
    memmap_speedups = {
        seq_len: values.get("tokenized_memmap", 0.0)
        / max(values.get("jsonl_byte_online", 0.0), 1e-9)
        for seq_len, values in batcher_by_seq.items()
    }
    profile_by_seq: dict[int, dict[str, dict[str, Any]]] = {}
    for row in profile_rows:
        profile_by_seq.setdefault(int(row["seq_len"]), {})[str(row["variant"])] = row
    local_speed_notes = {}
    for seq_len, variants in profile_by_seq.items():
        dense = variants.get("vanilla_dense")
        local = variants.get("vanilla_local")
        if dense and local:
            local_speed_notes[seq_len] = {
                "vanilla_local_train_tps_ratio": float(local["train_tokens_per_second"])
                / max(float(dense["train_tokens_per_second"]), 1e-9),
                "local_useful_attention_ratio": local["attention_proxy"]["useful_to_dense_ratio"],
                "compact_sliding_window_materialization": True,
            }
    carry_rows = []
    for row in segment_rows:
        tac = row["tac"]
        carry_rows.append(
            {
                "variant": row["variant"],
                "seq_len": row["seq_len"],
                "carry_minus_reset": tac["carry"]["value_accuracy"] - tac["reset"]["value_accuracy"],
                "carry_minus_shuffled": tac["carry"]["value_accuracy"] - tac["shuffled"]["value_accuracy"],
                "carry_minus_baseline": tac["carry"]["value_accuracy"]
                - row["baseline"]["carry"]["value_accuracy"],
            }
        )
    return {
        "memmap_speedups": memmap_speedups,
        "local_attention_notes": local_speed_notes,
        "segment_carry_deltas": carry_rows,
        "local_gate_status": {
            "tokenized_batcher_5x_gate": "pass"
            if memmap_speedups and min(memmap_speedups.values()) >= 5.0
            else "fail_initial_byte_memmap_only",
            "tokenized_batcher_functional": "pass"
            if memmap_speedups and min(memmap_speedups.values()) >= 1.0
            else "needs_followup",
            "local_attention_memory_shape": "pass_compact_logits",
            "local_attention_wall_clock": "fails_cpu_speed_gate",
            "segment_carry": "screen_only",
        },
    }


def format_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Local TAC Long-Context Efficiency Results",
        "",
        f"Date: {aggregate['date']}",
        f"Device: `{aggregate['device']}`",
        f"Prepared JSONL: `{aggregate['prepared_jsonl']}`",
        "",
        "## Batcher Throughput",
        "",
        "| Batcher | Seq len | Tokens/sec | Seconds |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in aggregate["batcher_results"]:
        lines.append(
            f"| `{row['batcher']}` | {row['seq_len']} | "
            f"{row['tokens_per_second']:.2f} | {row['seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Model Decomposition",
            "",
            "| Variant | Seq len | Window | Train tok/s | Prefill tok/s | Carried tok/s | Decode tok/s | Useful/dense attn |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate["model_profiles"]:
        window = row["attention_window_size"] if row["attention_window_size"] is not None else "dense"
        lines.append(
            f"| `{row['variant']}` | {row['seq_len']} | {window} | "
            f"{row['train_tokens_per_second']:.2f} | "
            f"{row['prefill_tokens_per_second']:.2f} | "
            f"{row['carried_query_tokens_per_second']:.2f} | "
            f"{row['decode_tokens_per_second']:.2f} | "
            f"{row['attention_proxy']['useful_to_dense_ratio']:.4f} |"
        )
    if aggregate["segment_carry_screen"]:
        lines.extend(
            [
                "",
                "## Segment Carry Screen",
                "",
                "| Variant | Seq len | Carry | Reset | Shuffled | Baseline | Decision |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in aggregate["segment_carry_screen"]:
            tac = row["tac"]
            baseline = row["baseline"]
            lines.append(
                f"| `{row['variant']}` | {row['seq_len']} | "
                f"{tac['carry']['value_accuracy']:.4f} | "
                f"{tac['reset']['value_accuracy']:.4f} | "
                f"{tac['shuffled']['value_accuracy']:.4f} | "
                f"{baseline['carry']['value_accuracy']:.4f} | "
                f"{row['decision']['status']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "```json",
            json.dumps(aggregate["interpretation"], indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
