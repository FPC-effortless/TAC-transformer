from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile opt-in content-read query gating for TAC memory reads."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/content_read_query_gating_smoke_2026_06_04"),
    )
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--content-store-size", type=int, default=8)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    base_config = TACConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=max(args.seq_len, args.context_len),
        state_update_type="gated",
        memory_read_type="content_addressed",
        content_store_size=args.content_store_size,
        content_read_steps=1,
        detach_identity_state=False,
    )
    base_model = TACTransformerLM(base_config).eval()
    state_dict = base_model.state_dict()
    context_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_len),
    )
    query_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
    )

    rows = []
    for top_k_value in args.top_k_values:
        top_k = None if top_k_value == 0 else top_k_value
        config = replace(base_config, content_read_query_top_k=top_k)
        model = TACTransformerLM(config).eval()
        model.load_state_dict(state_dict)
        rows.append(profile_variant(args, model, context_ids, query_ids, top_k_value))

    aggregate = {
        "schema": "content_read_query_gating_profile.v1",
        "date": "2026-06-04",
        "config": {
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "n_programs": args.n_programs,
            "seq_len": args.seq_len,
            "context_len": args.context_len,
            "batch_size": args.batch_size,
            "content_store_size": args.content_store_size,
            "warmup": args.warmup,
            "iters": args.iters,
            "torch_threads": torch.get_num_threads(),
        },
        "profiles": rows,
        "interpretation": interpret(rows),
    }
    (args.output_dir / "content_read_query_gating_profile.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(aggregate),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2), flush=True)


def profile_variant(
    args: argparse.Namespace,
    model: TACTransformerLM,
    context_ids: torch.Tensor,
    query_ids: torch.Tensor,
    top_k_value: int,
) -> dict[str, Any]:
    with torch.no_grad():
        context = model(
            context_ids,
            collect_auxiliary=False,
            update_content_memory=True,
        )
        for _ in range(args.warmup):
            model(
                query_ids,
                identity_states=context.identity_states,
                collect_auxiliary=False,
                update_content_memory=False,
            )
        start = time.perf_counter()
        for _ in range(args.iters):
            output = model(
                query_ids,
                identity_states=context.identity_states,
                collect_auxiliary=False,
                update_content_memory=False,
            )
        elapsed = time.perf_counter() - start

    identity_field = model.blocks[-1].identity_field
    content_read_queries = float(identity_field._last_content_read_queries.detach())
    content_read_query_fraction = float(
        identity_field._last_content_read_query_fraction.detach()
    )
    tokens = args.batch_size * args.seq_len * args.iters
    return {
        "variant": "full_read" if top_k_value == 0 else f"top_k_{top_k_value}",
        "content_read_query_top_k": None if top_k_value == 0 else top_k_value,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "content_read_queries_per_forward": content_read_queries,
        "content_read_query_fraction": content_read_query_fraction,
        "content_read_skipped_fraction": 1.0 - content_read_query_fraction,
        "logit_checksum": float(output.logits.mean().detach()),
    }


def interpret(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"decision": "no_rows"}
    baseline = rows[0]
    baseline_tps = float(baseline["tokens_per_second"])
    interpreted = []
    for row in rows:
        tps = float(row["tokens_per_second"])
        interpreted.append(
            {
                "variant": row["variant"],
                "tps_ratio_vs_full": tps / max(baseline_tps, 1e-9),
                "read_query_fraction": row["content_read_query_fraction"],
                "read_query_reduction": 1.0 - row["content_read_query_fraction"],
            }
        )
    return {
        "baseline": baseline["variant"],
        "rows": interpreted,
        "decision": "query_gating_reduces_memory_read_work_by_construction",
    }


def format_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Content Read Query Gating Profile",
        "",
        f"Date: {aggregate['date']}",
        "",
        "## Profiles",
        "",
        "| Variant | TPS | TPS vs full | Read fraction | Skipped fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    ratios = {
        row["variant"]: item["tps_ratio_vs_full"]
        for row, item in zip(aggregate["profiles"], aggregate["interpretation"]["rows"])
    }
    for row in aggregate["profiles"]:
        lines.append(
            "| {variant} | {tps:.2f} | {ratio:.4f} | {read:.4f} | {skip:.4f} |".format(
                variant=row["variant"],
                tps=row["tokens_per_second"],
                ratio=ratios[row["variant"]],
                read=row["content_read_query_fraction"],
                skip=row["content_read_skipped_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The top-k variants reduce content-addressed memory lookup positions by construction.",
            "Wall-clock speedup is hardware- and shape-dependent; the key gate is whether read work",
            "falls without enabling the gated path by default.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
