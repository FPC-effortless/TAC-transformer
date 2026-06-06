from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import ContentWritePolicy, TACTransformerLM, best_tac_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile whether TAC decode can benefit from an identity-output cache."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/identity_decode_cache_local"),
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--policies",
        default="query_skip,decode_state_skip",
        help="Comma-separated decode write policies to profile.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_identity_decode_cache_profile(args, device)
    (args.output_dir / "identity_decode_cache_profile.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


def run_identity_decode_cache_profile(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    config = best_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=max(args.seq_len, 1),
        content_store_size=8,
        content_read_steps=1,
    )
    model = TACTransformerLM(config).to(device)
    model.eval()
    context = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
    )
    decode_tokens = torch.randint(
        0,
        config.vocab_size,
        (args.decode_steps, args.batch_size, 1),
        device=device,
    )
    with torch.inference_mode():
        states = model(
            context,
            collect_auxiliary=False,
            write_policy=ContentWritePolicy.DENSE,
        ).identity_states
        profiles = []
        for policy in parse_policies(args.policies):
            for _ in range(args.warmup):
                decode_once(model, decode_tokens, states, write_policy=policy)
            sync(device)
            timer = IdentityFieldTimer(model)
            timer.install()
            started = time.perf_counter()
            assignments = []
            try:
                for _ in range(args.iters):
                    assignments.extend(
                        decode_once(model, decode_tokens, states, write_policy=policy)
                    )
                sync(device)
            finally:
                timer.remove()
            elapsed = max(time.perf_counter() - started, 1e-9)
            profiles.append(
                summarize_identity_decode_cache_profile(
                    {
                        "policy": policy.value,
                        "device": str(device),
                        "seq_len": args.seq_len,
                        "decode_steps": args.decode_steps,
                        "batch_size": args.batch_size,
                        "iters": args.iters,
                        "decode_seconds": elapsed / max(args.iters, 1),
                        "decode_tokens_per_second": (
                            args.decode_steps * args.batch_size * args.iters / elapsed
                        ),
                        "identity_field_seconds": timer.elapsed / max(args.iters, 1),
                        "identity_field_calls_per_iter": timer.calls / max(args.iters, 1),
                        "assignment_trace": assignments,
                    }
                )
            )
    best = max(profiles, key=lambda row: row["decode_tokens_per_second"])
    return {
        "profiles": profiles,
        "best_policy_by_tokens_per_second": best["policy"],
        "best_decode_tokens_per_second": best["decode_tokens_per_second"],
    }


class IdentityFieldTimer:
    def __init__(self, model: TACTransformerLM):
        self.model = model
        self.elapsed = 0.0
        self.calls = 0
        self._originals: list[tuple[Any, Any]] = []

    def install(self) -> None:
        for block in self.model.blocks:
            field = block.identity_field
            original = field.forward

            def timed_forward(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.elapsed += time.perf_counter() - started
                    self.calls += 1

            field.forward = timed_forward  # type: ignore[method-assign]
            self._originals.append((field, original))

    def remove(self) -> None:
        for field, original in self._originals:
            field.forward = original  # type: ignore[method-assign]
        self._originals = []


def decode_once(
    model: TACTransformerLM,
    tokens: torch.Tensor,
    states: Any,
    *,
    write_policy: ContentWritePolicy,
) -> list[list[int]]:
    current_states = states
    assignments = []
    for index in range(tokens.shape[0]):
        output = model(
            tokens[index],
            identity_states=current_states,
            collect_auxiliary=False,
            write_policy=write_policy,
        )
        current_states = output.identity_states
        token_activations = output.aux.token_program_activations
        if token_activations is None or token_activations.numel() == 0:
            assignments.append([])
        else:
            assignments.append(
                token_activations.argmax(dim=-1).reshape(-1).detach().cpu().tolist()
            )
    return assignments


def parse_policies(raw: str) -> list[ContentWritePolicy]:
    policies = []
    for name in raw.split(","):
        cleaned = name.strip()
        if cleaned:
            policies.append(ContentWritePolicy(cleaned))
    if not policies:
        raise ValueError("--policies must include at least one write policy")
    return policies


def summarize_identity_decode_cache_profile(row: dict[str, Any]) -> dict[str, Any]:
    decode_seconds = max(float(row["decode_seconds"]), 1e-9)
    identity_seconds = max(float(row["identity_field_seconds"]), 0.0)
    identity_fraction = min(identity_seconds / decode_seconds, 1.0)
    speedup_ceiling = 1.0 / max(1.0 - identity_fraction, 1e-6)
    trace = row.get("assignment_trace", [])
    switch_fraction = program_switch_fraction(trace)
    return {
        **{key: value for key, value in row.items() if key != "assignment_trace"},
        "identity_field_fraction": identity_fraction,
        "identity_cache_speedup_ceiling": speedup_ceiling,
        "program_switch_fraction": switch_fraction,
        "program_stable_fraction": 1.0 - switch_fraction,
        "meets_20_percent_speedup_ceiling": speedup_ceiling >= 1.2,
        "cache_safety_decision": (
            "prototype_guarded_cache"
            if speedup_ceiling >= 1.2 and switch_fraction <= 0.05
            else "diagnostic_only"
        ),
    }


def program_switch_fraction(trace: list[list[int]]) -> float:
    switches = 0
    comparisons = 0
    previous: list[int] | None = None
    for current in trace:
        if previous is not None and current and previous:
            width = min(len(previous), len(current))
            switches += sum(1 for idx in range(width) if previous[idx] != current[idx])
            comparisons += width
        previous = current
    if comparisons == 0:
        return 0.0
    return switches / comparisons


def format_markdown(result: dict[str, Any]) -> str:
    if "profiles" in result:
        rows = [
            (
                f"| `{profile['policy']}` | {profile['decode_tokens_per_second']:.2f} | "
                f"{profile['identity_field_fraction']:.4f} | "
                f"{profile['identity_cache_speedup_ceiling']:.4f} | "
                f"{profile['program_switch_fraction']:.4f} |"
            )
            for profile in result["profiles"]
        ]
        return "\n".join(
            [
                "# TAC Identity Decode State-Skip Profile",
                "",
                "| Policy | Decode tok/s | Identity-field fraction | Cache speedup ceiling | Program switch fraction |",
                "| --- | ---: | ---: | ---: | ---: |",
                *rows,
                "",
                f"Best policy by tok/s: `{result['best_policy_by_tokens_per_second']}`",
                "",
            ]
        )
    return "\n".join(
        [
            "# TAC Identity Decode Cache Profile",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Decode tok/s | {result['decode_tokens_per_second']:.2f} |",
            f"| Identity-field fraction | {result['identity_field_fraction']:.4f} |",
            f"| Cache speedup ceiling | {result['identity_cache_speedup_ceiling']:.4f} |",
            f"| Program switch fraction | {result['program_switch_fraction']:.4f} |",
            f"| Decision | `{result['cache_safety_decision']}` |",
            "",
        ]
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
