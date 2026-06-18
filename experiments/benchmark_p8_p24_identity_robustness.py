from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.serving import (
    TAC_BYTE_TOKEN_OFFSET,
    encode_tac_byte_tokens,
    load_tac_checkpoint_for_generation,
)


DEFAULT_P8_CHECKPOINT = (
    ROOT
    / "runs"
    / "kaggle_outputs"
    / "identity_ratio_p8_5k_v2_completed_jeffkolo_20260607"
    / "run5b_identity_ratio_p8_5k"
    / "last.pt"
)
DEFAULT_P24_CHECKPOINT = (
    ROOT
    / "runs"
    / "kaggle_outputs"
    / "identity_ratio_p24_5k_v2_completed_jeffkolo_20260607"
    / "run5b_identity_ratio_p24_5k"
    / "last.pt"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "runs" / "benchmarks" / "p8_p24_identity_robustness_2026_06_07"
)


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    target_cue: str
    target_value: str
    alternatives: tuple[str, ...]
    prefill_text: str
    distractor_texts: tuple[str, ...]
    query_text: str
    interference_pairs: int


@dataclass
class CaseScore:
    case_id: str
    interference_pairs: int
    target_cue: str
    target_value: str
    correct: bool
    forced_choice_rank: int
    forced_choice_probability: float
    forced_choice_margin: float
    full_vocab_rank: int
    memory_read_correct: bool | None
    memory_read_probability: float | None
    memory_read_margin: float | None
    program_memory_norm: float
    selected_programs: list[int]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run post-hoc p8/p24 identity retention, interference, active-context "
            "compression, and knockout robustness benchmarks."
        )
    )
    parser.add_argument("--p8-checkpoint", type=Path, default=DEFAULT_P8_CHECKPOINT)
    parser.add_argument("--p24-checkpoint", type=Path, default=DEFAULT_P24_CHECKPOINT)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="LABEL=CHECKPOINT",
        help=(
            "Additional or replacement benchmark variant. When provided, all "
            "variants are taken from these LABEL=CHECKPOINT entries."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--case-count", type=int, default=48)
    parser.add_argument("--knockout-case-count", type=int, default=12)
    parser.add_argument("--interference-levels", type=int, nargs="+", default=[0, 4, 8, 16])
    parser.add_argument("--active-context-budgets", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32, 48])
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    variants = parse_variant_args(args.variant)
    if not variants:
        variants = {
            "p8": Path(args.p8_checkpoint),
            "p24": Path(args.p24_checkpoint),
        }
    result = run_identity_robustness_variants(
        variants=variants,
        output_dir=args.output_dir,
        device=args.device,
        case_count=args.case_count,
        knockout_case_count=args.knockout_case_count,
        interference_levels=args.interference_levels,
        active_context_budgets=args.active_context_budgets,
    )
    print(json.dumps(result["decision"], indent=2), flush=True)


def parse_variant_args(entries: Sequence[str]) -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"variant must be LABEL=CHECKPOINT, got {entry!r}")
        label, checkpoint = entry.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("variant label must not be empty")
        if label in variants:
            raise ValueError(f"duplicate variant label: {label}")
        variants[label] = Path(checkpoint)
    return variants


def run_p8_p24_identity_robustness(
    *,
    p8_checkpoint: Path,
    p24_checkpoint: Path,
    output_dir: Path,
    device: str | torch.device = "cpu",
    case_count: int = 48,
    knockout_case_count: int = 12,
    interference_levels: Sequence[int] = (0, 4, 8, 16),
    active_context_budgets: Sequence[int] = (4, 8, 12, 16, 24, 32, 48),
) -> dict[str, Any]:
    return run_identity_robustness_variants(
        variants={
            "p8": Path(p8_checkpoint),
            "p24": Path(p24_checkpoint),
        },
        output_dir=output_dir,
        device=device,
        case_count=case_count,
        knockout_case_count=knockout_case_count,
        interference_levels=interference_levels,
        active_context_budgets=active_context_budgets,
        schema="p8_p24_identity_robustness.v1",
        scope="TAC-200 completed p8 vs p24 checkpoints",
        created_at="2026-06-07",
    )


def run_identity_robustness_variants(
    *,
    variants: dict[str, Path],
    output_dir: Path,
    device: str | torch.device = "cpu",
    case_count: int = 48,
    knockout_case_count: int = 12,
    interference_levels: Sequence[int] = (0, 4, 8, 16),
    active_context_budgets: Sequence[int] = (4, 8, 12, 16, 24, 32, 48),
    schema: str = "identity_robustness_multi_variant.v1",
    scope: str = "Completed identity-ratio checkpoints",
    created_at: str = "2026-06-08",
) -> dict[str, Any]:
    if not variants:
        raise ValueError("at least one variant is required")
    started = time.perf_counter()
    cases = build_retrieval_cases(
        case_count=case_count,
        interference_levels=interference_levels,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_results = {}
    for label, checkpoint in variants.items():
        model, metadata = load_tac_checkpoint_for_generation(
            checkpoint,
            model_type="tac",
            device=device,
        )
        variant_results[label] = evaluate_variant(
            model,
            metadata,
            cases,
            device=torch.device(device),
            active_context_budgets=active_context_budgets,
            knockout_case_count=knockout_case_count,
        )
        del model

    result = {
        "schema": schema,
        "created_at": created_at,
        "scope": scope,
        "checkpoints": {
            label: str(Path(checkpoint))
            for label, checkpoint in variants.items()
        },
        "protocol": {
            "case_count": int(case_count),
            "knockout_case_count": int(knockout_case_count),
            "interference_levels": [int(value) for value in interference_levels],
            "active_context_budgets": [int(value) for value in active_context_budgets],
            "retrieval_scoring": (
                "forced-choice next-byte prediction over one correct identity value "
                "and five candidate values; target fact is written in a prior segment "
                "and is absent from the active query segment."
            ),
        },
        "variants": variant_results,
        "comparison": compare_variants(variant_results),
        "decision": decide(variant_results),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "identity_robustness.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "RESULTS.md").write_text(format_markdown(result), encoding="utf-8")
    return result


def build_retrieval_cases(
    *,
    case_count: int,
    interference_levels: Sequence[int],
) -> list[RetrievalCase]:
    if case_count < 1:
        raise ValueError("case_count must be positive")
    levels = [int(value) for value in interference_levels]
    if not levels:
        raise ValueError("at least one interference level is required")
    cue_pool = list("abcdefghjkmnpqrstuvwxyz")
    value_pool = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
    cases: list[RetrievalCase] = []
    for index in range(int(case_count)):
        interference = levels[index % len(levels)]
        cue = cue_pool[(index * 5 + 3) % len(cue_pool)]
        value = value_pool[(index * 7 + 2) % len(value_pool)]
        alternatives = [value]
        cursor = 0
        while len(alternatives) < 6:
            candidate = value_pool[(index * 11 + cursor * 3 + 5) % len(value_pool)]
            cursor += 1
            if candidate not in alternatives:
                alternatives.append(candidate)
        support_pairs = [
            f"{cue}{value}",
            f"{cue_pool[(index + 4) % len(cue_pool)]}{value_pool[(index + 9) % len(value_pool)]}",
            f"{cue_pool[(index + 8) % len(cue_pool)]}{value_pool[(index + 13) % len(value_pool)]}",
        ]
        prefill = "identity memo: " + " ".join(support_pairs) + "\n"
        distractors = []
        remaining = interference
        chunk = 0
        while remaining > 0:
            pair_count = min(4, remaining)
            pairs = []
            for pair_index in range(pair_count):
                cue_index = (index * 13 + chunk * 7 + pair_index * 2 + 1) % len(cue_pool)
                distractor_cue = cue_pool[cue_index]
                if distractor_cue == cue:
                    distractor_cue = cue_pool[(cue_index + 1) % len(cue_pool)]
                distractor_value = value_pool[
                    (index * 17 + chunk * 5 + pair_index * 3 + 4) % len(value_pool)
                ]
                pairs.append(f"{distractor_cue}{distractor_value}")
            distractors.append("interference memo: " + " ".join(pairs) + "\n")
            remaining -= pair_count
            chunk += 1
        query = f"retrieve value for cue {cue}: {cue}"
        cases.append(
            RetrievalCase(
                case_id=f"case_{index:03d}_i{interference}",
                target_cue=cue,
                target_value=value,
                alternatives=tuple(alternatives),
                prefill_text=prefill,
                distractor_texts=tuple(distractors),
                query_text=query,
                interference_pairs=interference,
            )
        )
    return cases


def evaluate_variant(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    cases: list[RetrievalCase],
    *,
    device: torch.device,
    active_context_budgets: Sequence[int],
    knockout_case_count: int,
) -> dict[str, Any]:
    config = model.config
    carry_scores = [
        score_case(model, case, device=device, carry_state=True)
        for case in cases
    ]
    reset_scores = [
        score_case(model, case, device=device, carry_state=False)
        for case in cases
    ]
    compression = {}
    for budget in active_context_budgets:
        budget_scores = [
            score_case(
                model,
                case,
                device=device,
                carry_state=True,
                active_context_budget=int(budget),
            )
            for case in cases
        ]
        compression[str(int(budget))] = summarize_scores(budget_scores)

    knockout_cases = cases[: max(1, min(int(knockout_case_count), len(cases)))]
    knockout_baseline = [
        score_case(model, case, device=device, carry_state=True)
        for case in knockout_cases
    ]
    knockouts = []
    for program_id in range(int(config.n_programs)):
        with knockout_program(model, program_id):
            knockout_scores = [
                score_case(model, case, device=device, carry_state=True)
                for case in knockout_cases
            ]
        knockouts.append(
            summarize_knockout(
                program_id=program_id,
                baseline=knockout_baseline,
                knockout=knockout_scores,
            )
        )

    return {
        "metadata": {
            "checkpoint": metadata.get("checkpoint"),
            "checkpoint_step": metadata.get("checkpoint_step"),
            "best_eval_loss": metadata.get("best_eval_loss"),
            "parameter_counts": metadata.get("parameter_counts"),
            "config": {
                "n_programs": int(config.n_programs),
                "max_seq_len": int(config.max_seq_len),
                "content_store_size": int(config.content_store_size),
                "memory_read_type": config.memory_read_type,
                "identity_attention_type": config.identity_attention_type,
            },
        },
        "long_context_carry": {
            "carry": summarize_scores(carry_scores),
            "reset_no_carry": summarize_scores(reset_scores),
            "carry_minus_reset_accuracy": (
                summarize_scores(carry_scores)["accuracy"]
                - summarize_scores(reset_scores)["accuracy"]
            ),
            "carry_minus_reset_margin": (
                summarize_scores(carry_scores)["mean_margin"]
                - summarize_scores(reset_scores)["mean_margin"]
            ),
        },
        "interference_resistance": summarize_by_interference(carry_scores),
        "active_context_compression": compression,
        "minimum_context_for_50pct_accuracy": minimum_budget_for_accuracy(
            compression,
            threshold=0.5,
        ),
        "minimum_context_for_75pct_accuracy": minimum_budget_for_accuracy(
            compression,
            threshold=0.75,
        ),
        "program_knockout_robustness": summarize_knockouts(knockouts),
        "program_knockouts": knockouts,
    }


def score_case(
    model: torch.nn.Module,
    case: RetrievalCase,
    *,
    device: torch.device,
    carry_state: bool,
    active_context_budget: int | None = None,
) -> CaseScore:
    config = model.config
    states = None
    memory_read = None
    with torch.inference_mode():
        if carry_state:
            for segment in [case.prefill_text, *case.distractor_texts]:
                tokens = encode_tac_byte_tokens(
                    segment,
                    vocab_size=int(config.vocab_size),
                    append_eos=False,
                )
                tokens = tokens[-int(config.max_seq_len) :]
                output = model(
                    torch.tensor([tokens], dtype=torch.long, device=device),
                    identity_states=states,
                    collect_auxiliary=False,
                    collect_metrics=False,
                )
                states = output.identity_states
            memory_read = score_memory_read(model, states, case, device=device)

        query_tokens = encode_tac_byte_tokens(
            case.query_text,
            vocab_size=int(config.vocab_size),
            append_eos=False,
        )
        if active_context_budget is not None:
            query_tokens = query_tokens[-max(1, int(active_context_budget)) :]
        query_tokens = query_tokens[-int(config.max_seq_len) :]
        output = model(
            torch.tensor([query_tokens], dtype=torch.long, device=device),
            identity_states=states,
            collect_auxiliary=True,
            collect_metrics=True,
        )
    logits = output.logits[0, -1].detach().float().cpu()
    alternative_ids = [
        _byte_token_id(value)
        for value in case.alternatives
    ]
    target_id = _byte_token_id(case.target_value)
    forced = forced_choice_metrics(logits, target_id, alternative_ids)
    selected_programs = selected_programs_for_last_token(output)
    return CaseScore(
        case_id=case.case_id,
        interference_pairs=int(case.interference_pairs),
        target_cue=case.target_cue,
        target_value=case.target_value,
        correct=bool(forced["correct"]),
        forced_choice_rank=int(forced["rank"]),
        forced_choice_probability=float(forced["probability"]),
        forced_choice_margin=float(forced["margin"]),
        full_vocab_rank=int(full_vocab_rank(logits, target_id)),
        memory_read_correct=None if memory_read is None else bool(memory_read["correct"]),
        memory_read_probability=None if memory_read is None else float(memory_read["probability"]),
        memory_read_margin=None if memory_read is None else float(memory_read["margin"]),
        program_memory_norm=program_memory_norm(output.identity_states),
        selected_programs=selected_programs,
    )


def score_memory_read(
    model: torch.nn.Module,
    states: Any,
    case: RetrievalCase,
    *,
    device: torch.device,
) -> dict[str, float | bool] | None:
    if not states:
        return None
    cue_id = _byte_token_id(case.target_cue)
    try:
        logits = model.memory_read_logits(
            torch.tensor([cue_id], dtype=torch.long, device=device),
            states,
        )[0].detach().float().cpu()
    except Exception:
        return None
    alternative_ids = [_byte_token_id(value) for value in case.alternatives]
    return forced_choice_metrics(logits, _byte_token_id(case.target_value), alternative_ids)


def forced_choice_metrics(
    logits: torch.Tensor,
    target_id: int,
    alternative_ids: Sequence[int],
) -> dict[str, float | int | bool]:
    ids = torch.tensor([int(value) for value in alternative_ids], dtype=torch.long)
    values = logits[ids]
    probabilities = torch.softmax(values, dim=0)
    target_index = list(map(int, alternative_ids)).index(int(target_id))
    order = torch.argsort(values, descending=True).tolist()
    rank = order.index(target_index) + 1
    target_value = float(values[target_index])
    competitor_values = [
        float(value)
        for index, value in enumerate(values.tolist())
        if index != target_index
    ]
    return {
        "correct": rank == 1,
        "rank": rank,
        "probability": float(probabilities[target_index]),
        "margin": target_value - max(competitor_values),
    }


def selected_programs_for_last_token(output: Any) -> list[int]:
    mask = output.aux.token_selected_program_mask
    if mask is None:
        mask = output.aux.selected_program_mask[:, None, :]
    row = mask[0, -1].detach().cpu()
    return [int(index) for index in row.nonzero(as_tuple=False).flatten().tolist()]


def program_memory_norm(states: Any) -> float:
    if not states:
        return 0.0
    memory = states[-1].program_memory.detach().float().cpu()
    if memory.numel() == 0:
        return 0.0
    return float(memory.norm(dim=-1).mean())


def summarize_scores(scores: Sequence[CaseScore]) -> dict[str, Any]:
    rows = list(scores)
    if not rows:
        return {
            "cases": 0,
            "accuracy": 0.0,
            "mean_margin": 0.0,
            "mean_probability": 0.0,
            "mean_full_vocab_rank": 0.0,
            "memory_read_accuracy": None,
            "memory_read_mean_margin": None,
            "mean_program_memory_norm": 0.0,
        }
    memory_rows = [row for row in rows if row.memory_read_correct is not None]
    return {
        "cases": len(rows),
        "accuracy": _mean([1.0 if row.correct else 0.0 for row in rows]),
        "mean_margin": _mean([row.forced_choice_margin for row in rows]),
        "mean_probability": _mean([row.forced_choice_probability for row in rows]),
        "mean_full_vocab_rank": _mean([float(row.full_vocab_rank) for row in rows]),
        "memory_read_accuracy": None
        if not memory_rows
        else _mean([1.0 if row.memory_read_correct else 0.0 for row in memory_rows]),
        "memory_read_mean_margin": None
        if not memory_rows
        else _mean([float(row.memory_read_margin) for row in memory_rows if row.memory_read_margin is not None]),
        "memory_read_mean_probability": None
        if not memory_rows
        else _mean([
            float(row.memory_read_probability)
            for row in memory_rows
            if row.memory_read_probability is not None
        ]),
        "mean_program_memory_norm": _mean([row.program_memory_norm for row in rows]),
    }


def summarize_by_interference(scores: Sequence[CaseScore]) -> dict[str, Any]:
    groups: dict[int, list[CaseScore]] = {}
    for score in scores:
        groups.setdefault(score.interference_pairs, []).append(score)
    return {
        str(level): summarize_scores(rows)
        for level, rows in sorted(groups.items())
    }


def summarize_knockout(
    *,
    program_id: int,
    baseline: Sequence[CaseScore],
    knockout: Sequence[CaseScore],
) -> dict[str, Any]:
    base_summary = summarize_scores(baseline)
    knockout_summary = summarize_scores(knockout)
    return {
        "program": int(program_id),
        "baseline_accuracy": base_summary["accuracy"],
        "knockout_accuracy": knockout_summary["accuracy"],
        "accuracy_drop": base_summary["accuracy"] - knockout_summary["accuracy"],
        "baseline_margin": base_summary["mean_margin"],
        "knockout_margin": knockout_summary["mean_margin"],
        "margin_drop": base_summary["mean_margin"] - knockout_summary["mean_margin"],
        "baseline_memory_read_margin": base_summary["memory_read_mean_margin"],
        "knockout_memory_read_margin": knockout_summary["memory_read_mean_margin"],
        "memory_read_margin_drop": _optional_delta(
            base_summary["memory_read_mean_margin"],
            knockout_summary["memory_read_mean_margin"],
        ),
        "baseline_program_memory_norm": base_summary["mean_program_memory_norm"],
        "knockout_program_memory_norm": knockout_summary["mean_program_memory_norm"],
        "program_memory_norm_drop": (
            base_summary["mean_program_memory_norm"]
            - knockout_summary["mean_program_memory_norm"]
        ),
    }


def summarize_knockouts(knockouts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(knockouts)
    positive_accuracy_drops = [max(0.0, float(row["accuracy_drop"])) for row in rows]
    positive_margin_drops = [max(0.0, float(row["margin_drop"])) for row in rows]
    max_accuracy_drop = max(positive_accuracy_drops, default=0.0)
    max_margin_drop = max(positive_margin_drops, default=0.0)
    sum_accuracy_drop = sum(positive_accuracy_drops)
    sum_margin_drop = sum(positive_margin_drops)
    top_accuracy = max(rows, key=lambda row: float(row["accuracy_drop"]), default=None)
    top_margin = max(rows, key=lambda row: float(row["margin_drop"]), default=None)
    return {
        "programs_tested": len(rows),
        "mean_accuracy_drop": _mean([float(row["accuracy_drop"]) for row in rows]),
        "max_accuracy_drop": max_accuracy_drop,
        "accuracy_drop_concentration": (
            0.0 if sum_accuracy_drop <= 0 else max_accuracy_drop / sum_accuracy_drop
        ),
        "harmful_accuracy_programs_ge_5pct": sum(
            1 for row in rows if float(row["accuracy_drop"]) >= 0.05
        ),
        "mean_margin_drop": _mean([float(row["margin_drop"]) for row in rows]),
        "max_margin_drop": max_margin_drop,
        "margin_drop_concentration": (
            0.0 if sum_margin_drop <= 0 else max_margin_drop / sum_margin_drop
        ),
        "top_accuracy_drop_program": None if top_accuracy is None else int(top_accuracy["program"]),
        "top_margin_drop_program": None if top_margin is None else int(top_margin["program"]),
    }


def compare_variants(variant_results: dict[str, Any]) -> dict[str, Any]:
    ranking = rank_variants(variant_results)
    if "p8" not in variant_results or "p24" not in variant_results:
        return {"ranking": ranking}
    p8 = variant_results["p8"]
    p24 = variant_results["p24"]
    p8_carry = p8["long_context_carry"]["carry"]
    p24_carry = p24["long_context_carry"]["carry"]
    return {
        "p24_minus_p8_carry_accuracy": (
            p24_carry["accuracy"] - p8_carry["accuracy"]
        ),
        "p24_minus_p8_carry_margin": (
            p24_carry["mean_margin"] - p8_carry["mean_margin"]
        ),
        "p24_minus_p8_memory_read_accuracy": _optional_delta(
            p24_carry["memory_read_accuracy"],
            p8_carry["memory_read_accuracy"],
        ),
        "p24_min_context_for_50pct_minus_p8": _optional_delta(
            p24["minimum_context_for_50pct_accuracy"],
            p8["minimum_context_for_50pct_accuracy"],
        ),
        "p24_min_context_for_75pct_minus_p8": _optional_delta(
            p24["minimum_context_for_75pct_accuracy"],
            p8["minimum_context_for_75pct_accuracy"],
        ),
        "p24_minus_p8_knockout_max_accuracy_drop": (
            p24["program_knockout_robustness"]["max_accuracy_drop"]
            - p8["program_knockout_robustness"]["max_accuracy_drop"]
        ),
        "p24_minus_p8_knockout_accuracy_drop_concentration": (
            p24["program_knockout_robustness"]["accuracy_drop_concentration"]
            - p8["program_knockout_robustness"]["accuracy_drop_concentration"]
        ),
        "ranking": ranking,
    }


def decide(variant_results: dict[str, Any]) -> dict[str, Any]:
    comparison = compare_variants(variant_results)
    if "p8" not in variant_results or "p24" not in variant_results:
        ranking = comparison["ranking"]
        best_carry = ranking["best_carry_accuracy"]
        return {
            "status": "multi_variant_identity_robustness_ranked",
            "best_carry_accuracy": best_carry,
            "best_memory_read_accuracy": ranking["best_memory_read_accuracy"],
            "best_compression_50pct": ranking["best_minimum_context_for_50pct_accuracy"],
            "interpretation": (
                "This is a post-hoc synthetic byte-pair retrieval benchmark over "
                "completed checkpoints. It ranks identity carry, memory-read, "
                "active-context compression, and knockout behavior, and should be "
                "read alongside LM/specialization metrics rather than as a "
                "multi-seed capability estimate."
            ),
        }
    p8 = variant_results["p8"]
    p24 = variant_results["p24"]
    p8_accuracy = p8["long_context_carry"]["carry"]["accuracy"]
    p24_accuracy = p24["long_context_carry"]["carry"]["accuracy"]
    p8_reset = p8["long_context_carry"]["reset_no_carry"]["accuracy"]
    p24_reset = p24["long_context_carry"]["reset_no_carry"]["accuracy"]
    p24_better = p24_accuracy - p8_accuracy >= 0.05
    both_above_reset = (
        p8_accuracy > p8_reset + 0.05
        or p24_accuracy > p24_reset + 0.05
    )
    if p24_better and both_above_reset:
        status = "p24_identity_robustness_advantage_supported"
    elif both_above_reset:
        status = "identity_carry_detected_without_p24_advantage"
    else:
        status = "no_reliable_identity_carry_advantage_detected"
    return {
        "status": status,
        "p8_carry_accuracy": p8_accuracy,
        "p24_carry_accuracy": p24_accuracy,
        "p8_reset_accuracy": p8_reset,
        "p24_reset_accuracy": p24_reset,
        "p24_minus_p8_carry_accuracy": comparison["p24_minus_p8_carry_accuracy"],
        "interpretation": (
            "This is a post-hoc synthetic byte-pair retrieval benchmark over the "
            "completed p8/p24 checkpoints. It tests whether specialization is "
            "causally useful for identity carry; it should be read alongside the "
            "original LM/specialization metrics, not as a replacement for multi-seed "
            "external validation."
        ),
    }


def rank_variants(variant_results: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for label, result in variant_results.items():
        carry = result["long_context_carry"]["carry"]
        knockout = result["program_knockout_robustness"]
        rows.append(
            {
                "label": label,
                "carry_accuracy": carry["accuracy"],
                "carry_margin": carry["mean_margin"],
                "memory_read_accuracy": carry["memory_read_accuracy"],
                "memory_read_margin": carry["memory_read_mean_margin"],
                "minimum_context_for_50pct_accuracy": result[
                    "minimum_context_for_50pct_accuracy"
                ],
                "minimum_context_for_75pct_accuracy": result[
                    "minimum_context_for_75pct_accuracy"
                ],
                "knockout_max_accuracy_drop": knockout["max_accuracy_drop"],
                "knockout_max_margin_drop": knockout["max_margin_drop"],
                "knockout_accuracy_drop_concentration": knockout[
                    "accuracy_drop_concentration"
                ],
            }
        )

    def best(
        key: str,
        *,
        higher_is_better: bool = True,
        none_is_worst: bool = True,
    ) -> dict[str, Any] | None:
        candidates = [
            row for row in rows
            if row[key] is not None or not none_is_worst
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: float(row[key]),
        ) if higher_is_better else min(
            candidates,
            key=lambda row: float(row[key]),
        )

    return {
        "rows": rows,
        "best_carry_accuracy": best("carry_accuracy"),
        "best_carry_margin": best("carry_margin"),
        "best_memory_read_accuracy": best("memory_read_accuracy"),
        "best_memory_read_margin": best("memory_read_margin"),
        "best_minimum_context_for_50pct_accuracy": best(
            "minimum_context_for_50pct_accuracy",
            higher_is_better=False,
        ),
        "best_minimum_context_for_75pct_accuracy": best(
            "minimum_context_for_75pct_accuracy",
            higher_is_better=False,
        ),
        "largest_knockout_accuracy_drop": best("knockout_max_accuracy_drop"),
        "largest_knockout_margin_drop": best("knockout_max_margin_drop"),
    }


def minimum_budget_for_accuracy(
    compression: dict[str, dict[str, Any]],
    *,
    threshold: float,
) -> int | None:
    for budget in sorted(int(value) for value in compression):
        if compression[str(budget)]["accuracy"] >= float(threshold):
            return budget
    return None


@contextmanager
def knockout_program(model: torch.nn.Module, program_id: int) -> Iterator[None]:
    originals = []
    for block in model.blocks:
        identity = block.identity_field
        original = identity._compute_program_context
        originals.append((identity, original))

        def patched(
            self,
            hidden,
            selected_weights,
            selected_denominator,
            previous_memory,
            previous_engram_patterns,
            previous_engram_values,
            previous_engram_mask,
            previous_content_cues,
            previous_content_values,
            previous_content_mask,
            *,
            _original=original,
        ):
            masked_weights = selected_weights.clone()
            masked_weights[..., int(program_id)] = 0.0
            denominator = masked_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return _original(
                hidden,
                masked_weights,
                denominator,
                previous_memory,
                previous_engram_patterns,
                previous_engram_values,
                previous_engram_mask,
                previous_content_cues,
                previous_content_values,
                previous_content_mask,
            )

        identity._compute_program_context = MethodType(patched, identity)
    try:
        yield
    finally:
        for identity, original in originals:
            identity._compute_program_context = original


def format_markdown(result: dict[str, Any]) -> str:
    variants = result["variants"]
    labels = list(variants)
    lines = [
        "# Identity Robustness Benchmark",
        "",
        f"Decision: `{result['decision']['status']}`.",
        "",
        "This is a post-hoc benchmark over completed 5k checkpoints.",
        "The target fact is written in an earlier segment and is absent from the active query segment.",
        "",
        _wide_header("Metric", labels),
        _wide_separator(labels),
        _wide_metric_row(
            "Carry retrieval accuracy",
            labels,
            variants,
            lambda row: row["long_context_carry"]["carry"]["accuracy"],
        ),
        _wide_metric_row(
            "Reset retrieval accuracy",
            labels,
            variants,
            lambda row: row["long_context_carry"]["reset_no_carry"]["accuracy"],
        ),
        _wide_metric_row(
            "Carry mean margin",
            labels,
            variants,
            lambda row: row["long_context_carry"]["carry"]["mean_margin"],
        ),
        _wide_metric_row(
            "Memory-read accuracy",
            labels,
            variants,
            lambda row: row["long_context_carry"]["carry"]["memory_read_accuracy"],
        ),
        _wide_metric_row(
            "Min context for 50% acc",
            labels,
            variants,
            lambda row: row["minimum_context_for_50pct_accuracy"],
        ),
        _wide_metric_row(
            "Min context for 75% acc",
            labels,
            variants,
            lambda row: row["minimum_context_for_75pct_accuracy"],
        ),
        _wide_metric_row(
            "Max knockout accuracy drop",
            labels,
            variants,
            lambda row: row["program_knockout_robustness"]["max_accuracy_drop"],
        ),
        _wide_metric_row(
            "Knockout drop concentration",
            labels,
            variants,
            lambda row: row["program_knockout_robustness"]["accuracy_drop_concentration"],
        ),
        "",
        "## Interference",
        "",
        _wide_header("Interference pairs", labels),
        _wide_separator(labels),
    ]
    levels = sorted(
        {
            level
            for row in variants.values()
            for level in row["interference_resistance"]
        },
        key=lambda value: int(value),
    )
    for level in levels:
        values = [
            variants[label]["interference_resistance"].get(level, {}).get("accuracy")
            for label in labels
        ]
        lines.append(_wide_value_row(level, values))
    lines.extend(
        [
            "",
            "## Active Context Compression",
            "",
            _wide_header("Active query tokens", labels),
            _wide_separator(labels),
        ]
    )
    budgets = sorted(
        {
            budget
            for row in variants.values()
            for budget in row["active_context_compression"]
        },
        key=lambda value: int(value),
    )
    for budget in budgets:
        values = [
            variants[label]["active_context_compression"].get(budget, {}).get("accuracy")
            for label in labels
        ]
        lines.append(_wide_value_row(budget, values))
    ranking = result.get("comparison", {}).get("ranking")
    if ranking:
        lines.extend(
            [
                "",
                "## Ranking",
                "",
                "| Criterion | Variant | Value |",
                "| --- | --- | ---: |",
            ]
        )
        for key, name in [
            ("best_carry_accuracy", "Best carry accuracy"),
            ("best_carry_margin", "Best carry margin"),
            ("best_memory_read_accuracy", "Best memory-read accuracy"),
            ("best_memory_read_margin", "Best memory-read margin"),
            ("best_minimum_context_for_50pct_accuracy", "Smallest context for 50% acc"),
            ("largest_knockout_accuracy_drop", "Largest knockout accuracy drop"),
        ]:
            row = ranking.get(key)
            if row:
                value_key = {
                    "best_carry_accuracy": "carry_accuracy",
                    "best_carry_margin": "carry_margin",
                    "best_memory_read_accuracy": "memory_read_accuracy",
                    "best_memory_read_margin": "memory_read_margin",
                    "best_minimum_context_for_50pct_accuracy": "minimum_context_for_50pct_accuracy",
                    "largest_knockout_accuracy_drop": "knockout_max_accuracy_drop",
                }[key]
                lines.append(
                    f"| {name} | {row['label']} | {_format_value(row[value_key])} |"
                )
    lines.extend(
        [
            "",
            "Boundary:",
            "",
            "- Treat this as causal evidence only if carry beats reset and p24 beats p8 on retrieval or compression.",
            "- Program knockouts are interpreted by retrieval/memory drops, not just LM loss deltas.",
            "- This remains a single-checkpoint post-hoc benchmark; repeated seeds would still be required for a strong claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _metric_row(name: str, p8_value: Any, p24_value: Any) -> str:
    return f"| {name} | {_format_value(p8_value)} | {_format_value(p24_value)} |"


def _wide_header(first: str, labels: Sequence[str]) -> str:
    return "| " + " | ".join([first, *labels]) + " |"


def _wide_separator(labels: Sequence[str]) -> str:
    return "| " + " | ".join(["---", *(["---:"] * len(labels))]) + " |"


def _wide_metric_row(
    name: str,
    labels: Sequence[str],
    variants: dict[str, Any],
    getter: Any,
) -> str:
    return _wide_value_row(name, [getter(variants[label]) for label in labels])


def _wide_value_row(name: Any, values: Sequence[Any]) -> str:
    return "| " + " | ".join([str(name), *[_format_value(value) for value in values]]) + " |"


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def full_vocab_rank(logits: torch.Tensor, target_id: int) -> int:
    target_value = logits[int(target_id)]
    return int((logits > target_value).sum().item()) + 1


def _byte_token_id(character: str) -> int:
    encoded = str(character).encode("utf-8")
    if len(encoded) != 1:
        raise ValueError(f"expected one byte character, got {character!r}")
    return int(encoded[0]) + TAC_BYTE_TOKEN_OFFSET


def _mean(values: Sequence[float]) -> float:
    vals = [float(value) for value in values]
    return statistics.fmean(vals) if vals else 0.0


def _optional_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


if __name__ == "__main__":
    main()
