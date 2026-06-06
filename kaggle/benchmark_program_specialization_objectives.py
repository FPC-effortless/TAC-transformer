from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from types import MethodType
from typing import Any, Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACTransformerLM, best_tac_config
from tac_transformer.optimization import TACOptimizerConfig, build_tac_optimizer
from tac_transformer.training import (
    _weighted_auxiliary_loss,
    category_program_mi_loss,
    category_route_loss,
    selected_program_mi_loss,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "current_best": {
        "overrides": {},
        "category_route_weight": 0.0,
        "category_objective": "fixed",
    },
    "base_semantic": {
        "overrides": {"routing_type": "base_semantic", "routing_top_k": 2},
        "category_route_weight": 0.0,
        "category_objective": "fixed",
    },
    "authority_gated": {
        "overrides": {"routing_type": "authority_gated", "routing_top_k": 2},
        "category_route_weight": 0.0,
        "category_objective": "fixed",
    },
    "base_semantic_supervised_0p05": {
        "overrides": {
            "routing_type": "base_semantic",
            "routing_top_k": 2,
            "routing_load_balance_weight": 0.05,
        },
        "category_route_weight": 0.05,
        "category_objective": "fixed",
    },
    "base_semantic_supervised_0p1": {
        "overrides": {
            "routing_type": "base_semantic",
            "routing_top_k": 2,
            "routing_load_balance_weight": 0.05,
        },
        "category_route_weight": 0.1,
        "category_objective": "fixed",
    },
    "base_semantic_supervised_0p2": {
        "overrides": {
            "routing_type": "base_semantic",
            "routing_top_k": 2,
            "routing_load_balance_weight": 0.05,
        },
        "category_route_weight": 0.2,
        "category_objective": "fixed",
    },
    "base_semantic_supervised_0p5": {
        "overrides": {
            "routing_type": "base_semantic",
            "routing_top_k": 2,
            "routing_load_balance_weight": 0.05,
        },
        "category_route_weight": 0.5,
        "category_objective": "fixed",
    },
    "base_semantic_mi_0p1": {
        "overrides": {
            "routing_type": "base_semantic",
            "routing_top_k": 2,
            "routing_load_balance_weight": 0.05,
        },
        "category_route_weight": 0.1,
        "category_objective": "mi",
    },
    "base_semantic_mi_0p5": {
        "overrides": {
            "routing_type": "base_semantic",
            "routing_top_k": 2,
            "routing_load_balance_weight": 0.05,
        },
        "category_route_weight": 0.5,
        "category_objective": "mi",
    },
    "base_semantic_soft": {
        "overrides": {"routing_type": "base_semantic_soft", "routing_top_k": 2},
        "category_route_weight": 0.0,
        "category_objective": "fixed",
    },
    "base_semantic_soft_mi_0p5": {
        "overrides": {
            "routing_type": "base_semantic_soft",
            "routing_top_k": 2,
            "routing_load_balance_weight": 0.05,
        },
        "category_route_weight": 0.5,
        "category_objective": "mi",
    },
}


class LabeledTextBatcher:
    def __init__(
        self,
        path: str | Path,
        *,
        vocab_size: int,
        seq_len: int,
        max_records_per_category: int,
        seed: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.rng = random.Random(seed)
        self.records = _load_records(Path(path), max_records_per_category=max_records_per_category)
        self.categories = sorted({record["category"] for record in self.records})
        self.category_to_id = {
            category: index for index, category in enumerate(self.categories)
        }
        if not self.records:
            raise ValueError(f"no labeled records found in {path}")

    def next_batch(
        self,
        batch_size: int,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = []
        category_ids = []
        for _ in range(batch_size):
            record = self.rng.choice(self.records)
            rows.append(_window_tokens(record["text"], self.vocab_size, self.seq_len, self.rng))
            category_ids.append(self.category_to_id[record["category"]])
        batch = torch.tensor(rows, dtype=torch.long, device=device)
        categories = torch.tensor(category_ids, dtype=torch.long, device=device)
        return batch[:, :-1], batch[:, 1:], categories


def run_program_specialization_objectives(
    *,
    train_jsonl: str | Path,
    eval_jsonl: str | Path,
    variants: list[str],
    steps: int,
    batch_size: int,
    eval_records_per_category: int,
    train_records_per_category: int,
    seed: int,
    device: str | torch.device,
    vocab_size: int = 512,
    seq_len: int = 128,
    d_model: int = 32,
    n_layers: int = 1,
    n_heads: int = 4,
    n_programs: int = 12,
    learning_rate: float = 3e-4,
    include_knockouts: bool = False,
) -> dict[str, Any]:
    device = _select_device(device)
    runs = []
    for variant in variants:
        settings = VARIANTS[variant]
        torch.manual_seed(seed)
        batcher = LabeledTextBatcher(
            train_jsonl,
            vocab_size=vocab_size,
            seq_len=seq_len,
            max_records_per_category=train_records_per_category,
            seed=seed,
        )
        config = best_tac_config(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_programs=n_programs,
            max_seq_len=seq_len,
            energy_budget=4.0,
            **settings["overrides"],
        )
        model = TACTransformerLM(config).to(device)
        optimizer = build_tac_optimizer(
            model,
            TACOptimizerConfig(learning_rate=learning_rate),
        )
        aux_weights = {
            name: _default_aux_weight(name, model)
            for name in [
                "coherence",
                "program_reuse",
                "energy",
                "multi_token",
                "separation",
                "content_cue_separation",
                "content_gate_entropy",
                "routing_load_balance",
            ]
        }
        latest = {"loss": 0.0, "category_route_loss": 0.0}
        for _ in range(steps):
            input_ids, labels, category_ids = batcher.next_batch(batch_size, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids, labels=labels)
            if settings["category_objective"] == "selected_mi":
                route_loss = selected_program_mi_loss(
                    output.aux.program_activations,
                    output.aux.selected_program_mask,
                    category_ids,
                    n_categories=len(batcher.categories),
                )
            elif settings["category_objective"] == "mi":
                route_loss = category_program_mi_loss(
                    output.aux.token_program_activations,
                    category_ids,
                    n_categories=len(batcher.categories),
                )
            else:
                route_loss = category_route_loss(
                    output.aux.token_program_activations,
                    category_ids,
                )
            aux_loss = _weighted_auxiliary_loss(output, aux_weights)
            loss = output.loss + aux_loss + float(settings["category_route_weight"]) * route_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            latest = {
                "loss": float(loss.detach()),
                "category_route_loss": float(route_loss.detach()),
            }
        eval_report = evaluate_specialization(
            model,
            eval_jsonl,
            vocab_size=vocab_size,
            seq_len=seq_len,
            max_records_per_category=eval_records_per_category,
            device=device,
            include_knockouts=include_knockouts,
        )
        runs.append(
            {
                "variant": variant,
                "config_overrides": settings["overrides"],
                "category_route_weight": settings["category_route_weight"],
                "category_objective": settings["category_objective"],
                "latest_train": latest,
                **eval_report,
            }
        )
    ranked = sorted(
        runs,
        key=lambda row: (
            row["raw_activation_mi"]["normalized_mi"],
            row["raw_activation_mi"]["mi_bits"],
            -row["mean_eval_loss"],
        ),
        reverse=True,
    )
    return {
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        "include_knockouts": include_knockouts,
        "variants": variants,
        "runs": runs,
        "ranking_by_raw_activation_nmi": ranked,
    }


def run_program_specialization_objectives_multi_seed(
    *,
    train_jsonl: str | Path,
    eval_jsonl: str | Path,
    variants: list[str],
    steps: int,
    batch_size: int,
    eval_records_per_category: int,
    train_records_per_category: int,
    seeds: list[int],
    device: str | torch.device,
    include_knockouts: bool = False,
) -> dict[str, Any]:
    seed_reports = [
        run_program_specialization_objectives(
            train_jsonl=train_jsonl,
            eval_jsonl=eval_jsonl,
            variants=variants,
            steps=steps,
            batch_size=batch_size,
            eval_records_per_category=eval_records_per_category,
            train_records_per_category=train_records_per_category,
            seed=seed,
            device=device,
            include_knockouts=include_knockouts,
        )
        for seed in seeds
    ]
    rows = [
        {**row, "seed": report["seed"]}
        for report in seed_reports
        for row in report["runs"]
    ]
    by_variant = {
        variant: _aggregate_variant_rows(
            [row for row in rows if row["variant"] == variant]
        )
        for variant in variants
    }
    ranking_by_raw_activation_nmi = sorted(
        by_variant.values(),
        key=lambda row: (
            row["mean_raw_activation_nmi"],
            row["mean_raw_activation_mi_bits"],
            -row["mean_eval_loss"],
        ),
        reverse=True,
    )
    ranking_by_selected_route_nmi = sorted(
        by_variant.values(),
        key=lambda row: (
            row["mean_selected_route_nmi"],
            row["mean_selected_route_mi_bits"],
            -row["mean_eval_loss"],
        ),
        reverse=True,
    )
    return {
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "steps": steps,
        "batch_size": batch_size,
        "seeds": seeds,
        "include_knockouts": include_knockouts,
        "variants": variants,
        "seed_reports": seed_reports,
        "by_variant": by_variant,
        "ranking_by_raw_activation_nmi": ranking_by_raw_activation_nmi,
        "ranking_by_selected_route_nmi": ranking_by_selected_route_nmi,
    }


def _aggregate_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    result = {
        "variant": rows[0]["variant"],
        "runs": len(rows),
        "category_objective": rows[0].get("category_objective", "fixed"),
        "category_route_weight": rows[0]["category_route_weight"],
        "config_overrides": rows[0]["config_overrides"],
        "mean_eval_loss": mean(row["mean_eval_loss"] for row in rows),
        "mean_category_route_loss": mean(
            row["latest_train"]["category_route_loss"] for row in rows
        ),
        "mean_raw_activation_mi_bits": mean(
            row["raw_activation_mi"]["mi_bits"] for row in rows
        ),
        "mean_raw_activation_nmi": mean(
            row["raw_activation_mi"]["normalized_mi"] for row in rows
        ),
        "mean_raw_activation_entropy": mean(
            row["raw_activation_mi"]["program_entropy_bits"] for row in rows
        ),
        "mean_selected_route_mi_bits": mean(
            row["selected_route_mi"]["mi_bits"] for row in rows
        ),
        "mean_selected_route_nmi": mean(
            row["selected_route_mi"]["normalized_mi"] for row in rows
        ),
        "mean_selected_route_entropy": mean(
            row["selected_route_mi"]["program_entropy_bits"] for row in rows
        ),
        "seeds": [row["seed"] for row in rows],
    }
    knockout_spans = [
        max(
            (
                float(item["selectivity_span"])
                for item in row.get("knockout_selectivity", [])
            ),
            default=0.0,
        )
        for row in rows
    ]
    if any(span != 0.0 for span in knockout_spans):
        result["mean_top_knockout_selectivity_span"] = mean(knockout_spans)
        result["top_knockout_selectivity_by_seed"] = knockout_spans
    return result


def _default_aux_weight(name: str, model: TACTransformerLM) -> float:
    config = model.config
    weights = {
        "coherence": 0.05,
        "program_reuse": 0.05,
        "energy": 0.01,
        "multi_token": getattr(config, "multi_token_loss_weight", 0.0),
        "separation": getattr(config, "memory_separation_weight", 0.0),
        "content_cue_separation": getattr(config, "content_cue_separation_weight", 0.0),
        "content_gate_entropy": getattr(config, "content_gate_entropy_weight", 0.0),
        "routing_load_balance": getattr(config, "routing_load_balance_weight", 0.0),
    }
    return weights.get(name, 0.0)


def evaluate_specialization(
    model: TACTransformerLM,
    jsonl_path: str | Path,
    *,
    vocab_size: int,
    seq_len: int,
    max_records_per_category: int,
    device: torch.device,
    include_knockouts: bool = False,
) -> dict[str, Any]:
    records = _load_records(Path(jsonl_path), max_records_per_category=max_records_per_category)
    categories = sorted({record["category"] for record in records})
    category_index = {category: index for index, category in enumerate(categories)}
    raw_counts = [[0 for _ in range(model.config.n_programs)] for _ in categories]
    selected_counts = [[0 for _ in range(model.config.n_programs)] for _ in categories]
    losses = []
    baseline_losses: dict[int, float] = {}
    with torch.no_grad():
        for record_index, record in enumerate(records):
            tokens = _window_tokens(record["text"], vocab_size, seq_len, random.Random(0))
            tensor = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]
            output = model(tensor[:, :-1], labels=tensor[:, 1:])
            loss_value = float(output.loss.detach()) if output.loss is not None else 0.0
            losses.append(loss_value)
            baseline_losses[record_index] = loss_value
            raw_probs = _normalize(output.aux.token_program_activations[0])
            selected_probs = _normalize(
                output.aux.token_program_activations[0]
                * output.aux.token_selected_program_mask[0]
            )
            row = category_index[record["category"]]
            for value in raw_probs.argmax(dim=-1).detach().cpu().tolist():
                raw_counts[row][int(value)] += 1
            for value in selected_probs.argmax(dim=-1).detach().cpu().tolist():
                selected_counts[row][int(value)] += 1
    report = {
        "records": len(records),
        "categories": categories,
        "mean_eval_loss": mean(losses) if losses else 0.0,
        "raw_activation_mi": _mutual_information_from_counts(raw_counts, categories),
        "selected_route_mi": _mutual_information_from_counts(selected_counts, categories),
        "raw_activation_counts": raw_counts,
        "selected_route_counts": selected_counts,
    }
    if include_knockouts:
        category_knockouts = _evaluate_category_knockouts(
            model,
            records,
            baseline_losses,
            vocab_size=vocab_size,
            seq_len=seq_len,
            device=device,
        )
        report["category_knockouts"] = category_knockouts
        report["knockout_selectivity"] = _knockout_selectivity(category_knockouts)
    return report


def _evaluate_category_knockouts(
    model: TACTransformerLM,
    records: list[dict[str, str]],
    baseline_losses: dict[int, float],
    *,
    vocab_size: int,
    seq_len: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for program_id in range(model.config.n_programs):
        deltas = []
        by_category: dict[str, list[float]] = defaultdict(list)
        with _knockout_program(model, program_id):
            with torch.no_grad():
                for record_index, record in enumerate(records):
                    tokens = _window_tokens(
                        record["text"],
                        vocab_size,
                        seq_len,
                        random.Random(0),
                    )
                    tensor = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]
                    output = model(tensor[:, :-1], labels=tensor[:, 1:])
                    loss_value = float(output.loss.detach()) if output.loss is not None else 0.0
                    delta = loss_value - baseline_losses[record_index]
                    deltas.append(delta)
                    by_category[record["category"]].append(delta)
        rows.append(
            {
                "program": program_id,
                "loss_delta": mean(deltas) if deltas else 0.0,
                "by_category": {
                    category: {
                        "loss_delta": mean(values),
                        "records": len(values),
                    }
                    for category, values in sorted(by_category.items())
                },
            }
        )
    return rows


def _knockout_selectivity(knockouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for knockout in knockouts:
        by_category = {
            category: float(values["loss_delta"])
            for category, values in knockout.get("by_category", {}).items()
        }
        if not by_category:
            rows.append(
                {
                    "program": int(knockout["program"]),
                    "preferred_category": None,
                    "avoided_category": None,
                    "selectivity_span": 0.0,
                    "by_category": by_category,
                }
            )
            continue
        preferred = max(by_category, key=by_category.get)
        avoided = min(by_category, key=by_category.get)
        rows.append(
            {
                "program": int(knockout["program"]),
                "preferred_category": preferred,
                "avoided_category": avoided,
                "selectivity_span": by_category[preferred] - by_category[avoided],
                "by_category": by_category,
            }
        )
    return rows


@contextmanager
def _knockout_program(model: TACTransformerLM, program_id: int) -> Iterator[None]:
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
            masked_weights[..., program_id] = 0.0
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


def _load_records(path: Path, *, max_records_per_category: int) -> list[dict[str, str]]:
    records = []
    seen: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            category = str(row.get("domain") or "")
            text = str(row.get("text") or "")
            if not category or not text or seen[category] >= max_records_per_category:
                continue
            seen[category] += 1
            records.append({"category": category, "text": text})
    return records


def _window_tokens(text: str, vocab_size: int, seq_len: int, rng: random.Random) -> list[int]:
    tokens = [byte + 4 for byte in text.encode("utf-8", errors="replace")]
    tokens = [token for token in tokens if token < vocab_size] + [3]
    needed = seq_len + 1
    if len(tokens) >= needed:
        start = rng.randrange(len(tokens) - seq_len)
        return tokens[start : start + needed]
    return tokens + [3] * (needed - len(tokens))


def _normalize(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(0.0)
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _mutual_information_from_counts(
    counts: list[list[int]],
    categories: list[str],
) -> dict[str, Any]:
    n_programs = len(counts[0]) if counts else 0
    total = sum(sum(row) for row in counts)
    if total == 0:
        return {"mi_bits": 0.0, "normalized_mi": 0.0, "counts": counts}
    category_totals = [sum(row) for row in counts]
    program_totals = [sum(counts[c][p] for c in range(len(counts))) for p in range(n_programs)]
    mi = 0.0
    for category_id, row in enumerate(counts):
        for program_id, value in enumerate(row):
            if value == 0:
                continue
            joint = value / total
            category_prob = category_totals[category_id] / total
            program_prob = program_totals[program_id] / total
            mi += joint * math.log2(joint / (category_prob * program_prob))
    category_entropy = _entropy_bits(category_totals)
    program_entropy = _entropy_bits(program_totals)
    denominator = min(category_entropy, program_entropy)
    return {
        "mi_bits": mi,
        "normalized_mi": 0.0 if denominator == 0.0 else mi / denominator,
        "category_entropy_bits": category_entropy,
        "program_entropy_bits": program_entropy,
        "counts": counts,
        "categories": categories,
        "programs": list(range(n_programs)),
    }


def _entropy_bits(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count == 0:
            continue
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


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
        description="Screen objective-level TAC program-specialization solutions."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-records-per-category", type=int, default=64)
    parser.add_argument("--eval-records-per-category", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--include-knockouts", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seeds is None:
        report = run_program_specialization_objectives(
            train_jsonl=args.train_jsonl,
            eval_jsonl=args.eval_jsonl,
            variants=args.variants,
            steps=args.steps,
            batch_size=args.batch_size,
            train_records_per_category=args.train_records_per_category,
            eval_records_per_category=args.eval_records_per_category,
            seed=args.seed,
            device=args.device,
            include_knockouts=args.include_knockouts,
        )
    else:
        report = run_program_specialization_objectives_multi_seed(
            train_jsonl=args.train_jsonl,
            eval_jsonl=args.eval_jsonl,
            variants=args.variants,
            steps=args.steps,
            batch_size=args.batch_size,
            train_records_per_category=args.train_records_per_category,
            eval_records_per_category=args.eval_records_per_category,
            seeds=args.seeds,
            device=args.device,
            include_knockouts=args.include_knockouts,
        )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
