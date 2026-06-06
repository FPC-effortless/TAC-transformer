from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM


def evaluate_forced_programs(
    checkpoint: str | Path,
    jsonl_path: str | Path,
    *,
    max_records_per_category: int | None = 8,
    batch_size: int = 4,
    programs: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device = _select_device(device)
    checkpoint_path = Path(checkpoint)
    model, config, checkpoint_data = _load_checkpoint_model(checkpoint_path, device)
    records = _load_labeled_records(
        Path(jsonl_path),
        max_records_per_category=max_records_per_category,
    )
    if not records:
        raise ValueError(f"no labeled records found in {jsonl_path}")

    selected_programs = programs if programs is not None else list(range(config.n_programs))
    for program_id in selected_programs:
        if program_id < 0 or program_id >= config.n_programs:
            raise ValueError(f"program_id out of range: {program_id}")

    natural = _evaluate_records(
        model,
        config,
        records,
        batch_size=batch_size,
        device=device,
        forced_program=None,
    )
    forced = []
    for program_id in selected_programs:
        forced.append(
            _evaluate_records(
                model,
                config,
                records,
                batch_size=batch_size,
                device=device,
                forced_program=program_id,
            )
        )
    forced = [_attach_loss_deltas(row, natural) for row in forced]
    return _forced_program_report(
        checkpoint_path=checkpoint_path,
        checkpoint_data=checkpoint_data,
        jsonl_path=Path(jsonl_path),
        records=records,
        config=config,
        batch_size=batch_size,
        max_records_per_category=max_records_per_category,
        natural=natural,
        forced=forced,
    )


def evaluate_forced_programs_incremental(
    checkpoint: str | Path,
    jsonl_path: str | Path,
    *,
    output_dir: str | Path,
    max_records_per_category: int | None = 8,
    batch_size: int = 4,
    programs: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    device = _select_device(device)
    checkpoint_path = Path(checkpoint)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model, config, checkpoint_data = _load_checkpoint_model(checkpoint_path, device)
    records = _load_labeled_records(
        Path(jsonl_path),
        max_records_per_category=max_records_per_category,
    )
    if not records:
        raise ValueError(f"no labeled records found in {jsonl_path}")

    selected_programs = programs if programs is not None else list(range(config.n_programs))
    for program_id in selected_programs:
        if program_id < 0 or program_id >= config.n_programs:
            raise ValueError(f"program_id out of range: {program_id}")

    natural = _evaluate_records(
        model,
        config,
        records,
        batch_size=batch_size,
        device=device,
        forced_program=None,
    )
    (output_path / "natural.json").write_text(
        json.dumps(natural, indent=2) + "\n",
        encoding="utf-8",
    )

    forced = []
    for program_id in selected_programs:
        row = _evaluate_records(
            model,
            config,
            records,
            batch_size=batch_size,
            device=device,
            forced_program=program_id,
        )
        row = _attach_loss_deltas(row, natural)
        forced.append(row)
        (output_path / f"program_{program_id:02d}.json").write_text(
            json.dumps(row, indent=2) + "\n",
            encoding="utf-8",
        )
        partial_report = _forced_program_report(
            checkpoint_path=checkpoint_path,
            checkpoint_data=checkpoint_data,
            jsonl_path=Path(jsonl_path),
            records=records,
            config=config,
            batch_size=batch_size,
            max_records_per_category=max_records_per_category,
            natural=natural,
            forced=forced,
        )
        partial_report["complete"] = len(forced) == len(selected_programs)
        partial_report["completed_programs"] = [int(row["program"]) for row in forced]
        (output_path / "partial_report.json").write_text(
            json.dumps(partial_report, indent=2) + "\n",
            encoding="utf-8",
        )

    final_report = _forced_program_report(
        checkpoint_path=checkpoint_path,
        checkpoint_data=checkpoint_data,
        jsonl_path=Path(jsonl_path),
        records=records,
        config=config,
        batch_size=batch_size,
        max_records_per_category=max_records_per_category,
        natural=natural,
        forced=forced,
    )
    final_report["complete"] = True
    final_report["completed_programs"] = [int(row["program"]) for row in forced]
    return final_report


def _forced_program_report(
    *,
    checkpoint_path: Path,
    checkpoint_data: dict[str, Any],
    jsonl_path: Path,
    records: list[dict[str, Any]],
    config: TACConfig,
    batch_size: int,
    max_records_per_category: int | None,
    natural: dict[str, Any],
    forced: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(forced, key=lambda row: row["loss"])
    natural_loss = float(natural["loss"])
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_data.get("step", 0)),
        "best_eval_loss": _optional_float(checkpoint_data.get("best_eval_loss")),
        "data": str(jsonl_path),
        "records": len(records),
        "categories": sorted({record["category"] for record in records}),
        "max_records_per_category": max_records_per_category,
        "batch_size": batch_size,
        "config": {
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "n_layers": config.n_layers,
            "n_programs": config.n_programs,
            "routing_type": config.routing_type,
            "memory_read_type": config.memory_read_type,
            "identity_attention_type": config.identity_attention_type,
        },
        "natural": natural,
        "forced_programs": forced,
        "ranking_by_loss": ranked,
        "category_program_rankings": _category_program_rankings(natural, forced),
        "summary": {
            "best_forced_program": ranked[0]["program"] if ranked else None,
            "best_forced_loss": ranked[0]["loss"] if ranked else None,
            "worst_forced_program": ranked[-1]["program"] if ranked else None,
            "worst_forced_loss": ranked[-1]["loss"] if ranked else None,
            "forced_loss_range": (
                ranked[-1]["loss"] - ranked[0]["loss"] if len(ranked) > 1 else 0.0
            ),
            "forced_loss_std": _std([row["loss"] for row in forced]),
            "forced_loss_variance": _variance([row["loss"] for row in forced]),
            "best_minus_natural_loss": (
                ranked[0]["loss"] - natural_loss if ranked else None
            ),
            "worst_minus_natural_loss": (
                ranked[-1]["loss"] - natural_loss if ranked else None
            ),
            "forced_programs_better_than_natural": sum(
                1 for row in forced if row["loss"] < natural_loss
            ),
        },
    }


def _attach_loss_deltas(
    forced_row: dict[str, Any],
    natural_row: dict[str, Any],
) -> dict[str, Any]:
    row = dict(forced_row)
    natural_loss = float(natural_row["loss"])
    row["loss_delta_vs_natural"] = float(row["loss"]) - natural_loss
    by_category = {}
    for category, values in row.get("by_category", {}).items():
        category_values = dict(values)
        natural_category_loss = float(
            natural_row.get("by_category", {})
            .get(category, {})
            .get("loss", natural_loss)
        )
        category_values["loss_delta_vs_natural"] = (
            float(category_values["loss"]) - natural_category_loss
        )
        by_category[category] = category_values
    row["by_category"] = by_category
    return row


def _category_program_rankings(
    natural: dict[str, Any],
    forced: list[dict[str, Any]],
) -> dict[str, Any]:
    rankings = {}
    categories = sorted(natural.get("by_category", {}).keys())
    for category in categories:
        rows = [
            {
                "program": int(row["program"]),
                "loss": float(row["by_category"][category]["loss"]),
                "loss_delta_vs_natural": float(
                    row["by_category"][category]["loss_delta_vs_natural"]
                ),
            }
            for row in forced
            if category in row.get("by_category", {})
        ]
        ranked = sorted(rows, key=lambda row: row["loss"])
        natural_loss = float(natural["by_category"][category]["loss"])
        rankings[category] = {
            "natural_loss": natural_loss,
            "best_program": ranked[0]["program"] if ranked else None,
            "best_forced_loss": ranked[0]["loss"] if ranked else None,
            "best_delta_vs_natural": (
                ranked[0]["loss_delta_vs_natural"] if ranked else None
            ),
            "worst_program": ranked[-1]["program"] if ranked else None,
            "worst_forced_loss": ranked[-1]["loss"] if ranked else None,
            "worst_delta_vs_natural": (
                ranked[-1]["loss_delta_vs_natural"] if ranked else None
            ),
            "forced_loss_range": (
                ranked[-1]["loss"] - ranked[0]["loss"] if len(ranked) > 1 else 0.0
            ),
            "forced_loss_std": _std([row["loss"] for row in ranked]),
            "forced_loss_variance": _variance([row["loss"] for row in ranked]),
            "ranking_by_loss": ranked,
        }
    return rankings


def _evaluate_records(
    model: TACTransformerLM,
    config: TACConfig,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    forced_program: int | None,
) -> dict[str, Any]:
    losses = []
    correct = 0.0
    total = 0
    routing_load_std = []
    program_memory_cosine = []
    by_category: dict[str, list[float]] = defaultdict(list)
    natural_top_counts = [0 for _ in range(config.n_programs)]
    context = _force_program(model, forced_program) if forced_program is not None else _nullcontext()
    with context:
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                input_ids, labels = _batch_tensors(
                    [record["text"] for record in batch],
                    config,
                    device,
                )
                output = model(input_ids, labels=labels)
                per_token_loss = F.cross_entropy(
                    output.logits.reshape(-1, config.vocab_size),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape(labels.shape)
                per_record_loss = per_token_loss.mean(dim=-1)
                losses.extend(float(value) for value in per_record_loss.detach().cpu())
                predictions = output.logits.argmax(dim=-1)
                correct += float((predictions == labels).float().sum().detach())
                total += labels.numel()
                routing_load_std.append(
                    float(
                        output.aux.metrics.get(
                            "routing_load_std",
                            output.logits.new_zeros(()),
                        ).detach()
                    )
                )
                program_memory_cosine.append(
                    float(
                        output.aux.metrics.get(
                            "program_memory_cosine",
                            output.logits.new_zeros(()),
                        ).detach()
                    )
                )
                if forced_program is None:
                    scores = output.aux.program_activations * output.aux.selected_program_mask
                    fallback = scores.sum(dim=-1, keepdim=True) <= 0
                    scores = torch.where(fallback, output.aux.program_activations, scores)
                    top_programs = scores.argmax(dim=-1).detach().cpu().tolist()
                    for program_id in top_programs:
                        natural_top_counts[int(program_id)] += 1
                for record, loss in zip(batch, per_record_loss.detach().cpu().tolist()):
                    by_category[record["category"]].append(float(loss))

    result: dict[str, Any] = {
        "program": forced_program,
        "mode": "natural" if forced_program is None else "forced",
        "records": len(records),
        "loss": mean(losses) if losses else 0.0,
        "token_accuracy": correct / max(total, 1),
        "routing_load_std": mean(routing_load_std) if routing_load_std else 0.0,
        "program_memory_cosine": mean(program_memory_cosine) if program_memory_cosine else 0.0,
        "by_category": {
            category: {
                "loss": mean(values),
                "records": len(values),
            }
            for category, values in sorted(by_category.items())
        },
    }
    if forced_program is None:
        result["top_program_counts"] = natural_top_counts
        total_counts = sum(natural_top_counts)
        result["top_program_distribution"] = [
            count / max(total_counts, 1)
            for count in natural_top_counts
        ]
    return result


@contextmanager
def _force_program(model: TACTransformerLM, program_id: int) -> Iterator[None]:
    originals = []
    for block in model.blocks:
        identity = block.identity_field
        original = identity._route_programs
        originals.append((identity, original))

        def patched(self, stability, activations=None, *, _program_id=program_id):
            del activations
            routed = torch.zeros_like(stability)
            routed[..., _program_id] = 1.0
            return routed

        identity._route_programs = MethodType(patched, identity)
    try:
        yield
    finally:
        for identity, original in originals:
            identity._route_programs = original


@contextmanager
def _nullcontext() -> Iterator[None]:
    yield


def _load_labeled_records(
    path: Path,
    *,
    max_records_per_category: int | None,
) -> list[dict[str, Any]]:
    records = []
    seen: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            category = str(row.get("domain") or _infer_category(text) or "")
            if not category:
                continue
            if max_records_per_category is not None and seen[category] >= max_records_per_category:
                continue
            seen[category] += 1
            records.append(
                {
                    "id": str(row.get("record_id") or f"{category}_{seen[category]}"),
                    "category": category,
                    "text": text,
                }
            )
    return records


def _batch_tensors(
    texts: list[str],
    config: TACConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    rows = []
    needed = config.max_seq_len + 1
    for text in texts:
        tokens = _encode_text(text, config.vocab_size) + [3]
        if len(tokens) < needed:
            tokens = tokens + [3] * (needed - len(tokens))
        rows.append(tokens[:needed])
    tensor = torch.tensor(rows, dtype=torch.long, device=device)
    return tensor[:, :-1], tensor[:, 1:]


def _encode_text(text: str, vocab_size: int) -> list[int]:
    tokens = [byte + 4 for byte in text.encode("utf-8", errors="replace")]
    return [token for token in tokens if token < vocab_size]


def _infer_category(text: str) -> str | None:
    match = re.search(r'<record type="hard_([^"]+)">', text)
    if match:
        return match.group(1)
    return None


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


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return (sum((value - average) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return sum((value - average) ** 2 for value in values) / (len(values) - 1)


def write_forced_program_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "program",
                "category",
                "records",
                "loss",
                "loss_delta_vs_natural",
                "token_accuracy",
            ],
        )
        writer.writeheader()
        for row in report.get("forced_programs", []):
            writer.writerow(
                {
                    "program": row["program"],
                    "category": "__all__",
                    "records": row["records"],
                    "loss": row["loss"],
                    "loss_delta_vs_natural": row["loss_delta_vs_natural"],
                    "token_accuracy": row["token_accuracy"],
                }
            )
            for category, values in row.get("by_category", {}).items():
                writer.writerow(
                    {
                        "program": row["program"],
                        "category": category,
                        "records": values["records"],
                        "loss": values["loss"],
                        "loss_delta_vs_natural": values["loss_delta_vs_natural"],
                        "token_accuracy": "",
                    }
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TAC checkpoint while forcing every token through one program at a time."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--max-records-per-category", type=int, default=8)
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="Ignore --max-records-per-category and evaluate all labeled records.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--programs", type=int, nargs="+", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument(
        "--incremental-dir",
        type=Path,
        default=None,
        help="Write natural/program partial JSON files after each forced program.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print the JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    kwargs = {
        "max_records_per_category": None if args.all_records else args.max_records_per_category,
        "batch_size": args.batch_size,
        "programs": args.programs,
        "device": args.device,
    }
    if args.incremental_dir is not None:
        report = evaluate_forced_programs_incremental(
            args.checkpoint,
            args.jsonl,
            output_dir=args.incremental_dir,
            **kwargs,
        )
    else:
        report = evaluate_forced_programs(
            args.checkpoint,
            args.jsonl,
            **kwargs,
        )
    text = json.dumps(report, indent=2)
    if not args.quiet:
        print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.csv_output is not None:
        write_forced_program_csv(report, args.csv_output)


if __name__ == "__main__":
    main()
