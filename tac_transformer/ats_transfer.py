from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import torch

from .phase_d_benchmarks import (
    extract_phase_d_answer,
    generate_phase_d_completion,
    load_phase_d_checkpoint_model,
)


ATS_TRAIN_DOMAINS = ("navigation", "inventory")
ATS_TEST_DOMAINS = ("lab_protocol", "incident_response")
ATS_TASK_IDS = ("cross_domain_identity_transfer", "two_program_sequential")

_DOMAIN_TERMS = {
    "navigation": {
        "entity": "waypoint",
        "relation": "route",
        "processor": "mapper",
        "resolver": "pilot",
        "prefix": "nav",
    },
    "inventory": {
        "entity": "part",
        "relation": "bin",
        "processor": "scanner",
        "resolver": "packer",
        "prefix": "inv",
    },
    "lab_protocol": {
        "entity": "sample",
        "relation": "assay",
        "processor": "prep",
        "resolver": "reader",
        "prefix": "lab",
    },
    "incident_response": {
        "entity": "ticket",
        "relation": "escalation",
        "processor": "triage",
        "resolver": "responder",
        "prefix": "ir",
    },
}


def build_ats_transfer_suite(
    *,
    seed: int,
    examples_per_domain: int = 4,
    train_domains: Sequence[str] = ATS_TRAIN_DOMAINS,
    test_domains: Sequence[str] = ATS_TEST_DOMAINS,
) -> dict[str, Any]:
    """Build an ATS-style transfer suite with disjoint train/test domains."""

    if examples_per_domain <= 0:
        raise ValueError("examples_per_domain must be positive")
    train_domain_list = tuple(str(domain) for domain in train_domains)
    test_domain_list = tuple(str(domain) for domain in test_domains)
    overlap = set(train_domain_list) & set(test_domain_list)
    if overlap:
        raise ValueError(f"train/test domains must be disjoint: {sorted(overlap)}")
    for domain in train_domain_list + test_domain_list:
        if domain not in _DOMAIN_TERMS:
            raise ValueError(f"unknown ATS transfer domain: {domain}")

    rng = random.Random(int(seed))
    examples: list[dict[str, Any]] = []
    for split, domains in (("train", train_domain_list), ("test", test_domain_list)):
        for domain in domains:
            for index in range(int(examples_per_domain)):
                invariant = f"stable_identity_{index % max(examples_per_domain, 1)}"
                examples.append(
                    _build_cross_domain_identity_example(
                        seed=seed,
                        split=split,
                        domain=domain,
                        index=index,
                        invariant=invariant,
                        rng=rng,
                    )
                )
                examples.append(
                    _build_two_program_sequential_example(
                        seed=seed,
                        split=split,
                        domain=domain,
                        index=index,
                        invariant=invariant,
                        rng=rng,
                    )
                )

    return {
        "phase": "D",
        "schema": "ats_transfer_suite.v1",
        "seed": int(seed),
        "train_domains": list(train_domain_list),
        "test_domains": list(test_domain_list),
        "task_ids": list(ATS_TASK_IDS),
        "examples_per_domain": int(examples_per_domain),
        "example_count": len(examples),
        "examples": examples,
    }


def build_ats_oracle_predictions(
    examples: Iterable[Mapping[str, Any]],
    *,
    control_id: str,
) -> list[dict[str, Any]]:
    """Return exact answers for every ATS example as a positive control."""

    return [
        {
            "example_id": str(example["id"]),
            "control_id": str(control_id),
            "prediction": str(example["answer"]),
        }
        for example in examples
    ]


def build_ats_surface_baseline_predictions(
    examples: Iterable[Mapping[str, Any]],
    *,
    control_id: str,
) -> list[dict[str, Any]]:
    """Surface baseline: memorizes train-domain surfaces and fails held-out domains."""

    rows = []
    for example in examples:
        split = str(example.get("split"))
        rows.append(
            {
                "example_id": str(example["id"]),
                "control_id": str(control_id),
                "prediction": str(example["answer"]) if split == "train" else "",
            }
        )
    return rows


def score_ats_transfer_predictions(
    examples: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Score ATS transfer predictions by control, split, and task."""

    example_rows = [dict(example) for example in examples]
    prediction_by_key = {
        (str(row.get("control_id")), str(row.get("example_id"))): row
        for row in predictions
        if isinstance(row, Mapping)
    }
    controls = sorted({control for control, _ in prediction_by_key})
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for control_id in controls:
        for example in example_rows:
            grouped[
                (
                    control_id,
                    str(example.get("split")),
                    str(example.get("task_id")),
                )
            ].append(example)

    rows = []
    for (control_id, split, task_id), task_examples in sorted(grouped.items()):
        correct = 0
        missing = 0
        for example in task_examples:
            prediction = prediction_by_key.get((control_id, str(example["id"])))
            if prediction is None:
                missing += 1
                continue
            correct += int(
                _normalise_answer(prediction.get("prediction"))
                == _normalise_answer(example.get("answer"))
            )
        total = len(task_examples)
        rows.append(
            {
                "schema": "ats_transfer_score_row.v1",
                "control_id": control_id,
                "split": split,
                "task_id": task_id,
                "primary_metric": "exact_match",
                "primary_score": correct / total if total else 0.0,
                "correct_count": correct,
                "example_count": total,
                "missing_prediction_count": missing,
            }
        )
    return rows


def run_ats_checkpoint_predictions(
    *,
    examples: Iterable[Mapping[str, Any]],
    checkpoint_path: str | Path,
    control_id: str,
    seed: int,
    output_jsonl: str | Path | None = None,
    model_type: str = "auto",
    device: str | torch.device = "cpu",
    precision: str = "fp32",
    max_new_tokens: int = 32,
    answer_extraction: str = "first_token",
) -> dict[str, Any]:
    """Run a TAC or vanilla checkpoint over ATS transfer examples."""

    example_rows = [dict(example) for example in examples]
    model, metadata = load_phase_d_checkpoint_model(
        checkpoint_path,
        model_type=model_type,
        device=device,
    )
    rows = []
    for example in example_rows:
        result = generate_phase_d_completion(
            model,
            str(example.get("prompt", "")),
            max_new_tokens=max_new_tokens,
            device=device,
            precision=precision,
        )
        raw_completion = str(result["completion"])
        rows.append(
            {
                "schema": "ats_transfer_checkpoint_prediction.v1",
                "example_id": str(example["id"]),
                "task_id": str(example.get("task_id")),
                "family": str(example.get("family")),
                "split": str(example.get("split")),
                "domain": str(example.get("domain")),
                "control_id": str(control_id),
                "seed": int(seed),
                "prediction": extract_phase_d_answer(
                    raw_completion,
                    mode=answer_extraction,
                ),
                "raw_completion": raw_completion,
                "answer_extraction": answer_extraction,
                "generated_token_count": int(result["generated_token_count"]),
                "prompt_token_count": int(result["prompt_token_count"]),
                "truncated_prompt_token_count": int(
                    result["truncated_prompt_token_count"]
                ),
                "context_window": int(result["context_window"]),
                "tokens_per_second": float(result["tokens_per_second"]),
                "wall_clock_seconds": float(result["wall_clock_seconds"]),
                "checkpoint": metadata["checkpoint"],
                "checkpoint_step": metadata["checkpoint_step"],
                "model_type": metadata["model_type"],
            }
        )
    if output_jsonl is not None:
        _write_jsonl(Path(output_jsonl), rows)
    return {
        "schema": "ats_transfer_checkpoint_predictions.v1",
        "seed": int(seed),
        "control_id": str(control_id),
        "checkpoint": metadata["checkpoint"],
        "checkpoint_step": metadata["checkpoint_step"],
        "model_type": metadata["model_type"],
        "prediction_count": len(rows),
        "answer_extraction": answer_extraction,
        "max_new_tokens": int(max_new_tokens),
        "rows": rows,
    }


def ats_example_to_prepared_row(example: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an ATS transfer example to the existing prepared JSONL contract."""

    prompt = str(example.get("prompt", ""))
    answer = str(example.get("answer", ""))
    text = f"{prompt}{answer}\n"
    return {
        "record_id": str(example["id"]),
        "source": "ats_transfer_supervised",
        "domain": str(example.get("domain")),
        "split": str(example.get("split")),
        "task_id": str(example.get("task_id")),
        "family": str(example.get("family")),
        "latent_invariant": str(example.get("latent_invariant")),
        "answer": answer,
        "text": text,
    }


def stage_ats_transfer_training_corpus(
    *,
    output_dir: str | Path,
    seed: int,
    examples_per_domain: int = 128,
    train_domains: Sequence[str] = ATS_TRAIN_DOMAINS,
    test_domains: Sequence[str] = ATS_TEST_DOMAINS,
) -> dict[str, Any]:
    """Write ATS transfer train/eval prepared JSONL files for existing trainers."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suite = build_ats_transfer_suite(
        seed=seed,
        examples_per_domain=examples_per_domain,
        train_domains=train_domains,
        test_domains=test_domains,
    )
    rows = [ats_example_to_prepared_row(example) for example in suite["examples"]]
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "test"]
    train_path = output / "train.prepared.jsonl"
    eval_path = output / "eval.prepared.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    (output / "ats_transfer_suite.json").write_text(
        json.dumps(suite, indent=2) + "\n",
        encoding="utf-8",
    )

    train_domain_set = set(suite["train_domains"])
    test_domain_set = set(suite["test_domains"])
    duplicate_count = len(rows) - len({row["record_id"] for row in rows})
    leakage = {
        "test_domain_rows_in_train": sum(
            1 for row in train_rows if row["domain"] in test_domain_set
        ),
        "train_domain_rows_in_eval": sum(
            1 for row in eval_rows if row["domain"] in train_domain_set
        ),
        "duplicate_record_ids": duplicate_count,
    }
    max_prompt_bytes = max(
        (len(str(example["prompt"]).encode("utf-8")) for example in suite["examples"]),
        default=0,
    )
    max_answer_bytes = max(
        (len(str(example["answer"]).encode("utf-8")) for example in suite["examples"]),
        default=0,
    )
    max_text_bytes = max(
        (len(str(row["text"]).encode("utf-8")) for row in rows),
        default=0,
    )
    checks = {
        "has_train_records": bool(train_rows),
        "has_eval_records": bool(eval_rows),
        "no_domain_leakage": (
            leakage["test_domain_rows_in_train"] == 0
            and leakage["train_domain_rows_in_eval"] == 0
        ),
        "unique_record_ids": duplicate_count == 0,
        "fits_256_byte_context": max_text_bytes <= 256,
    }
    manifest = {
        "schema": "ats_transfer_training_corpus.v1",
        "seed": int(seed),
        "examples_per_domain": int(examples_per_domain),
        "train_domains": suite["train_domains"],
        "test_domains": suite["test_domains"],
        "task_ids": suite["task_ids"],
        "train_records": len(train_rows),
        "eval_records": len(eval_rows),
        "total_records": len(rows),
        "train_jsonl": str(train_path),
        "eval_jsonl": str(eval_path),
        "suite_json": str(output / "ats_transfer_suite.json"),
        "max_prompt_bytes": max_prompt_bytes,
        "max_answer_bytes": max_answer_bytes,
        "max_text_bytes": max_text_bytes,
        "leakage": leakage,
        "recommended_commands": _ats_training_commands(train_path, eval_path),
        "decision": {
            "status": (
                "ats_transfer_training_corpus_staged"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
        },
    }
    (output / "ats_transfer_training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(
        _format_ats_training_corpus_markdown(manifest),
        encoding="utf-8",
    )
    return manifest


def aggregate_ats_transfer_results(
    score_rows: Iterable[Mapping[str, Any]],
    *,
    oracle_control_id: str = "identity_oracle",
    baseline_control_id: str = "surface_baseline",
    min_oracle_test_score: float = 0.95,
    min_baseline_train_score: float = 0.95,
    max_baseline_test_score: float = 0.25,
) -> dict[str, Any]:
    rows = [dict(row) for row in score_rows]
    controls: dict[str, dict[str, Any]] = {}
    for control_id in sorted({str(row["control_id"]) for row in rows}):
        control_rows = [row for row in rows if str(row["control_id"]) == control_id]
        split_summary = {}
        for split in sorted({str(row["split"]) for row in control_rows}):
            split_rows = [row for row in control_rows if str(row["split"]) == split]
            split_summary[split] = {
                "mean_score": mean(float(row["primary_score"]) for row in split_rows),
                "task_count": len(split_rows),
                "tasks": {
                    str(row["task_id"]): float(row["primary_score"])
                    for row in split_rows
                },
            }
        controls[control_id] = {"splits": split_summary}

    oracle_test = _split_score(controls, oracle_control_id, "test")
    baseline_train = _split_score(controls, baseline_control_id, "train")
    baseline_test = _split_score(controls, baseline_control_id, "test")
    checks = {
        "oracle_transfers_to_test": oracle_test >= min_oracle_test_score,
        "surface_baseline_learns_train": baseline_train >= min_baseline_train_score,
        "surface_baseline_fails_test": baseline_test <= max_baseline_test_score,
        "required_tasks_present": set(ATS_TASK_IDS).issubset(
            {str(row["task_id"]) for row in rows}
        ),
    }
    return {
        "schema": "ats_transfer_aggregate.v1",
        "controls": controls,
        "score_rows": rows,
        "decision": {
            "status": (
                "ats_transfer_benchmark_valid"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "thresholds": {
                "min_oracle_test_score": min_oracle_test_score,
                "min_baseline_train_score": min_baseline_train_score,
                "max_baseline_test_score": max_baseline_test_score,
            },
            "scope": (
                "This validates the benchmark and scoring contract, not a TAC "
                "capability win. The identity oracle and surface baseline are "
                "controls used to prove that held-out domain transfer is measured."
            ),
        },
    }


def aggregate_ats_checkpoint_run_results(
    checkpoint_runs: Iterable[Mapping[str, Any]],
    *,
    tac_control_id: str = "tac_base_ats_5k",
    vanilla_control_id: str = "vanilla_base_ats_5k",
    min_train_score: float = 0.95,
    min_tac_test_score: float = 0.95,
    min_tac_test_advantage: float = 0.10,
) -> dict[str, Any]:
    """Aggregate TAC-vs-vanilla ATS checkpoint runs into a promotion decision."""

    runs = [dict(run) for run in checkpoint_runs if isinstance(run, Mapping)]
    controls: dict[str, dict[str, Any]] = {}
    all_score_rows: list[dict[str, Any]] = []
    for run in runs:
        control_id = str(run.get("control_id") or "")
        if not control_id:
            continue
        score_rows = [
            dict(row)
            for row in run.get("score_rows", [])
            if isinstance(row, Mapping)
        ]
        all_score_rows.extend(score_rows)
        controls[control_id] = {
            "model_type": str(run.get("model_type") or ""),
            "checkpoint": str(run.get("checkpoint") or ""),
            "checkpoint_step": run.get("checkpoint_step"),
            "prediction_count": int(run.get("prediction_count") or 0),
            "splits": _split_summary(score_rows),
        }

    missing_controls = [
        control_id
        for control_id in (tac_control_id, vanilla_control_id)
        if control_id not in controls
    ]
    tac_train = _split_score(controls, tac_control_id, "train")
    tac_test = _split_score(controls, tac_control_id, "test")
    vanilla_train = _split_score(controls, vanilla_control_id, "train")
    vanilla_test = _split_score(controls, vanilla_control_id, "test")
    test_advantage = tac_test - vanilla_test
    missing_predictions = sum(
        int(row.get("missing_prediction_count") or 0) for row in all_score_rows
    )
    observed_tasks = {str(row.get("task_id")) for row in all_score_rows}
    checks = {
        "required_controls_present": not missing_controls,
        "tac_learns_train": tac_train >= min_train_score,
        "vanilla_control_learns_train": vanilla_train >= min_train_score,
        "tac_transfers_to_test": tac_test >= min_tac_test_score,
        "tac_beats_vanilla_test": test_advantage >= min_tac_test_advantage,
        "no_missing_predictions": missing_predictions == 0,
        "required_tasks_present": set(ATS_TASK_IDS).issubset(observed_tasks),
    }
    if missing_controls:
        status = "pending"
        reason = "Missing required TAC or vanilla checkpoint run outputs."
    elif all(checks.values()):
        status = "ats_external_transfer_promote"
        reason = "TAC cleared train fit, held-out transfer, and vanilla advantage gates."
    else:
        status = "ats_external_transfer_fail"
        reason = "Required external ATS transfer gates did not all pass."

    return {
        "schema": "ats_checkpoint_run_aggregate.v1",
        "controls": controls,
        "score_rows": all_score_rows,
        "decision": {
            "status": status,
            "reason": reason,
            "checks": checks,
            "missing_controls": missing_controls,
            "metrics": {
                "tac_train_score": tac_train,
                "tac_test_score": tac_test,
                "vanilla_train_score": vanilla_train,
                "vanilla_test_score": vanilla_test,
                "tac_test_advantage": test_advantage,
                "missing_prediction_count": missing_predictions,
            },
            "thresholds": {
                "min_train_score": min_train_score,
                "min_tac_test_score": min_tac_test_score,
                "min_tac_test_advantage": min_tac_test_advantage,
            },
            "scope": (
                "This is the external TAC-175 checkpoint comparison gate. "
                "It requires both TAC and parameter-matched vanilla runs before "
                "claiming pass or fail, and it only promotes TAC when held-out "
                "ATS exact-match transfer clears threshold and beats vanilla."
            ),
        },
    }


def format_ats_checkpoint_run_markdown(aggregate: Mapping[str, Any]) -> str:
    decision = aggregate["decision"]
    lines = [
        "# ATS Checkpoint Transfer Comparison",
        "",
        f"Decision: `{decision['status']}`",
        "",
        f"Reason: {decision['reason']}",
        "",
        "| Control | Model | Step | Train | Test |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for control_id, control in aggregate.get("controls", {}).items():
        splits = control.get("splits", {})
        train = _format_score(splits.get("train", {}).get("mean_score"))
        test = _format_score(splits.get("test", {}).get("mean_score"))
        step = control.get("checkpoint_step")
        lines.append(
            "| {control} | {model} | {step} | {train} | {test} |".format(
                control=control_id,
                model=control.get("model_type") or "unknown",
                step="n/a" if step is None else step,
                train=train,
                test=test,
            )
        )
    metrics = decision.get("metrics", {})
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- TAC test advantage: `{_format_score(metrics.get('tac_test_advantage'))}`",
            f"- Missing predictions: `{metrics.get('missing_prediction_count', 0)}`",
            "",
            "## Scope",
            "",
            decision["scope"],
            "",
        ]
    )
    return "\n".join(lines)


def write_ats_transfer_artifacts(
    *,
    output_dir: str | Path,
    suite: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "ats_transfer_suite.json").write_text(
        json.dumps(suite, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output / "ats_transfer_predictions.jsonl", predictions)
    _write_jsonl(output / "ats_transfer_score_rows.jsonl", score_rows)
    (output / "ats_transfer_aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(
        format_ats_transfer_markdown(aggregate),
        encoding="utf-8",
    )


def format_ats_transfer_markdown(aggregate: Mapping[str, Any]) -> str:
    decision = aggregate["decision"]
    lines = [
        "# ATS Transfer Benchmark",
        "",
        f"Decision: `{decision['status']}`",
        "",
        "| Control | Train | Test |",
        "| --- | ---: | ---: |",
    ]
    for control_id, control in aggregate.get("controls", {}).items():
        splits = control.get("splits", {})
        train = _format_score(splits.get("train", {}).get("mean_score"))
        test = _format_score(splits.get("test", {}).get("mean_score"))
        lines.append(f"| {control_id} | {train} | {test} |")
    lines.extend(["", "## Scope", "", decision["scope"], ""])
    return "\n".join(lines)


def _split_summary(score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    split_summary = {}
    for split in sorted({str(row["split"]) for row in score_rows}):
        split_rows = [row for row in score_rows if str(row["split"]) == split]
        split_summary[split] = {
            "mean_score": mean(float(row["primary_score"]) for row in split_rows),
            "task_count": len(split_rows),
            "tasks": {
                str(row["task_id"]): float(row["primary_score"])
                for row in split_rows
            },
        }
    return split_summary


def _build_cross_domain_identity_example(
    *,
    seed: int,
    split: str,
    domain: str,
    index: int,
    invariant: str,
    rng: random.Random,
) -> dict[str, Any]:
    terms = _DOMAIN_TERMS[domain]
    source = f"{terms['prefix']}_{terms['entity']}_{rng.randrange(100, 999)}"
    answer = f"{terms['prefix']}_identity_{index}_{rng.randrange(1000, 9999)}"
    distractor = f"{terms['prefix']}_distractor_{rng.randrange(1000, 9999)}"
    prompt = (
        f"<d>{domain}\n"
        f"<i>{invariant}\n"
        f"<map>{source}->{answer}\n"
        f"<noise>{terms['relation']}->{distractor}\n"
        "<q>identity token?\n"
        "<a>"
    )
    return {
        "id": f"ats_{split}_{domain}_identity_{seed}_{index:04d}",
        "task_id": "cross_domain_identity_transfer",
        "family": "ats_transfer",
        "split": split,
        "domain": domain,
        "latent_invariant": invariant,
        "prompt": prompt,
        "answer": answer,
        "metadata": {
            "source": source,
            "distractor": distractor,
            "requires_transfer": split == "test",
        },
    }


def _build_two_program_sequential_example(
    *,
    seed: int,
    split: str,
    domain: str,
    index: int,
    invariant: str,
    rng: random.Random,
) -> dict[str, Any]:
    terms = _DOMAIN_TERMS[domain]
    observation = f"{terms['prefix']}_obs_{rng.randrange(100, 999)}"
    bridge = f"{terms['prefix']}_bridge_{index}_{rng.randrange(100, 999)}"
    answer = f"{terms['prefix']}_final_{index}_{rng.randrange(1000, 9999)}"
    prompt = (
        f"<d>{domain}\n"
        f"<i>{invariant}\n"
        f"<p1>{terms['processor']}:{observation}->{bridge}\n"
        f"<p2>{terms['resolver']}:{bridge}->{answer}\n"
        "<q>final token?\n"
        "<a>"
    )
    return {
        "id": f"ats_{split}_{domain}_sequential_{seed}_{index:04d}",
        "task_id": "two_program_sequential",
        "family": "ood_multi_step",
        "split": split,
        "domain": domain,
        "latent_invariant": invariant,
        "prompt": prompt,
        "answer": answer,
        "metadata": {
            "observation": observation,
            "bridge": bridge,
            "programs": [terms["processor"], terms["resolver"]],
            "requires_coalition": True,
            "requires_transfer": split == "test",
        },
    }


def _split_score(
    controls: Mapping[str, Any],
    control_id: str,
    split: str,
) -> float:
    try:
        return float(controls[control_id]["splits"][split]["mean_score"])
    except KeyError:
        return 0.0


def _normalise_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip("`'\" .,:;")


def _format_score(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _ats_training_commands(train_path: Path, eval_path: Path) -> dict[str, str]:
    common = (
        f"--train-jsonl {train_path} "
        f"--eval-jsonl {eval_path} "
        "--scale base --seq-len 176 --steps 5000 --device auto"
    )
    return {
        "tac_base": (
            "python kaggle/train_best_tac_agentic.py "
            f"{common} "
            "--precision fp32 --min-healthy-gradient-norm 1e-12 "
            "--fail-on-unhealthy-optimization "
            "--routing-type base_semantic --routing-top-k 2 "
            "--category-route-weight 0.5 --category-route-objective selected_mi"
        ),
        "vanilla_parameter_matched_base": (
            "python kaggle/train_vanilla_baseline.py "
            f"{common} --baseline-mode parameter_matched"
        ),
        "score_checkpoint": (
            "python experiments/run_ats_checkpoint_predictions.py "
            "--suite-json <output_dir>/ats_transfer_suite.json "
            "--checkpoint <checkpoint.pt> --control-id <control_id> "
            "--output-dir <score_output_dir> --max-new-tokens 24"
        ),
    }


def _format_ats_training_corpus_markdown(manifest: Mapping[str, Any]) -> str:
    decision = manifest["decision"]
    leakage = manifest["leakage"]
    lines = [
        "# ATS Transfer Training Corpus",
        "",
        f"Decision: `{decision['status']}`",
        "",
        f"- Train records: `{manifest['train_records']}`",
        f"- Eval records: `{manifest['eval_records']}`",
        f"- Train domains: `{', '.join(manifest['train_domains'])}`",
        f"- Eval domains: `{', '.join(manifest['test_domains'])}`",
        f"- Max text bytes: `{manifest['max_text_bytes']}`",
        f"- Test-domain rows in train: `{leakage['test_domain_rows_in_train']}`",
        f"- Train-domain rows in eval: `{leakage['train_domain_rows_in_eval']}`",
        "",
        "## Commands",
        "",
    ]
    for name, command in manifest.get("recommended_commands", {}).items():
        lines.extend([f"### {name}", "", "```bash", str(command), "```", ""])
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
