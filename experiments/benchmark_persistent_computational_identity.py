from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/persistent_computational_identity_2026_06_05")
RULES = ["copy", "successor", "predecessor", "affine_jump"]
LOW_TOKEN = 4


@dataclass(frozen=True)
class IdentitySession:
    identity_id: str
    rule: str
    support_inputs: list[int]
    support_targets: list[int]
    query_inputs: list[int]
    query_targets: list[int]


def apply_rule_bank(
    values: torch.Tensor,
    programs: Sequence[str],
    *,
    vocab_size: int,
) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("values must be a 1D tensor")
    if len(programs) != int(values.numel()):
        raise ValueError("programs length must match values")
    span = _span(vocab_size)
    normalized = values.to(torch.long) - LOW_TOKEN
    outputs = []
    for value, program in zip(normalized.tolist(), programs):
        if program == "copy":
            transformed = value
        elif program == "successor":
            transformed = value + 1
        elif program == "predecessor":
            transformed = value - 1
        elif program == "affine_jump":
            transformed = value * 5 + 2
        else:
            raise ValueError(f"unknown program: {program}")
        outputs.append(LOW_TOKEN + (int(transformed) % span))
    return torch.tensor(outputs, dtype=torch.long, device=values.device)


def build_identity_probe_suite(
    *,
    seeds: Sequence[int],
    identities_per_seed: int,
    queries_per_identity: int,
    vocab_size: int,
) -> dict[str, Any]:
    if identities_per_seed < len(RULES) or identities_per_seed % len(RULES) != 0:
        raise ValueError("identities_per_seed must be a positive multiple of rule count")
    if queries_per_identity < 1:
        raise ValueError("queries_per_identity must be positive")
    support_values = _diagnostic_support_values(vocab_size)
    sessions: list[IdentitySession] = []
    for seed in seeds:
        for identity_index in range(identities_per_seed):
            rule = RULES[identity_index % len(RULES)]
            identity_id = f"seed{int(seed)}_identity{identity_index:02d}"
            support_inputs = support_values
            support_targets = _targets(support_inputs, rule, vocab_size)
            query_inputs = _query_values(
                seed=int(seed),
                identity_index=identity_index,
                count=queries_per_identity,
                vocab_size=vocab_size,
                forbidden=set(support_values),
            )
            query_targets = _targets(query_inputs, rule, vocab_size)
            sessions.append(
                IdentitySession(
                    identity_id=identity_id,
                    rule=rule,
                    support_inputs=support_inputs,
                    support_targets=support_targets,
                    query_inputs=query_inputs,
                    query_targets=query_targets,
                )
            )
    return {
        "schema": "persistent_identity_probe_suite.v1",
        "rules": list(RULES),
        "vocab_size": int(vocab_size),
        "seeds": [int(seed) for seed in seeds],
        "identities_per_seed": int(identities_per_seed),
        "queries_per_identity": int(queries_per_identity),
        "support_examples_per_identity": len(support_values),
        "sessions": [asdict(session) for session in sessions],
    }


def evaluate_persistent_identity_solver(suite: dict[str, Any]) -> dict[str, Any]:
    correct = 0
    total = 0
    inferred_rules: dict[str, str] = {}
    for session in suite["sessions"]:
        inferred = infer_rule_from_support(
            session["support_inputs"],
            session["support_targets"],
            vocab_size=suite["vocab_size"],
        )
        inferred_rules[session["identity_id"]] = inferred
        predictions = _targets(session["query_inputs"], inferred, suite["vocab_size"])
        for prediction, target in zip(predictions, session["query_targets"]):
            correct += int(prediction == target)
            total += 1
    return {
        "schema": "persistent_identity_solver.v1",
        "accuracy": correct / float(total),
        "correct_count": correct,
        "example_count": total,
        "inferred_rule_count": len(inferred_rules),
    }


def evaluate_stateless_reset_solver(suite: dict[str, Any]) -> dict[str, Any]:
    prior_rule = _majority_rule(suite)
    correct = 0
    total = 0
    for session in suite["sessions"]:
        predictions = _targets(session["query_inputs"], prior_rule, suite["vocab_size"])
        for prediction, target in zip(predictions, session["query_targets"]):
            correct += int(prediction == target)
            total += 1
    return {
        "schema": "stateless_reset_solver.v1",
        "chosen_prior_rule": prior_rule,
        "accuracy": correct / float(total),
        "correct_count": correct,
        "example_count": total,
    }


def evaluate_global_persistent_solver(suite: dict[str, Any]) -> dict[str, Any]:
    last = suite["sessions"][-1]
    global_rule = infer_rule_from_support(
        last["support_inputs"],
        last["support_targets"],
        vocab_size=suite["vocab_size"],
    )
    correct = 0
    total = 0
    for session in suite["sessions"]:
        predictions = _targets(session["query_inputs"], global_rule, suite["vocab_size"])
        for prediction, target in zip(predictions, session["query_targets"]):
            correct += int(prediction == target)
            total += 1
    return {
        "schema": "global_persistent_without_identity_solver.v1",
        "global_rule_after_overwrite": global_rule,
        "accuracy": correct / float(total),
        "correct_count": correct,
        "example_count": total,
    }


def evaluate_memory_only_solver(suite: dict[str, Any]) -> dict[str, Any]:
    correct = 0
    total = 0
    missing = 0
    for session in suite["sessions"]:
        memory = dict(zip(session["support_inputs"], session["support_targets"]))
        for query, target in zip(session["query_inputs"], session["query_targets"]):
            total += 1
            if query not in memory:
                missing += 1
                continue
            correct += int(memory[query] == target)
    return {
        "schema": "memory_only_identity_solver.v1",
        "accuracy": correct / float(total),
        "correct_count": correct,
        "example_count": total,
        "missing_prediction_count": missing,
        "missing_prediction_rate": missing / float(total),
    }


def infer_rule_from_support(
    support_inputs: Sequence[int],
    support_targets: Sequence[int],
    *,
    vocab_size: int,
) -> str:
    matches = []
    for rule in RULES:
        if _targets(support_inputs, rule, vocab_size) == list(support_targets):
            matches.append(rule)
    if len(matches) != 1:
        raise ValueError(f"support examples do not identify a unique rule: {matches}")
    return matches[0]


def run_persistent_identity_probe(
    *,
    seeds: Sequence[int],
    identities_per_seed: int,
    queries_per_identity: int,
    vocab_size: int,
) -> dict[str, Any]:
    suite = build_identity_probe_suite(
        seeds=seeds,
        identities_per_seed=identities_per_seed,
        queries_per_identity=queries_per_identity,
        vocab_size=vocab_size,
    )
    persistent = evaluate_persistent_identity_solver(suite)
    stateless = evaluate_stateless_reset_solver(suite)
    global_only = evaluate_global_persistent_solver(suite)
    memory_only = evaluate_memory_only_solver(suite)
    best_non_identity = max(
        stateless["accuracy"],
        global_only["accuracy"],
        memory_only["accuracy"],
    )
    metrics = {
        "persistent_identity_accuracy": persistent["accuracy"],
        "stateless_reset_accuracy": stateless["accuracy"],
        "global_persistent_without_identity_accuracy": global_only["accuracy"],
        "memory_only_unseen_accuracy": memory_only["accuracy"],
        "memory_only_missing_prediction_rate": memory_only["missing_prediction_rate"],
        "best_non_identity_accuracy": best_non_identity,
        "persistent_advantage_over_best_non_identity": (
            persistent["accuracy"] - best_non_identity
        ),
        "support_examples_per_identity": suite["support_examples_per_identity"],
        "heldout_queries": persistent["example_count"],
    }
    theorem = theorem_bound_for_suite(suite)
    decision = _decision(metrics)
    return {
        "schema": "persistent_computational_identity.v1",
        "intelligence_metric": (
            "Held-out exact-match accuracy on latent-rule tasks where support "
            "observations reveal an identity-specific computation and later "
            "queries omit the rule."
        ),
        "suite_summary": {
            "seeds": suite["seeds"],
            "rule_families": suite["rules"],
            "identity_count": len(suite["sessions"]),
            "identities_per_seed": suite["identities_per_seed"],
            "queries_per_identity": suite["queries_per_identity"],
            "vocab_size": suite["vocab_size"],
        },
        "theorem": theorem,
        "controls": {
            "persistent_identity": persistent,
            "stateless_reset": stateless,
            "global_persistent_without_identity": global_only,
            "memory_only_without_computation": memory_only,
        },
        "metrics": metrics,
        "decision": decision,
    }


def theorem_bound_for_suite(suite: dict[str, Any]) -> dict[str, Any]:
    counts = {rule: 0 for rule in RULES}
    for session in suite["sessions"]:
        counts[session["rule"]] += 1
    total = sum(counts.values())
    max_prior = max(counts.values()) / float(total)
    return {
        "schema": "persistent_identity_constructive_bound.v1",
        "assumptions": [
            "The hidden computation rule is identity-specific.",
            "Later query prompts omit the hidden rule and contain held-out input values.",
            "Rule families are balanced and query values are chosen so rule outputs are disambiguated.",
            "Diagnostic support examples uniquely identify the rule for a persistent identity state.",
        ],
        "rule_count": len(RULES),
        "rule_prior": {rule: counts[rule] / float(total) for rule in RULES},
        "stateless_upper_bound": max_prior,
        "constructive_persistent_accuracy": 1.0,
        "proved_advantage_lower_bound": 1.0 - max_prior,
        "claim": (
            "For this task family, a reset/stateless policy that does not see "
            "identity history cannot infer the hidden rule and is bounded by "
            "the best rule prior. A persistent computational identity can store "
            "the inferred rule per identity and apply it to held-out queries."
        ),
    }


def format_persistent_identity_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    theorem = result["theorem"]
    lines = [
        "# Persistent Computational Identity",
        "",
        f"- Decision: `{result['decision']['status']}`",
        f"- Reason: {result['decision']['reason']}",
        f"- Intelligence metric: {result['intelligence_metric']}",
        "",
        "## Theorem Bound",
        "",
        f"- Rule count: `{theorem['rule_count']}`",
        f"- Stateless upper bound: `{theorem['stateless_upper_bound']:.4f}`",
        f"- Constructive persistent accuracy: `{theorem['constructive_persistent_accuracy']:.4f}`",
        f"- Proved advantage lower bound: `{theorem['proved_advantage_lower_bound']:.4f}`",
        "",
        "## Empirical Controls",
        "",
        "| Control | Accuracy |",
        "| --- | ---: |",
        f"| Persistent identity | {metrics['persistent_identity_accuracy']:.4f} |",
        f"| Stateless reset | {metrics['stateless_reset_accuracy']:.4f} |",
        f"| Global persistent without identity | {metrics['global_persistent_without_identity_accuracy']:.4f} |",
        f"| Memory only without computation | {metrics['memory_only_unseen_accuracy']:.4f} |",
        "",
        "## Interpretation",
        "",
        result["decision"]["recommendation"],
        "",
        "## Boundary",
        "",
        (
            "This proves the bounded task family and validates the measurement "
            "harness. It does not prove that every persistent-state mechanism "
            "improves every intelligence benchmark, and it does not yet prove "
            "that current external TAC checkpoints have learned the mechanism."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Prove a bounded persistent computational identity advantage."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--identities-per-seed", type=int, default=8)
    parser.add_argument("--queries-per-identity", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=64)
    args = parser.parse_args(argv)

    result = run_persistent_identity_probe(
        seeds=args.seeds,
        identities_per_seed=args.identities_per_seed,
        queries_per_identity=args.queries_per_identity,
        vocab_size=args.vocab_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "persistent_computational_identity.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_persistent_identity_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2))
    return result


def _decision(metrics: dict[str, Any]) -> dict[str, Any]:
    persistent_ok = metrics["persistent_identity_accuracy"] >= 0.95
    stateless_ok = metrics["stateless_reset_accuracy"] <= 0.30
    memory_ok = metrics["memory_only_unseen_accuracy"] <= 0.05
    advantage_ok = metrics["persistent_advantage_over_best_non_identity"] >= 0.45
    if persistent_ok and stateless_ok and memory_ok and advantage_ok:
        return {
            "status": "persistent_computational_identity_proved",
            "reason": (
                "Persistent identity cleared the held-out latent-computation "
                "task while reset, non-identity persistence, and memory-only "
                "controls remained bounded."
            ),
            "recommendation": (
                "Treat persistent computational identity as a required capability "
                "for tasks with identity-specific latent rules across episodes. "
                "Use it selectively with verifier and cost gates before promoting "
                "it as a default model path."
            ),
        }
    return {
        "status": "persistent_computational_identity_not_proved",
        "reason": (
            "One or more empirical gates failed: persistent accuracy, reset "
            "bound, memory-only bound, or advantage over non-identity controls."
        ),
        "recommendation": (
            "Do not promote the mechanism from this run. Inspect the failed "
            "control and revise the benchmark or identity update mechanism."
        ),
    }


def _targets(values: Sequence[int], rule: str, vocab_size: int) -> list[int]:
    tensor = torch.tensor(list(values), dtype=torch.long)
    return apply_rule_bank(tensor, [rule] * len(values), vocab_size=vocab_size).tolist()


def _majority_rule(suite: dict[str, Any]) -> str:
    counts = {rule: 0 for rule in RULES}
    for session in suite["sessions"]:
        counts[session["rule"]] += 1
    return max(RULES, key=lambda rule: (counts[rule], -RULES.index(rule)))


def _diagnostic_support_values(vocab_size: int) -> list[int]:
    span = _span(vocab_size)
    values = [LOW_TOKEN + 1, LOW_TOKEN + 6]
    if any(value >= vocab_size for value in values) or span < 16:
        raise ValueError("vocab_size must be at least 20 for diagnostic support")
    return values


def _query_values(
    *,
    seed: int,
    identity_index: int,
    count: int,
    vocab_size: int,
    forbidden: set[int],
) -> list[int]:
    span = _span(vocab_size)
    values: list[int] = []
    cursor = (seed * 13 + identity_index * 17 + 9) % span
    while len(values) < count:
        candidate = LOW_TOKEN + cursor
        cursor = (cursor + 7) % span
        if candidate in forbidden:
            continue
        values.append(candidate)
    return values


def _span(vocab_size: int) -> int:
    span = int(vocab_size) - LOW_TOKEN
    if span < 16:
        raise ValueError("vocab_size must leave at least sixteen non-special tokens")
    return span


if __name__ == "__main__":
    main()
