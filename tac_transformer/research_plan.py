"""Research-plan contracts for advancing the TAC Run 5B result.

The helpers in this module are intentionally plain-data builders.  They make
the next TAC research stage auditable by freezing the exact Run 5B reference
configuration, carrying unresolved gaps forward, and emitting replication and
capability-evaluation protocols that can be generated from local artifacts.
"""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Iterable


TAC_CONTROL_V1_CONFIG: dict[str, Any] = {
    "identity_attention_type": "identity_first",
    "memory_read_type": "content_addressed",
    "content_read_steps": 2,
    "content_read_gate_type": "synthesis",
    "program_memory_update_type": "program_conditioned",
    "memory_allocation_type": "creb",
    "memory_allocation_k": 6,
    "memory_separation_weight": 0.1,
    "routing_type": "base_semantic",
    "routing_top_k": 2,
    "category_route_objective": "mi",
    "category_route_weight": 0.1,
}

TAC_CONTROL_V1_REFERENCE = {
    "name": "TAC-Control-v1",
    "source": "Run 5B program-conditioned CREB-k6 checkpoint",
    "checkpoint_step": 10000,
    "tokens_seen": 183_600_000,
}

PHASE_B_DEFAULT_SEEDS = (11, 23, 37)

PHASE_B_SUCCESS_CRITERIA = {
    "program_memory_cosine_max": 0.25,
    "selected_route_mi_min": 0.15,
    "max_knockout_loss_delta_min": 0.05,
    "eval_accuracy_min": 0.93,
}

PHASE_C_SUCCESS_CRITERIA = {
    "min_seed_count": 2,
    "min_alignment_similarity": 0.80,
}


def audit_tac_control_v1(
    manifest: dict[str, Any],
    tac_summary: dict[str, Any],
    external_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Phase A freeze audit for TAC-Control-v1.

    Parameters are parsed from already-loaded JSON artifacts.  Missing optional
    fields produce open evidence notes instead of exceptions so the same audit
    can be used while remote artifacts are still arriving.
    """

    latest_metrics = _coerce_dict(tac_summary.get("latest_metrics"))
    train_metrics = _coerce_dict(latest_metrics.get("train"))
    eval_metrics = _coerce_dict(latest_metrics.get("eval"))
    specialization = _specialization_metrics(external_validation or {}, tac_summary)

    config_checks = {
        key: {
            "expected": expected,
            "actual": _manifest_value(manifest, key),
            "passes": _values_match(_manifest_value(manifest, key), expected),
        }
        for key, expected in TAC_CONTROL_V1_CONFIG.items()
    }

    content_hit = _first_number(
        eval_metrics.get("content_addressed_hit"),
        train_metrics.get("content_addressed_hit"),
        latest_metrics.get("content_addressed_hit"),
    )
    content_gate = _first_number(
        eval_metrics.get("content_synthesis_gate"),
        train_metrics.get("content_synthesis_gate"),
        latest_metrics.get("content_synthesis_gate"),
    )
    content_live = (
        config_checks["memory_read_type"]["passes"]
        and content_hit is not None
        and content_hit > 0.0
        and content_gate is not None
        and content_gate > 0.0
    )
    identity_first_live = config_checks["identity_attention_type"]["passes"]

    checkpoint_step = _first_int(
        tac_summary.get("checkpoint_step"),
        tac_summary.get("completed_steps"),
        latest_metrics.get("step"),
        TAC_CONTROL_V1_REFERENCE["checkpoint_step"],
    )
    tokens_seen = _first_int(
        tac_summary.get("checkpoint_tokens_seen"),
        tac_summary.get("tokens_seen"),
        latest_metrics.get("tokens_seen"),
        TAC_CONTROL_V1_REFERENCE["tokens_seen"],
    )

    optimization_health = _optimization_health(tac_summary, latest_metrics)
    route_mi = _first_number(
        specialization.get("selected_route_mi"),
        specialization.get("mi_bits"),
        specialization.get("route_mi"),
    )
    program_memory_cosine = _first_number(
        specialization.get("program_memory_cosine"),
        specialization.get("program_memory_cosine_max"),
        specialization.get("program_memory_cosine_mean"),
        eval_metrics.get("program_memory_cosine"),
        train_metrics.get("program_memory_cosine"),
        latest_metrics.get("program_memory_cosine"),
    )
    max_knockout_delta = _first_number(
        specialization.get("max_knockout_loss_delta"),
        specialization.get("knockout_loss_delta_max"),
    )

    all_config_passed = all(check["passes"] for check in config_checks.values())
    specialization_present = route_mi is not None and route_mi > 0.0
    fair_checkpoint = checkpoint_step == TAC_CONTROL_V1_REFERENCE["checkpoint_step"]
    freeze_ready = (
        all_config_passed
        and content_live
        and identity_first_live
        and specialization_present
        and fair_checkpoint
    )

    resolved_gaps = {
        "content_addressed_store": {
            "status": "resolved" if content_live else "unresolved",
            "evidence": {
                "memory_read_type": config_checks["memory_read_type"]["actual"],
                "content_addressed_hit": content_hit,
                "content_synthesis_gate": content_gate,
            },
        },
        "identity_first_path": {
            "status": "resolved" if identity_first_live else "unresolved",
            "evidence": {
                "identity_attention_type": config_checks["identity_attention_type"][
                    "actual"
                ]
            },
        },
        "fair_token_checkpoint": {
            "status": "resolved" if fair_checkpoint else "unresolved",
            "evidence": {
                "checkpoint_step": checkpoint_step,
                "tokens_seen": tokens_seen,
            },
        },
        "specialization_signal": {
            "status": "resolved" if specialization_present else "unresolved",
            "evidence": {
                "selected_route_mi": route_mi,
                "program_memory_cosine": program_memory_cosine,
                "max_knockout_loss_delta": max_knockout_delta,
            },
        },
    }

    open_gaps = {
        "multi_hop": {
            "status": "open",
            "question": "Does BASE routing still dominate multi-hop behavior under the frozen Run 5B configuration?",
            "next_action": "Run controlled multi-hop chain retrieval with TAC-Control-v1, TAC shuffled-state, and parameter-matched vanilla baselines.",
        },
        "long_context": {
            "status": "open",
            "question": "Does program specialization improve retrieval and synthesis beyond the short-context training regime?",
            "next_action": "Evaluate 2k/4k/8k retrieval ladders with matched token budgets.",
        },
        "seed_stability": {
            "status": "open",
            "question": "Are learned program identities stable across independent seeds?",
            "next_action": "Run Phase B seeds and align programs by memory vectors, route distributions, and knockout profiles.",
        },
        "decode_economics": {
            "status": "open",
            "question": "Does any capability gain justify the observed TAC decode penalty?",
            "next_action": "Pair all Phase D scores with wall-clock, token/s, and cost-normalized deltas.",
        },
    }

    return {
        "phase": "A",
        "decision": {
            "status": "freeze_ready" if freeze_ready else "blocked",
            "reason": (
                "Run 5B step 10000 satisfies the TAC-Control-v1 freeze contract."
                if freeze_ready
                else "One or more TAC-Control-v1 freeze checks are still unresolved."
            ),
        },
        "reference": {
            **TAC_CONTROL_V1_REFERENCE,
            "checkpoint_step": checkpoint_step,
            "tokens_seen": tokens_seen,
        },
        "config": config_checks,
        "frozen_config": dict(TAC_CONTROL_V1_CONFIG),
        "config_checks": config_checks,
        "metrics": {
            "eval_loss": _first_number(eval_metrics.get("loss"), eval_metrics.get("eval_loss")),
            "eval_accuracy": _first_number(
                eval_metrics.get("accuracy"), eval_metrics.get("eval_accuracy")
            ),
            "content_addressed_hit": content_hit,
            "content_synthesis_gate": content_gate,
            "selected_route_mi": route_mi,
            "program_memory_cosine": program_memory_cosine,
            "max_knockout_loss_delta": max_knockout_delta,
            "optimization_health": optimization_health,
        },
        "resolved_gaps": resolved_gaps,
        "open_gaps": open_gaps,
    }


def build_phase_b_replication_plan(
    seeds: Iterable[int] = PHASE_B_DEFAULT_SEEDS,
    output_root: str = "runs/phase_b_tac_control_v1",
) -> dict[str, Any]:
    """Return the deterministic Phase B replication plan."""

    seed_list = [int(seed) for seed in seeds]
    runs = []
    for seed in seed_list:
        run_name = f"tac_control_v1_seed_{seed}"
        runs.append(
            {
                "seed": seed,
                "name": run_name,
                "output_dir": f"{output_root}/{run_name}",
                "command": _phase_b_command(seed, f"{output_root}/{run_name}"),
            }
        )

    return {
        "phase": "B",
        "objective": "Replicate TAC-Control-v1 specialization across independent seeds before capability claims.",
        "seeds": seed_list,
        "frozen_config": dict(TAC_CONTROL_V1_CONFIG),
        "checkpoint_policy": {
            "primary_step": TAC_CONTROL_V1_REFERENCE["checkpoint_step"],
            "fair_token_budget": TAC_CONTROL_V1_REFERENCE["tokens_seen"],
            "use_best_pt_for": "loss-only diagnostic, not specialization claims",
        },
        "success_criteria": dict(PHASE_B_SUCCESS_CRITERIA),
        "runs": runs,
        "analysis": {
            "program_alignment": [
                "memory-vector cosine matching",
                "selected-route distribution matching",
                "knockout-profile matching",
            ],
            "aggregate_rule": "Report median and full seed table; do not collapse away failed seeds.",
        },
    }


def build_phase_b_kaggle_kernel(
    *,
    seed: int,
    code_dataset: str,
    data_dataset: str,
    owner: str = "jeffkolo",
    kernel_slug_prefix: str = "tac-control-v1-phase-b",
) -> dict[str, Any]:
    """Build Kaggle kernel metadata and script text for one Phase B seed."""

    seed = int(seed)
    code_file = "run_tac_control_v1_phase_b.py"
    slug = f"{kernel_slug_prefix}-seed-{seed}-20k"
    metadata = {
        "id": f"{owner}/{slug}",
        "title": f"TAC-Control-v1 Phase B Seed {seed} 20k",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [code_dataset, data_dataset],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    return {
        "seed": seed,
        "code_file": code_file,
        "metadata": metadata,
        "script": _phase_b_kaggle_script(seed),
    }


def summarize_phase_b_seed_result(
    *,
    seed: int,
    final_summary: dict[str, Any],
    metrics_rows: Iterable[dict[str, Any]] | None = None,
    run_manifest: dict[str, Any] | None = None,
    specialization_report: dict[str, Any] | None = None,
    success_criteria: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize one Phase B seed against the frozen TAC-Control-v1 gates."""

    criteria = dict(success_criteria or PHASE_B_SUCCESS_CRITERIA)
    primary_metrics = _phase_b_primary_metrics(final_summary, metrics_rows)
    eval_metrics = _coerce_dict(primary_metrics.get("eval"))
    specialization = _phase_b_specialization(final_summary, specialization_report)

    eval_accuracy = _first_number(
        eval_metrics.get("accuracy"),
        primary_metrics.get("eval_accuracy"),
        final_summary.get("eval_accuracy"),
    )
    eval_loss = _first_number(
        eval_metrics.get("loss"),
        primary_metrics.get("eval_loss"),
        final_summary.get("best_eval_loss"),
    )
    program_memory_cosine = _first_number(
        eval_metrics.get("program_memory_cosine"),
        primary_metrics.get("program_memory_cosine"),
        final_summary.get("program_memory_cosine"),
    )
    selected_route_mi = _first_number(
        specialization.get("mi_bits"),
        specialization.get("selected_route_mi"),
    )
    max_knockout_delta = _first_number(
        specialization.get("max_knockout_loss_delta"),
        specialization.get("knockout_loss_delta_max"),
    )
    tokens_seen = _first_int(
        primary_metrics.get("tokens_seen"),
        final_summary.get("tokens_seen"),
    )
    checkpoint_step = _first_int(
        primary_metrics.get("step"),
        specialization.get("checkpoint_step"),
        final_summary.get("completed_steps"),
    )

    gates = {
        "eval_accuracy": _min_gate(
            eval_accuracy,
            criteria["eval_accuracy_min"],
        ),
        "program_memory_cosine": _max_gate(
            program_memory_cosine,
            criteria["program_memory_cosine_max"],
        ),
        "selected_route_mi": _min_gate(
            selected_route_mi,
            criteria["selected_route_mi_min"],
        ),
        "max_knockout_loss_delta": _phase_b_knockout_gate(
            max_knockout_delta,
            criteria["max_knockout_loss_delta_min"],
            bool(specialization.get("run_knockouts", False)),
        ),
    }
    if run_manifest is not None:
        gates["frozen_config"] = _phase_b_config_gate(run_manifest)

    failed_gates = [
        name for name, gate in gates.items() if gate.get("status") == "fail"
    ]
    pending_gates = [
        name for name, gate in gates.items() if gate.get("status") == "pending"
    ]
    if failed_gates:
        status = "fail"
    elif pending_gates:
        status = "pending_knockout"
    else:
        status = "pass"

    evidence_gaps = [
        f"{name} evidence is pending"
        for name in pending_gates
    ]
    hard_failures = [
        f"{name} failed"
        for name in failed_gates
    ]

    return {
        "seed": int(seed),
        "status": status,
        "metrics": {
            "checkpoint_step": checkpoint_step,
            "tokens_seen": tokens_seen,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "program_memory_cosine": program_memory_cosine,
            "selected_route_mi": selected_route_mi,
            "max_knockout_loss_delta": max_knockout_delta,
            "tokens_per_second": _first_number(primary_metrics.get("tokens_per_second")),
            "optimization_health_status": _coerce_dict(
                primary_metrics.get("optimization_health")
            ).get("status"),
            "completed_steps": _first_int(final_summary.get("completed_steps")),
            "stopped_for_time": bool(final_summary.get("stopped_for_time", False)),
        },
        "gates": gates,
        "specialization": specialization,
        "evidence_gaps": evidence_gaps,
        "hard_failures": hard_failures,
    }


def aggregate_phase_b_seed_results(
    seed_results: Iterable[dict[str, Any]],
    *,
    required_pass_count: int = 2,
) -> dict[str, Any]:
    """Aggregate Phase B seed summaries into the seed-stability decision."""

    rows = list(seed_results)
    passed = [row for row in rows if row.get("status") == "pass"]
    failed = [row for row in rows if row.get("status") == "fail"]
    pending = [
        row
        for row in rows
        if row.get("status") not in {"pass", "fail"}
    ]
    ready_for_phase_d = (
        len(passed) >= required_pass_count
        and not failed
        and not pending
    )
    if ready_for_phase_d:
        status = "pass"
        reason = "Phase B has enough complete passing seed evidence for Phase D."
    elif failed:
        status = "fail"
        reason = "At least one Phase B seed failed a hard gate."
    else:
        status = "pending"
        reason = "Phase B is still waiting on seed completion or knockout evidence."

    metric_names = [
        "eval_accuracy",
        "program_memory_cosine",
        "selected_route_mi",
        "max_knockout_loss_delta",
        "tokens_per_second",
    ]
    aggregates = {
        name: _phase_b_metric_aggregate(rows, name)
        for name in metric_names
    }
    return {
        "phase": "B",
        "decision": {
            "status": status,
            "reason": reason,
            "ready_for_phase_d": ready_for_phase_d,
            "required_pass_count": required_pass_count,
        },
        "summary": {
            "seed_count": len(rows),
            "passed_seed_count": len(passed),
            "failed_seed_count": len(failed),
            "pending_seed_count": len(pending),
            "seeds": [row.get("seed") for row in rows],
        },
        "aggregates": aggregates,
        "seeds": rows,
    }


def format_phase_b_seed_results_markdown(result: dict[str, Any]) -> str:
    """Format a Phase B seed aggregation result as Markdown."""

    decision = _coerce_dict(result.get("decision"))
    summary = _coerce_dict(result.get("summary"))
    lines = [
        "# TAC-Control-v1 Phase B Seed Results",
        "",
        f"Decision: `{decision.get('status')}`",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Ready for Phase D: `{decision.get('ready_for_phase_d')}`",
        f"- Passed seeds: `{summary.get('passed_seed_count')}`",
        f"- Pending seeds: `{summary.get('pending_seed_count')}`",
        f"- Failed seeds: `{summary.get('failed_seed_count')}`",
        "",
        "| Seed | Status | Eval Acc | Memory Cos | Route MI | Knockout Delta |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result.get("seeds", []):
        metrics = _coerce_dict(row.get("metrics"))
        lines.append(
            "| Seed {seed} | {status} | {accuracy} | {cosine} | {mi} | {knockout} |".format(
                seed=row.get("seed"),
                status=row.get("status"),
                accuracy=_format_metric(metrics.get("eval_accuracy")),
                cosine=_format_metric(metrics.get("program_memory_cosine")),
                mi=_format_metric(metrics.get("selected_route_mi")),
                knockout=_format_metric(metrics.get("max_knockout_loss_delta")),
            )
        )

    pending = [
        row
        for row in result.get("seeds", [])
        if row.get("evidence_gaps")
    ]
    if pending:
        lines.extend(["", "## Evidence Gaps", ""])
        for row in pending:
            lines.append(
                f"- Seed {row.get('seed')}: {'; '.join(row.get('evidence_gaps', []))}"
            )
    return "\n".join(lines)


def build_phase_c_identity_stability_protocol() -> dict[str, Any]:
    """Return the Phase C program-identity stability protocol."""

    return {
        "phase": "C",
        "objective": "Determine whether TAC-Control-v1 learns stable program roles across independent seeds.",
        "inputs": [
            "Phase B seed final_summary.json artifacts",
            "step-10000 program_specialization.json reports",
            "program_memory_summary from post-hoc checkpoint analysis when needed",
        ],
        "alignment_components": [
            "memory_vector",
            "selected_route_distribution",
            "knockout_profile",
        ],
        "success_criteria": dict(PHASE_C_SUCCESS_CRITERIA),
        "decision_gate": "Stable program identities across at least two passing Phase B seeds",
        "blocked_by": "Phase B complete passing seed evidence",
        "reporting_rule": "Report the full alignment table; do not hide seed-specific role permutations.",
    }


def summarize_phase_c_identity_seed(
    *,
    seed: int,
    specialization_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract program-role signatures for Phase C seed-stability alignment."""

    report = _coerce_dict(specialization_report)
    categories = _phase_c_categories(report)
    route_profiles = _phase_c_route_profiles(report, categories)
    memory_profiles = _phase_c_memory_profiles(report)
    knockout_profiles = _phase_c_knockout_profiles(report, categories)
    programs = sorted(
        set(route_profiles)
        | set(memory_profiles)
        | set(knockout_profiles)
    )
    profiles = []
    for program in programs:
        route = route_profiles.get(program, {})
        knockout = knockout_profiles.get(program, {})
        preferred = None
        if route:
            preferred = max(route, key=route.get)
        profiles.append(
            {
                "program": program,
                "preferred_category": preferred,
                "route_profile": route,
                "memory_vector": memory_profiles.get(program),
                "knockout_profile": knockout,
            }
        )

    component_coverage = {
        "route": bool(route_profiles),
        "memory": bool(memory_profiles),
        "knockout": bool(knockout_profiles),
    }
    missing_components = [
        name for name, present in component_coverage.items() if not present
    ]
    if not profiles:
        status = "missing"
        reason = "No program identity profiles were found in the specialization report."
    elif missing_components:
        status = "pending_evidence"
        reason = "Program identity profiles are present but one or more alignment components are missing."
    else:
        status = "ready"
        reason = "Program identity profiles include route, memory, and knockout evidence."

    mutual_information = _coerce_dict(report.get("mutual_information"))
    return {
        "seed": int(seed),
        "status": status,
        "reason": reason,
        "checkpoint_step": _first_int(
            report.get("checkpoint_step"),
            TAC_CONTROL_V1_REFERENCE["checkpoint_step"],
        ),
        "records": len(report.get("records", [])) if isinstance(report.get("records"), list) else _first_int(report.get("records")),
        "categories": categories,
        "mi_bits": _first_number(mutual_information.get("mi_bits"), report.get("mi_bits")),
        "normalized_mi": _first_number(
            mutual_information.get("normalized_mi"),
            report.get("normalized_mi"),
        ),
        "component_coverage": component_coverage,
        "missing_components": missing_components,
        "program_profiles": profiles,
    }


def aggregate_phase_c_identity_stability_results(
    seed_profiles: Iterable[dict[str, Any]],
    *,
    phase_b_decision: dict[str, Any] | None = None,
    min_seed_count: int = PHASE_C_SUCCESS_CRITERIA["min_seed_count"],
    min_alignment_similarity: float = PHASE_C_SUCCESS_CRITERIA[
        "min_alignment_similarity"
    ],
) -> dict[str, Any]:
    """Aggregate Phase C program-identity alignment across independent seeds."""

    seeds = [row for row in seed_profiles if isinstance(row, dict)]
    ready_seeds = [
        row
        for row in seeds
        if row.get("status") == "ready" and row.get("program_profiles")
    ]
    phase_b_ready = bool(
        _coerce_dict(phase_b_decision or {}).get("ready_for_phase_d", False)
    )
    component_coverage = {
        component: bool(ready_seeds)
        and all(
            bool(_coerce_dict(row.get("component_coverage")).get(component))
            for row in ready_seeds
        )
        for component in ("route", "memory", "knockout")
    }
    missing_components = [
        name for name, present in component_coverage.items() if not present
    ]
    pairwise = []
    if len(ready_seeds) >= min_seed_count:
        reference = ready_seeds[0]
        for candidate in ready_seeds[1:]:
            pairwise.append(
                _phase_c_align_seed_pair(
                    reference,
                    candidate,
                )
            )

    similarities = [
        _first_number(row.get("mean_similarity"))
        for row in pairwise
    ]
    similarity_numbers = [value for value in similarities if value is not None]
    min_similarity = min(similarity_numbers) if similarity_numbers else None
    mean_similarity = mean(similarity_numbers) if similarity_numbers else None
    alignment_passes = (
        min_similarity is not None
        and min_similarity >= float(min_alignment_similarity)
    )

    if not phase_b_ready:
        status = "blocked_by_phase_b"
        reason = "Phase C remains blocked until Phase B has enough complete passing seed evidence."
    elif len(ready_seeds) < min_seed_count:
        status = "pending"
        reason = "Phase C is waiting for enough seeds with complete identity-profile evidence."
    elif missing_components:
        status = "pending_evidence"
        reason = "Phase C is missing route, memory, or knockout components required for identity-stability claims."
    elif alignment_passes:
        status = "pass"
        reason = "Program identities align across independent seeds above the stability threshold."
    else:
        status = "fail"
        reason = "Program identities do not align across independent seeds above the stability threshold."

    return {
        "phase": "C",
        "decision": {
            "status": status,
            "reason": reason,
            "phase_b_ready": phase_b_ready,
            "passes_identity_stability_gate": bool(
                phase_b_ready
                and len(ready_seeds) >= min_seed_count
                and not missing_components
                and alignment_passes
            ),
            "min_seed_count": min_seed_count,
            "min_alignment_similarity": min_alignment_similarity,
            "missing_components": missing_components,
        },
        "summary": {
            "seed_count": len(seeds),
            "ready_seed_count": len(ready_seeds),
            "pending_seed_count": len(seeds) - len(ready_seeds),
            "seeds": [row.get("seed") for row in seeds],
        },
        "component_coverage": component_coverage,
        "alignment": {
            "reference_seed": ready_seeds[0].get("seed") if ready_seeds else None,
            "mean_similarity": mean_similarity,
            "min_similarity": min_similarity,
        },
        "pairwise_alignments": pairwise,
        "seeds": seeds,
    }


def format_phase_c_identity_stability_markdown(result: dict[str, Any]) -> str:
    """Format Phase C identity-stability aggregation as Markdown."""

    decision = _coerce_dict(result.get("decision"))
    alignment = _coerce_dict(result.get("alignment"))
    summary = _coerce_dict(result.get("summary"))
    lines = [
        "# Phase C Identity Stability",
        "",
        f"Decision: `{decision.get('status')}`",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Passes identity-stability gate: `{decision.get('passes_identity_stability_gate')}`",
        f"- Ready seeds: `{summary.get('ready_seed_count')}`",
        f"- Mean alignment similarity: `{_format_metric(alignment.get('mean_similarity'))}`",
        f"- Minimum alignment similarity: `{_format_metric(alignment.get('min_similarity'))}`",
        "",
        "| Seed | Status | Route | Memory | Knockout | MI bits | Programs |",
        "| ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in result.get("seeds", []):
        coverage = _coerce_dict(row.get("component_coverage"))
        lines.append(
            "| Seed {seed} | {status} | {route} | {memory} | {knockout} | {mi} | {programs} |".format(
                seed=row.get("seed"),
                status=row.get("status"),
                route=coverage.get("route"),
                memory=coverage.get("memory"),
                knockout=coverage.get("knockout"),
                mi=_format_metric(row.get("mi_bits")),
                programs=len(row.get("program_profiles", [])),
            )
        )

    if result.get("pairwise_alignments"):
        lines.extend(["", "## Pairwise Alignments", ""])
        for alignment_row in result.get("pairwise_alignments", []):
            lines.append(
                "- Seed {candidate} vs Seed {reference}: mean similarity `{score}`".format(
                    candidate=alignment_row.get("candidate_seed"),
                    reference=alignment_row.get("reference_seed"),
                    score=_format_metric(alignment_row.get("mean_similarity")),
                )
            )
            for match in alignment_row.get("matches", [])[:8]:
                lines.append(
                    "  - P{reference_program} -> P{candidate_program}: `{similarity}`".format(
                        reference_program=match.get("reference_program"),
                        candidate_program=match.get("candidate_program"),
                        similarity=_format_metric(match.get("similarity")),
                    )
                )
    missing = decision.get("missing_components") or []
    if missing:
        lines.extend(["", "## Evidence Gaps", ""])
        lines.append("- Missing components: " + ", ".join(str(item) for item in missing))
    return "\n".join(lines)


def build_phase_d_benchmark_protocol() -> dict[str, Any]:
    """Return the Phase D consequence benchmark protocol."""

    memory_tasks = [
        {
            "id": "multi_hop_chain_retrieval",
            "family": "memory_intensive",
            "primary_metric": "exact_chain_success",
            "why": "Directly tests whether TAC specialization fixes the unresolved multi-hop routing question.",
        },
        {
            "id": "long_context_retrieval_4096",
            "family": "memory_intensive",
            "primary_metric": "target_recall_at_4096",
            "why": "Checks whether content-addressed memory survives longer contexts than the training distribution.",
        },
        {
            "id": "episodic_fact_update",
            "family": "memory_intensive",
            "primary_metric": "post_update_accuracy",
            "why": "Separates useful writable memory from static lexical memorization.",
        },
    ]
    agentic_tasks = [
        {
            "id": "tool_selection",
            "family": "agentic",
            "primary_metric": "valid_tool_success",
            "why": "Measures route-to-action consequences under agent-like decisions.",
        },
        {
            "id": "delayed_goal_binding",
            "family": "agentic",
            "primary_metric": "goal_consistency",
            "why": "Tests whether identity programs support role-stable behavior over delayed evidence.",
        },
    ]
    controls = [
        {
            "id": "parameter_matched_vanilla",
            "description": "Vanilla transformer matched on train tokens, parameter count, tokenizer, and evaluation harness.",
        },
        {
            "id": "tac_shuffled_state",
            "description": "TAC-Control-v1 with shuffled program-memory state at evaluation time.",
        },
        {
            "id": "tac_base_routing_ablation",
            "description": "TAC-Control-v1 with routing constrained to BASE where the harness supports it.",
        },
        {
            "id": "loss_matched_run5b_bestpt",
            "description": "Run 5B best.pt diagnostic to separate lower loss from specialization.",
        },
    ]

    return {
        "phase": "D",
        "objective": "Test whether Run 5B specialization buys capability, not just prettier internal metrics.",
        "suites": [
            {
                "id": "memory_intensive",
                "description": "Retrieval, writable-memory, and multi-hop consequence checks.",
                "tasks": memory_tasks,
            },
            {
                "id": "agentic",
                "description": "Route-to-action and delayed-goal behavior checks.",
                "tasks": agentic_tasks,
            },
        ],
        "tasks": [*memory_tasks, *agentic_tasks],
        "controls": controls,
        "metrics": [
            "task_primary_score",
            "loss",
            "accuracy",
            "selected_route_mi",
            "program_memory_cosine",
            "wall_clock_seconds",
            "tokens_per_second",
            "cost_normalized_score",
        ],
        "decision_gate": "TAC > parameter-matched vanilla",
        "minimum_evidence": {
            "required_task_families": ["memory_intensive", "agentic"],
            "required_seed_policy": "At least two passing Phase B seeds or an explicit seed-instability report.",
            "decode_penalty_rule": "Capability lift must be reported beside the 8-10x decode-cost penalty.",
        },
    }


def aggregate_phase_d_benchmark_results(
    rows: Iterable[dict[str, Any]],
    *,
    phase_b_decision: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate Phase D task rows into the consequence decision gate."""

    protocol = protocol or build_phase_d_benchmark_protocol()
    task_family = {
        task["id"]: task["family"]
        for task in protocol.get("tasks", [])
        if isinstance(task, dict) and "id" in task
    }
    required_families = list(
        _coerce_dict(protocol.get("minimum_evidence")).get(
            "required_task_families",
            ["memory_intensive", "agentic"],
        )
    )
    rows_list = [row for row in rows if isinstance(row, dict)]
    task_results = _phase_d_task_results(rows_list, task_family)
    family_results = _phase_d_family_results(task_results, required_families)

    phase_b_ready = bool(
        _coerce_dict(phase_b_decision or {}).get("ready_for_phase_d", False)
    )
    missing_families = [
        family
        for family in required_families
        if not _coerce_dict(family_results.get(family)).get("has_required_controls")
    ]
    families_pass = all(
        _coerce_dict(family_results.get(family)).get("passes", False)
        for family in required_families
    )
    if not phase_b_ready:
        status = "blocked_by_phase_b"
        reason = "Phase D remains blocked until Phase B has enough complete passing seed evidence."
    elif missing_families:
        status = "pending"
        reason = "Phase D is missing required TAC or parameter-matched vanilla task evidence."
    elif families_pass:
        status = "pass"
        reason = "TAC-Control-v1 beats the parameter-matched vanilla control across required task families."
    else:
        status = "fail"
        reason = "TAC-Control-v1 does not beat the parameter-matched vanilla control across required task families."

    return {
        "phase": "D",
        "decision": {
            "status": status,
            "reason": reason,
            "ready_for_phase_d": phase_b_ready,
            "passes_decision_gate": bool(phase_b_ready and families_pass and not missing_families),
            "decision_gate": protocol.get("decision_gate"),
            "missing_families": missing_families,
        },
        "families": family_results,
        "tasks": task_results,
        "rows": rows_list,
    }


def format_phase_d_benchmark_results_markdown(result: dict[str, Any]) -> str:
    """Format Phase D benchmark aggregation as Markdown."""

    decision = _coerce_dict(result.get("decision"))
    lines = [
        "# Phase D Benchmark Results",
        "",
        f"Decision: `{decision.get('status')}`",
        "",
        f"- Reason: {decision.get('reason')}",
        f"- Decision gate: `{decision.get('decision_gate')}`",
        f"- Passes decision gate: `{decision.get('passes_decision_gate')}`",
        "",
        "| Family | TAC Mean | Vanilla Mean | TAC Advantage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, row in result.get("families", {}).items():
        family_row = _coerce_dict(row)
        lines.append(
            "| {family} | {tac} | {vanilla} | {advantage} |".format(
                family=family,
                tac=_format_metric(family_row.get("tac_mean_score")),
                vanilla=_format_metric(family_row.get("vanilla_mean_score")),
                advantage=_format_metric(family_row.get("tac_advantage")),
            )
        )
    lines.extend(
        [
            "",
            "| Task | Family | TAC Mean | parameter_matched_vanilla Mean | Advantage |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for task_id, row in result.get("tasks", {}).items():
        task_row = _coerce_dict(row)
        lines.append(
            "| {task} | {family} | {tac} | {vanilla} | {advantage} |".format(
                task=task_id,
                family=task_row.get("family"),
                tac=_format_metric(task_row.get("tac_mean_score")),
                vanilla=_format_metric(task_row.get("vanilla_mean_score")),
                advantage=_format_metric(task_row.get("tac_advantage")),
            )
        )
    return "\n".join(lines)


def format_tac_research_plan_markdown(
    audit: dict[str, Any],
    phase_b_plan: dict[str, Any],
    phase_c_protocol: dict[str, Any],
    phase_d_protocol: dict[str, Any],
) -> str:
    """Format the next-stage TAC research contract as Markdown."""

    reference = _coerce_dict(audit.get("reference"))
    decision = _coerce_dict(audit.get("decision"))
    metrics = _coerce_dict(audit.get("metrics"))
    resolved_gaps = _coerce_dict(audit.get("resolved_gaps"))
    open_gaps = _coerce_dict(audit.get("open_gaps"))

    lines = [
        "# TAC Next-Stage Research Contract",
        "",
        "## Phase A Freeze",
        "",
        f"- Reference: {reference.get('name', 'TAC-Control-v1')}",
        f"- Checkpoint step: {reference.get('checkpoint_step')}",
        f"- Tokens seen: {reference.get('tokens_seen')}",
        f"- Decision: {decision.get('status')}",
        f"- Route MI: {_format_metric(metrics.get('selected_route_mi'))}",
        f"- Program-memory cosine: {_format_metric(metrics.get('program_memory_cosine'))}",
        f"- content_addressed hit: {_format_metric(metrics.get('content_addressed_hit'))}",
        "",
        "### Resolved Gaps",
        "",
    ]

    for gap_id, gap in resolved_gaps.items():
        gap_dict = _coerce_dict(gap)
        lines.append(f"- {gap_id}: {gap_dict.get('status')}")

    lines.extend(["", "### Open Gaps", ""])
    for gap_id, gap in open_gaps.items():
        gap_dict = _coerce_dict(gap)
        label = "multi-hop" if gap_id == "multi_hop" else gap_id.replace("_", "-")
        lines.append(f"- {label}: {gap_dict.get('question')}")

    lines.extend(
        [
            "",
            "## Phase B Replication",
            "",
            f"- Seeds: {', '.join(str(seed) for seed in phase_b_plan.get('seeds', []))}",
            f"- Success criteria: route MI >= {phase_b_plan['success_criteria']['selected_route_mi_min']}, program-memory cosine <= {phase_b_plan['success_criteria']['program_memory_cosine_max']}",
            "- Frozen config includes content_addressed memory reads and identity_first attention.",
            "",
            "### Commands",
            "",
        ]
    )
    for run in phase_b_plan.get("runs", []):
        lines.append(f"- Seed {run['seed']}: `{run['command']}`")

    lines.extend(
        [
            "",
            "## Phase C Identity Stability",
            "",
            f"- Decision gate: {phase_c_protocol.get('decision_gate')}",
            f"- Blocked by: {phase_c_protocol.get('blocked_by')}",
            f"- Alignment components: {', '.join(phase_c_protocol.get('alignment_components', []))}",
            f"- Minimum seeds: {phase_c_protocol['success_criteria']['min_seed_count']}",
            f"- Minimum alignment similarity: {phase_c_protocol['success_criteria']['min_alignment_similarity']}",
            "- Report role permutations rather than assuming program ids are stable.",
        ]
    )

    task_ids = _phase_d_task_ids(phase_d_protocol)
    control_ids = [control["id"] for control in phase_d_protocol.get("controls", [])]
    lines.extend(
        [
            "",
            "## Phase D Benchmark Protocol",
            "",
            f"- Decision gate: {phase_d_protocol.get('decision_gate')}",
            f"- Tasks: {', '.join(task_ids)}",
            f"- Controls: {', '.join(control_ids)}",
            "- Report every capability score beside wall-clock, tokens/s, and cost-normalized deltas.",
            "",
        ]
    )

    return "\n".join(lines)


def _phase_b_command(seed: int, output_dir: str) -> str:
    parts = [
        "python",
        "kaggle/train_best_tac_agentic.py",
        "--preset run5b_capability",
        "--steps 20000",
        f"--seed {seed}",
        "--identity-attention-type identity_first",
        "--memory-read-type content_addressed",
        "--content-read-steps 2",
        "--content-read-gate-type synthesis",
        "--program-memory-update-type program_conditioned",
        "--memory-allocation-type creb",
        "--memory-allocation-k 6",
        "--memory-separation-weight 0.1",
        "--routing-type base_semantic",
        "--routing-top-k 2",
        "--category-route-objective selected_mi",
        "--category-route-weight 0.1",
        "--precision fp32",
        f"--output-dir {output_dir}",
    ]
    return " ".join(parts)


def _phase_b_kaggle_script(seed: int) -> str:
    command = [
        'sys.executable',
        '"-m"',
        '"torch.distributed.run"',
        '"--standalone"',
        '"--nproc_per_node=2"',
        'str(code_root / "kaggle" / "train_best_tac_agentic.py")',
        '"--preset"',
        '"run5b_capability"',
        '"--train-jsonl"',
        'str(train_jsonl)',
        '"--eval-jsonl"',
        'str(eval_jsonl)',
        '"--specialization-jsonl"',
        'str(specialization_jsonl)',
        '"--scale"',
        '"base"',
        '"--steps"',
        '"20000"',
        '"--seed"',
        f'"{seed}"',
        '"--identity-attention-type"',
        '"identity_first"',
        '"--memory-read-type"',
        '"content_addressed"',
        '"--content-read-steps"',
        '"2"',
        '"--content-read-gate-type"',
        '"synthesis"',
        '"--program-memory-update-type"',
        '"program_conditioned"',
        '"--memory-allocation-type"',
        '"creb"',
        '"--memory-allocation-k"',
        '"6"',
        '"--memory-separation-weight"',
        '"0.1"',
        '"--routing-type"',
        '"base_semantic"',
        '"--routing-top-k"',
        '"2"',
        '"--category-route-objective"',
        '"selected_mi"',
        '"--category-route-weight"',
        '"0.1"',
        '"--precision"',
        '"fp32"',
        '"--batch-size"',
        '"12"',
        '"--grad-accum-steps"',
        '"3"',
        '"--eval-every"',
        '"500"',
        '"--eval-batches"',
        '"8"',
        '"--checkpoint-every"',
        '"250"',
        '"--output-dir"',
        'str(OUTPUT_DIR)',
        '"--device"',
        '"auto"',
        '"--max-seconds"',
        '"30600"',
        '"--stop-buffer-seconds"',
        '"1200"',
        '"--specialization-checkpoints"',
        '"2000"',
        '"5000"',
        '"10000"',
        '"20000"',
        '"--specialization-checkpoint-max-records-per-category"',
        '"16"',
        '"--analyze-specialization-at-end"',
        '"--specialization-max-records-per-category"',
        '"64"',
        '"--specialization-device"',
        '"cpu"',
        '"--skip-end-specialization-on-time-stop"',
    ]
    command_block = ",\n        ".join(command)
    return f'''from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import torch


ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
CODE_WORK = WORKING_ROOT / "tac_control_v1_phase_b_code"
DATA_WORK = WORKING_ROOT / "tac_control_v1_phase_b_data"
SEED = {seed}
OUTPUT_DIR = WORKING_ROOT / "tac_control_v1_seed_{seed}"


def main() -> None:
    started = time.perf_counter()
    code_root = _prepare_code_root()
    data_root = _prepare_data_root()
    train_jsonl = _find_file("train.prepared.jsonl", preferred="tac-run5b-capability-data")
    eval_jsonl = _find_file("eval.prepared.jsonl", preferred="tac-run5b-capability-data")
    specialization_jsonl = _find_file("hard_agentic_eval.generated.jsonl")
    _require_dual_t4()

    command = [
        {command_block}
    ]

    print(
        json.dumps(
            {{
                "event": "phase_b_seed_start",
                "seed": SEED,
                "code_root": str(code_root),
                "data_root": str(data_root),
                "train_jsonl": str(train_jsonl),
                "eval_jsonl": str(eval_jsonl),
                "specialization_jsonl": str(specialization_jsonl),
                "output_dir": str(OUTPUT_DIR),
                "cuda_devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "command": command,
            }},
            indent=2,
        ),
        flush=True,
    )
    result = subprocess.run(command, cwd=code_root, check=False)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {{
                "event": "phase_b_seed_complete",
                "seed": SEED,
                "returncode": result.returncode,
                "elapsed_seconds": elapsed,
                "output_dir": str(OUTPUT_DIR),
            }},
            indent=2,
        ),
        flush=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _prepare_code_root() -> Path:
    for candidate in [ROOT, *ROOT.parents]:
        if _is_code_root(candidate):
            return candidate

    for train_script in sorted(INPUT_ROOT.rglob("kaggle/train_best_tac_agentic.py")):
        source_root = train_script.parents[1]
        if _is_code_root(source_root):
            if CODE_WORK.exists():
                shutil.rmtree(CODE_WORK)
            shutil.copytree(
                source_root,
                CODE_WORK,
                ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"),
            )
            return CODE_WORK

    bundle_zip = next(INPUT_ROOT.rglob("best-tac-agentic-training-bundle.zip"), None)
    if bundle_zip is not None:
        if CODE_WORK.exists():
            shutil.rmtree(CODE_WORK)
        CODE_WORK.mkdir(parents=True, exist_ok=True)
        with ZipFile(bundle_zip) as archive:
            archive.extractall(CODE_WORK)
        if _is_code_root(CODE_WORK):
            return CODE_WORK

    visible = [
        str(path.relative_to(INPUT_ROOT))
        for path in sorted(INPUT_ROOT.rglob("*"))[:80]
    ]
    raise FileNotFoundError(
        f"Could not locate TAC code root under /kaggle/input. Visible: {{visible}}"
    )


def _is_code_root(path: Path) -> bool:
    return (
        (path / "kaggle" / "train_best_tac_agentic.py").exists()
        and (path / "tac_transformer" / "__init__.py").exists()
    )


def _prepare_data_root() -> Path:
    if DATA_WORK.exists():
        return DATA_WORK
    DATA_WORK.mkdir(parents=True, exist_ok=True)
    for archive_path in sorted(INPUT_ROOT.rglob("prepared_corpus_agentic_hard_upload.zip")):
        with ZipFile(archive_path) as archive:
            archive.extractall(DATA_WORK)
        break
    for eval_path in sorted(INPUT_ROOT.rglob("hard_agentic_eval.generated.jsonl")):
        target = DATA_WORK / "hard_agentic_eval.generated.jsonl"
        if not target.exists():
            shutil.copy2(eval_path, target)
        break
    return DATA_WORK


def _find_file(name: str, *, preferred: str | None = None) -> Path:
    roots = [DATA_WORK, INPUT_ROOT]
    matches = [
        path
        for root in roots
        if root.exists()
        for path in sorted(root.rglob(name))
    ]
    if preferred is not None:
        preferred_matches = [
            path for path in matches if preferred in str(path).replace("\\\\", "/")
        ]
        if preferred_matches:
            return preferred_matches[0]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {{name}} under /kaggle/input")


def _require_dual_t4() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase B.")
    device_count = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    if device_count < 2:
        raise RuntimeError(f"Phase B requires dual T4 GPUs, found {{device_names}}")
    if not all("T4" in name for name in device_names[:2]):
        raise RuntimeError(f"Phase B expected T4 GPUs, found {{device_names}}")


if __name__ == "__main__":
    main()
'''


def _phase_d_task_ids(protocol: dict[str, Any]) -> list[str]:
    suites = protocol.get("suites")
    if isinstance(suites, list):
        return [
            task["id"]
            for suite in suites
            if isinstance(suite, dict)
            for task in suite.get("tasks", [])
            if isinstance(task, dict) and "id" in task
        ]
    return [
        task["id"]
        for task in protocol.get("tasks", [])
        if isinstance(task, dict) and "id" in task
    ]


def _phase_b_primary_metrics(
    final_summary: dict[str, Any],
    metrics_rows: Iterable[dict[str, Any]] | None,
) -> dict[str, Any]:
    rows = [row for row in metrics_rows or [] if isinstance(row, dict)]
    for row in rows:
        if _first_int(row.get("step")) == TAC_CONTROL_V1_REFERENCE["checkpoint_step"]:
            return row
    for row in rows:
        if _first_int(row.get("tokens_seen")) == TAC_CONTROL_V1_REFERENCE["tokens_seen"]:
            return row
    latest = _coerce_dict(final_summary.get("latest_metrics"))
    if latest:
        return latest
    return {}


def _phase_b_specialization(
    final_summary: dict[str, Any],
    specialization_report: dict[str, Any] | None,
) -> dict[str, Any]:
    standalone = _phase_b_specialization_from_report(specialization_report)
    if standalone:
        return standalone
    return _phase_b_specialization_from_summary(final_summary)


def _phase_b_specialization_from_summary(final_summary: dict[str, Any]) -> dict[str, Any]:
    checkpoints = [
        row
        for row in final_summary.get("specialization_checkpoints", [])
        if isinstance(row, dict) and row.get("enabled")
    ]
    selected = None
    for row in checkpoints:
        if _first_int(row.get("checkpoint_step")) == TAC_CONTROL_V1_REFERENCE["checkpoint_step"]:
            selected = row
            break
    if selected is None and checkpoints:
        selected = checkpoints[-1]
    embedded = selected or _coerce_dict(final_summary.get("specialization_analysis"))
    if not embedded:
        return {}
    deltas = _phase_b_ablation_deltas(embedded.get("top_ablation_loss_deltas", []))
    return {
        "source": "final_summary",
        "label": embedded.get("label"),
        "checkpoint_step": _first_int(embedded.get("checkpoint_step")),
        "records": _first_int(embedded.get("records")),
        "mi_bits": _first_number(embedded.get("mi_bits")),
        "normalized_mi": _first_number(embedded.get("normalized_mi")),
        "run_knockouts": bool(embedded.get("run_knockouts", False)),
        "max_knockout_loss_delta": max(deltas) if deltas else None,
        "max_knockout_selectivity_span": None,
    }


def _phase_b_specialization_from_report(
    report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    mutual_information = report.get("mutual_information")
    if isinstance(mutual_information, dict):
        mi_bits = _first_number(mutual_information.get("mi_bits"))
        normalized_mi = _first_number(mutual_information.get("normalized_mi"))
    else:
        mi_bits = _first_number(report.get("mi_bits"))
        normalized_mi = _first_number(report.get("normalized_mi"))
    deltas = _phase_b_ablation_deltas(report.get("ablations", []))
    records = report.get("records")
    return {
        "source": "specialization_report",
        "label": report.get("label"),
        "checkpoint_step": _first_int(
            report.get("checkpoint_step"),
            TAC_CONTROL_V1_REFERENCE["checkpoint_step"],
        ),
        "records": len(records) if isinstance(records, list) else _first_int(records),
        "mi_bits": mi_bits,
        "normalized_mi": normalized_mi,
        "run_knockouts": bool(deltas),
        "max_knockout_loss_delta": max(deltas) if deltas else None,
        "max_knockout_selectivity_span": _phase_b_knockout_selectivity_span(report),
    }


def _phase_b_ablation_deltas(rows: Any) -> list[float]:
    deltas: list[float] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        value = _first_number(
            row.get("loss_delta"),
            row.get("delta_loss"),
            row.get("mean_loss_delta"),
        )
        if value is not None:
            deltas.append(value)
    return deltas


def _phase_b_knockout_selectivity_span(report: dict[str, Any]) -> float | None:
    metrics = report.get("specialization_metrics")
    candidates = (
        metrics.get("knockout_selectivity")
        if isinstance(metrics, dict)
        else report.get("knockout_selectivity")
    )
    spans = [
        abs(float(row.get("selectivity_span", 0.0)))
        for row in candidates or []
        if isinstance(row, dict)
    ]
    return max(spans) if spans else None


def _min_gate(value: float | None, threshold: float) -> dict[str, Any]:
    if value is None:
        return {"status": "fail", "passes": False, "value": value, "threshold": threshold}
    passes = value >= threshold
    return {
        "status": "pass" if passes else "fail",
        "passes": passes,
        "value": value,
        "threshold": threshold,
    }


def _max_gate(value: float | None, threshold: float) -> dict[str, Any]:
    if value is None:
        return {"status": "fail", "passes": False, "value": value, "threshold": threshold}
    passes = value <= threshold
    return {
        "status": "pass" if passes else "fail",
        "passes": passes,
        "value": value,
        "threshold": threshold,
    }


def _phase_b_knockout_gate(
    value: float | None,
    threshold: float,
    run_knockouts: bool,
) -> dict[str, Any]:
    if value is None or not run_knockouts:
        return {
            "status": "pending",
            "passes": False,
            "value": value,
            "threshold": threshold,
        }
    passes = value >= threshold
    return {
        "status": "pass" if passes else "fail",
        "passes": passes,
        "value": value,
        "threshold": threshold,
    }


def _phase_b_config_gate(run_manifest: dict[str, Any]) -> dict[str, Any]:
    checks = {
        key: _values_match(_manifest_value(run_manifest, key), expected)
        for key, expected in TAC_CONTROL_V1_CONFIG.items()
    }
    passes = all(checks.values())
    return {
        "status": "pass" if passes else "fail",
        "passes": passes,
        "checks": checks,
    }


def _phase_b_metric_aggregate(
    rows: Iterable[dict[str, Any]],
    metric_name: str,
) -> dict[str, float | None]:
    values = [
        _first_number(_coerce_dict(row.get("metrics")).get(metric_name))
        for row in rows
    ]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(numbers),
        "median": median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _phase_d_task_results(
    rows: Iterable[dict[str, Any]],
    task_family: dict[str, str],
) -> dict[str, Any]:
    task_ids = sorted({row.get("task_id") for row in rows if row.get("task_id")})
    results = {}
    for task_id in task_ids:
        task_rows = [row for row in rows if row.get("task_id") == task_id]
        tac_scores = _phase_d_scores(task_rows, "tac_control_v1")
        vanilla_scores = _phase_d_scores(task_rows, "parameter_matched_vanilla")
        tac_mean = mean(tac_scores) if tac_scores else None
        vanilla_mean = mean(vanilla_scores) if vanilla_scores else None
        advantage = (
            tac_mean - vanilla_mean
            if tac_mean is not None and vanilla_mean is not None
            else None
        )
        results[str(task_id)] = {
            "task_id": task_id,
            "family": task_family.get(str(task_id), "unknown"),
            "tac_mean_score": tac_mean,
            "vanilla_mean_score": vanilla_mean,
            "tac_advantage": advantage,
            "has_required_controls": tac_mean is not None and vanilla_mean is not None,
            "passes": advantage is not None and advantage > 0.0,
            "controls": _phase_d_control_results(task_rows),
        }
    return results


def _phase_d_family_results(
    task_results: dict[str, Any],
    required_families: Iterable[str],
) -> dict[str, Any]:
    families = sorted(
        set(required_families)
        | {
            str(_coerce_dict(row).get("family"))
            for row in task_results.values()
            if _coerce_dict(row).get("family")
        }
    )
    results = {}
    for family in families:
        family_tasks = [
            _coerce_dict(row)
            for row in task_results.values()
            if _coerce_dict(row).get("family") == family
        ]
        tac_values = [
            _first_number(row.get("tac_mean_score"))
            for row in family_tasks
        ]
        vanilla_values = [
            _first_number(row.get("vanilla_mean_score"))
            for row in family_tasks
        ]
        tac_numbers = [value for value in tac_values if value is not None]
        vanilla_numbers = [value for value in vanilla_values if value is not None]
        tac_mean = mean(tac_numbers) if tac_numbers else None
        vanilla_mean = mean(vanilla_numbers) if vanilla_numbers else None
        advantage = (
            tac_mean - vanilla_mean
            if tac_mean is not None and vanilla_mean is not None
            else None
        )
        has_required_controls = any(
            bool(row.get("has_required_controls")) for row in family_tasks
        )
        results[family] = {
            "task_count": len(family_tasks),
            "has_required_controls": has_required_controls,
            "tac_mean_score": tac_mean,
            "vanilla_mean_score": vanilla_mean,
            "tac_advantage": advantage,
            "passes": has_required_controls and advantage is not None and advantage > 0.0,
        }
    return results


def _phase_d_control_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    controls = sorted({row.get("control_id") for row in rows if row.get("control_id")})
    results = {}
    for control in controls:
        scores = _phase_d_scores(rows, str(control))
        results[str(control)] = {
            "n": len(scores),
            "mean_primary_score": mean(scores) if scores else None,
        }
    return results


def _phase_d_scores(rows: Iterable[dict[str, Any]], control_id: str) -> list[float]:
    scores = []
    for row in rows:
        if row.get("control_id") != control_id:
            continue
        score = _first_number(
            row.get("primary_score"),
            row.get("task_primary_score"),
            row.get("score"),
        )
        if score is not None:
            scores.append(score)
    return scores


def _phase_c_categories(report: dict[str, Any]) -> list[str]:
    mutual_information = _coerce_dict(report.get("mutual_information"))
    categories = report.get("categories") or mutual_information.get("categories") or []
    if isinstance(categories, list):
        return [str(category) for category in categories]
    return []


def _phase_c_route_profiles(
    report: dict[str, Any],
    categories: list[str],
) -> dict[int, dict[str, float]]:
    metrics = _coerce_dict(report.get("specialization_metrics"))
    selected = _phase_c_selectivity_profiles(
        metrics.get("selected_route_selectivity"),
        categories,
    )
    if selected:
        return selected
    category = _phase_c_category_route_histogram_profiles(report, categories)
    if category:
        return category
    return _phase_c_mutual_information_profiles(report, categories)


def _phase_c_category_route_histogram_profiles(
    report: dict[str, Any],
    categories: list[str],
) -> dict[int, dict[str, float]]:
    histogram = _coerce_dict(report.get("category_route_histogram"))
    by_category = _coerce_dict(histogram.get("by_category"))
    programs = histogram.get("programs") or []
    if not by_category or not isinstance(programs, list):
        return {}
    profiles: dict[int, dict[str, float]] = {
        int(program): {} for program in programs
    }
    for category in categories:
        row = _coerce_dict(by_category.get(category))
        frequencies = row.get("selected_route_frequency")
        if not isinstance(frequencies, list):
            frequencies = row.get("selected_top_program_frequency")
        if not isinstance(frequencies, list):
            continue
        for index, value in enumerate(frequencies):
            if index in profiles:
                profiles[index][category] = float(value)
    return {program: profile for program, profile in profiles.items() if profile}


def _phase_c_mutual_information_profiles(
    report: dict[str, Any],
    categories: list[str],
) -> dict[int, dict[str, float]]:
    mutual_information = _coerce_dict(report.get("mutual_information"))
    counts = mutual_information.get("counts")
    if not isinstance(counts, list):
        return {}
    profiles: dict[int, dict[str, float]] = {}
    n_programs = max((len(row) for row in counts if isinstance(row, list)), default=0)
    for program in range(n_programs):
        values = []
        for row in counts:
            if isinstance(row, list) and program < len(row):
                values.append(float(row[program]))
            else:
                values.append(0.0)
        total = sum(values)
        if total <= 0.0:
            continue
        profiles[program] = {
            categories[index] if index < len(categories) else str(index): value / total
            for index, value in enumerate(values)
        }
    return profiles


def _phase_c_memory_profiles(report: dict[str, Any]) -> dict[int, list[float]]:
    summary = _coerce_dict(report.get("program_memory_summary"))
    rows = summary.get("programs")
    if isinstance(rows, list):
        profiles = {}
        for index, row in enumerate(rows):
            row_dict = _coerce_dict(row)
            vector = row_dict.get("mean_vector") or row_dict.get("vector")
            if not isinstance(vector, list):
                continue
            profiles[int(row_dict.get("program", index))] = [
                float(value) for value in vector
            ]
        if profiles:
            return profiles

    vectors = report.get("program_memory_vectors")
    if isinstance(vectors, list):
        return {
            index: [float(value) for value in vector]
            for index, vector in enumerate(vectors)
            if isinstance(vector, list)
        }
    return {}


def _phase_c_knockout_profiles(
    report: dict[str, Any],
    categories: list[str],
) -> dict[int, dict[str, float]]:
    metrics = _coerce_dict(report.get("specialization_metrics"))
    selectivity = _phase_c_selectivity_profiles(
        metrics.get("knockout_selectivity"),
        categories,
    )
    if selectivity:
        return selectivity
    profiles = {}
    for row in report.get("ablations", []) or []:
        row_dict = _coerce_dict(row)
        by_category = _coerce_dict(row_dict.get("by_category"))
        values = {}
        for category in categories:
            category_row = by_category.get(category)
            if isinstance(category_row, dict):
                number = _first_number(category_row.get("loss_delta"))
            else:
                number = _first_number(category_row)
            if number is not None:
                values[category] = number
        if values:
            profiles[int(row_dict.get("program", len(profiles)))] = values
    return profiles


def _phase_c_selectivity_profiles(
    rows: Any,
    categories: list[str],
) -> dict[int, dict[str, float]]:
    profiles = {}
    for index, row in enumerate(rows or []):
        row_dict = _coerce_dict(row)
        by_category = _coerce_dict(row_dict.get("by_category"))
        if not by_category:
            continue
        values = {}
        for category in categories or sorted(by_category):
            number = _first_number(by_category.get(category))
            if number is not None:
                values[str(category)] = number
        if values:
            profiles[int(row_dict.get("program", index))] = values
    return profiles


def _phase_c_align_seed_pair(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reference_profiles = [
        _coerce_dict(row)
        for row in reference.get("program_profiles", [])
        if isinstance(row, dict)
    ]
    candidate_profiles = [
        _coerce_dict(row)
        for row in candidate.get("program_profiles", [])
        if isinstance(row, dict)
    ]
    matrix = [
        [
            _phase_c_profile_similarity(ref_profile, candidate_profile)
            for candidate_profile in candidate_profiles
        ]
        for ref_profile in reference_profiles
    ]
    assignments = _phase_c_max_assignment(matrix)
    matches = []
    for ref_index, candidate_index, similarity in assignments:
        ref_profile = reference_profiles[ref_index]
        candidate_profile = candidate_profiles[candidate_index]
        matches.append(
            {
                "reference_program": ref_profile.get("program"),
                "candidate_program": candidate_profile.get("program"),
                "reference_preferred_category": ref_profile.get("preferred_category"),
                "candidate_preferred_category": candidate_profile.get("preferred_category"),
                "similarity": similarity,
            }
        )
    similarities = [match["similarity"] for match in matches]
    return {
        "reference_seed": reference.get("seed"),
        "candidate_seed": candidate.get("seed"),
        "mean_similarity": mean(similarities) if similarities else None,
        "min_similarity": min(similarities) if similarities else None,
        "matches": matches,
    }


def _phase_c_profile_similarity(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    similarities = []
    route = _phase_c_dict_similarity(
        _coerce_dict(reference.get("route_profile")),
        _coerce_dict(candidate.get("route_profile")),
    )
    if route is not None:
        similarities.append(route)
    memory = _phase_c_vector_similarity(
        reference.get("memory_vector"),
        candidate.get("memory_vector"),
    )
    if memory is not None:
        similarities.append(memory)
    knockout = _phase_c_dict_similarity(
        _coerce_dict(reference.get("knockout_profile")),
        _coerce_dict(candidate.get("knockout_profile")),
    )
    if knockout is not None:
        similarities.append(knockout)
    return mean(similarities) if similarities else 0.0


def _phase_c_dict_similarity(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> float | None:
    keys = sorted(set(reference) | set(candidate))
    if not keys:
        return None
    return _phase_c_vector_cosine(
        [_first_number(reference.get(key)) or 0.0 for key in keys],
        [_first_number(candidate.get(key)) or 0.0 for key in keys],
    )


def _phase_c_vector_similarity(reference: Any, candidate: Any) -> float | None:
    if not isinstance(reference, list) or not isinstance(candidate, list):
        return None
    length = min(len(reference), len(candidate))
    if length == 0:
        return None
    return _phase_c_vector_cosine(
        [float(value) for value in reference[:length]],
        [float(value) for value in candidate[:length]],
    )


def _phase_c_vector_cosine(reference: list[float], candidate: list[float]) -> float:
    dot = sum(left * right for left, right in zip(reference, candidate))
    left_norm = math.sqrt(sum(value * value for value in reference))
    right_norm = math.sqrt(sum(value * value for value in candidate))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _phase_c_max_assignment(matrix: list[list[float]]) -> list[tuple[int, int, float]]:
    if not matrix or not matrix[0]:
        return []
    n_ref = len(matrix)
    n_candidate = len(matrix[0])
    if n_ref > 16 or n_candidate > 16:
        return _phase_c_greedy_assignment(matrix)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def solve(index: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if index >= n_ref:
            return 0.0, ()
        best_score = -math.inf
        best_path: tuple[int, ...] = ()
        for candidate_index in range(n_candidate):
            if used_mask & (1 << candidate_index):
                continue
            child_score, child_path = solve(
                index + 1,
                used_mask | (1 << candidate_index),
            )
            score = matrix[index][candidate_index] + child_score
            if score > best_score:
                best_score = score
                best_path = (candidate_index, *child_path)
        return best_score, best_path

    _, path = solve(0, 0)
    return [
        (ref_index, candidate_index, matrix[ref_index][candidate_index])
        for ref_index, candidate_index in enumerate(path)
    ]


def _phase_c_greedy_assignment(matrix: list[list[float]]) -> list[tuple[int, int, float]]:
    candidates = [
        (score, ref_index, candidate_index)
        for ref_index, row in enumerate(matrix)
        for candidate_index, score in enumerate(row)
    ]
    matches = []
    used_ref = set()
    used_candidate = set()
    for score, ref_index, candidate_index in sorted(candidates, reverse=True):
        if ref_index in used_ref or candidate_index in used_candidate:
            continue
        used_ref.add(ref_index)
        used_candidate.add(candidate_index)
        matches.append((ref_index, candidate_index, score))
    return sorted(matches)


def _manifest_value(manifest: dict[str, Any], key: str) -> Any:
    config = _coerce_dict(manifest.get("config"))
    if key in config:
        return config[key]
    if key in manifest:
        return manifest[key]

    args = _coerce_dict(manifest.get("args"))
    if key in args:
        return args[key]

    hyphen_key = key.replace("_", "-")
    if hyphen_key in config:
        return config[hyphen_key]
    if hyphen_key in manifest:
        return manifest[hyphen_key]
    if hyphen_key in args:
        return args[hyphen_key]
    return None


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _specialization_metrics(
    external_validation: dict[str, Any], tac_summary: dict[str, Any]
) -> dict[str, Any]:
    candidates = [
        external_validation.get("specialization"),
        external_validation.get("specialization_metrics"),
        external_validation.get("tac_specialization"),
        tac_summary.get("specialization"),
        tac_summary.get("specialization_metrics"),
    ]
    for candidate in candidates:
        candidate_dict = _coerce_dict(candidate)
        if candidate_dict:
            return candidate_dict
    return {}


def _optimization_health(
    tac_summary: dict[str, Any], latest_metrics: dict[str, Any]
) -> dict[str, Any]:
    health = _coerce_dict(tac_summary.get("optimization_health"))
    if health:
        return health
    health = _coerce_dict(latest_metrics.get("optimization_health"))
    if health:
        return health
    return {"status": "unknown"}


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_int(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _coerce_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_metric(value: Any) -> str:
    number = _first_number(value)
    if number is None:
        return "n/a"
    return f"{number:.6g}"
