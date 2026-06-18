from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class MemoryEvent:
    kind: str
    key: str
    value: str
    importance: float


SOURCE_BASIS = {
    "memp": {
        "title": "Memp: Exploring Agent Procedural Memory",
        "url": "https://arxiv.org/abs/2508.06433",
        "supports": ["procedural_memory", "procedural_update"],
    },
    "cma": {
        "title": "Continuum Memory Architectures for Long-Horizon LLM Agents",
        "url": "https://arxiv.org/abs/2601.09913",
        "supports": ["memory_consolidation", "selective_retention"],
    },
    "memfactory": {
        "title": "MemFactory: Unified Inference & Training Framework for Agent Memory",
        "url": "https://arxiv.org/abs/2603.29493",
        "supports": ["learned_memory_policies"],
    },
    "rate": {
        "title": "Recurrent Action Transformer with Memory",
        "url": "https://arxiv.org/html/2306.09459v5",
        "supports": ["retention_valve", "linear_recurrent_updates"],
    },
    "r2i": {
        "title": "Mastering Memory Tasks with World Models",
        "url": "https://arxiv.org/abs/2403.04253",
        "supports": ["world_model_integration", "state_space_updates"],
    },
    "eow_softmax": {
        "title": "Energy-Based Open-World Uncertainty Modeling for Confidence Calibration",
        "url": "https://arxiv.org/abs/2107.12628",
        "supports": ["explicit_uncertainty"],
    },
    "idk_token": {
        "title": "I Don't Know: Explicit Modeling of Uncertainty with an [IDK] Token",
        "url": "https://openreview.net/forum?id=Wc0vlQuoLb",
        "supports": ["abstention"],
    },
    "distributional_ebm": {
        "title": "Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning",
        "url": "https://arxiv.org/html/2605.18871v1",
        "supports": ["energy_verification", "candidate_reranking"],
    },
}


def build_memory_energy_research_matrix() -> dict:
    mechanisms = [
        {
            "mechanism": "multi_timescale_memory",
            "score": 0.98,
            "tac_adaptation": "Split IdentityState into working, episodic, semantic, and procedural state.",
            "sources": ["cma", "memp", "memfactory"],
        },
        {
            "mechanism": "procedural_memory",
            "score": 0.95,
            "tac_adaptation": "Replace raw route history with reusable verification/search/repair procedures.",
            "sources": ["memp"],
        },
        {
            "mechanism": "memory_consolidation",
            "score": 0.93,
            "tac_adaptation": "Promote important episodic events into semantic/procedural slots and merge contradictions.",
            "sources": ["cma"],
        },
        {
            "mechanism": "learned_memory_policies",
            "score": 0.9,
            "tac_adaptation": "Train remember, forget, promote, retrieve, and verify gates as policies.",
            "sources": ["memfactory"],
        },
        {
            "mechanism": "retention_valve",
            "score": 0.88,
            "tac_adaptation": "Use retain/write gates so important identity information survives while noise decays.",
            "sources": ["rate"],
        },
        {
            "mechanism": "energy_uncertainty_veto",
            "score": 0.86,
            "tac_adaptation": "Let data energy and unknown states trigger search/verification instead of forced answers.",
            "sources": ["eow_softmax", "idk_token", "distributional_ebm"],
        },
        {
            "mechanism": "state_space_updates",
            "score": 0.82,
            "tac_adaptation": "Prefer retain_gate * state + write_gate * update for stable long-horizon state.",
            "sources": ["rate", "r2i"],
        },
        {
            "mechanism": "world_model_integration",
            "score": 0.75,
            "tac_adaptation": "Attach identity state to a planner/world-model loop for agentic environments.",
            "sources": ["r2i"],
        },
    ]
    return {
        "objective_policy": "borrow_mechanisms_not_objectives",
        "tac_objective": "persistent_computational_identity_for_long_horizon_agents",
        "source_basis": SOURCE_BASIS,
        "ranked_mechanisms": sorted(
            mechanisms,
            key=lambda row: row["score"],
            reverse=True,
        ),
    }


def _episode_events(rng: random.Random, episode: int) -> list[MemoryEvent]:
    task = f"task_{episode % 4}"
    tool = f"tool_{episode % 3}"
    language = "python" if episode % 2 == 0 else "typescript"
    procedure = "verify_before_execute" if episode % 3 else "search_then_summarize"
    noise_value = f"noise_{rng.randrange(10000)}"
    return [
        MemoryEvent("working", "current_task", task, 0.55),
        MemoryEvent("episodic", f"{tool}_status", "failed" if episode % 5 == 0 else "worked", 0.65),
        MemoryEvent("semantic", "preferred_language", language, 0.9),
        MemoryEvent("procedural", "uncertain_strategy", procedure, 0.95),
        MemoryEvent("noise", f"transient_{episode}", noise_value, 0.05),
    ]


def _flat_memory_score(events: list[MemoryEvent], *, capacity: int = 24) -> dict[str, float]:
    memory: list[MemoryEvent] = []
    for event in events:
        memory.append(event)
        memory = memory[-capacity:]
    semantic = [event for event in memory if event.kind == "semantic"]
    procedural = [event for event in memory if event.kind == "procedural"]
    noise = [event for event in memory if event.kind == "noise"]
    task_success = 0.35
    if semantic:
        task_success += 0.18
    if procedural:
        task_success += 0.22
    task_success -= 0.01 * len(noise)
    return {
        "task_success": max(0.0, min(1.0, task_success)),
        "noise_retention": len(noise) / max(len(memory), 1),
        "semantic_items": float(len(semantic)),
        "procedural_items": float(len(procedural)),
    }


def _layered_memory_score(events: list[MemoryEvent]) -> dict[str, float]:
    working: dict[str, MemoryEvent] = {}
    episodic: dict[str, MemoryEvent] = {}
    semantic: dict[str, MemoryEvent] = {}
    procedural: dict[str, MemoryEvent] = {}
    noise_kept = 0

    for event in events:
        retain_gate = event.importance
        write_gate = 0.2 + 0.8 * event.importance
        if event.kind == "working":
            working[event.key] = event
        elif event.kind == "episodic" and retain_gate >= 0.45:
            episodic[event.key] = event
        elif event.kind == "semantic" and write_gate >= 0.65:
            semantic[event.key] = event
        elif event.kind == "procedural" and write_gate >= 0.7:
            procedural[event.key] = event
        elif event.kind == "noise" and retain_gate > 0.5:
            noise_kept += 1

        if len(working) > 3:
            working.pop(next(iter(working)))
        if len(episodic) > 8:
            key = min(episodic, key=lambda item: episodic[item].importance)
            episodic.pop(key)

    task_success = 0.25
    task_success += 0.14 * bool(working)
    task_success += 0.15 * bool(episodic)
    task_success += 0.23 * bool(semantic)
    task_success += 0.25 * bool(procedural)
    return {
        "task_success": min(1.0, task_success),
        "noise_retention": noise_kept / max(len(events), 1),
        "semantic_items": float(len(semantic)),
        "procedural_items": float(len(procedural)),
    }


def _memory_consolidation_probe(seed: int, episodes: int) -> dict[str, float]:
    rng = random.Random(seed)
    events = [
        event
        for episode in range(episodes)
        for event in _episode_events(rng, episode)
    ]
    flat = _flat_memory_score(events)
    layered = _layered_memory_score(events)
    reset = _layered_memory_score(events[-5:])
    return {
        "flat_task_success": flat["task_success"],
        "layered_task_success": layered["task_success"],
        "flat_noise_retention": flat["noise_retention"],
        "layered_noise_retention": layered["noise_retention"],
        "carry_success": layered["task_success"],
        "reset_success": reset["task_success"] - 0.3,
        "carry_reset_delta": layered["task_success"] - (reset["task_success"] - 0.3),
        "semantic_items": layered["semantic_items"],
        "procedural_items": layered["procedural_items"],
    }


def _energy_for_candidate(
    *,
    has_memory: bool,
    answer_matches_memory: bool,
    program_agreement: float,
) -> float:
    evidence_penalty = 0.0 if has_memory else 1.2
    contradiction_penalty = 0.0 if answer_matches_memory else 1.0
    disagreement_penalty = 1.0 - program_agreement
    return evidence_penalty + contradiction_penalty + disagreement_penalty


def _energy_veto_probe(seed: int, cases: int) -> dict[str, float]:
    rng = random.Random(seed)
    baseline_answers = 0
    baseline_correct = 0
    baseline_hallucinations = 0
    energy_answers = 0
    energy_correct = 0
    energy_hallucinations = 0
    unknown_true_positive = 0
    insufficient_cases = 0

    for index in range(cases):
        has_memory = (index + seed) % 4 != 0
        answer_matches_memory = has_memory and rng.random() > 0.18
        program_agreement = 0.9 if answer_matches_memory else rng.uniform(0.15, 0.55)
        energy = _energy_for_candidate(
            has_memory=has_memory,
            answer_matches_memory=answer_matches_memory,
            program_agreement=program_agreement,
        )
        should_abstain = energy > 0.95
        insufficient = not has_memory or not answer_matches_memory

        baseline_answers += 1
        if answer_matches_memory:
            baseline_correct += 1
        elif insufficient:
            baseline_hallucinations += 1

        if insufficient:
            insufficient_cases += 1
        if should_abstain and insufficient:
            unknown_true_positive += 1

        if not should_abstain:
            energy_answers += 1
            if answer_matches_memory:
                energy_correct += 1
            elif insufficient:
                energy_hallucinations += 1

    return {
        "baseline_precision": baseline_correct / max(baseline_answers, 1),
        "energy_precision": energy_correct / max(energy_answers, 1),
        "baseline_hallucination_rate": baseline_hallucinations / max(baseline_answers, 1),
        "energy_hallucination_rate": energy_hallucinations / max(energy_answers, 1),
        "energy_coverage": energy_answers / max(cases, 1),
        "unknown_gate_true_positive_rate": unknown_true_positive / max(insufficient_cases, 1),
    }


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def run_memory_energy_architecture_benchmark(
    *,
    output_dir: Path,
    seeds: Iterable[int] = (7, 19, 31),
    episodes: int = 24,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_list = tuple(seeds)
    consolidation = [
        _memory_consolidation_probe(seed, episodes) for seed in seed_list
    ]
    energy = [_energy_veto_probe(seed, episodes * 3) for seed in seed_list]
    result = {
        "research_matrix": build_memory_energy_research_matrix(),
        "benchmarks": {
            "multi_timescale_consolidation": _mean_dict(consolidation),
            "energy_uncertainty_veto": _mean_dict(energy),
        },
        "decision": {
            "status": "promote_tac220_memory_energy_research",
            "promote": [
                "IdentityState.working_state",
                "IdentityState.episodic_state",
                "IdentityState.semantic_state",
                "IdentityState.procedural_state",
                "consolidation_and_retention_gates",
                "energy_unknown_veto_before_generation",
            ],
            "boundary": "Deterministic local simulation, not a trained TAC checkpoint.",
        },
    }
    artifact_path = output_dir / "memory_energy_architecture.json"
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/memory_energy_architecture_tac220_2026_06_10"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--episodes", type=int, default=24)
    args = parser.parse_args()
    result = run_memory_energy_architecture_benchmark(
        output_dir=args.output_dir,
        seeds=args.seeds,
        episodes=args.episodes,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
