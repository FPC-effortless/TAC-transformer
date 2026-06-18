from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


V02_METRIC_SCHEMA = "tac_v02_metrics.v1"
REQUIRED_V02_METRICS = (
    "train_loss",
    "eval_loss",
    "perplexity",
    "routing_entropy",
    "program_utilization",
    "memory_utilization",
    "state_carry_score",
)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def normalize_v02_metrics(
    *,
    model_name: str,
    step: int,
    train_metrics: Mapping[str, Any],
    eval_metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    eval_metrics = eval_metrics or {}
    routing_entropy = _float_or_none(
        train_metrics.get("metric_route_entropy")
        or train_metrics.get("metric_routing_entropy")
        or eval_metrics.get("metric_route_entropy")
        or eval_metrics.get("metric_routing_entropy")
    )
    program_utilization = _float_or_none(
        train_metrics.get("metric_active_expert_fraction")
        or eval_metrics.get("metric_active_expert_fraction")
        or train_metrics.get("active_expert_fraction")
        or eval_metrics.get("active_expert_fraction")
    )
    memory_utilization = _float_or_none(
        train_metrics.get("metric_selected_identity_state_norm")
        or eval_metrics.get("metric_selected_identity_state_norm")
        or train_metrics.get("program_memory_cosine")
        or eval_metrics.get("program_memory_cosine")
    )
    state_carry_score = _float_or_none(
        train_metrics.get("metric_decision_continuity_agreement")
        or eval_metrics.get("metric_decision_continuity_agreement")
        or train_metrics.get("program_memory_cosine")
        or eval_metrics.get("program_memory_cosine")
    )
    payload = {
        "schema": V02_METRIC_SCHEMA,
        "model_name": model_name,
        "step": int(step),
        "train_loss": _float_or_none(
            train_metrics.get("next_token_loss") or train_metrics.get("loss")
        ),
        "eval_loss": _float_or_none(eval_metrics.get("loss")),
        "perplexity": _float_or_none(eval_metrics.get("perplexity")),
        "routing_entropy": routing_entropy,
        "program_utilization": program_utilization,
        "memory_utilization": memory_utilization,
        "state_carry_score": state_carry_score,
        "raw_train_metrics": dict(train_metrics),
        "raw_eval_metrics": dict(eval_metrics),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


@dataclass
class V02MetricLogger:
    path: Path
    records: list[dict[str, Any]] = field(default_factory=list)

    def append(self, record: Mapping[str, Any]) -> None:
        normalized = dict(record)
        if normalized.get("schema") != V02_METRIC_SCHEMA:
            raise ValueError(f"record schema must be {V02_METRIC_SCHEMA!r}")
        missing = [name for name in REQUIRED_V02_METRICS if name not in normalized]
        if missing:
            raise ValueError(f"missing v0.2 metrics: {missing}")
        self.records.append(normalized)
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": V02_METRIC_SCHEMA,
            "records": self.records,
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_v02_metrics(path: Path, records: list[Mapping[str, Any]]) -> dict[str, Any]:
    logger = V02MetricLogger(path)
    for record in records:
        logger.append(record)
    return {"schema": V02_METRIC_SCHEMA, "records": logger.records}

