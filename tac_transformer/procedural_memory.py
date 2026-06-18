from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional


@dataclass
class ProceduralStep:
    action: str
    observation: str = ""
    success: bool = False
    repair_delta: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProceduralMemoryRecord:
    record_id: int
    task_key: str
    procedure_trace: list[ProceduralStep]
    success_score: float
    usage_count: int = 0
    last_used_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProceduralMemoryRead:
    records: list[ProceduralMemoryRecord]
    query_key: str
    top_score: float


class ProceduralMemoryStore:
    """External memory for repair procedures and verifier feedback."""

    def __init__(self, max_records: int = 256):
        if max_records < 1:
            raise ValueError("max_records must be at least 1")
        self.max_records = max_records
        self._records: list[ProceduralMemoryRecord] = []
        self._next_record_id = 0

    @property
    def records(self) -> list[ProceduralMemoryRecord]:
        return list(self._records)

    def write(
        self,
        *,
        task_key: str,
        procedure_trace: Iterable[ProceduralStep],
        success_score: float,
        step: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProceduralMemoryRecord:
        if not task_key:
            raise ValueError("task_key must be non-empty")
        if not 0.0 <= success_score <= 1.0:
            raise ValueError("success_score must be between 0 and 1")
        trace = list(procedure_trace)
        if not trace:
            raise ValueError("procedure_trace must contain at least one step")

        record = ProceduralMemoryRecord(
            record_id=self._next_record_id,
            task_key=task_key,
            procedure_trace=trace,
            success_score=success_score,
            last_used_step=step,
            metadata=dict(metadata or {}),
        )
        self._next_record_id += 1
        self._records.append(record)
        self._evict_if_needed()
        return record

    def read(
        self,
        task_key: str,
        *,
        top_k: int = 1,
        min_success_score: float = 0.0,
        step: int = 0,
    ) -> ProceduralMemoryRead:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= min_success_score <= 1.0:
            raise ValueError("min_success_score must be between 0 and 1")

        matches = [
            record
            for record in self._records
            if record.task_key == task_key and record.success_score >= min_success_score
        ]
        matches.sort(
            key=lambda record: (
                record.success_score,
                record.usage_count,
                record.last_used_step,
            ),
            reverse=True,
        )
        selected = matches[:top_k]
        for record in selected:
            record.usage_count += 1
            record.last_used_step = step
        top_score = 0.0 if not selected else selected[0].success_score
        return ProceduralMemoryRead(
            records=selected,
            query_key=task_key,
            top_score=top_score,
        )

    def update_success(
        self,
        record_id: int,
        *,
        success: bool,
        step: int = 0,
    ) -> ProceduralMemoryRecord:
        record = self.get(record_id)
        n = record.usage_count + 1
        record.success_score = record.success_score + (
            float(success) - record.success_score
        ) / n
        record.usage_count = n
        record.last_used_step = step
        return record

    def get(self, record_id: int) -> ProceduralMemoryRecord:
        for record in self._records:
            if record.record_id == record_id:
                return record
        raise KeyError(record_id)

    def _evict_if_needed(self) -> None:
        if len(self._records) <= self.max_records:
            return
        self._records.sort(
            key=lambda record: (
                record.success_score,
                record.usage_count,
                record.last_used_step,
            )
        )
        del self._records[: len(self._records) - self.max_records]
