from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


AUTHORITY_REPORT_SCHEMA = "tac_transformer.authority_report.v1"
CURRICULUM_REPORT_SCHEMA = "tac_transformer.curriculum_report.v1"

TRUSTED_AUTHORITY_MODES = frozenset(
    {
        "verified_memory",
        "verified_execution",
        "exact_memory",
        "retrieved_evidence",
    }
)
PROPOSAL_AUTHORITY_MODES = frozenset({"proposal", "proposal_only"})
GUESS_AUTHORITY_MODES = frozenset({"guess", "unverified_guess"})
REJECTED_AUTHORITY_MODES = frozenset({"rejected", "abstain"})


def _require_non_empty(value: str, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(_jsonable(row), sort_keys=True) + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class AuthorityEvent:
    case_id: str
    domain: str
    authority_mode: str
    accepted: bool
    correct: bool | None = None
    source_domain: str | None = None
    program_id: str | None = None
    confidence: float | None = None
    allow_cross_domain: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _require_non_empty(self.case_id, "case_id"))
        object.__setattr__(self, "domain", _require_non_empty(self.domain, "domain"))
        object.__setattr__(
            self,
            "authority_mode",
            _require_non_empty(self.authority_mode, "authority_mode"),
        )
        if self.source_domain is not None:
            object.__setattr__(
                self,
                "source_domain",
                _require_non_empty(self.source_domain, "source_domain"),
            )
        if self.program_id is not None:
            object.__setattr__(
                self,
                "program_id",
                _require_non_empty(self.program_id, "program_id"),
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        object.__setattr__(self, "metadata", _jsonable(dict(self.metadata)))

    @property
    def is_trusted(self) -> bool:
        return self.authority_mode in TRUSTED_AUTHORITY_MODES

    @property
    def is_proposal(self) -> bool:
        return self.authority_mode in PROPOSAL_AUTHORITY_MODES

    @property
    def is_guess(self) -> bool:
        return self.authority_mode in GUESS_AUTHORITY_MODES

    @property
    def is_rejected(self) -> bool:
        return (not self.accepted) or self.authority_mode in REJECTED_AUTHORITY_MODES

    @property
    def is_false_authority(self) -> bool:
        return self.accepted and self.is_trusted and self.correct is False

    @property
    def is_cross_domain_authority_violation(self) -> bool:
        if not self.accepted or not self.is_trusted or self.allow_cross_domain:
            return False
        return self.source_domain is not None and self.source_domain != self.domain

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "authority_mode": self.authority_mode,
            "accepted": self.accepted,
            "correct": self.correct,
            "source_domain": self.source_domain,
            "program_id": self.program_id,
            "confidence": self.confidence,
            "allow_cross_domain": self.allow_cross_domain,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "AuthorityEvent":
        return cls(
            case_id=str(payload["case_id"]),
            domain=str(payload["domain"]),
            authority_mode=str(payload["authority_mode"]),
            accepted=bool(payload["accepted"]),
            correct=payload.get("correct"),
            source_domain=payload.get("source_domain"),
            program_id=payload.get("program_id"),
            confidence=payload.get("confidence"),
            allow_cross_domain=bool(payload.get("allow_cross_domain", False)),
            metadata=payload.get("metadata") or {},
        )


@dataclass(frozen=True)
class AuthorityReport:
    run_id: str
    events: tuple[AuthorityEvent, ...]
    schema: str = AUTHORITY_REPORT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_non_empty(self.run_id, "run_id"))
        object.__setattr__(self, "events", tuple(self.events))

    def accepted_trusted_events(self) -> list[AuthorityEvent]:
        return [
            event
            for event in self.events
            if event.accepted and event.is_trusted
        ]

    def false_authority_events(self) -> list[AuthorityEvent]:
        return [event for event in self.events if event.is_false_authority]

    def cross_domain_authority_violations(self) -> list[AuthorityEvent]:
        return [
            event
            for event in self.events
            if event.is_cross_domain_authority_violation
        ]

    def to_manifest(self) -> dict[str, Any]:
        trusted_events = self.accepted_trusted_events()
        trusted_with_verdict = [
            event
            for event in trusted_events
            if event.correct is not None
        ]
        trusted_correct = [
            event
            for event in trusted_with_verdict
            if event.correct is True
        ]
        trusted_accuracy = (
            len(trusted_correct) / len(trusted_with_verdict)
            if trusted_with_verdict
            else None
        )
        authority_mode_counts = Counter(event.authority_mode for event in self.events)
        domain_counts = Counter(event.domain for event in self.events)
        correct_count = sum(1 for event in self.events if event.correct is True)

        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "event_count": len(self.events),
            "accepted_event_count": sum(1 for event in self.events if event.accepted),
            "rejected_event_count": sum(1 for event in self.events if event.is_rejected),
            "correct_event_count": correct_count,
            "trusted_event_count": len(trusted_events),
            "trusted_correct_count": len(trusted_correct),
            "trusted_accuracy": trusted_accuracy,
            "false_authority_count": len(self.false_authority_events()),
            "cross_domain_authority_violation_count": len(
                self.cross_domain_authority_violations()
            ),
            "proposal_event_count": sum(1 for event in self.events if event.is_proposal),
            "guess_event_count": sum(1 for event in self.events if event.is_guess),
            "authority_mode_counts": dict(sorted(authority_mode_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
        }

    def save_artifacts(self, output_dir: str | Path) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / "manifest.json"
        events_path = output_path / "authority_events.jsonl"

        _write_json(manifest_path, self.to_manifest())
        _write_jsonl(events_path, (event.to_json() for event in self.events))

        return {
            "manifest": manifest_path,
            "authority_events": events_path,
        }


@dataclass(frozen=True)
class VerifierCase:
    case_id: str
    domain: str
    expected: Any
    observed: Any
    authority_mode: str
    source_domain: str | None = None
    program_id: str | None = None
    accepted: bool = True
    confidence: float | None = None
    allow_cross_domain: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _require_non_empty(self.case_id, "case_id"))
        object.__setattr__(self, "domain", _require_non_empty(self.domain, "domain"))
        object.__setattr__(
            self,
            "authority_mode",
            _require_non_empty(self.authority_mode, "authority_mode"),
        )
        object.__setattr__(self, "expected", _jsonable(self.expected))
        object.__setattr__(self, "observed", _jsonable(self.observed))
        object.__setattr__(self, "metadata", _jsonable(dict(self.metadata)))

    @property
    def correct(self) -> bool:
        return self.expected == self.observed

    def verify(self) -> AuthorityEvent:
        return AuthorityEvent(
            case_id=self.case_id,
            domain=self.domain,
            authority_mode=self.authority_mode,
            accepted=self.accepted,
            correct=self.correct,
            source_domain=self.source_domain,
            program_id=self.program_id,
            confidence=self.confidence,
            allow_cross_domain=self.allow_cross_domain,
            metadata=self.metadata,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "expected": self.expected,
            "observed": self.observed,
            "correct": self.correct,
            "authority_mode": self.authority_mode,
            "source_domain": self.source_domain,
            "program_id": self.program_id,
            "accepted": self.accepted,
            "confidence": self.confidence,
            "allow_cross_domain": self.allow_cross_domain,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CurriculumReport:
    curriculum_id: str
    cases: tuple[VerifierCase, ...]
    authority_report: AuthorityReport
    schema: str = CURRICULUM_REPORT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "curriculum_id",
            _require_non_empty(self.curriculum_id, "curriculum_id"),
        )
        object.__setattr__(self, "cases", tuple(self.cases))

    def _domain_accuracy(self) -> dict[str, float]:
        totals: dict[str, int] = defaultdict(int)
        solved: dict[str, int] = defaultdict(int)
        for case in self.cases:
            totals[case.domain] += 1
            solved[case.domain] += int(case.correct)
        return {
            domain: solved[domain] / totals[domain]
            for domain in sorted(totals)
        }

    def to_manifest(self) -> dict[str, Any]:
        solved_case_count = sum(1 for case in self.cases if case.correct)
        return {
            "schema": self.schema,
            "curriculum_id": self.curriculum_id,
            "case_count": len(self.cases),
            "solved_case_count": solved_case_count,
            "accuracy": solved_case_count / len(self.cases) if self.cases else None,
            "domain_accuracy": self._domain_accuracy(),
            "authority": self.authority_report.to_manifest(),
        }

    def save_artifacts(self, output_dir: str | Path) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / "manifest.json"
        cases_path = output_path / "cases.jsonl"
        authority_events_path = output_path / "authority_events.jsonl"

        _write_json(manifest_path, self.to_manifest())
        _write_jsonl(cases_path, (case.to_json() for case in self.cases))
        _write_jsonl(
            authority_events_path,
            (event.to_json() for event in self.authority_report.events),
        )

        return {
            "manifest": manifest_path,
            "cases": cases_path,
            "authority_events": authority_events_path,
        }


def build_authority_report(
    run_id: str,
    events: Iterable[AuthorityEvent],
) -> AuthorityReport:
    return AuthorityReport(run_id=run_id, events=tuple(events))


def build_curriculum_report(
    curriculum_id: str,
    cases: Iterable[VerifierCase],
) -> CurriculumReport:
    case_tuple = tuple(cases)
    authority_report = build_authority_report(
        curriculum_id,
        (case.verify() for case in case_tuple),
    )
    return CurriculumReport(
        curriculum_id=curriculum_id,
        cases=case_tuple,
        authority_report=authority_report,
    )


def load_authority_events(path: str | Path) -> list[AuthorityEvent]:
    events: list[AuthorityEvent] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(AuthorityEvent.from_json(json.loads(line)))
    return events
