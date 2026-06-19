from __future__ import annotations

from dataclasses import dataclass

from .structure_types import LifecyclePhase, StructureLifecycleStats, StructureObject


@dataclass
class StructureLifecycleDecision:
    structure_id: int
    phase: LifecyclePhase
    survival_score: float
    should_retire: bool
    reason: str


class StructureLifecycleScorer:
    """NSF-style lifecycle scoring for explicit structure objects."""

    def __init__(
        self,
        *,
        usage_horizon: int = 100,
        retire_threshold: float = 0.2,
        strengthen_threshold: float = 0.7,
    ):
        if usage_horizon < 1:
            raise ValueError("usage_horizon must be at least 1")
        if not 0.0 <= retire_threshold <= 1.0:
            raise ValueError("retire_threshold must be between 0 and 1")
        if not 0.0 <= strengthen_threshold <= 1.0:
            raise ValueError("strengthen_threshold must be between 0 and 1")
        self.usage_horizon = usage_horizon
        self.retire_threshold = retire_threshold
        self.strengthen_threshold = strengthen_threshold

    def score(self, stats: StructureLifecycleStats) -> float:
        usage = min(1.0, stats.usage_count / self.usage_horizon)
        success = _clamp01(stats.success_rate)
        transfer = _clamp01(stats.transfer_gain)
        robustness = (
            _clamp01(1.0 - stats.reset_sensitivity)
            + _clamp01(1.0 - stats.shuffle_sensitivity)
            + _clamp01(stats.attack_recovery)
            + _clamp01(stats.shift_retention)
        ) / 4.0
        score = 0.25 * usage + 0.30 * success + 0.20 * transfer + 0.25 * robustness
        return _clamp01(score)

    def decide(
        self,
        structure: StructureObject,
        stats: StructureLifecycleStats,
    ) -> StructureLifecycleDecision:
        score = self.score(stats)
        structure.survival_score = score

        if stats.usage_count == 0:
            phase = LifecyclePhase.appear
            reason = "new structure has no survival evidence yet"
        elif score < self.retire_threshold:
            phase = LifecyclePhase.retire
            reason = "low survival score after lifecycle evidence"
        elif stats.success_rate < 0.25 and stats.usage_count >= 3:
            phase = LifecyclePhase.decay
            reason = "low observed success rate"
        elif stats.transfer_gain >= 0.5:
            phase = LifecyclePhase.specialize
            reason = "strong transfer gain"
        elif score >= self.strengthen_threshold:
            phase = LifecyclePhase.strengthen
            reason = "high survival score"
        else:
            phase = LifecyclePhase.survive
            reason = "structure remains viable"

        return StructureLifecycleDecision(
            structure_id=structure.structure_id,
            phase=phase,
            survival_score=score,
            should_retire=phase == LifecyclePhase.retire,
            reason=reason,
        )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
