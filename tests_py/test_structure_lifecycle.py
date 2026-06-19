from __future__ import annotations

import unittest

from tac_transformer.structure_lifecycle import (
    StructureLifecycleDecision,
    StructureLifecycleScorer,
)
from tac_transformer.structure_types import (
    LifecyclePhase,
    StructureLifecycleStats,
    StructureObject,
)


class TestStructureLifecycleScorer(unittest.TestCase):
    def test_new_structure_is_appear_phase(self):
        scorer = StructureLifecycleScorer()
        obj = StructureObject(structure_id=1)
        decision = scorer.decide(obj, StructureLifecycleStats())
        self.assertIsInstance(decision, StructureLifecycleDecision)
        self.assertEqual(decision.phase, LifecyclePhase.appear)
        self.assertFalse(decision.should_retire)

    def test_strong_transfer_structure_specializes(self):
        scorer = StructureLifecycleScorer()
        obj = StructureObject(structure_id=2)
        stats = StructureLifecycleStats(
            usage_count=20,
            success_rate=0.9,
            transfer_gain=0.7,
            attack_recovery=0.9,
            shift_retention=0.8,
        )
        decision = scorer.decide(obj, stats)
        self.assertEqual(decision.phase, LifecyclePhase.specialize)
        self.assertGreater(decision.survival_score, 0.0)
        self.assertEqual(obj.survival_score, decision.survival_score)

    def test_low_survival_structure_retires(self):
        scorer = StructureLifecycleScorer(retire_threshold=0.3)
        obj = StructureObject(structure_id=3)
        stats = StructureLifecycleStats(
            usage_count=10,
            success_rate=0.0,
            reset_sensitivity=1.0,
            shuffle_sensitivity=1.0,
            attack_recovery=0.0,
            shift_retention=0.0,
        )
        decision = scorer.decide(obj, stats)
        self.assertEqual(decision.phase, LifecyclePhase.retire)
        self.assertTrue(decision.should_retire)

    def test_reset_and_shuffle_sensitivity_lower_score(self):
        scorer = StructureLifecycleScorer()
        stable = StructureLifecycleStats(
            usage_count=20,
            success_rate=0.8,
            transfer_gain=0.2,
            reset_sensitivity=0.0,
            shuffle_sensitivity=0.0,
            attack_recovery=0.8,
            shift_retention=0.8,
        )
        brittle = StructureLifecycleStats(
            usage_count=20,
            success_rate=0.8,
            transfer_gain=0.2,
            reset_sensitivity=1.0,
            shuffle_sensitivity=1.0,
            attack_recovery=0.0,
            shift_retention=0.0,
        )
        self.assertGreater(scorer.score(stable), scorer.score(brittle))

    def test_invalid_constructor_values_are_rejected(self):
        with self.assertRaises(ValueError):
            StructureLifecycleScorer(usage_horizon=0)
        with self.assertRaises(ValueError):
            StructureLifecycleScorer(retire_threshold=-0.1)
        with self.assertRaises(ValueError):
            StructureLifecycleScorer(strengthen_threshold=1.1)


if __name__ == "__main__":
    unittest.main()
