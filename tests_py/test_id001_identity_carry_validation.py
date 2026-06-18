import json
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_id001_identity_carry_validation import (
    build_id001_protocol,
    evaluate_id001_identity_carry_validation,
    format_id001_markdown,
    run_id001_identity_carry_validation,
)


class ID001IdentityCarryValidationTests(unittest.TestCase):
    def test_protocol_freezes_identity_carry_controls(self):
        protocol = build_id001_protocol()

        self.assertEqual(protocol["schema"], "id001_identity_carry_protocol.v1")
        self.assertEqual(protocol["gate"], "ID001")
        self.assertEqual(protocol["risk"], "identity_carry_value")
        self.assertEqual(protocol["mechanisms"], ["IdentityState", "IdentityField"])
        self.assertEqual(
            protocol["required_controls"],
            ["carried_identity", "reset_identity", "shuffled_identity", "identity_knockout"],
        )
        self.assertIn("carried > reset", protocol["success_criteria"])
        self.assertIn("knockout hurts", protocol["success_criteria"])

    def test_evaluator_blocks_when_effect_families_are_missing(self):
        result = evaluate_id001_identity_carry_validation(
            [
                self._row("structure_memory", "carried_identity", 0.80),
                self._row("structure_memory", "reset_identity", 0.45),
                self._row("structure_memory", "shuffled_identity", 0.40),
                self._row("structure_memory", "identity_knockout", 0.25),
            ]
        )

        self.assertEqual(result["decision"]["status"], "blocked")
        self.assertIn("procedural_memory", result["decision"]["missing_effects"])
        self.assertFalse(result["decision"]["passes_identity_gate"])

    def test_evaluator_validates_carried_identity_advantage(self):
        rows = []
        for effect, carried, reset, shuffled, knockout in (
            ("structure_memory", 0.84, 0.48, 0.44, 0.22),
            ("procedural_memory", 0.78, 0.39, 0.34, 0.20),
            ("identity_carry", 0.88, 0.42, 0.37, 0.18),
        ):
            rows.extend(
                [
                    self._row(effect, "carried_identity", carried),
                    self._row(effect, "reset_identity", reset),
                    self._row(effect, "shuffled_identity", shuffled),
                    self._row(effect, "identity_knockout", knockout),
                ]
            )

        result = evaluate_id001_identity_carry_validation(rows)

        self.assertEqual(result["decision"]["status"], "validated")
        self.assertTrue(result["decision"]["passes_identity_gate"])
        self.assertGreater(result["metrics"]["mean_carry_reset_delta"], 0.0)
        self.assertGreater(result["metrics"]["mean_carry_shuffled_delta"], 0.0)
        self.assertGreater(result["metrics"]["mean_knockout_drop"], 0.0)
        for effect in ("structure_memory", "procedural_memory", "identity_carry"):
            self.assertTrue(result["effects"][effect]["passes"])

    def test_evaluator_rejects_when_shuffled_matches_carried(self):
        rows = []
        for effect, carried, reset, shuffled, knockout in (
            ("structure_memory", 0.84, 0.48, 0.44, 0.22),
            ("procedural_memory", 0.78, 0.39, 0.79, 0.20),
            ("identity_carry", 0.88, 0.42, 0.37, 0.18),
        ):
            rows.extend(
                [
                    self._row(effect, "carried_identity", carried),
                    self._row(effect, "reset_identity", reset),
                    self._row(effect, "shuffled_identity", shuffled),
                    self._row(effect, "identity_knockout", knockout),
                ]
            )

        result = evaluate_id001_identity_carry_validation(rows)

        self.assertEqual(result["decision"]["status"], "not_validated")
        self.assertFalse(result["effects"]["procedural_memory"]["passes"])
        self.assertLess(result["effects"]["procedural_memory"]["carry_shuffled_delta"], 0.0)

    def test_runner_writes_json_and_markdown(self):
        rows = []
        for effect, carried, reset, shuffled, knockout in (
            ("structure_memory", 0.84, 0.48, 0.44, 0.22),
            ("procedural_memory", 0.78, 0.39, 0.34, 0.20),
            ("identity_carry", 0.88, 0.42, 0.37, 0.18),
        ):
            rows.extend(
                [
                    self._row(effect, "carried_identity", carried),
                    self._row(effect, "reset_identity", reset),
                    self._row(effect, "shuffled_identity", shuffled),
                    self._row(effect, "identity_knockout", knockout),
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "rows.json"
            input_path.write_text(json.dumps(rows), encoding="utf-8")
            result = run_id001_identity_carry_validation(
                output_dir=Path(tmp) / "out",
                results_path=input_path,
            )

            self.assertEqual(result["schema"], "id001_identity_carry_validation.v1")
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertTrue(Path(result["markdown_path"]).exists())
            markdown = format_id001_markdown(result)
            self.assertIn("# ID001 Identity Carry Validation", markdown)
            self.assertIn("validated", markdown)

    def _row(self, effect: str, control: str, score: float) -> dict:
        return {
            "task_id": f"{effect}-{control}",
            "effect": effect,
            "control": control,
            "primary_score": score,
            "mechanisms": ["IdentityState", "IdentityField"],
        }


if __name__ == "__main__":
    unittest.main()
