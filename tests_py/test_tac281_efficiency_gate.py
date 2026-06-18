import json
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_tac281_efficiency_gate import (
    build_tac281_protocol,
    evaluate_tac281_efficiency_gate,
    format_tac281_gate_markdown,
    run_tac281_efficiency_gate,
)


class TAC281EfficiencyGateTests(unittest.TestCase):
    def test_protocol_requires_three_efficiency_variants(self):
        protocol = build_tac281_protocol()

        self.assertEqual(protocol["schema"], "tac281_efficiency_gate_protocol.v1")
        self.assertEqual(protocol["gate"], "TAC-281")
        self.assertEqual(protocol["risk"], "lm_efficiency_penalty")
        self.assertEqual(
            protocol["required_variants"],
            ["late_bottleneck", "small_adapter", "auxiliary_mechanism"],
        )
        self.assertIn("mechanism wins >= 3 of 4", protocol["success_criteria"])
        self.assertIn("LM loss gap shrinks by at least 30%", protocol["success_criteria"])
        self.assertIn("speed penalty reduced", protocol["success_criteria"])

    def test_gate_blocks_until_all_variants_are_accounted_for(self):
        result = evaluate_tac281_efficiency_gate(
            {
                "schema": "tac_v02_tac281_variant_decision.v1",
                "decision": {"status": "do_not_scale_yet"},
                "variants": [self._variant("small_adapter", "not_scale_ready")],
            }
        )

        self.assertEqual(result["decision"]["status"], "blocked")
        self.assertIn("late_bottleneck", result["decision"]["missing_variants"])
        self.assertIn("auxiliary_mechanism", result["decision"]["missing_variants"])
        self.assertFalse(result["decision"]["passes_efficiency_gate"])

    def test_gate_validates_when_any_complete_variant_is_scale_ready(self):
        summary = {
            "schema": "tac_v02_tac281_variant_decision.v1",
            "decision": {"status": "scale_to_112m"},
            "variants": [
                self._variant("late_bottleneck", "not_scale_ready"),
                self._variant("small_adapter", "scale_ready"),
                self._variant("auxiliary_mechanism", "not_scale_ready"),
            ],
        }

        result = evaluate_tac281_efficiency_gate(summary)

        self.assertEqual(result["decision"]["status"], "validated")
        self.assertTrue(result["decision"]["passes_efficiency_gate"])
        self.assertEqual(result["decision"]["scale_ready_variants"], ["small_adapter"])
        self.assertGreaterEqual(result["variants"]["small_adapter"]["lm_gap_shrink_fraction"], 0.30)

    def test_gate_rejects_complete_non_scale_ready_result(self):
        summary = {
            "schema": "tac_v02_tac281_variant_decision.v1",
            "decision": {"status": "do_not_scale_yet"},
            "variants": [
                self._variant("late_bottleneck", "not_scale_ready"),
                self._variant("small_adapter", "not_scale_ready"),
                self._variant("auxiliary_mechanism", "not_scale_ready"),
            ],
        }

        result = evaluate_tac281_efficiency_gate(summary)

        self.assertEqual(result["decision"]["status"], "not_validated")
        self.assertFalse(result["decision"]["passes_efficiency_gate"])
        self.assertEqual(result["decision"]["scale_ready_variants"], [])

    def test_runner_writes_json_and_markdown(self):
        summary = {
            "schema": "tac_v02_tac281_variant_decision.v1",
            "decision": {"status": "scale_to_112m"},
            "variants": [
                self._variant("late_bottleneck", "not_scale_ready"),
                self._variant("small_adapter", "scale_ready"),
                self._variant("auxiliary_mechanism", "not_scale_ready"),
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "tac281_variant_decision.json"
            input_path.write_text(json.dumps(summary), encoding="utf-8")
            result = run_tac281_efficiency_gate(
                output_dir=Path(tmp) / "out",
                decision_path=input_path,
            )

            self.assertEqual(result["schema"], "tac281_efficiency_gate.v1")
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertTrue(Path(result["markdown_path"]).exists())
            markdown = format_tac281_gate_markdown(result)
            self.assertIn("# TAC-281 Efficiency Gate", markdown)
            self.assertIn("validated", markdown)

    def _variant(self, variant: str, status: str) -> dict:
        scale_ready = status == "scale_ready"
        return {
            "variant": variant,
            "status": status,
            "checks": {
                "mechanism_wins_ge_3_of_4": True,
                "carry_advantage_positive": True,
                "knockout_delta_positive": scale_ready,
                "lm_gap_shrinks_enough": True,
                "speed_penalty_reduced": True,
            },
            "lm": {
                "gap_shrink_fraction": 0.45 if scale_ready else 0.20,
                "required_gap_shrink_fraction": 0.30,
            },
            "speed": {
                "original_speed_penalty": 14.19,
                "current_speed_penalty": 4.35,
            },
            "mechanisms": {
                "tac_win_families": 3,
                "tac_carry_advantage": 0.04,
                "bottleneck_knockout_delta": 0.02 if scale_ready else 0.0,
            },
        }


if __name__ == "__main__":
    unittest.main()
