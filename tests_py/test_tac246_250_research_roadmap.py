import json
import tempfile
import unittest
from pathlib import Path


class TAC246To250ResearchRoadmapTests(unittest.TestCase):
    def _assert_common_contract(self, result, schema, metric_keys):
        self.assertEqual(result["schema"], schema)
        self.assertIn("method", result)
        self.assertIn("per_seed", result)
        self.assertIn("metrics", result)
        self.assertIn("decision", result)
        self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
        self.assertTrue(Path(result["artifact_path"]).exists())
        for key in metric_keys:
            self.assertIn(key, result["metrics"])
        json.dumps(result)

    def test_tac246_algorithm_transfer_matrix_contract(self):
        from experiments.benchmark_tac246_algorithm_transfer_matrix import (
            run_tac246_algorithm_transfer_matrix,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac246_algorithm_transfer_matrix(
                output_dir=Path(tmp),
                seeds=(7,),
                source_algorithms=("sorting",),
                target_algorithms=("graph_search",),
                train_steps=1,
                transfer_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac246_algorithm_transfer_matrix.v1",
                (
                    "cross_algorithm_transfer_accuracy",
                    "transfer_advantage_over_fresh",
                    "transfer_advantage_over_randomized",
                    "negative_transfer_rate",
                    "program_reuse_rate",
                    "selectivity_retention",
                ),
            )
            self.assertEqual(result["method"]["task"], "algorithm_transfer_matrix")

    def test_tac247_algorithm_transfer_break_controls_contract(self):
        from experiments.benchmark_tac247_algorithm_transfer_break_controls import (
            run_tac247_algorithm_transfer_break_controls,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac247_algorithm_transfer_break_controls(
                output_dir=Path(tmp),
                seeds=(7,),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac247_algorithm_transfer_break_controls.v1",
                (
                    "clean_transfer_accuracy",
                    "scrambled_label_accuracy",
                    "surface_cue_control_accuracy",
                    "program_knockout_drop",
                    "causal_transfer_gap",
                    "shortcut_resistance",
                ),
            )
            self.assertEqual(result["method"]["task"], "algorithm_transfer_break_controls")

    def test_tac248_context_compression_scaling_contract(self):
        from experiments.benchmark_tac248_context_compression_scaling import (
            run_tac248_context_compression_scaling,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac248_context_compression_scaling(
                output_dir=Path(tmp),
                seeds=(7,),
                compression_ratios=(10,),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac248_context_compression_scaling.v1",
                (
                    "max_validated_compression_ratio",
                    "mean_accuracy_gap",
                    "equal_accuracy_token_savings",
                    "collapse_ratio",
                    "state_knockout_drop",
                    "compression_curve_slope",
                ),
            )
            self.assertEqual(result["method"]["task"], "context_compression_scaling")

    def test_tac249_context_compression_stress_contract(self):
        from experiments.benchmark_tac249_context_compression_stress import (
            run_tac249_context_compression_stress,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac249_context_compression_stress(
                output_dir=Path(tmp),
                seeds=(7,),
                distractor_counts=(0,),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac249_context_compression_stress.v1",
                (
                    "stress_tac_accuracy",
                    "stress_transformer_accuracy",
                    "stress_accuracy_gap",
                    "distractor_resilience",
                    "collision_failure_rate",
                    "state_knockout_drop",
                ),
            )
            self.assertEqual(result["method"]["task"], "context_compression_stress")

    def test_tac250_program_composition_hardening_contract(self):
        from experiments.benchmark_tac250_program_composition_hardening import (
            run_tac250_program_composition_hardening,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac250_program_composition_hardening(
                output_dir=Path(tmp),
                seeds=(7,),
                composition_depths=(2,),
                train_steps=1,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac250_program_composition_hardening.v1",
                (
                    "composed_accuracy",
                    "single_program_accuracy",
                    "composition_advantage",
                    "depth_generalization_accuracy",
                    "targeted_knockout_gap",
                    "composition_consistency",
                ),
            )
            self.assertEqual(result["method"]["task"], "program_composition_hardening")

    def test_prd_contains_pending_tac246_through_tac250_tickets(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        for ticket_id in ("TAC-246", "TAC-247", "TAC-248", "TAC-249", "TAC-250"):
            self.assertIn(ticket_id, tickets)
            self.assertEqual(tickets[ticket_id]["status"], "pending")
            self.assertGreaterEqual(len(tickets[ticket_id]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
