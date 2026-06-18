import json
import tempfile
import unittest
from pathlib import Path


class TAC256To260AcademicRoadmapTests(unittest.TestCase):
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

    def test_tac256_architecture_paper_readiness_contract(self):
        from experiments.benchmark_tac256_architecture_paper_readiness import (
            run_tac256_architecture_paper_readiness,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac256_architecture_paper_readiness(
                output_dir=Path(tmp),
                seeds=(7,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac256_architecture_paper_readiness.v1",
                (
                    "causal_program_score",
                    "reproduction_score",
                    "ablation_strength",
                    "mechanistic_clarity",
                    "paper_readiness_score",
                    "recommended_venue_tier",
                ),
            )
            self.assertEqual(result["method"]["task"], "architecture_paper_readiness")

    def test_tac257_algorithm_transfer_paper_readiness_contract(self):
        from experiments.benchmark_tac257_algorithm_transfer_paper_readiness import (
            run_tac257_algorithm_transfer_paper_readiness,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac257_algorithm_transfer_paper_readiness(
                output_dir=Path(tmp),
                seeds=(7,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac257_algorithm_transfer_paper_readiness.v1",
                (
                    "transfer_effect_size",
                    "control_survival_score",
                    "task_coverage_score",
                    "negative_transfer_safety",
                    "citation_potential_score",
                    "paper_readiness_score",
                ),
            )
            self.assertEqual(result["method"]["task"], "algorithm_transfer_paper_readiness")

    def test_tac258_context_compression_paper_readiness_contract(self):
        from experiments.benchmark_tac258_context_compression_paper_readiness import (
            run_tac258_context_compression_paper_readiness,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac258_context_compression_paper_readiness(
                output_dir=Path(tmp),
                seeds=(7,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac258_context_compression_paper_readiness.v1",
                (
                    "max_validated_compression",
                    "realistic_workload_score",
                    "stress_survival_score",
                    "state_dependency_score",
                    "scaling_boundary_clarity",
                    "paper_readiness_score",
                ),
            )
            self.assertEqual(result["method"]["task"], "context_compression_paper_readiness")

    def test_tac259_transfer_compression_scaling_study_contract(self):
        from experiments.benchmark_tac259_transfer_compression_scaling_study import (
            run_tac259_transfer_compression_scaling_study,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac259_transfer_compression_scaling_study(
                output_dir=Path(tmp),
                seeds=(7,),
                program_counts=(4,),
                compression_ratios=(10,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac259_transfer_compression_scaling_study.v1",
                (
                    "transfer_compression_correlation",
                    "program_specialization_slope",
                    "context_requirement_slope",
                    "unified_claim_score",
                    "scaling_law_clarity",
                    "paper_readiness_score",
                ),
            )
            self.assertEqual(result["method"]["task"], "transfer_compression_scaling_study")

    def test_tac260_composition_publication_gate_contract(self):
        from experiments.benchmark_tac260_composition_publication_gate import (
            run_tac260_composition_publication_gate,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac260_composition_publication_gate(
                output_dir=Path(tmp),
                seeds=(7,),
                composition_depths=(2,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_common_contract(
                result,
                "tac260_composition_publication_gate.v1",
                (
                    "composition_advantage",
                    "depth_generalization_accuracy",
                    "causal_composition_score",
                    "new_capability_score",
                    "publication_gate_score",
                    "recommended_action",
                ),
            )
            self.assertEqual(result["method"]["task"], "composition_publication_gate")

    def test_prd_contains_pending_tac256_through_tac260_tickets(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        for ticket_id in ("TAC-256", "TAC-257", "TAC-258", "TAC-259", "TAC-260"):
            self.assertIn(ticket_id, tickets)
            self.assertEqual(tickets[ticket_id]["status"], "pending")
            self.assertGreaterEqual(len(tickets[ticket_id]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
