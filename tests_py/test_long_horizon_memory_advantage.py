import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_long_horizon_memory_advantage as bench
from kaggle import make_agentic_training_bundle


class LongHorizonMemoryAdvantageTests(unittest.TestCase):
    def test_benchmark_answers_attachment_question_with_equal_resource_controls(self):
        result = bench.run_long_horizon_memory_advantage_benchmark(
            train_seeds=[1, 2],
            eval_seeds=[101],
            model_seeds=[5],
            identities_per_seed=8,
            examples_per_task=2,
            horizon_windows=4,
            vocab_size=64,
            training_steps=260,
            context_budgets=[6, 10, 14, 22, 38, 62],
        )

        self.assertEqual(result["schema"], "long_horizon_memory_advantage.v1")
        self.assertIn("persistent computational identity", result["primary_question"])
        self.assertEqual(
            result["decision"]["status"],
            "controlled_long_horizon_memory_advantage_observed",
        )
        self.assertTrue(result["resource_contract"]["same_task_rows"])
        self.assertTrue(result["resource_contract"]["same_parameter_budget_contract"])
        self.assertFalse(result["boundary"]["claims_external_checkpoint_result"])

        control_ids = {control["id"] for control in result["controls"]}
        self.assertEqual(
            control_ids,
            {
                "tac_carried_identity_state",
                "transformer_window",
                "transformer_retrieval",
                "transformer_memory_db",
                "tac_reset_state",
                "tac_shuffled_state",
            },
        )

        tokens = result["tokens_required_for_target_success"]
        self.assertEqual(tokens["tac_carried_identity_state"], 6)
        self.assertGreater(tokens["transformer_window"], tokens["tac_carried_identity_state"])
        self.assertGreater(tokens["transformer_retrieval"], tokens["tac_carried_identity_state"])
        self.assertGreater(tokens["transformer_memory_db"], tokens["tac_carried_identity_state"])
        self.assertGreaterEqual(result["aggregate_metrics"]["tac_carried_accuracy_mean"], 0.90)
        self.assertGreaterEqual(
            result["aggregate_metrics"]["advantage_at_tac_context_budget"],
            0.50,
        )

    def test_context_curve_and_days_curve_have_expected_shape(self):
        result = bench.run_long_horizon_memory_advantage_benchmark(
            train_seeds=[1, 2],
            eval_seeds=[101],
            model_seeds=[5],
            identities_per_seed=8,
            examples_per_task=2,
            horizon_windows=4,
            vocab_size=64,
            training_steps=260,
            context_budgets=[6, 10, 14, 22, 38, 62],
        )

        context_rows = result["graphs"]["context_tokens_required_vs_task_success"]
        day_rows = result["graphs"]["days_since_instruction_vs_accuracy"]
        self.assertTrue(context_rows)
        self.assertTrue(day_rows)
        self.assertIn("Context Tokens Required vs Task Success", result["target_graph"])
        self.assertEqual(result["fixed_low_context_budget"], 6)

        tac_curve = [
            row for row in context_rows if row["control_id"] == "tac_carried_identity_state"
        ]
        db_curve = [
            row for row in context_rows if row["control_id"] == "transformer_memory_db"
        ]
        self.assertEqual(tac_curve[0]["context_tokens"], 6)
        self.assertGreaterEqual(tac_curve[0]["task_success"], 0.90)
        self.assertLess(db_curve[0]["task_success"], 0.90)
        self.assertGreaterEqual(db_curve[2]["task_success"], 0.90)

    def test_cli_writes_artifacts_and_bundle_includes_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--train-seeds",
                    "1",
                    "2",
                    "--eval-seeds",
                    "101",
                    "--model-seeds",
                    "5",
                    "--identities-per-seed",
                    "8",
                    "--examples-per-task",
                    "2",
                    "--horizon-windows",
                    "4",
                    "--training-steps",
                    "260",
                    "--context-budgets",
                    "6",
                    "10",
                    "14",
                    "22",
                    "38",
                    "62",
                ]
            )

            artifact = json.loads(
                (output_dir / "long_horizon_memory_advantage.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")
            with (output_dir / "context_tokens_required_vs_task_success.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                context_rows = list(csv.DictReader(handle))

        self.assertEqual(artifact["schema"], "long_horizon_memory_advantage.v1")
        self.assertIn("Controlled Long-Horizon Memory Advantage", markdown)
        self.assertTrue(context_rows)
        self.assertIn(
            "experiments/benchmark_long_horizon_memory_advantage.py",
            make_agentic_training_bundle.FILES,
        )


if __name__ == "__main__":
    unittest.main()
