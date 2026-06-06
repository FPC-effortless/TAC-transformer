import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_trained_identity_collapse_recovery as bench


class TrainedIdentityCollapseRecoveryTests(unittest.TestCase):
    def test_trained_model_recovers_identity_under_collapse_pressure(self):
        result = bench.run_trained_identity_collapse_recovery_probe(
            train_seeds=[1, 2, 3],
            eval_seeds=[101],
            identities_per_seed=8,
            examples_per_task=3,
            vocab_size=64,
            training_steps=160,
            model_seeds=[5, 7, 11],
            collapse_pressure=0.04,
            gradient_noise_std=0.015,
        )

        self.assertEqual(
            result["decision"]["status"],
            "trained_identity_collapse_recovery_proved",
        )
        self.assertFalse(result["training_contract"]["hidden_rule_labels_used_for_loss"])
        self.assertTrue(result["training_contract"]["collapse_pressure_applied"])
        self.assertTrue(result["training_contract"]["gradient_noise_injected"])

        metrics = result["aggregate_metrics"]
        self.assertGreaterEqual(metrics["trained_accuracy_mean"], 0.90)
        self.assertGreaterEqual(metrics["trained_accuracy_min"], 0.85)
        self.assertLessEqual(metrics["solver_gap_mean"], 0.10)
        self.assertLessEqual(metrics["best_non_identity_control_accuracy"], 0.35)
        self.assertGreaterEqual(metrics["trained_advantage_over_control"], 0.60)
        self.assertGreater(metrics["state_separation_margin_min"], 0.15)
        self.assertGreaterEqual(metrics["route_agreement_min"], 0.90)

    def test_collapse_only_control_stays_near_baseline(self):
        suite = bench.build_trained_identity_suite(
            train_seeds=[1],
            eval_seeds=[101],
            identities_per_seed=4,
            examples_per_task=2,
            vocab_size=64,
        )

        control = bench.evaluate_non_identity_controls(suite)

        self.assertLessEqual(control["best_non_identity_control_accuracy"], 0.35)
        self.assertEqual(control["solver_accuracy"], 1.0)
        self.assertAlmostEqual(control["solver_advantage"], 0.75)

    def test_cli_writes_trained_collapse_recovery_artifacts(self):
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
                    "--identities-per-seed",
                    "4",
                    "--examples-per-task",
                    "2",
                    "--training-steps",
                    "120",
                    "--model-seeds",
                    "5",
                    "7",
                ]
            )

            artifact = json.loads(
                (output_dir / "trained_identity_collapse_recovery.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "trained_identity_collapse_recovery.v1")
        self.assertIn("Trained Identity Collapse Recovery", markdown)
        self.assertIn("collapse pressure", markdown)


if __name__ == "__main__":
    unittest.main()
