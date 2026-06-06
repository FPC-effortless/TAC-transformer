import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_identity_interference_stress as bench


class IdentityInterferenceStressTests(unittest.TestCase):
    def test_stress_suite_maps_collision_shift_pressure_and_scaling(self):
        result = bench.run_identity_interference_stress_probe(
            train_seeds=[1, 2],
            eval_seeds=[101],
            model_seeds=[5, 7],
            identities_per_seed=8,
            examples_per_task=2,
            vocab_size=64,
            training_steps=120,
            pressure_values=[0.04, 0.5, 20.0],
        )

        self.assertEqual(result["schema"], "identity_interference_stress.v1")
        self.assertEqual(result["decision"]["status"], "identity_interference_stress_boundary_mapped")
        self.assertEqual(
            set(result["scenarios"].keys()),
            {
                "identity_collision",
                "distribution_shift",
                "adversarial_pressure_sweep",
                "scaled_load",
            },
        )

        for scenario_name in ["identity_collision", "distribution_shift", "scaled_load"]:
            scenario = result["scenarios"][scenario_name]
            self.assertEqual(scenario["status"], "passed")
            self.assertGreaterEqual(scenario["metrics"]["trained_accuracy_mean"], 0.90)
            self.assertGreaterEqual(scenario["metrics"]["route_agreement_min"], 0.90)
            self.assertGreater(scenario["metrics"]["state_separation_margin_min"], 0.15)

        pressure = result["scenarios"]["adversarial_pressure_sweep"]
        self.assertEqual(pressure["status"], "boundary_observed")
        self.assertEqual(pressure["phase_transition_pressure"], 20.0)
        self.assertGreaterEqual(pressure["by_pressure"]["0.04"]["trained_accuracy_mean"], 0.90)
        self.assertLess(pressure["by_pressure"]["20.0"]["trained_accuracy_mean"], 0.90)

    def test_distribution_shift_holds_out_task_families_from_training(self):
        scenario = bench.run_distribution_shift_scenario(
            train_seeds=[1, 2],
            eval_seeds=[101],
            model_seeds=[5],
            identities_per_seed=8,
            examples_per_task=2,
            vocab_size=64,
            training_steps=120,
        )

        self.assertEqual(scenario["status"], "passed")
        self.assertEqual(
            scenario["train_task_families"],
            ["transfer_learning", "agent_memory"],
        )
        self.assertEqual(
            scenario["eval_task_families"],
            ["multi_hop_reasoning", "language_like_instruction"],
        )

    def test_cli_writes_interference_stress_artifacts(self):
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
                    "4",
                    "--examples-per-task",
                    "2",
                    "--training-steps",
                    "100",
                    "--pressure-values",
                    "0.04",
                    "20.0",
                ]
            )

            artifact = json.loads(
                (output_dir / "identity_interference_stress.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "identity_interference_stress.v1")
        self.assertIn("Identity Interference Stress", markdown)
        self.assertIn("phase_transition_pressure", markdown)


if __name__ == "__main__":
    unittest.main()
