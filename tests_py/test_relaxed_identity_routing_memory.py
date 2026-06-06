import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_relaxed_identity_routing_memory as bench


class RelaxedIdentityRoutingMemoryTests(unittest.TestCase):
    def test_relaxed_router_memory_passes_long_horizon_gate(self):
        result = bench.run_relaxed_identity_routing_memory_probe(
            train_seeds=[1, 2],
            eval_seeds=[101],
            model_seeds=[5, 7],
            identities_per_seed=8,
            examples_per_task=2,
            horizon_windows=4,
            vocab_size=64,
            training_steps=140,
            collapse_pressure=0.02,
            memory_noise_std=0.01,
        )

        self.assertEqual(result["schema"], "relaxed_identity_routing_memory.v1")
        self.assertEqual(
            result["decision"]["status"],
            "relaxed_identity_routing_memory_promote_candidate",
        )
        self.assertFalse(result["training_contract"]["explicit_route_labels_used_for_loss"])
        self.assertFalse(result["training_contract"]["hidden_rule_labels_used_for_loss"])
        self.assertTrue(result["training_contract"]["trainable_memory_subsystem"])
        self.assertTrue(result["training_contract"]["soft_routing"])

        metrics = result["aggregate_metrics"]
        self.assertGreaterEqual(metrics["carried_accuracy_mean"], 0.90)
        self.assertGreaterEqual(metrics["horizon_tail_accuracy_mean"], 0.90)
        self.assertLessEqual(metrics["reset_accuracy_mean"], 0.35)
        self.assertLessEqual(metrics["shuffled_memory_accuracy_mean"], 0.35)
        self.assertGreaterEqual(metrics["carried_advantage_over_best_control"], 0.55)
        self.assertGreaterEqual(metrics["route_rule_nmi_min"], 0.75)
        self.assertGreaterEqual(metrics["route_consistency_min"], 0.90)

    def test_suite_uses_multi_window_queries_without_training_labels(self):
        suite = bench.build_relaxed_identity_sequence_suite(
            seeds=[3],
            identities_per_seed=8,
            examples_per_task=2,
            horizon_windows=3,
            vocab_size=64,
        )

        self.assertEqual(suite["schema"], "relaxed_identity_sequence_suite.v1")
        self.assertEqual(suite["horizon_windows"], 3)
        self.assertGreater(len(suite["rows"]), len(suite["identity_support"]))
        self.assertFalse(suite["training_contract"]["query_rows_include_rule_label"])
        self.assertFalse(suite["training_contract"]["route_labels_available_to_model"])
        self.assertEqual(
            sorted({row["horizon_window"] for row in suite["rows"]}),
            [0, 1, 2],
        )

    def test_cli_writes_relaxed_routing_memory_artifacts(self):
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
                    "--horizon-windows",
                    "3",
                    "--training-steps",
                    "120",
                ]
            )

            artifact = json.loads(
                (output_dir / "relaxed_identity_routing_memory.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "relaxed_identity_routing_memory.v1")
        self.assertIn("Relaxed Identity Routing Memory", markdown)
        self.assertIn("shuffled memory", markdown)


if __name__ == "__main__":
    unittest.main()
