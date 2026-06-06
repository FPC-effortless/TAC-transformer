import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_live_persistent_identity_state_bridge as bench


class LivePersistentIdentityStateBridgeTests(unittest.TestCase):
    def test_live_state_bridge_proves_carried_identity_advantage(self):
        result = bench.run_live_state_bridge_probe(
            seeds=[2, 3],
            identities_per_seed=8,
            examples_per_task=4,
            vocab_size=64,
        )

        self.assertEqual(
            result["decision"]["status"],
            "live_persistent_identity_state_bridge_proved",
        )
        self.assertEqual(result["state_adapter"]["hidden_rule_labels_used"], False)
        self.assertEqual(result["state_adapter"]["uses_identity_keyed_state"], True)
        self.assertEqual(
            set(result["task_metrics"].keys()),
            {
                "transfer_learning",
                "multi_hop_reasoning",
                "agent_memory",
                "language_like_instruction",
            },
        )
        for metrics in result["task_metrics"].values():
            self.assertGreaterEqual(metrics["carried_identity_state_accuracy"], 0.95)
            self.assertLessEqual(metrics["best_non_identity_accuracy"], 0.35)
            self.assertGreaterEqual(metrics["carried_state_advantage"], 0.60)

    def test_reset_state_loses_support_observations(self):
        suite = bench.build_live_state_suite(
            seeds=[5],
            identities_per_seed=4,
            examples_per_task=3,
            vocab_size=64,
        )

        carried = bench.evaluate_live_state_adapter(suite, control="carried_identity_state")
        reset = bench.evaluate_live_state_adapter(suite, control="reset_per_query_state")

        self.assertEqual(carried["accuracy"], 1.0)
        self.assertLessEqual(reset["accuracy"], 0.35)
        self.assertGreater(carried["state_update_count"], 0)
        self.assertEqual(reset["state_update_count"], 0)

    def test_cli_writes_live_state_bridge_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "7",
                    "--identities-per-seed",
                    "4",
                    "--examples-per-task",
                    "2",
                ]
            )

            artifact = json.loads(
                (output_dir / "live_persistent_identity_state_bridge.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "live_persistent_identity_state_bridge.v1")
        self.assertIn("Live Persistent Identity State Bridge", markdown)
        self.assertIn("not a trained checkpoint", markdown)


if __name__ == "__main__":
    unittest.main()
