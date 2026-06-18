import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_multihop_reasoning_advantage as bench
from kaggle import make_agentic_training_bundle


class MultiHopReasoningAdvantageTests(unittest.TestCase):
    def test_reasoning_benchmark_holds_direct_recall_constant(self):
        result = bench.run_multihop_reasoning_advantage_benchmark(
            train_seeds=[1, 2],
            eval_seeds=[101],
            model_seeds=[5, 7],
            identities_per_seed=6,
            examples_per_task=4,
            chain_lengths=[1, 2, 3],
            distractors_per_identity=2,
        )

        self.assertEqual(result["schema"], "multihop_reasoning_advantage.v1")
        self.assertFalse(result["selection_contract"]["uses_target_labels_for_selection"])
        self.assertFalse(result["boundary"]["claims_external_checkpoint_result"])
        self.assertEqual(
            result["decision"]["status"],
            "controlled_multihop_reasoning_advantage_observed",
        )

        direct = result["by_chain_length"]["1"]
        self.assertEqual(direct["tac_carried_identity_state_accuracy"], 1.0)
        self.assertEqual(direct["recall_oracle_accuracy"], 1.0)
        self.assertEqual(direct["direct_recall_regression"], 0.0)

        aggregate = result["aggregate_metrics"]
        self.assertGreaterEqual(aggregate["tac_multihop_accuracy_mean"], 0.80)
        self.assertGreaterEqual(aggregate["reasoning_lift_over_best_recall_control"], 0.30)
        self.assertGreaterEqual(aggregate["tac_multihop_accuracy_min_seed"], 0.70)

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
                    "6",
                    "--examples-per-task",
                    "4",
                    "--chain-lengths",
                    "1",
                    "2",
                    "3",
                ]
            )

            artifact = json.loads(
                (output_dir / "multihop_reasoning_advantage.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "multihop_reasoning_advantage.v1")
        self.assertIn("Controlled Multi-Hop Reasoning Advantage", markdown)
        self.assertIn(
            "experiments/benchmark_multihop_reasoning_advantage.py",
            make_agentic_training_bundle.FILES,
        )


if __name__ == "__main__":
    unittest.main()
