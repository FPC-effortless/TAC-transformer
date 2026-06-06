import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments import benchmark_persistent_computational_identity as bench


class PersistentComputationalIdentityTests(unittest.TestCase):
    def test_persistent_identity_beats_reset_and_memory_controls(self):
        result = bench.run_persistent_identity_probe(
            seeds=[3, 5],
            identities_per_seed=8,
            queries_per_identity=6,
            vocab_size=64,
        )

        self.assertEqual(
            result["decision"]["status"],
            "persistent_computational_identity_proved",
        )
        self.assertGreaterEqual(result["metrics"]["persistent_identity_accuracy"], 0.95)
        self.assertLessEqual(result["metrics"]["stateless_reset_accuracy"], 0.30)
        self.assertLessEqual(result["metrics"]["memory_only_unseen_accuracy"], 0.05)
        self.assertGreaterEqual(
            result["metrics"]["persistent_advantage_over_best_non_identity"],
            0.45,
        )

    def test_theorem_bound_matches_balanced_rule_prior(self):
        result = bench.run_persistent_identity_probe(
            seeds=[7],
            identities_per_seed=8,
            queries_per_identity=4,
            vocab_size=64,
        )
        theorem = result["theorem"]

        self.assertEqual(theorem["rule_count"], 4)
        self.assertAlmostEqual(theorem["stateless_upper_bound"], 0.25)
        self.assertAlmostEqual(theorem["constructive_persistent_accuracy"], 1.0)
        self.assertAlmostEqual(theorem["proved_advantage_lower_bound"], 0.75)

    def test_identity_keying_prevents_global_persistence_overwrite(self):
        suite = bench.build_identity_probe_suite(
            seeds=[11],
            identities_per_seed=4,
            queries_per_identity=5,
            vocab_size=64,
        )

        persistent = bench.evaluate_persistent_identity_solver(suite)
        global_only = bench.evaluate_global_persistent_solver(suite)

        self.assertEqual(persistent["accuracy"], 1.0)
        self.assertLessEqual(global_only["accuracy"], 0.30)
        self.assertGreater(
            persistent["accuracy"] - global_only["accuracy"],
            0.65,
        )

    def test_program_computation_generalizes_to_unseen_values(self):
        values = torch.tensor([4, 5, 6, 7], dtype=torch.long)
        programs = ["copy", "successor", "predecessor", "affine_jump"]
        targets = bench.apply_rule_bank(values, programs, vocab_size=64)

        self.assertEqual(targets.tolist()[0], 4)
        self.assertEqual(targets.tolist()[1], 6)
        self.assertEqual(targets.tolist()[2], 5)
        self.assertEqual(targets.tolist()[3], 21)

    def test_cli_writes_json_and_markdown_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "1",
                    "--identities-per-seed",
                    "4",
                    "--queries-per-identity",
                    "3",
                ]
            )

            artifact = json.loads(
                (
                    output_dir / "persistent_computational_identity.json"
                ).read_text(encoding="utf-8")
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(
            artifact["schema"],
            "persistent_computational_identity.v1",
        )
        self.assertIn("Persistent Computational Identity", markdown)
        self.assertIn(artifact["decision"]["status"], markdown)


if __name__ == "__main__":
    unittest.main()
