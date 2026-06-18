import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_forced_identity_objective as bench


class ForcedIdentityObjectiveTests(unittest.TestCase):
    def test_batch_shapes_and_targets_are_valid(self):
        import random
        import torch

        batch = bench.make_batch(
            random.Random(7),
            batch_size=3,
            n_pairs=4,
            device=torch.device("cpu"),
        )

        self.assertEqual(batch["support"].shape, (3, 9))
        self.assertEqual(batch["query"].shape, (3, 2))
        self.assertEqual(batch["full"].shape, (3, 12))
        self.assertTrue((batch["target"] >= bench.VALUE_START).all())

    def test_smoke_run_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "forced"

            report = bench.run_forced_identity_objective(
                output_dir=output_dir,
                steps=2,
                batch_size=4,
                eval_batches=1,
                seeds=[7],
                variants=["forced_state"],
            )

            self.assertEqual(report["schema"], "forced_identity_objective.v1")
            self.assertIn("forced_state", report["aggregate"])
            self.assertTrue((output_dir / "forced_identity_objective.json").exists())
            saved = json.loads((output_dir / "forced_identity_objective.json").read_text())
            self.assertEqual(saved["schema"], report["schema"])
            self.assertTrue((output_dir / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
