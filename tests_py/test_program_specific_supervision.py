import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_program_specific_supervision as bench


class ProgramSpecificSupervisionTests(unittest.TestCase):
    def test_batch_assigns_key_values_to_programs(self):
        import random
        import torch

        batch = bench.make_program_batch(
            random.Random(7),
            batch_size=5,
            n_pairs=3,
            device=torch.device("cpu"),
        )

        self.assertEqual(batch["support"].shape, (5, 7))
        self.assertEqual(batch["query"].shape, (5, 2))
        self.assertEqual(batch["support_program_targets"].shape, (5, 7))
        self.assertTrue((batch["target_program"] >= 0).all())
        self.assertTrue((batch["target_program"] < bench.N_PROGRAMS).all())
        for row, program in zip(batch["query"], batch["target_program"]):
            self.assertEqual(int(row[-1]) - bench.KEY_START, int(program))

    def test_smoke_run_writes_program_supervision_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "program_supervision"

            report = bench.run_program_specific_supervision(
                output_dir=output_dir,
                base_steps=2,
                head_steps=2,
                batch_size=4,
                eval_batches=1,
                knockout_batches=1,
                n_pairs=2,
                seeds=[7],
                variants=["program_supervised"],
                torch_threads=1,
            )

            self.assertEqual(report["schema"], "program_specific_supervision.v1")
            self.assertIn("program_supervised", report["aggregate"])
            aggregate = report["aggregate"]["program_supervised"]
            self.assertIn("target_slot_bridge_accuracy", aggregate)
            self.assertIn("targeted_knockout_drop", aggregate)
            self.assertIn("route_target_selected_rate", aggregate)
            self.assertTrue((output_dir / "program_specific_supervision.json").exists())
            saved = json.loads(
                (output_dir / "program_specific_supervision.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved["schema"], report["schema"])
            self.assertTrue((output_dir / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
