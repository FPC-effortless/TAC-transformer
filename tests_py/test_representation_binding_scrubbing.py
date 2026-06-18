import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_representation_binding_scrubbing as bench


class RepresentationBindingScrubbingTests(unittest.TestCase):
    def test_program_subspace_masks_partition_model_dimensions(self):
        import torch

        masks = bench.program_subspace_masks(
            d_model=32,
            n_programs=8,
            device=torch.device("cpu"),
        )

        self.assertEqual(masks.shape, (8, 32))
        self.assertTrue((masks.sum(dim=0) == 1).all())
        self.assertTrue((masks.sum(dim=1) == 4).all())

    def test_smoke_run_writes_binding_scrubbing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "binding_scrubbing"

            report = bench.run_representation_binding_scrubbing(
                output_dir=output_dir,
                base_steps=2,
                head_steps=2,
                batch_size=4,
                eval_batches=1,
                knockout_batches=1,
                n_pairs=2,
                seeds=[7],
                variants=["subspace_bound"],
                torch_threads=1,
            )

            self.assertEqual(report["schema"], "representation_binding_scrubbing.v1")
            self.assertIn("subspace_bound", report["aggregate"])
            aggregate = report["aggregate"]["subspace_bound"]
            self.assertIn("bound_slot_bridge_accuracy", aggregate["binding"])
            self.assertIn("slot_subspace_projection_drop", aggregate["scrubbing"])
            self.assertIn("attention_ablation_drop", aggregate["scrubbing"])
            self.assertTrue((output_dir / "representation_binding_scrubbing.json").exists())
            saved = json.loads(
                (output_dir / "representation_binding_scrubbing.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved["schema"], report["schema"])
            self.assertTrue((output_dir / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
