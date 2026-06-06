import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_local_tac_efficiency_matrix as bench
from kaggle import make_agentic_training_bundle


class LocalTacEfficiencyMatrixTests(unittest.TestCase):
    def test_benchmark_writes_local_efficiency_matrix_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--vocab-size",
                    "128",
                    "--d-model",
                    "24",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--n-programs",
                    "8",
                    "--seq-len",
                    "32",
                    "--batch-size",
                    "1",
                    "--warmup",
                    "0",
                    "--iters",
                    "1",
                    "--learning-rate",
                    "1e-4",
                    "--torch-threads",
                    "1",
                    "--skip-compile",
                ]
            )

            artifact = json.loads(
                (output_dir / "local_tac_efficiency_matrix.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "local_tac_efficiency_matrix.v1")
        self.assertEqual(artifact["ticket"], "TAC-192")
        self.assertIn("environment", artifact)
        self.assertIn("decision", artifact)

        variants = {row["variant"]: row for row in artifact["variants"]}
        for required in [
            "eager_full_aux",
            "eager_metrics_deferred",
            "eager_aux_every_2",
            "eager_aux_every_4",
        ]:
            self.assertEqual(variants[required]["status"], "completed")
            self.assertGreater(variants[required]["tokens_per_second"], 0.0)
            self.assertIn("speed_ratio_vs_baseline", variants[required])
            self.assertIn("capability_proxy", variants[required])
            self.assertIn(
                "eval_loss_delta_vs_baseline",
                variants[required]["capability_proxy"],
            )

        self.assertEqual(variants["torch_compile_reduce_overhead"]["status"], "skipped")
        self.assertEqual(variants["triton_identity_kernel"]["status"], "not_applicable")
        self.assertEqual(variants["foreach_identity_ops"]["status"], "deferred")
        self.assertEqual(variants["routing_cache_or_hard_routing"]["status"], "deferred")
        self.assertEqual(variants["parameter_reallocation"]["status"], "deferred")
        self.assertEqual(variants["eager_aux_every_4"]["auxiliary_loss_cadence"], 4)
        self.assertLessEqual(
            artifact["decision"]["accepted_loss_delta_threshold"],
            0.25,
        )
        self.assertIn("Local TAC Efficiency Matrix", markdown)

    def test_bundle_includes_local_efficiency_matrix_benchmark(self):
        instructions = make_agentic_training_bundle._instructions()

        self.assertIn(
            "experiments/benchmark_local_tac_efficiency_matrix.py",
            make_agentic_training_bundle.FILES,
        )
        self.assertIn("benchmark_local_tac_efficiency_matrix.py", instructions)


if __name__ == "__main__":
    unittest.main()
