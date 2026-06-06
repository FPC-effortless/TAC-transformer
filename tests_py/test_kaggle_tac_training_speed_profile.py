import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_kaggle_tac_training_speed_profile as bench
from kaggle import make_agentic_training_bundle, train_best_tac_agentic
from tac_transformer import kaggle_fast_tac_config, kaggle_fast_tac_training_kwargs


class KaggleTacTrainingSpeedProfileTests(unittest.TestCase):
    def test_fast_preset_keeps_tac_memory_path_but_reduces_read_work(self):
        config = kaggle_fast_tac_config(vocab_size=512)
        training = kaggle_fast_tac_training_kwargs()

        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.routing_top_k, 2)
        self.assertEqual(config.n_programs, 12)
        self.assertEqual(config.memory_read_type, "content_addressed")
        self.assertEqual(config.content_read_steps, 2)
        self.assertEqual(config.content_read_gate_type, "synthesis")
        self.assertEqual(config.identity_attention_type, "identity_first")
        self.assertEqual(config.memory_adapter_type, "gated_residual")
        self.assertEqual(config.content_read_query_top_k, 8)
        self.assertEqual(config.attention_window_size, 128)
        self.assertEqual(training["category_route_objective"], "selected_mi")
        self.assertAlmostEqual(training["category_route_weight"], 0.5)
        self.assertEqual(training["precision"], "fp32")
        self.assertAlmostEqual(training["min_healthy_gradient_norm"], 1e-12)
        self.assertEqual(training["fail_on_unhealthy_optimization"], 1)

    def test_trainer_selects_fast_preset_and_allows_attention_window_override(self):
        args = train_best_tac_agentic.parse_args(
            [
                "--preset",
                "kaggle_fast_tac",
                "--scale",
                "smoke",
                "--attention-window-size",
                "96",
            ]
        )
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.routing_top_k, 2)
        self.assertEqual(config.n_programs, 12)
        self.assertEqual(config.content_read_query_top_k, 8)
        self.assertEqual(config.attention_window_size, 96)
        self.assertEqual(args.category_route_objective, "selected_mi")
        self.assertAlmostEqual(args.category_route_weight, 0.5)
        self.assertEqual(args.precision, "fp32")
        self.assertTrue(args.fail_on_unhealthy_optimization)
        self.assertEqual(
            train_best_tac_agentic.target_routing_mode(args),
            ("base_semantic", 2),
        )

    def test_benchmark_writes_speed_profile_artifact(self):
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
                    "64",
                    "--batch-size",
                    "1",
                    "--warmup",
                    "0",
                    "--iters",
                    "1",
                    "--torch-threads",
                    "1",
                ]
            )

            artifact = json.loads(
                (output_dir / "kaggle_tac_training_speed_profile.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "kaggle_tac_training_speed_profile.v1")
        self.assertEqual(
            artifact["decision"]["status"],
            "kaggle_fast_tac_profile_ready_for_external_validation",
        )
        self.assertLessEqual(
            artifact["structural_gate"]["fast_tac_content_read_query_fraction"],
            0.25,
        )
        self.assertIn("vanilla_gap", artifact["interpretation"])
        self.assertIn("Kaggle TAC Training Speed Profile", markdown)

    def test_bundle_includes_speed_profile_benchmark_and_command(self):
        instructions = make_agentic_training_bundle._instructions()

        self.assertIn(
            "experiments/benchmark_kaggle_tac_training_speed_profile.py",
            make_agentic_training_bundle.FILES,
        )
        self.assertIn("--preset kaggle_fast_tac", instructions)
        self.assertIn("benchmark_kaggle_tac_training_speed_profile.py", instructions)


if __name__ == "__main__":
    unittest.main()
