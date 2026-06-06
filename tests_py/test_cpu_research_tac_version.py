import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_cpu_research_tac_version as bench
from kaggle import make_agentic_training_bundle, train_best_tac_agentic
from tac_transformer import (
    best_tac_config,
    cpu_research_tac_config,
    cpu_research_tac_training_kwargs,
    kaggle_fast_tac_config,
)


class CpuResearchTacVersionTests(unittest.TestCase):
    def test_cpu_research_preset_is_opt_in_and_keeps_main_presets_unchanged(self):
        config = cpu_research_tac_config(vocab_size=512)
        training = cpu_research_tac_training_kwargs()
        fast = kaggle_fast_tac_config(vocab_size=512)
        best = best_tac_config(vocab_size=512)

        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.routing_top_k, 1)
        self.assertEqual(config.n_programs, 8)
        self.assertEqual(config.memory_read_type, "content_addressed")
        self.assertEqual(config.content_read_steps, 1)
        self.assertEqual(config.content_read_query_top_k, 4)
        self.assertEqual(config.attention_window_size, 64)
        self.assertEqual(config.memory_adapter_type, "residual")
        self.assertEqual(config.identity_attention_type, "identity_first")

        self.assertEqual(training["aux_loss_cadence"], 4)
        self.assertEqual(training["torch_threads"], 1)
        self.assertEqual(training["torch_interop_threads"], 1)
        self.assertEqual(training["precision"], "fp32")
        self.assertEqual(training["fail_on_unhealthy_optimization"], 1)

        self.assertEqual(fast.routing_top_k, 2)
        self.assertEqual(fast.n_programs, 12)
        self.assertEqual(fast.content_read_steps, 2)
        self.assertEqual(fast.memory_adapter_type, "gated_residual")
        self.assertNotEqual(best.routing_type, config.routing_type)

    def test_trainer_can_select_cpu_research_preset_with_thread_defaults(self):
        args = train_best_tac_agentic.parse_args(
            [
                "--preset",
                "cpu_research_tac",
                "--scale",
                "smoke",
            ]
        )
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.routing_top_k, 1)
        self.assertEqual(config.n_programs, 8)
        self.assertEqual(config.content_read_steps, 1)
        self.assertEqual(config.content_read_query_top_k, 4)
        self.assertEqual(config.attention_window_size, 64)
        self.assertEqual(args.aux_loss_cadence, 4)
        self.assertEqual(args.torch_threads, 1)
        self.assertEqual(args.torch_interop_threads, 1)
        self.assertEqual(args.precision, "fp32")
        self.assertTrue(args.fail_on_unhealthy_optimization)

    def test_benchmark_writes_cpu_research_version_artifact(self):
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
                    "--seq-len",
                    "32",
                    "--batch-size",
                    "1",
                    "--warmup",
                    "0",
                    "--iters",
                    "1",
                    "--torch-threads",
                    "1",
                    "--interop-threads",
                    "1",
                ]
            )

            artifact = json.loads(
                (output_dir / "cpu_research_tac_version.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "cpu_research_tac_version.v1")
        self.assertEqual(artifact["ticket"], "TAC-193")
        self.assertEqual(artifact["model_version"]["preset"], "cpu_research_tac")
        self.assertIn("hard_lower_k_routing", artifact["applied_cpu_tactics"])
        self.assertIn("sparse_content_reads", artifact["applied_cpu_tactics"])
        self.assertIn("profiles", artifact)
        variants = {row["variant"] for row in artifact["profiles"]}
        self.assertIn("kaggle_fast_tac_reference", variants)
        self.assertIn("kaggle_fast_tac_aux_every_4", variants)
        self.assertIn("cpu_research_arch_full_aux", variants)
        self.assertIn("cpu_research_tac", variants)
        by_variant = {row["variant"]: row for row in artifact["profiles"]}
        self.assertEqual(by_variant["kaggle_fast_tac_reference"]["auxiliary_loss_cadence"], 1)
        self.assertEqual(by_variant["kaggle_fast_tac_aux_every_4"]["auxiliary_loss_cadence"], 4)
        self.assertEqual(by_variant["cpu_research_arch_full_aux"]["auxiliary_loss_cadence"], 1)
        self.assertEqual(by_variant["cpu_research_tac"]["auxiliary_loss_cadence"], 4)
        self.assertIn("combination_analysis", artifact)
        self.assertIn("combined_speed_ratio_vs_fast_full_aux", artifact["combination_analysis"])
        self.assertIn("aux_every_4_speed_ratio_on_fast_tac", artifact["combination_analysis"])
        self.assertIn("cpu_architecture_speed_ratio_with_full_aux", artifact["combination_analysis"])
        self.assertIn("aux_every_4_speed_ratio_on_cpu_architecture", artifact["combination_analysis"])
        self.assertIn("prior_local_efficiency_references", artifact)
        aux_ref = artifact["prior_local_efficiency_references"][0]
        self.assertEqual(aux_ref["variant"], "tac_aux_every_4")
        self.assertAlmostEqual(aux_ref["tokens_per_second"], 2276.89)
        self.assertAlmostEqual(aux_ref["speed_ratio_vs_full_aux_tac"], 1.15)
        self.assertAlmostEqual(aux_ref["eval_loss_delta_vs_full_aux_tac"], -0.0006)
        self.assertFalse(artifact["boundary"]["changes_main_tac_architecture"])
        self.assertIn("CPU Research TAC Version", markdown)
        self.assertIn("Same-Run Ablation", markdown)
        self.assertIn("Combination Analysis", markdown)
        self.assertIn("Prior Local Efficiency Reference", markdown)
        self.assertIn("tac_aux_every_4", markdown)

    def test_bundle_includes_cpu_research_benchmark_and_command(self):
        instructions = make_agentic_training_bundle._instructions()

        self.assertIn(
            "experiments/benchmark_cpu_research_tac_version.py",
            make_agentic_training_bundle.FILES,
        )
        self.assertIn("--preset cpu_research_tac", instructions)
        self.assertIn("benchmark_cpu_research_tac_version.py", instructions)


if __name__ == "__main__":
    unittest.main()
