import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments import benchmark_memory_advantage_model_version as bench
from kaggle import make_agentic_training_bundle, train_best_tac_agentic
from tac_transformer import (
    TACTransformerLM,
    memory_advantage_config,
    memory_advantage_training_kwargs,
    run5b_best_capability_fast_config,
    run5b_best_capability_fast_training_kwargs,
)
from tac_transformer.training import (
    count_parameters,
    parameter_matched_baseline_config,
)


class MemoryAdvantageModelVersionTests(unittest.TestCase):
    def test_memory_advantage_preset_composes_research_promoted_mechanisms(self):
        config = memory_advantage_config(vocab_size=512)
        training = memory_advantage_training_kwargs()
        counts = count_parameters(TACTransformerLM(config))
        vanilla_config = parameter_matched_baseline_config(config)

        self.assertEqual(config.norm_type, "rmsnorm")
        self.assertEqual(config.mlp_type, "swiglu")
        self.assertEqual(config.position_type, "rope")
        self.assertEqual(config.program_compute_type, "linear_expert")
        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.routing_top_k, 2)
        self.assertEqual(config.n_programs, 24)
        self.assertAlmostEqual(config.routing_load_balance_weight, 0.05)
        self.assertEqual(config.program_memory_update_type, "program_conditioned")
        self.assertEqual(config.memory_allocation_type, "creb")
        self.assertEqual(config.memory_allocation_k, 6)
        self.assertEqual(config.memory_read_type, "content_addressed")
        self.assertEqual(config.content_read_steps, 2)
        self.assertEqual(config.content_read_gate_type, "synthesis")
        self.assertEqual(config.coalition_context_type, "program_memory_graph")
        self.assertEqual(config.memory_adapter_type, "gated_residual")
        self.assertEqual(config.identity_attention_type, "identity_first")
        self.assertFalse(config.detach_identity_state)

        self.assertEqual(training["precision"], "fp32")
        self.assertEqual(training["category_route_objective"], "selected_mi")
        self.assertAlmostEqual(training["category_route_weight"], 0.5)
        self.assertAlmostEqual(training["min_healthy_gradient_norm"], 1e-12)
        self.assertEqual(training["fail_on_unhealthy_optimization"], 1)
        self.assertLessEqual(counts["identity_field"] / counts["total"], 0.6)
        self.assertGreater(vanilla_config.d_model, 0)

    def test_memory_advantage_preset_runs_a_small_forward_pass(self):
        config = memory_advantage_config(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=16,
            content_store_size=4,
            memory_allocation_k=2,
        )
        model = TACTransformerLM(config)
        output = model(torch.tensor([[1, 2, 3, 4, 5]]))

        self.assertEqual(output.logits.shape, (1, 5, 64))
        self.assertEqual(float(output.aux.metrics["routing_type"]), 5.0)
        self.assertEqual(float(output.aux.metrics["memory_allocation_type"]), 1.0)

    def test_train_cli_can_select_memory_advantage_preset(self):
        args = train_best_tac_agentic.parse_args(
            [
                "--preset",
                "memory_advantage",
                "--scale",
                "smoke",
            ]
        )
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.n_programs, 24)
        self.assertEqual(config.program_memory_update_type, "program_conditioned")
        self.assertEqual(config.memory_allocation_type, "creb")
        self.assertEqual(config.coalition_context_type, "program_memory_graph")
        self.assertEqual(args.category_route_objective, "selected_mi")
        self.assertAlmostEqual(args.category_route_weight, 0.5)
        self.assertEqual(args.precision, "fp32")
        self.assertTrue(args.fail_on_unhealthy_optimization)

    def test_run5b_best_capability_fast_preset_merges_capability_and_speed_lessons(self):
        config = run5b_best_capability_fast_config(vocab_size=512)
        training = run5b_best_capability_fast_training_kwargs()

        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.routing_top_k, 2)
        self.assertEqual(config.n_programs, 24)
        self.assertEqual(config.program_memory_update_type, "program_conditioned")
        self.assertEqual(config.memory_allocation_type, "creb")
        self.assertEqual(config.memory_allocation_k, 6)
        self.assertEqual(config.memory_read_type, "content_addressed")
        self.assertEqual(config.content_read_steps, 2)
        self.assertEqual(config.content_read_gate_type, "cue_match")
        self.assertEqual(config.content_read_query_top_k, 8)
        self.assertEqual(config.coalition_context_type, "program_memory_graph")
        self.assertEqual(config.memory_adapter_type, "gated_residual")
        self.assertEqual(config.identity_attention_type, "identity_first")
        self.assertEqual(config.attention_window_size, 128)

        self.assertEqual(training["category_route_objective"], "selected_mi")
        self.assertAlmostEqual(training["category_route_weight"], 0.1)
        self.assertEqual(training["aux_loss_cadence"], 4)
        self.assertEqual(training["precision"], "fp32")
        self.assertAlmostEqual(training["min_healthy_gradient_norm"], 1e-12)
        self.assertEqual(training["fail_on_unhealthy_optimization"], 1)

    def test_train_cli_can_select_run5b_best_capability_fast_preset(self):
        args = train_best_tac_agentic.parse_args(
            [
                "--preset",
                "run5b_best_capability_fast",
                "--scale",
                "smoke",
            ]
        )
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.n_programs, 24)
        self.assertEqual(config.content_read_gate_type, "cue_match")
        self.assertEqual(config.coalition_context_type, "program_memory_graph")
        self.assertEqual(config.attention_window_size, 128)
        self.assertEqual(args.category_route_objective, "selected_mi")
        self.assertAlmostEqual(args.category_route_weight, 0.1)
        self.assertEqual(args.aux_loss_cadence, 4)
        self.assertEqual(args.precision, "fp32")
        self.assertTrue(args.fail_on_unhealthy_optimization)

    def test_run5b_best_capability_fast_trainer_smoke_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.prepared.jsonl"
            eval_path = tmp_path / "eval.prepared.jsonl"
            rows = [
                {"domain": "tool_choice", "text": "choose calculator and verify"},
                {"domain": "repair_after_failure", "text": "repair test then verify"},
                {"domain": "tool_choice", "text": "choose shell command"},
                {"domain": "repair_after_failure", "text": "diagnose error and patch"},
            ]
            payload = "\n".join(json.dumps(row) for row in rows) + "\n"
            train_path.write_text(payload, encoding="utf-8")
            eval_path.write_text(payload, encoding="utf-8")

            train_best_tac_agentic.main(
                [
                    "--preset",
                    "run5b_best_capability_fast",
                    "--scale",
                    "smoke",
                    "--d-model",
                    "32",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--seq-len",
                    "16",
                    "--batch-size",
                    "1",
                    "--steps",
                    "2",
                    "--eval-every",
                    "1",
                    "--eval-batches",
                    "1",
                    "--checkpoint-every",
                    "1",
                    "--train-jsonl",
                    str(train_path),
                    "--eval-jsonl",
                    str(eval_path),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--device",
                    "cpu",
                ]
            )

            manifest = json.loads(
                (tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (tmp_path / "out" / "final_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["preset"], "run5b_best_capability_fast")
        self.assertEqual(manifest["config"]["content_read_gate_type"], "cue_match")
        self.assertEqual(manifest["config"]["attention_window_size"], 128)
        self.assertEqual(manifest["category_route_objective"], "selected_mi")
        self.assertAlmostEqual(manifest["category_route_weight"], 0.1)
        self.assertEqual(manifest["aux_loss_cadence"], 4)
        self.assertEqual(manifest["precision"], "fp32")
        self.assertTrue(manifest["optimization_health"]["fail_on_unhealthy_optimization"])
        self.assertEqual(summary["completed_steps"], 2)

    def test_benchmark_manifest_frames_attachment_question_and_controls(self):
        result = bench.run_memory_advantage_model_version(
            vocab_size=128,
            d_model=32,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=32,
            content_store_size=4,
            memory_allocation_k=2,
        )

        self.assertEqual(result["schema"], "memory_advantage_model_version.v1")
        self.assertEqual(result["decision"]["status"], "memory_advantage_model_version_ready")
        self.assertIn("persistent computational identity", result["primary_question"])
        self.assertEqual(result["model_version"]["preset"], "memory_advantage")
        self.assertIn("Context Tokens Required vs Task Success", result["target_graphs"])
        self.assertIn("Days Since Instruction vs Accuracy", result["target_graphs"])

        control_ids = {control["id"] for control in result["equal_resource_controls"]}
        self.assertIn("parameter_matched_vanilla_window", control_ids)
        self.assertIn("parameter_matched_vanilla_retrieval", control_ids)
        self.assertIn("parameter_matched_vanilla_memory_db", control_ids)
        self.assertIn("current_best_tac", control_ids)
        self.assertFalse(result["boundary"]["claims_trained_checkpoint_advantage"])
        self.assertGreater(result["parameter_counts"]["memory_advantage_tac"]["total"], 0)
        self.assertGreater(result["parameter_counts"]["parameter_matched_vanilla"]["total"], 0)

    def test_cli_writes_memory_advantage_artifacts_and_bundle_includes_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--vocab-size",
                    "128",
                    "--d-model",
                    "32",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--n-programs",
                    "8",
                    "--max-seq-len",
                    "32",
                    "--content-store-size",
                    "4",
                    "--memory-allocation-k",
                    "2",
                ]
            )

            artifact = json.loads(
                (output_dir / "memory_advantage_model_version.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "memory_advantage_model_version.v1")
        self.assertIn("Memory Advantage Model Version", markdown)
        self.assertIn(
            "experiments/benchmark_memory_advantage_model_version.py",
            make_agentic_training_bundle.FILES,
        )
        instructions = make_agentic_training_bundle._instructions()
        self.assertIn("--preset run5b_best_capability_fast", instructions)
        self.assertIn("--aux-loss-cadence 4", instructions)


if __name__ == "__main__":
    unittest.main()
