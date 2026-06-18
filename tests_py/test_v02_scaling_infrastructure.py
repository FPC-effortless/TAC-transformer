from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from configs.tac_v02_50m import (
    TAC_V02_50M_AUXILIARY_MECHANISM_CONFIG,
    TAC_V02_50M_AUXILIARY_MECHANISM_PARAMS,
    TAC_V02_50M_LATE_BOTTLENECK_CONFIG,
    TAC_V02_50M_LATE_BOTTLENECK_PARAMS,
    TAC_V02_50M_PARAMS,
    TAC_V02_50M_SMALL_ADAPTER_CONFIG,
    TAC_V02_50M_SMALL_ADAPTER_PARAMS,
    TRANSFORMER_V02_50M_PARAMS,
)
from configs.tac_v02_112m import TAC_V02_112M_CONFIG, TAC_V02_112M_PARAMS
from experiments.stage_v02_kaggle_workflow import stage_v02_kaggle_workflow
from scripts.build_v02_datasets import build_dataset
from scripts.run_v02_checkpoint_mechanism_retests import build_probe_cases, compare_results
from scripts.summarize_tac281_variants import variant_decision
from scripts.train_v02_lm import (
    build_model,
    parse_args as parse_train_v02_args,
    train_v02_lm,
)
from tac_transformer import TACConfig, TACTransformerLM
from tac_transformer.v02_logging import (
    REQUIRED_V02_METRICS,
    V02_METRIC_SCHEMA,
    normalize_v02_metrics,
    write_v02_metrics,
)
from transformer_112m import (
    TRANSFORMER_V02_112M_CONFIG,
    TRANSFORMER_V02_112M_PARAMS,
)


class V02ScalingInfrastructureTest(unittest.TestCase):
    def test_tac_config_locks_requested_dimensions_and_parameter_target(self):
        self.assertEqual(TAC_V02_112M_CONFIG.vocab_size, 8192)
        self.assertEqual(TAC_V02_112M_CONFIG.d_model, 512)
        self.assertEqual(TAC_V02_112M_CONFIG.n_layers, 8)
        self.assertEqual(TAC_V02_112M_CONFIG.n_heads, 8)
        self.assertEqual(
            TAC_V02_112M_CONFIG.lm_readout_type,
            "slot_conditioned_program_bottleneck",
        )
        self.assertLess(abs(TAC_V02_112M_PARAMS - 112_000_000) / 112_000_000, 0.01)

    def test_transformer_baseline_is_parameter_matched(self):
        self.assertEqual(TRANSFORMER_V02_112M_CONFIG.vocab_size, 8192)
        self.assertEqual(TRANSFORMER_V02_112M_CONFIG.d_model, 512)
        self.assertEqual(TRANSFORMER_V02_112M_CONFIG.n_layers, 8)
        self.assertEqual(TRANSFORMER_V02_112M_CONFIG.n_heads, 8)
        self.assertLess(
            abs(TRANSFORMER_V02_112M_PARAMS - TAC_V02_112M_PARAMS)
            / TAC_V02_112M_PARAMS,
            0.01,
        )

    def test_50m_pilot_configs_are_inside_requested_range(self):
        self.assertGreaterEqual(TAC_V02_50M_PARAMS, 30_000_000)
        self.assertLessEqual(TAC_V02_50M_PARAMS, 50_000_000)
        self.assertLess(
            abs(TRANSFORMER_V02_50M_PARAMS - TAC_V02_50M_PARAMS)
            / TAC_V02_50M_PARAMS,
            0.02,
        )

    def test_tac281_variant_configs_are_30m_50m_and_distinct(self):
        for params in (
            TAC_V02_50M_LATE_BOTTLENECK_PARAMS,
            TAC_V02_50M_SMALL_ADAPTER_PARAMS,
            TAC_V02_50M_AUXILIARY_MECHANISM_PARAMS,
        ):
            self.assertGreaterEqual(params, 30_000_000)
            self.assertLessEqual(params, 50_000_000)
        self.assertEqual(
            TAC_V02_50M_LATE_BOTTLENECK_CONFIG.tac_active_layer_start,
            3,
        )
        self.assertEqual(TAC_V02_50M_SMALL_ADAPTER_CONFIG.n_programs, 12)
        self.assertEqual(TAC_V02_50M_SMALL_ADAPTER_CONFIG.lm_readout_type, "hidden")
        self.assertGreater(
            TAC_V02_50M_AUXILIARY_MECHANISM_CONFIG.routing_load_balance_weight,
            0.05,
        )

    def test_tac_active_layer_start_skips_early_identity_work(self):
        config = TACConfig(
            vocab_size=64,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_programs=4,
            max_seq_len=12,
            program_compute_type="low_rank_linear_expert",
            program_expert_rank=8,
            tac_active_layer_start=1,
            identity_attention_type="identity_first",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        output = model(torch.randint(0, config.vocab_size, (2, 8)))
        self.assertEqual(
            float(output.aux.metrics["lm_readout_type"]),
            0.0,
        )
        self.assertLess(
            float(output.aux.metrics["active_expert_fraction"]),
            1.0,
        )

    def test_train_v02_lm_builds_tac281_variants(self):
        for model_name in (
            "tac_50m_late_bottleneck",
            "tac_50m_small_adapter",
            "tac_50m_auxiliary_mechanism",
        ):
            model, config, family, estimated = build_model(model_name)
            self.assertIsInstance(model, TACTransformerLM)
            self.assertEqual(family, "tac")
            self.assertGreaterEqual(estimated, 30_000_000)
            self.assertLessEqual(estimated, 50_000_000)
            self.assertGreaterEqual(config.tac_active_layer_start, 0)

    def test_v02_lm_readout_uses_native_program_bottleneck(self):
        torch.manual_seed(7)
        config = TACConfig(
            vocab_size=64,
            d_model=32,
            n_layers=1,
            n_heads=4,
            n_programs=6,
            max_seq_len=12,
            program_compute_type="low_rank_linear_expert",
            program_expert_rank=8,
            lm_readout_type="slot_conditioned_program_bottleneck",
            routing_type="base",
            routing_top_k=2,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        before = model(input_ids)
        self.assertEqual(float(before.aux.metrics["lm_readout_type"]), 1.0)
        self.assertGreater(
            float(before.aux.metrics["lm_program_bottleneck_selected_mass"]),
            0.0,
        )
        with torch.no_grad():
            field = model.blocks[-1].identity_field
            field.program_expert_down.zero_()
            field.program_expert_up.zero_()
            field.program_expert_bias.zero_()
        after = model(input_ids)
        self.assertFalse(torch.allclose(before.logits, after.logits))

    def test_v02_metrics_schema_contains_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics_v02.json"
            record = normalize_v02_metrics(
                model_name="tac_112m",
                step=10,
                train_metrics={
                    "loss": 2.5,
                    "metric_routing_entropy": 1.1,
                    "metric_active_expert_fraction": 0.5,
                    "metric_selected_identity_state_norm": 0.25,
                    "metric_decision_continuity_agreement": 0.8,
                },
                eval_metrics={"loss": 2.3, "perplexity": 9.97},
            )
            payload = write_v02_metrics(path, [record])
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], V02_METRIC_SCHEMA)
            self.assertEqual(loaded["schema"], V02_METRIC_SCHEMA)
            for field in REQUIRED_V02_METRICS:
                self.assertIn(field, loaded["records"][0])

    def test_dataset_builder_keeps_holdouts_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = build_dataset(
                output_dir=Path(tmp),
                per_source_limit=1,
                long_horizon_count=20,
                holdout_rate=0.2,
                offline_synthetic_only=True,
                seed=7,
            )
            self.assertGreater(manifest["counts"]["train"], 0)
            self.assertGreater(manifest["counts"]["eval"], 0)
            self.assertGreater(manifest["counts"]["validation_holdout"], 0)
            self.assertIn("validation_holdout", manifest["paths"])
            self.assertTrue(Path(manifest["paths"]["validation_holdout"]).exists())

    def test_train_v02_lm_smoke_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            rows = [
                {"text": "alpha beta gamma delta"},
                {"text": "repair plan verify continue"},
                {"text": "memory carries state across chunks"},
                {"text": "compression keeps useful context"},
            ]
            train.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            eval_path.write_text(
                "\n".join(json.dumps(row) for row in rows[:2]) + "\n",
                encoding="utf-8",
            )
            args = parse_train_v02_args(
                [
                    "--model",
                    "smoke_tac",
                    "--train-jsonl",
                    str(train),
                    "--eval-jsonl",
                    str(eval_path),
                    "--output-dir",
                    str(root / "out"),
                    "--steps",
                    "1",
                    "--eval-every",
                    "1",
                    "--eval-batches",
                    "1",
                    "--batch-size",
                    "1",
                    "--grad-accum-steps",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            summary = train_v02_lm(args)
            metrics_path = Path(summary["metrics_path"])
            self.assertTrue(metrics_path.exists())
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], V02_METRIC_SCHEMA)
            self.assertEqual(payload["records"][0]["model_name"], "smoke_tac")

    def test_stage_v02_kaggle_workflow_smoke_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = stage_v02_kaggle_workflow(
                output_root=Path(tmp),
                owner="unit-test-owner",
                date_slug="2099-01-01",
                smoke=True,
                push=False,
            )
            self.assertEqual(manifest["schema"], "tac_v02_kaggle_workflow.v1")
            self.assertEqual(len(manifest["kernels"]), 8)
            self.assertTrue(Path(manifest["source_bundle"]).exists())
            names = {row["name"] for row in manifest["kernels"]}
            self.assertEqual(
                names,
                {
                    "mechanism-reproduction",
                    "lm-50m-pilot",
                    "lm-112m-pilot",
                    "checkpoint-retests",
                    "tac281-variants",
                    "tac281-late-bottleneck",
                    "tac281-small-adapter",
                    "tac281-auxiliary-mechanism",
                },
            )

    def test_stage_v02_kaggle_workflow_can_filter_kernels(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = stage_v02_kaggle_workflow(
                output_root=Path(tmp),
                owner="unit-test-owner",
                date_slug="2099-01-02",
                smoke=False,
                push=False,
                kernel_filter=("mechanism-reproduction",),
            )
            self.assertEqual(len(manifest["kernels"]), 1)
            self.assertEqual(manifest["kernels"][0]["name"], "mechanism-reproduction")
            self.assertEqual(manifest["kernel_filter"], ["mechanism-reproduction"])

    def test_stage_v02_kaggle_workflow_can_stage_checkpoint_retests(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = stage_v02_kaggle_workflow(
                output_root=Path(tmp),
                owner="unit-test-owner",
                date_slug="2099-01-03",
                smoke=False,
                push=False,
                kernel_filter=("checkpoint-retests",),
                checkpoint_kernel_source="unit/source-kernel",
            )
            kernel = manifest["kernels"][0]
            self.assertEqual(kernel["name"], "checkpoint-retests")
            self.assertEqual(kernel["metadata"]["kernel_sources"], ["unit/source-kernel"])

    def test_stage_v02_kaggle_workflow_can_stage_tac281_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = stage_v02_kaggle_workflow(
                output_root=Path(tmp),
                owner="unit-test-owner",
                date_slug="2099-01-04",
                smoke=True,
                push=False,
                kernel_filter=("tac281-variants",),
            )
            kernel = manifest["kernels"][0]
            self.assertEqual(kernel["name"], "tac281-variants")
            joined_commands = "\n".join(kernel["commands"])
            self.assertIn("tac_50m_late_bottleneck", joined_commands)
            self.assertIn("tac_50m_small_adapter", joined_commands)
            self.assertIn("tac_50m_auxiliary_mechanism", joined_commands)
            self.assertIn("summarize_tac281_variants.py", joined_commands)

    def test_stage_v02_kaggle_workflow_can_stage_split_tac281_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = stage_v02_kaggle_workflow(
                output_root=Path(tmp),
                owner="unit-test-owner",
                date_slug="2099-01-05",
                smoke=True,
                push=False,
                kernel_filter=("tac281-small-adapter",),
            )
            kernel = manifest["kernels"][0]
            self.assertEqual(kernel["name"], "tac281-small-adapter")
            self.assertEqual(kernel["metadata"]["enable_internet"], "true")
            joined_commands = "\n".join(kernel["commands"])
            self.assertIn("transformer_50m", joined_commands)
            self.assertIn("tac_50m_small_adapter", joined_commands)
            self.assertIn("summarize_tac281_variants.py", joined_commands)

    def test_tac280_probe_families_and_decision_math(self):
        cases = build_probe_cases(cases_per_family=2)
        self.assertEqual(len(cases), 8)
        self.assertEqual(
            {case.family for case in cases},
            {
                "persistent_state_carry",
                "repair_trace_reuse",
                "compression_structure_reuse",
                "noisy_key_retrieval",
            },
        )
        transformer = {
            "overall": {"full_context_loss": 2.0, "reset_loss": 2.5},
            "by_family": {
                family: {"full_context_loss": 2.0, "reset_loss": 2.5, "cases": 2}
                for family in {case.family for case in cases}
            },
        }
        tac = {
            "overall": {"carry_loss": 1.5, "reset_loss": 2.2, "knockout_loss": 2.0},
            "by_family": {
                family: {
                    "carry_loss": 1.5,
                    "reset_loss": 2.2,
                    "knockout_loss": 2.0,
                    "cases": 2,
                }
                for family in {case.family for case in cases}
            },
        }
        comparison = compare_results(transformer, tac)
        self.assertEqual(comparison["decision"]["status"], "mechanism_advantage")

    def test_tac281_decision_requires_mechanisms_gap_and_speed(self):
        retest = {
            "comparison": {
                "decision": {
                    "status": "mechanism_advantage",
                    "tac_win_families": 3,
                    "carry_positive_families": 3,
                },
                "overall": {
                    "tac_carry_advantage": 0.05,
                    "bottleneck_knockout_delta": 1.25,
                },
            }
        }
        decision = variant_decision(
            transformer_summary={
                "best_eval_loss": 1.0,
                "latest_metrics": {"elapsed_seconds": 100.0},
            },
            variant_summary={
                "model": "tac_50m_late_bottleneck",
                "best_eval_loss": 1.25,
                "latest_metrics": {"elapsed_seconds": 700.0},
            },
            retest=retest,
            min_gap_shrink=0.30,
            original_transformer_best_loss=1.0,
            original_tac_best_loss=1.5,
            original_transformer_runtime=100.0,
            original_tac_runtime=1000.0,
        )
        self.assertEqual(decision["status"], "scale_ready")


if __name__ == "__main__":
    unittest.main()
