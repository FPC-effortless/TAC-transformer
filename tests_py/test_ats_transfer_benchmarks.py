import unittest
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from experiments import benchmark_ats_transfer_suite as ats_bench
from tac_transformer import (
    TACConfig,
    TACTransformerLM,
    aggregate_ats_transfer_results,
    build_ats_oracle_predictions,
    build_ats_surface_baseline_predictions,
    build_ats_transfer_suite,
    aggregate_ats_checkpoint_run_results,
    run_ats_checkpoint_predictions,
    score_ats_transfer_predictions,
)
from tac_transformer.training import JsonlTextBatcher


class ATSTransferBenchmarkTests(unittest.TestCase):
    def test_ats_suite_has_disjoint_domains_and_required_task_families(self):
        suite = build_ats_transfer_suite(seed=31, examples_per_domain=2)

        train_domains = set(suite["train_domains"])
        test_domains = set(suite["test_domains"])
        task_ids = {example["task_id"] for example in suite["examples"]}

        self.assertFalse(train_domains & test_domains)
        self.assertIn("cross_domain_identity_transfer", task_ids)
        self.assertIn("two_program_sequential", task_ids)
        self.assertGreaterEqual(len(suite["examples"]), 8)
        self.assertTrue(all(example["latent_invariant"] for example in suite["examples"]))
        self.assertTrue(all(example["answer"] for example in suite["examples"]))
        self.assertTrue(
            all(len(example["prompt"].encode("utf-8")) <= 220 for example in suite["examples"])
        )

    def test_ats_scoring_validates_transfer_gap_over_surface_baseline(self):
        suite = build_ats_transfer_suite(seed=37, examples_per_domain=2)
        oracle_predictions = build_ats_oracle_predictions(
            suite["examples"],
            control_id="identity_oracle",
        )
        baseline_predictions = build_ats_surface_baseline_predictions(
            suite["examples"],
            control_id="surface_baseline",
        )
        rows = score_ats_transfer_predictions(
            suite["examples"],
            oracle_predictions + baseline_predictions,
        )
        aggregate = aggregate_ats_transfer_results(rows)

        self.assertEqual(
            aggregate["decision"]["status"],
            "ats_transfer_benchmark_valid",
        )
        self.assertEqual(
            aggregate["controls"]["identity_oracle"]["splits"]["test"]["mean_score"],
            1.0,
        )
        self.assertEqual(
            aggregate["controls"]["surface_baseline"]["splits"]["train"]["mean_score"],
            1.0,
        )
        self.assertLessEqual(
            aggregate["controls"]["surface_baseline"]["splits"]["test"]["mean_score"],
            0.25,
        )

    def test_ats_transfer_benchmark_probe_writes_valid_decision(self):
        report = ats_bench.run_ats_transfer_suite_probe(
            seed=41,
            examples_per_domain=2,
        )

        self.assertEqual(
            report["decision"]["status"],
            "ats_transfer_benchmark_valid",
        )
        self.assertIn("cross_domain_identity_transfer", report["suite"]["task_ids"])
        self.assertIn("two_program_sequential", report["suite"]["task_ids"])
        self.assertGreater(report["scores"]["oracle_test_advantage"], 0.5)

    def test_ats_checkpoint_prediction_runner_writes_scoring_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = build_ats_transfer_suite(seed=43, examples_per_domain=1)
            examples = suite["examples"][:2]
            checkpoint_path = root / "tac.pt"
            config = TACConfig(
                vocab_size=260,
                d_model=8,
                n_heads=2,
                n_layers=1,
                n_programs=4,
                max_seq_len=64,
            )
            model = TACTransformerLM(config)
            torch.save(
                {
                    "step": 3,
                    "best_eval_loss": 1.0,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                },
                checkpoint_path,
            )
            output_jsonl = root / "predictions.jsonl"

            payload = run_ats_checkpoint_predictions(
                examples=examples,
                checkpoint_path=checkpoint_path,
                control_id="tac_smoke",
                seed=43,
                output_jsonl=output_jsonl,
                max_new_tokens=2,
                device="cpu",
            )
            rows = [
                json.loads(line)
                for line in output_jsonl.read_text(encoding="utf-8").splitlines()
            ]
            scores = score_ats_transfer_predictions(examples, rows)

            self.assertEqual(payload["prediction_count"], len(examples))
            self.assertEqual(payload["model_type"], "tac")
            self.assertEqual(rows[0]["control_id"], "tac_smoke")
            self.assertIn("raw_completion", rows[0])
            self.assertIn("tokens_per_second", rows[0])
            self.assertGreater(len(scores), 0)

    def test_ats_supervised_corpus_stages_disjoint_prepared_jsonl(self):
        from tac_transformer import stage_ats_transfer_training_corpus

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            manifest = stage_ats_transfer_training_corpus(
                output_dir=root,
                seed=47,
                examples_per_domain=2,
            )
            train_path = root / "train.prepared.jsonl"
            eval_path = root / "eval.prepared.jsonl"
            train_rows = [
                json.loads(line)
                for line in train_path.read_text(encoding="utf-8").splitlines()
            ]
            eval_rows = [
                json.loads(line)
                for line in eval_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(manifest["schema"], "ats_transfer_training_corpus.v1")
            self.assertEqual(manifest["train_records"], len(train_rows))
            self.assertEqual(manifest["eval_records"], len(eval_rows))
            self.assertGreater(len(train_rows), 0)
            self.assertGreater(len(eval_rows), 0)
            self.assertTrue(
                {row["domain"] for row in train_rows}.issubset(
                    set(manifest["train_domains"])
                )
            )
            self.assertTrue(
                {row["domain"] for row in eval_rows}.issubset(
                    set(manifest["test_domains"])
                )
            )
            self.assertFalse(
                {row["domain"] for row in train_rows}
                & {row["domain"] for row in eval_rows}
            )
            self.assertTrue(
                all(row["text"].endswith(row["answer"] + "\n") for row in train_rows)
            )
            self.assertLessEqual(manifest["max_text_bytes"], 256)
            self.assertEqual(manifest["leakage"]["test_domain_rows_in_train"], 0)
            self.assertEqual(manifest["leakage"]["train_domain_rows_in_eval"], 0)
            tac_command = manifest["recommended_commands"]["tac_base"]
            self.assertIn("--seq-len 176", tac_command)
            self.assertIn("--precision fp32", tac_command)
            self.assertIn("--min-healthy-gradient-norm 1e-12", tac_command)
            self.assertIn("--fail-on-unhealthy-optimization", tac_command)
            self.assertIn("--category-route-objective selected_mi", tac_command)

            x, y = JsonlTextBatcher(
                train_path,
                seq_len=256,
                vocab_size=512,
                seed=1,
            ).next_batch(batch_size=2)
            self.assertEqual(tuple(x.shape), (2, 256))
            self.assertEqual(tuple(y.shape), (2, 256))

    def test_ats_answer_copy_probe_reports_tac_and_vanilla_controls(self):
        from experiments import benchmark_ats_answer_copy_training as answer_probe

        report = answer_probe.run_ats_answer_copy_training_probe(
            seed=53,
            examples_per_domain=1,
            train_steps=1,
            learning_rate=0.001,
            max_seq_len=176,
            d_model=16,
            n_heads=2,
            n_layers=1,
            n_programs=4,
        )

        self.assertEqual(report["schema"], "ats_answer_copy_training.v1")
        self.assertEqual(set(report["controls"]), {"tac_answer_only", "vanilla_answer_only"})
        self.assertIn("train", report["scores"]["tac_answer_only"])
        self.assertIn("test", report["scores"]["vanilla_answer_only"])
        self.assertEqual(report["suite"]["train_domains"], ["navigation", "inventory"])
        self.assertEqual(report["suite"]["test_domains"], ["lab_protocol", "incident_response"])

    def test_ats_checkpoint_run_aggregate_requires_tac_advantage(self):
        tac_run = {
            "schema": "ats_checkpoint_prediction_run.v1",
            "control_id": "tac_base_ats_5k",
            "model_type": "tac",
            "checkpoint_step": 5000,
            "prediction_count": 4,
            "score_rows": [
                _score_row("tac_base_ats_5k", "train", "cross_domain_identity_transfer", 1.0),
                _score_row("tac_base_ats_5k", "train", "two_program_sequential", 1.0),
                _score_row("tac_base_ats_5k", "test", "cross_domain_identity_transfer", 1.0),
                _score_row("tac_base_ats_5k", "test", "two_program_sequential", 1.0),
            ],
        }
        vanilla_run = {
            "schema": "ats_checkpoint_prediction_run.v1",
            "control_id": "vanilla_base_ats_5k",
            "model_type": "vanilla",
            "checkpoint_step": 5000,
            "prediction_count": 4,
            "score_rows": [
                _score_row("vanilla_base_ats_5k", "train", "cross_domain_identity_transfer", 1.0),
                _score_row("vanilla_base_ats_5k", "train", "two_program_sequential", 1.0),
                _score_row("vanilla_base_ats_5k", "test", "cross_domain_identity_transfer", 0.25),
                _score_row("vanilla_base_ats_5k", "test", "two_program_sequential", 0.25),
            ],
        }

        aggregate = aggregate_ats_checkpoint_run_results([tac_run, vanilla_run])
        pending = aggregate_ats_checkpoint_run_results([tac_run])

        self.assertEqual(
            aggregate["decision"]["status"],
            "ats_external_transfer_promote",
        )
        self.assertEqual(aggregate["controls"]["tac_base_ats_5k"]["splits"]["test"]["mean_score"], 1.0)
        self.assertTrue(aggregate["decision"]["checks"]["tac_beats_vanilla_test"])
        self.assertEqual(pending["decision"]["status"], "pending")
        self.assertIn("vanilla_base_ats_5k", pending["decision"]["missing_controls"])

def _score_row(control_id: str, split: str, task_id: str, score: float) -> dict:
    return {
        "schema": "ats_transfer_score_row.v1",
        "control_id": control_id,
        "split": split,
        "task_id": task_id,
        "primary_metric": "exact_match",
        "primary_score": score,
        "correct_count": int(score),
        "example_count": 1,
        "missing_prediction_count": 0,
    }


if __name__ == "__main__":
    unittest.main()
