import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from tac_transformer import AgenticScratchpadState, ScratchpadItem
from tac_transformer.model import TACConfig, TACTransformerLM, VanillaTransformerLM
from tac_transformer.phase_d_benchmarks import (
    build_phase_d_task_suite,
    extract_phase_d_answer,
    load_jsonl,
    load_phase_d_checkpoint_model,
    phase_d_text_to_token_ids,
    phase_d_token_ids_to_text,
    run_phase_d_checkpoint_predictions,
    run_phase_d_scratchpad_state_predictions,
    score_phase_d_predictions,
    stage_phase_d_benchmark_suite,
)
from experiments.aggregate_phase_d_benchmarks import discover_phase_d_row_sources
from experiments import benchmark_live_phase_d_scratchpad_policy as live_phase_d_policy
from experiments.run_phase_d_benchmark_matrix import (
    discover_phase_d_seed_checkpoint,
    run_phase_d_benchmark_matrix,
)
from experiments.stage_phase_d_suite_dataset import stage_phase_d_suite_dataset


class PhaseDBenchmarkHarnessTests(unittest.TestCase):
    def test_task_suite_covers_required_phase_d_tasks(self):
        suite = build_phase_d_task_suite(seed=11, examples_per_task=2, context_length=512)

        task_ids = {row["task_id"] for row in suite["examples"]}

        self.assertEqual(suite["seed"], 11)
        self.assertIn("multi_hop_chain_retrieval", task_ids)
        self.assertIn("long_context_retrieval_4096", task_ids)
        self.assertIn("episodic_fact_update", task_ids)
        self.assertIn("tool_selection", task_ids)
        self.assertIn("delayed_goal_binding", task_ids)
        self.assertTrue(all(row["answer"] for row in suite["examples"]))

        long_context = [
            row
            for row in suite["examples"]
            if row["task_id"] == "long_context_retrieval_4096"
        ][0]
        self.assertGreaterEqual(len(long_context["prompt"]), 512)

    def test_score_predictions_outputs_phase_d_rows_and_counts_missing_answers(self):
        suite = build_phase_d_task_suite(seed=11, examples_per_task=1, context_length=256)
        predictions = []
        for example in suite["examples"]:
            prediction = example["answer"]
            if example["task_id"] == "tool_selection":
                prediction = "wrong_tool"
            if example["task_id"] == "delayed_goal_binding":
                continue
            predictions.append(
                {
                    "example_id": example["id"],
                    "control_id": "tac_control_v1",
                    "prediction": prediction,
                    "tokens_per_second": 400.0,
                    "wall_clock_seconds": 1.5,
                }
            )

        rows = score_phase_d_predictions(
            suite["examples"],
            predictions,
            control_id="tac_control_v1",
            seed=11,
        )

        by_task = {row["task_id"]: row for row in rows}
        self.assertEqual(by_task["multi_hop_chain_retrieval"]["primary_score"], 1.0)
        self.assertEqual(by_task["tool_selection"]["primary_score"], 0.0)
        self.assertEqual(by_task["delayed_goal_binding"]["primary_score"], 0.0)
        self.assertEqual(by_task["multi_hop_chain_retrieval"]["primary_metric"], "exact_match")
        self.assertEqual(by_task["multi_hop_chain_retrieval"]["control_id"], "tac_control_v1")
        self.assertEqual(by_task["multi_hop_chain_retrieval"]["seed"], 11)
        self.assertEqual(by_task["multi_hop_chain_retrieval"]["tokens_per_second"], 400.0)

    def test_stage_phase_d_benchmark_suite_writes_manifest_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            manifest = stage_phase_d_benchmark_suite(
                output_dir=output_dir,
                seeds=[11, 23],
                examples_per_task=1,
                context_length=256,
            )

            self.assertEqual(manifest["phase"], "D")
            self.assertEqual(manifest["seeds"], [11, 23])
            self.assertTrue((output_dir / "phase_d_benchmark_manifest.json").exists())
            self.assertTrue((output_dir / "RESULTS.md").exists())
            self.assertTrue((output_dir / "seed_11" / "tasks.jsonl").exists())
            self.assertTrue((output_dir / "seed_23" / "tasks.jsonl").exists())
            self.assertGreater(manifest["example_count"], 0)

    def test_phase_d_byte_tokenizer_matches_training_contract(self):
        token_ids = phase_d_text_to_token_ids("A", vocab_size=260, append_eos=True)

        self.assertEqual(token_ids, [69, 3])
        self.assertEqual(phase_d_token_ids_to_text(token_ids), "A")
        self.assertEqual(
            extract_phase_d_answer(" value_123\nextra text", mode="first_token"),
            "value_123",
        )

    def test_checkpoint_prediction_runner_writes_scoring_compatible_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = build_phase_d_task_suite(
                seed=11,
                examples_per_task=1,
                context_length=64,
            )
            tasks_path = root / "tasks.jsonl"
            with tasks_path.open("w", encoding="utf-8") as handle:
                for row in suite["examples"]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
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
            _write_test_checkpoint(checkpoint_path, model, step=5)

            output_path = root / "predictions.jsonl"
            payload = run_phase_d_checkpoint_predictions(
                tasks_jsonl=tasks_path,
                checkpoint_path=checkpoint_path,
                control_id="tac_control_v1_seed_11",
                seed=11,
                output_jsonl=output_path,
                max_new_tokens=2,
                device="cpu",
            )

            rows = load_jsonl(output_path)
            scored = score_phase_d_predictions(
                suite["examples"],
                rows,
                control_id="tac_control_v1_seed_11",
                seed=11,
            )

            self.assertEqual(payload["prediction_count"], len(suite["examples"]))
            self.assertEqual(payload["model_type"], "tac")
            self.assertEqual(len(rows), len(suite["examples"]))
            self.assertEqual(rows[0]["model_type"], "tac")
            self.assertIn("raw_completion", rows[0])
            self.assertIn("tokens_per_second", rows[0])
            self.assertEqual(len(scored), len({row["task_id"] for row in suite["examples"]}))

    def test_checkpoint_loader_infers_vanilla_state_dicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "vanilla.pt"
            config = TACConfig(
                vocab_size=260,
                d_model=8,
                n_heads=2,
                n_layers=1,
                n_programs=4,
                max_seq_len=32,
            )
            model = VanillaTransformerLM(config)
            _write_test_checkpoint(checkpoint_path, model, step=7)

            loaded, metadata = load_phase_d_checkpoint_model(checkpoint_path)

            self.assertIsInstance(loaded, VanillaTransformerLM)
            self.assertEqual(metadata["model_type"], "vanilla")
            self.assertEqual(metadata["checkpoint_step"], 7)

    def test_phase_d_scratchpad_state_predictions_use_verified_state(self):
        suite = build_phase_d_task_suite(seed=5, examples_per_task=1, context_length=64)
        example = suite["examples"][0]
        state = AgenticScratchpadState(
            items=(
                ScratchpadItem(
                    item_id="answer",
                    kind="answer",
                    payload=example["answer"],
                    utility=1.0,
                    confidence=1.0,
                    verified=True,
                ),
                ScratchpadItem(
                    item_id="unverified",
                    kind="simulation",
                    payload="wrong_answer",
                    utility=1.0,
                    confidence=1.0,
                    imagined=True,
                    verified=False,
                ),
            ),
            budget=2,
            step=1,
        )

        report = run_phase_d_scratchpad_state_predictions(
            examples=[example],
            scratchpad_by_example={example["id"]: state},
            control_id="scratchpad_state",
            seed=5,
        )
        rows = report["rows"]
        scored = score_phase_d_predictions(
            [example],
            rows,
            control_id="scratchpad_state",
            seed=5,
        )

        self.assertEqual(rows[0]["prediction"], example["answer"])
        self.assertIn("Verified scratchpad", rows[0]["augmented_prompt"])
        self.assertNotIn("wrong_answer", rows[0]["augmented_prompt"])
        self.assertEqual(scored[0]["primary_score"], 1.0)

    def test_live_phase_d_policy_builds_verified_scratchpad_advantage(self):
        report = live_phase_d_policy.run_live_phase_d_scratchpad_policy_probe(
            seed=7,
            examples_per_task=1,
            context_length=64,
            train_steps=80,
        )

        self.assertEqual(
            report["decision"]["status"],
            "live_phase_d_scratchpad_policy_proved",
        )
        self.assertGreater(
            report["scores"]["scratchpad_mean_score"],
            report["scores"]["no_scratchpad_mean_score"],
        )
        self.assertGreaterEqual(report["scores"]["scratchpad_mean_score"], 0.95)
        self.assertEqual(report["scratchpad"]["hypothesis_contamination_rate"], 0.0)
        self.assertEqual(report["scratchpad"]["unverified_prompt_leak_count"], 0)
        self.assertEqual(report["policy"]["scratchpad_selection_score"], 1.0)

    def test_phase_d_benchmark_matrix_prefers_fair_seed_checkpoint_and_writes_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_dir = root / "suite"
            phase_b_dir = root / "phase_b"
            output_dir = root / "phase_d_predictions"
            stage_phase_d_benchmark_suite(
                output_dir=suite_dir,
                seeds=[11],
                examples_per_task=1,
                context_length=64,
            )
            config = TACConfig(
                vocab_size=260,
                d_model=8,
                n_heads=2,
                n_layers=1,
                n_programs=4,
                max_seq_len=64,
            )
            seed_run_dir = phase_b_dir / "seed_11" / "tac_control_v1_seed_11"
            fair_checkpoint = (
                seed_run_dir
                / "specialization_checkpoints"
                / "step_010000"
                / "checkpoint.pt"
            )
            best_checkpoint = seed_run_dir / "best.pt"
            _write_test_checkpoint(fair_checkpoint, TACTransformerLM(config), step=10000)
            _write_test_checkpoint(best_checkpoint, TACTransformerLM(config), step=20000)
            vanilla_checkpoint = root / "vanilla" / "best.pt"
            _write_test_checkpoint(vanilla_checkpoint, VanillaTransformerLM(config), step=20000)

            self.assertEqual(
                discover_phase_d_seed_checkpoint(phase_b_dir, seed=11),
                fair_checkpoint,
            )

            payload = run_phase_d_benchmark_matrix(
                suite_dir=suite_dir,
                phase_b_dir=phase_b_dir,
                output_dir=output_dir,
                vanilla_checkpoint=vanilla_checkpoint,
                seeds=[11],
                max_new_tokens=1,
                device="cpu",
            )

            combined_rows = load_jsonl(output_dir / "phase_d_benchmark_rows.jsonl")
            control_ids = {row["control_id"] for row in combined_rows}

            self.assertEqual(payload["decision"]["status"], "completed")
            self.assertEqual(payload["seed_count"], 1)
            self.assertEqual(payload["row_count"], 10)
            self.assertEqual(len(combined_rows), 10)
            self.assertIn("tac_control_v1_seed_11", control_ids)
            self.assertIn("parameter_matched_vanilla", control_ids)
            self.assertTrue((output_dir / "phase_d_prediction_matrix.json").exists())
            self.assertIn(
                output_dir / "phase_d_benchmark_rows.jsonl",
                discover_phase_d_row_sources(output_dir),
            )

            second_payload = run_phase_d_benchmark_matrix(
                suite_dir=suite_dir,
                phase_b_dir=phase_b_dir,
                output_dir=output_dir,
                vanilla_checkpoint=vanilla_checkpoint,
                seeds=[11],
                max_new_tokens=1,
                device="cpu",
            )

            self.assertTrue(
                all(run.get("skipped") == "existing_score" for run in second_payload["runs"])
            )

    def test_phase_d_benchmark_matrix_reports_pending_when_checkpoints_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_dir = root / "suite"
            output_dir = root / "out"
            stage_phase_d_benchmark_suite(
                output_dir=suite_dir,
                seeds=[11],
                examples_per_task=1,
                context_length=64,
            )
            config = TACConfig(
                vocab_size=260,
                d_model=8,
                n_heads=2,
                n_layers=1,
                n_programs=4,
                max_seq_len=64,
            )
            vanilla_checkpoint = root / "vanilla" / "best.pt"
            _write_test_checkpoint(vanilla_checkpoint, VanillaTransformerLM(config), step=20000)

            payload = run_phase_d_benchmark_matrix(
                suite_dir=suite_dir,
                phase_b_dir=root / "missing_phase_b",
                output_dir=output_dir,
                vanilla_checkpoint=vanilla_checkpoint,
                seeds=[11],
                max_new_tokens=1,
                device="cpu",
            )

            self.assertEqual(payload["decision"]["status"], "pending")
            self.assertGreaterEqual(len(payload["missing"]), 1)
            self.assertEqual(payload["row_count"], 0)
            self.assertTrue((output_dir / "phase_d_prediction_matrix.json").exists())

    def test_phase_d_benchmark_matrix_blocks_when_phase_b_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_dir = root / "suite"
            phase_b_dir = root / "phase_b"
            output_dir = root / "phase_d_predictions"
            phase_b_results = root / "phase_b_seed_results.json"
            stage_phase_d_benchmark_suite(
                output_dir=suite_dir,
                seeds=[11],
                examples_per_task=1,
                context_length=64,
            )
            config = TACConfig(
                vocab_size=260,
                d_model=8,
                n_heads=2,
                n_layers=1,
                n_programs=4,
                max_seq_len=64,
            )
            checkpoint = (
                phase_b_dir
                / "seed_11"
                / "tac_control_v1_seed_11"
                / "specialization_checkpoints"
                / "step_010000"
                / "checkpoint.pt"
            )
            _write_test_checkpoint(checkpoint, TACTransformerLM(config), step=10000)
            phase_b_results.write_text(
                json.dumps(
                    {
                        "phase": "B",
                        "decision": {
                            "status": "fail",
                            "ready_for_phase_d": False,
                            "reason": "At least one Phase B seed failed a hard gate.",
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = run_phase_d_benchmark_matrix(
                suite_dir=suite_dir,
                phase_b_dir=phase_b_dir,
                phase_b_results=phase_b_results,
                output_dir=output_dir,
                vanilla_checkpoint=None,
                seeds=[11],
                max_new_tokens=1,
                device="cpu",
            )

            self.assertEqual(payload["decision"]["status"], "blocked_by_phase_b")
            self.assertEqual(payload["decision"]["phase_b_status"], "fail")
            self.assertEqual(payload["run_count"], 0)
            self.assertEqual(payload["row_count"], 0)
            self.assertEqual(load_jsonl(output_dir / "phase_d_benchmark_rows.jsonl"), [])
            self.assertFalse((output_dir / "seed_11").exists())

    def test_stage_phase_d_suite_dataset_copies_suite_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite_dir = root / "suite"
            dataset_dir = root / "dataset"
            stage_phase_d_benchmark_suite(
                output_dir=suite_dir,
                seeds=[11],
                examples_per_task=1,
                context_length=64,
            )

            payload = stage_phase_d_suite_dataset(
                suite_dir=suite_dir,
                output_dir=dataset_dir,
                dataset_id="jeffkolo/tac-control-v1-phase-d-suite-test",
            )
            metadata = json.loads(
                (dataset_dir / "dataset-metadata.json").read_text(encoding="utf-8")
            )

            self.assertEqual(payload["dataset_id"], "jeffkolo/tac-control-v1-phase-d-suite-test")
            self.assertEqual(metadata["id"], "jeffkolo/tac-control-v1-phase-d-suite-test")
            self.assertTrue((dataset_dir / "seed_11" / "tasks.jsonl").exists())
            self.assertTrue((dataset_dir / "phase_d_benchmark_manifest.json").exists())
            self.assertGreater(payload["file_count"], 0)


def _write_test_checkpoint(path: Path, model: torch.nn.Module, *, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_eval_loss": 1.0,
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
        },
        path,
    )


if __name__ == "__main__":
    unittest.main()
