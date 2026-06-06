import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments import benchmark_parallel_program_trajectories as bench
from tac_transformer.training import ChunkedRecallBatcher


class ParallelProgramTrajectoryTests(unittest.TestCase):
    def test_parallel_verifier_selects_bridge_candidate_without_targets(self):
        batch = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=7,
            task_variant="multi_hop",
        ).next_batch(batch_size=4)
        logits = bench.build_controlled_first_hop_logits(
            batch,
            task="multi_hop",
            vocab_size=32,
            bridge_rank=1,
            distractor_rank=0,
        )

        greedy = bench.evaluate_trajectory_selection(
            logits,
            batch,
            top_k=1,
            max_steps=2,
        )
        parallel = bench.evaluate_trajectory_selection(
            logits,
            batch,
            top_k=3,
            max_steps=2,
        )

        self.assertEqual(greedy["accuracy"], 0.0)
        self.assertEqual(parallel["accuracy"], 1.0)
        self.assertEqual(parallel["used_target_labels_for_selection"], False)
        self.assertGreater(parallel["mean_selected_score"], greedy["mean_selected_score"])

    def test_parallel_probe_preserves_direct_and_improves_multihop(self):
        result = bench.run_parallel_trajectory_probe(
            tasks=["single_key", "multi_hop"],
            seeds=[3, 5],
            batch_size=16,
            vocab_size=48,
            seq_len=8,
            top_k=3,
            max_steps=2,
            min_multihop_gain=0.75,
            max_direct_regression=0.0,
        )

        self.assertEqual(result["decision"]["status"], "parallel_trajectory_probe_promote")
        self.assertEqual(result["by_task"]["single_key"]["parallel_accuracy"], 1.0)
        self.assertEqual(result["by_task"]["single_key"]["direct_regression"], 0.0)
        self.assertEqual(result["by_task"]["multi_hop"]["greedy_accuracy"], 0.0)
        self.assertEqual(result["by_task"]["multi_hop"]["parallel_accuracy"], 1.0)

    def test_cli_writes_json_and_markdown_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "1",
                    "2",
                    "--batch-size",
                    "8",
                    "--top-k",
                    "3",
                ]
            )

            artifact = json.loads(
                (output_dir / "parallel_program_trajectories.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "parallel_program_trajectories.v1")
        self.assertIn("Parallel Program Trajectories", markdown)
        self.assertIn(artifact["decision"]["status"], markdown)


if __name__ == "__main__":
    unittest.main()
