import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_live_tac_runtime_search as bench
from kaggle import make_agentic_training_bundle
from tac_transformer.runtime_search import (
    TACRuntimeSearchConfig,
    run_tac_runtime_search,
)
from tac_transformer.training import ChunkedRecallBatcher


class LiveTACRuntimeSearchTests(unittest.TestCase):
    def test_runtime_search_selects_verified_bridge_without_target_labels(self):
        batch = ChunkedRecallBatcher(
            vocab_size=48,
            seq_len=8,
            seed=3,
            task_variant="multi_hop",
        ).next_batch(batch_size=8)
        logits = bench.build_controlled_runtime_logits(
            batch,
            task="multi_hop",
            vocab_size=48,
            bridge_rank=1,
            distractor_rank=0,
        )

        greedy = run_tac_runtime_search(
            logits,
            batch,
            config=TACRuntimeSearchConfig(top_k=1, max_steps=2),
        )
        searched = run_tac_runtime_search(
            logits,
            batch,
            config=TACRuntimeSearchConfig(top_k=3, max_steps=2),
        )

        self.assertEqual(greedy.accuracy, 0.0)
        self.assertEqual(searched.accuracy, 1.0)
        self.assertFalse(searched.uses_target_labels_for_selection)
        self.assertEqual(searched.hypothesis_contamination, 0.0)
        self.assertGreater(searched.committed_scratchpad_items, 0)

    def test_runtime_search_benchmark_writes_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "3",
                    "5",
                    "--tasks",
                    "single_key",
                    "multi_hop",
                    "--batch-size",
                    "8",
                    "--top-k",
                    "3",
                    "--max-steps",
                    "2",
                ]
            )

            artifact = json.loads(
                (output_dir / "live_tac_runtime_search.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "live_tac_runtime_search.v1")
        self.assertEqual(artifact["decision"]["status"], "runtime_search_useful")
        self.assertIn("Live TAC-State Runtime Search", markdown)
        self.assertFalse(artifact["selection_contract"]["uses_target_labels_for_selection"])
        self.assertIn(
            "experiments/benchmark_live_tac_runtime_search.py",
            make_agentic_training_bundle.FILES,
        )


if __name__ == "__main__":
    unittest.main()
