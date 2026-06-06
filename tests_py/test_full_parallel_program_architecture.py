import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_full_parallel_program_architecture as bench


class FullParallelProgramArchitectureTests(unittest.TestCase):
    def test_full_probe_covers_all_pasted_ideas(self):
        result = bench.run_full_architecture_probe(
            seeds=[3, 5],
            batch_size=16,
            vocab_size=48,
            seq_len=8,
            top_k=3,
            stochastic_samples=12,
        )

        self.assertEqual(
            result["decision"]["status"],
            "full_parallel_program_architecture_promote",
        )
        self.assertEqual(
            set(result["ideas"].keys()),
            {
                "parallel_reasoning_trajectories",
                "program_disagreement_signal",
                "integrated_verifiers",
                "specialized_computation",
                "stochastic_path_exploration",
            },
        )
        for idea in result["ideas"].values():
            self.assertEqual(idea["status"], "promote_candidate")
            self.assertFalse(idea["uses_target_labels_for_selection"])

    def test_integrated_verifier_beats_confidence_only_on_multihop(self):
        result = bench.run_full_architecture_probe(
            seeds=[7],
            batch_size=12,
            vocab_size=48,
            seq_len=8,
            top_k=3,
            stochastic_samples=10,
        )
        verifier = result["ideas"]["integrated_verifiers"]["metrics"]

        self.assertEqual(verifier["confidence_only_accuracy"], 0.0)
        self.assertEqual(verifier["structural_verifier_accuracy"], 1.0)
        self.assertGreater(verifier["accuracy_delta"], 0.9)

    def test_disagreement_and_stochastic_exploration_are_measurable_signals(self):
        result = bench.run_full_architecture_probe(
            seeds=[11],
            batch_size=10,
            vocab_size=48,
            seq_len=8,
            top_k=3,
            stochastic_samples=16,
        )
        disagreement = result["ideas"]["program_disagreement_signal"]["metrics"]
        stochastic = result["ideas"]["stochastic_path_exploration"]["metrics"]

        self.assertGreaterEqual(disagreement["failure_detection_auc"], 0.99)
        self.assertGreater(
            disagreement["multi_hop_mean_disagreement"],
            disagreement["single_key_mean_disagreement"],
        )
        self.assertEqual(stochastic["greedy_accuracy"], 0.0)
        self.assertGreaterEqual(stochastic["stochastic_accuracy"], 0.9)
        self.assertGreaterEqual(stochastic["mean_unique_candidate_fraction"], 0.9)

    def test_specialized_computation_beats_memory_only_raw_retrieval(self):
        result = bench.run_specialized_computation_probe(
            seeds=[13],
            batch_size=24,
            vocab_size=64,
        )

        self.assertLess(result["memory_only_accuracy"], 0.6)
        self.assertEqual(result["program_computation_accuracy"], 1.0)
        self.assertGreater(result["accuracy_delta"], 0.4)

    def test_cli_writes_full_architecture_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "1",
                    "--batch-size",
                    "8",
                    "--stochastic-samples",
                    "16",
                ]
            )

            artifact = json.loads(
                (output_dir / "full_parallel_program_architecture.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "full_parallel_program_architecture.v1")
        self.assertIn("Full Parallel Program Architecture Probe", markdown)
        self.assertIn(artifact["decision"]["status"], markdown)


if __name__ == "__main__":
    unittest.main()
