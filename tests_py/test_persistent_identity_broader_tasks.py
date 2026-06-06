import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_persistent_identity_broader_tasks as bench


class PersistentIdentityBroaderTasksTests(unittest.TestCase):
    def test_bridge_probe_promotes_all_broader_task_families(self):
        result = bench.run_broader_task_bridge_probe(
            seeds=[3, 5],
            identities_per_seed=8,
            examples_per_task=4,
            vocab_size=64,
        )

        self.assertEqual(
            result["decision"]["status"],
            "persistent_identity_broader_task_bridge_proved",
        )
        self.assertEqual(
            set(result["task_metrics"].keys()),
            {
                "transfer_learning",
                "multi_hop_reasoning",
                "agent_memory",
                "language_like_instruction",
            },
        )
        for metrics in result["task_metrics"].values():
            self.assertGreaterEqual(metrics["persistent_identity_accuracy"], 0.95)
            self.assertLessEqual(metrics["best_non_identity_accuracy"], 0.35)
            self.assertGreaterEqual(metrics["persistent_advantage"], 0.60)

    def test_language_like_rows_are_prompted_but_marked_as_proxy(self):
        suite = bench.build_broader_task_suite(
            seeds=[7],
            identities_per_seed=4,
            examples_per_task=3,
            vocab_size=64,
        )
        language_rows = [
            row for row in suite["rows"] if row["task_family"] == "language_like_instruction"
        ]

        self.assertTrue(language_rows)
        self.assertTrue(all("prompt" in row for row in language_rows))
        self.assertTrue(all("Return only" in row["prompt"] for row in language_rows))
        self.assertEqual(suite["real_world_benchmark_status"], "proxy_not_real_world")

    def test_multi_hop_requires_identity_rule_composition(self):
        suite = bench.build_broader_task_suite(
            seeds=[11],
            identities_per_seed=4,
            examples_per_task=3,
            vocab_size=64,
        )
        persistent = bench.evaluate_solver(suite, solver="persistent_identity")
        stateless = bench.evaluate_solver(suite, solver="stateless_reset")

        multi_persistent = persistent["by_task"]["multi_hop_reasoning"]["accuracy"]
        multi_stateless = stateless["by_task"]["multi_hop_reasoning"]["accuracy"]
        self.assertEqual(multi_persistent, 1.0)
        self.assertLessEqual(multi_stateless, 0.35)

    def test_agent_memory_needs_identity_keying_not_global_state(self):
        suite = bench.build_broader_task_suite(
            seeds=[13],
            identities_per_seed=8,
            examples_per_task=3,
            vocab_size=64,
        )
        persistent = bench.evaluate_solver(suite, solver="persistent_identity")
        global_only = bench.evaluate_solver(
            suite,
            solver="global_persistent_without_identity",
        )

        self.assertEqual(persistent["by_task"]["agent_memory"]["accuracy"], 1.0)
        self.assertLessEqual(global_only["by_task"]["agent_memory"]["accuracy"], 0.35)

    def test_cli_writes_bridge_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            bench.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "1",
                    "--identities-per-seed",
                    "4",
                    "--examples-per-task",
                    "2",
                ]
            )

            artifact = json.loads(
                (output_dir / "persistent_identity_broader_tasks.json").read_text(
                    encoding="utf-8"
                )
            )
            markdown = (output_dir / "RESULTS.md").read_text(encoding="utf-8")

        self.assertEqual(artifact["schema"], "persistent_identity_broader_tasks.v1")
        self.assertIn("Persistent Identity Broader Tasks", markdown)
        self.assertIn(artifact["decision"]["status"], markdown)


if __name__ == "__main__":
    unittest.main()
