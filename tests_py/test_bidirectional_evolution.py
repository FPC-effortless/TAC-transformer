import unittest

from kaggle.benchmark_bidirectional_evolution import (
    aggregate_evolutionary_results,
    bin_index,
    binary_entropy,
    failure_flags,
)


def _fake_run(
    candidate,
    *,
    task="multi_hop",
    carry=0.5,
    reset=0.2,
    shuffled=0.25,
    baseline=0.1,
    cosine=0.4,
    dead_rate=0.0,
    gate=0.5,
    energy=2.0,
    tps=100.0,
    baseline_tps=200.0,
):
    probe = {
        "carry": {
            "value_accuracy": carry,
            "used_energy": energy,
            "active_programs": 2.0,
            "program_memory_cosine": cosine,
            "memory_allocation_dead_rate": dead_rate,
            "memory_allocation_load_std": 0.1,
            "content_synthesis_gate": gate,
        },
        "reset": {"value_accuracy": reset},
        "shuffled": {"value_accuracy": shuffled},
    }
    return {
        "candidate": candidate,
        "variant": candidate,
        "task": task,
        "seed": 11,
        "decision": {"status": "effective"},
        "tac": {
            "train": {"tokens_per_second": tps},
            "chunked_probe": probe,
        },
        "baseline": {
            "train": {"tokens_per_second": baseline_tps},
            "chunked_probe": {"carry": {"value_accuracy": baseline}},
        },
    }


class BidirectionalEvolutionTest(unittest.TestCase):
    def test_entropy_and_bins_are_bounded(self):
        self.assertEqual(binary_entropy(0.0), 0.0)
        self.assertEqual(binary_entropy(1.0), 0.0)
        self.assertAlmostEqual(binary_entropy(0.5), 1.0)
        self.assertEqual(bin_index(-1.0, 5), 0)
        self.assertEqual(bin_index(1.0, 5), 4)

    def test_failure_flags_catch_saturation_collapse_and_dead_programs(self):
        metrics = {
            "mean_content_synthesis_gate": 0.94,
            "mean_program_memory_cosine": 0.98,
            "mean_memory_allocation_dead_rate": 0.75,
        }

        flags = failure_flags(
            metrics,
            program_collapse_threshold=0.95,
            dead_program_threshold=0.5,
        )

        self.assertTrue(flags["gate_saturated"])
        self.assertTrue(flags["programs_collapsed"])
        self.assertTrue(flags["dead_programs"])

    def test_aggregate_scores_pareto_front_and_map_elites(self):
        runs = [
            _fake_run("direct", carry=0.6, reset=0.1, shuffled=0.2, cosine=0.8, gate=0.5),
            _fake_run("novel", carry=0.45, reset=0.2, shuffled=0.25, cosine=0.2, gate=0.5),
            _fake_run("collapsed", carry=0.7, reset=0.69, shuffled=0.68, cosine=0.99, gate=0.95),
        ]

        aggregate = aggregate_evolutionary_results(
            runs,
            tasks={"multi_hop": {"task_variant": "multi_hop", "seq_len": 16}},
            candidates={"direct": {}, "novel": {}, "collapsed": {}},
            map_bins=4,
            program_collapse_threshold=0.95,
        )

        ranking = aggregate["ranking_by_survival_score"]
        by_name = {row["candidate"]: row for row in ranking}

        self.assertIn("map_elites", aggregate)
        self.assertGreaterEqual(aggregate["map_elites"]["filled_cells"], 2)
        self.assertTrue(by_name["collapsed"]["constraint_violated"])
        self.assertFalse(by_name["direct"]["constraint_violated"])
        self.assertGreater(by_name["novel"]["behavioral_novelty"], 0.0)
        self.assertTrue(aggregate["pareto_front"])


if __name__ == "__main__":
    unittest.main()
