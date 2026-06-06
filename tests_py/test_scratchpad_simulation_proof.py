import unittest

from experiments import benchmark_scratchpad_simulation_proof as bench


class ScratchpadSimulationProofTest(unittest.TestCase):
    def test_scratchpad_simulation_and_teaching_controls_show_required_gains(self):
        report = bench.run_scratchpad_simulation_proof(example_count=24, seed=7)

        self.assertEqual(report["decision"]["status"], "mechanisms_proved")
        self.assertGreater(
            report["scratchpad"]["scratchpad_score"],
            report["scratchpad"]["no_scratchpad_score"],
        )
        self.assertGreater(
            report["simulation"]["simulation_score"],
            report["simulation"]["no_simulation_score"],
        )
        self.assertGreater(
            report["teaching"]["teaching_score"],
            report["teaching"]["no_teaching_score"],
        )
        self.assertEqual(report["scratchpad"]["hypothesis_contamination_rate"], 0.0)

    def test_scratchpad_simulation_report_markdown_names_controls(self):
        report = bench.run_scratchpad_simulation_proof(example_count=8, seed=11)
        markdown = bench.format_markdown(report)

        self.assertIn("no_scratchpad", markdown)
        self.assertIn("no_simulation", markdown)
        self.assertIn("no_teaching", markdown)
        self.assertIn("mechanisms_proved", markdown)


if __name__ == "__main__":
    unittest.main()
