import json
import tempfile
import unittest
from pathlib import Path


class TACS010StructureSuiteReplicationTests(unittest.TestCase):
    def test_tacs010_structure_suite_replication_contract(self):
        from experiments.benchmark_tacs010_structure_suite_replication import (
            run_tacs010_structure_suite_replication,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tacs010_structure_suite_replication(
                output_dir=Path(tmp),
                seeds=(7, 19),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )

            self.assertEqual(result["schema"], "tacs010_structure_suite_replication.v1")
            self.assertEqual(result["method"]["task"], "structure_suite_replication")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            self.assertIn("matrix", result)
            self.assertGreaterEqual(len(result["matrix"]), 5)
            for key in (
                "seed_count",
                "benchmarks_passed",
                "benchmark_pass_rate",
                "mean_structure_advantage",
                "mean_knockout_drop",
                "mean_survival_score",
                "mean_transfer_gain",
                "ablation_failure_rate",
                "replication_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tacs010_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-S010", tickets)
        self.assertEqual(tickets["TAC-S010"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-S010"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
