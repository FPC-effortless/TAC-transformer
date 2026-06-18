import json
import tempfile
import unittest
from pathlib import Path

from tac_transformer.research_directions import (
    StructureMemoryRecord,
    structure_memory_score,
    update_structure_memory,
)


class TAC277StructureMemoryTests(unittest.TestCase):
    def test_structure_memory_updates_usage_survival_and_transfer_edges(self):
        record = StructureMemoryRecord(structure_id="plant_family")

        updated = update_structure_memory(
            record,
            task_descriptor="tree_few_shot",
            success=True,
            reset_drop=0.42,
            knockout_drop=0.39,
            transfer_to="fruit_color_family",
            transfer_gain=0.11,
        )

        self.assertEqual(updated.structure_id, "plant_family")
        self.assertEqual(updated.success_count, 1)
        self.assertEqual(updated.failure_count, 0)
        self.assertIn("tree_few_shot", updated.task_descriptors)
        self.assertEqual(updated.transfer_edges["fruit_color_family"]["count"], 1)
        self.assertGreater(updated.survival_score, 0.0)
        self.assertGreater(updated.reuse_score, 0.0)
        self.assertGreater(structure_memory_score(updated), structure_memory_score(record))

    def test_tac277_structure_memory_benchmark_contract(self):
        from experiments.benchmark_tac277_structure_memory import (
            run_tac277_structure_memory,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac277_structure_memory(
                output_dir=Path(tmp),
                seeds=(7,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )

            self.assertEqual(result["schema"], "tac277_structure_memory.v1")
            self.assertEqual(result["method"]["task"], "structure_memory")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            for key in (
                "memory_records",
                "mean_success_rate",
                "mean_survival_score",
                "mean_reuse_score",
                "mean_reset_sensitivity",
                "mean_knockout_sensitivity",
                "transfer_edge_count",
                "structure_memory_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac277_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-277", tickets)
        self.assertEqual(tickets["TAC-277"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-277"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
