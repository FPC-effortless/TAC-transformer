import json
import tempfile
import unittest
from pathlib import Path


class StructureHardGateTests(unittest.TestCase):
    def _assert_contract(self, result, *, schema, task, keys):
        self.assertEqual(result["schema"], schema)
        self.assertEqual(result["method"]["task"], task)
        self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
        self.assertTrue(Path(result["artifact_path"]).exists())
        for key in keys:
            self.assertIn(key, result["metrics"])
        json.dumps(result)

    def test_tacs002_memory_attack_contract(self):
        from experiments.benchmark_tacs002_structure_memory_attack import (
            run_tacs002_structure_memory_attack,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tacs002_structure_memory_attack(
                output_dir=Path(tmp),
                seeds=(7,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_contract(
                result,
                schema="tacs002_structure_memory_attack.v1",
                task="structure_memory_attack",
                keys=(
                    "clean_memory_score",
                    "attacked_memory_score",
                    "recovered_memory_score",
                    "attack_drop",
                    "recovery_fraction",
                    "survival_after_recovery",
                    "transfer_edges_recovered",
                ),
            )

    def test_tacs003_distribution_shift_contract(self):
        from experiments.benchmark_tacs003_distribution_shift import (
            run_tacs003_distribution_shift,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tacs003_distribution_shift(
                output_dir=Path(tmp),
                seeds=(7,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_contract(
                result,
                schema="tacs003_distribution_shift.v1",
                task="structure_distribution_shift",
                keys=(
                    "clean_target_accuracy",
                    "shifted_target_accuracy",
                    "target_shift_retention",
                    "clean_family_accuracy",
                    "shifted_family_accuracy",
                    "family_shift_retention",
                    "source_shift_retention",
                    "shift_survival_score",
                ),
            )

    def test_tacs101_ab_transfer_contract(self):
        from experiments.benchmark_tacs101_structure_ab_transfer import (
            run_tacs101_structure_ab_transfer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tacs101_structure_ab_transfer(
                output_dir=Path(tmp),
                seeds=(7,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_contract(
                result,
                schema="tacs101_structure_ab_transfer.v1",
                task="structure_ab_transfer",
                keys=(
                    "source_structure_accuracy",
                    "target_transfer_accuracy",
                    "fresh_target_accuracy",
                    "transfer_gain",
                    "learning_speed_gain",
                    "structure_reuse_score",
                    "transfer_knockout_drop",
                ),
            )

    def test_tacs102_abc_transfer_chain_contract(self):
        from experiments.benchmark_tacs102_structure_abc_transfer_chain import (
            run_tacs102_structure_abc_transfer_chain,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tacs102_structure_abc_transfer_chain(
                output_dir=Path(tmp),
                seeds=(7,),
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )
            self._assert_contract(
                result,
                schema="tacs102_structure_abc_transfer_chain.v1",
                task="structure_abc_transfer_chain",
                keys=(
                    "task_a_accuracy",
                    "task_b_transfer_accuracy",
                    "task_c_chain_accuracy",
                    "fresh_c_accuracy",
                    "chain_transfer_gain",
                    "chain_retention",
                    "chain_reuse_score",
                    "chain_knockout_drop",
                ),
            )

    def test_prd_contains_structure_hard_gate_tickets(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        for ticket_id in ("TAC-S002", "TAC-S003", "TAC-S101", "TAC-S102"):
            self.assertIn(ticket_id, tickets)
            self.assertEqual(tickets[ticket_id]["status"], "pending")
            self.assertGreaterEqual(len(tickets[ticket_id]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
