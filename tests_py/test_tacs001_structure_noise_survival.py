import json
import tempfile
import unittest
from pathlib import Path


class TACS001StructureNoiseSurvivalTests(unittest.TestCase):
    def test_tacs001_structure_noise_survival_contract(self):
        from experiments.benchmark_tacs001_structure_noise_survival import (
            run_tacs001_structure_noise_survival,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tacs001_structure_noise_survival(
                output_dir=Path(tmp),
                seeds=(7,),
                source_examples=12,
                target_shots=3,
                eval_examples=12,
                steps=30,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )

            self.assertEqual(result["schema"], "tacs001_structure_noise_survival.v1")
            self.assertEqual(result["method"]["task"], "structure_noise_survival")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            for key in (
                "clean_target_accuracy",
                "noisy_target_accuracy",
                "target_noise_retention",
                "clean_family_accuracy",
                "noisy_family_accuracy",
                "family_noise_retention",
                "source_noise_retention",
                "noise_recovery_score",
                "structure_memory_survival_score",
                "noise_survival_score",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tacs001_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-S001", tickets)
        self.assertEqual(tickets["TAC-S001"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-S001"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
