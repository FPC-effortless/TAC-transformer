import json
import tempfile
import unittest
from pathlib import Path

import torch

from tac_transformer.research_directions import (
    CONCEPT_RELATION_TYPES,
    adaptive_concept_volume_loss,
    concept_relation_loss,
    concept_subsumption_loss,
)


class TAC274AdaptiveConceptVolumeLossTests(unittest.TestCase):
    def test_adaptive_volume_loss_prefers_anisotropic_region_and_backprops(self):
        embeddings = torch.tensor(
            [[2.0, 0.0], [1.5, 0.1], [0.0, 0.0]],
            dtype=torch.float32,
        )
        concept_ids = torch.zeros(3, dtype=torch.long)
        mean = torch.zeros(1, 2, requires_grad=True)
        anisotropic_log_vars = torch.log(torch.tensor([[4.0, 0.25]], dtype=torch.float32))
        spherical_log_vars = torch.zeros(1, 2)

        anisotropic_loss = adaptive_concept_volume_loss(
            embeddings,
            concept_ids,
            mean,
            anisotropic_log_vars,
        )
        spherical_loss = adaptive_concept_volume_loss(
            embeddings,
            concept_ids,
            mean.detach(),
            spherical_log_vars,
        )

        self.assertLess(float(anisotropic_loss.detach()), float(spherical_loss))
        anisotropic_loss.backward()
        self.assertIsNotNone(mean.grad)

    def test_subsumption_penalizes_child_outside_or_larger_than_parent(self):
        good_means = torch.tensor([[0.0, 0.0], [0.1, 0.1]], dtype=torch.float32)
        good_log_vars = torch.log(torch.tensor([[2.0, 2.0], [0.2, 0.2]], dtype=torch.float32))
        bad_means = torch.tensor([[0.0, 0.0], [4.0, 4.0]], dtype=torch.float32)
        bad_log_vars = torch.log(torch.tensor([[0.2, 0.2], [2.0, 2.0]], dtype=torch.float32))

        good_loss = concept_subsumption_loss(
            good_means,
            good_log_vars,
            torch.tensor([1]),
            torch.tensor([0]),
        )
        bad_loss = concept_subsumption_loss(
            bad_means,
            bad_log_vars,
            torch.tensor([1]),
            torch.tensor([0]),
        )

        self.assertLess(float(good_loss), float(bad_loss))

    def test_relation_loss_distinguishes_overlap_and_disjoint_pairs(self):
        means_good = torch.tensor(
            [[0.0, 0.0], [0.2, 0.0], [5.0, 0.0], [8.0, 0.0]],
            dtype=torch.float32,
        )
        log_vars_good = torch.log(torch.tensor([[1.0, 1.0]] * 4, dtype=torch.float32))
        means_bad = torch.tensor(
            [[0.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.2, 0.0]],
            dtype=torch.float32,
        )
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        relation_types = torch.tensor(
            [
                CONCEPT_RELATION_TYPES["overlaps"],
                CONCEPT_RELATION_TYPES["disjoint"],
            ],
            dtype=torch.long,
        )

        good_loss = concept_relation_loss(
            means_good,
            log_vars_good,
            pairs,
            relation_types,
        )
        bad_loss = concept_relation_loss(
            means_bad,
            log_vars_good,
            pairs,
            relation_types,
        )

        self.assertLess(float(good_loss), float(bad_loss))

    def test_tac274_adaptive_concept_volume_benchmark_contract(self):
        from experiments.benchmark_tac274_adaptive_concept_volume_loss import (
            run_tac274_adaptive_concept_volume_loss,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_tac274_adaptive_concept_volume_loss(
                output_dir=Path(tmp),
                seeds=(7,),
                examples_per_concept=16,
                steps=30,
                eval_batches=1,
                batch_size=2,
                torch_threads=1,
                smoke=True,
            )

            self.assertEqual(result["schema"], "tac274_adaptive_concept_volume_loss.v1")
            self.assertEqual(result["method"]["task"], "adaptive_concept_volume_loss")
            self.assertEqual(result["method"]["assigned_label"], "TAC-274 because TAC-273 already exists in prd.json")
            self.assertIn(result["decision"]["status"], {"validated", "not_validated", "blocked"})
            self.assertTrue(Path(result["artifact_path"]).exists())
            for key in (
                "adaptive_eval_loss",
                "fixed_isotropic_eval_loss",
                "adaptive_loss_advantage",
                "adaptive_assignment_accuracy",
                "shape_logvar_correlation",
                "hierarchy_subsumption_loss",
                "relation_loss",
                "program_knockout_drop_proxy",
                "reset_accuracy_proxy",
                "lm_collapse_proxy",
            ):
                self.assertIn(key, result["metrics"])
            json.dumps(result)

    def test_prd_contains_pending_tac274_ticket(self):
        prd = json.loads(Path("prd.json").read_text(encoding="utf-8"))
        tickets = {ticket["id"]: ticket for ticket in prd["tickets"]}
        self.assertIn("TAC-274", tickets)
        self.assertEqual(tickets["TAC-274"]["status"], "pending")
        self.assertGreaterEqual(len(tickets["TAC-274"]["acceptance"]), 5)


if __name__ == "__main__":
    unittest.main()
