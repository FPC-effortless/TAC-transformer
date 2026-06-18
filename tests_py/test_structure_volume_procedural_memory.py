import unittest

import torch

from tac_transformer.research_directions import (
    ProceduralMemoryRecord,
    ProceduralMemoryStore,
    adapt_procedural_memory_after_feedback,
    two_level_structure_route,
)


class StructureVolumeProceduralMemoryTest(unittest.TestCase):
    def test_two_level_structure_route_selects_family_then_specialist(self):
        embeddings = torch.tensor([[0.10, 0.05], [4.20, 0.10]])
        family_means = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
        family_log_vars = torch.zeros_like(family_means)
        specialist_means = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 2.0]],
                [[4.0, 0.0], [4.0, 2.0]],
            ]
        )

        route = two_level_structure_route(
            embeddings,
            family_means,
            family_log_vars,
            specialist_means,
        )

        self.assertEqual(route.family_ids.tolist(), [0, 1])
        self.assertEqual(route.specialist_ids.tolist(), [0, 0])
        self.assertEqual(route.family_scores.shape, (2, 2))
        self.assertEqual(route.specialist_scores.shape, (2, 2))
        self.assertGreater(float(route.family_confidence.min()), 0.99)

    def test_two_level_structure_route_rejects_impossible_top_k(self):
        embeddings = torch.tensor([[0.0, 0.0]])
        family_means = torch.tensor([[0.0, 0.0]])
        family_log_vars = torch.zeros_like(family_means)
        specialist_means = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])

        with self.assertRaisesRegex(ValueError, "top_k must be <= specialists per family"):
            two_level_structure_route(
                embeddings,
                family_means,
                family_log_vars,
                specialist_means,
                top_k=3,
            )

    def test_procedural_memory_feedback_moves_retrieval_to_expected_family(self):
        store = ProceduralMemoryStore()
        store.write(
            ProceduralMemoryRecord(
                procedure_id="schema_patch_old",
                family_id="schema",
                task_descriptor="repair schema drift",
                steps=("edit schema",),
                embedding=torch.tensor([0.95, 0.05]),
                success_rate=0.95,
            )
        )
        store.write(
            ProceduralMemoryRecord(
                procedure_id="routing_patch",
                family_id="routing",
                task_descriptor="repair routing drift",
                steps=("edit route",),
                embedding=torch.tensor([0.70, 0.20]),
                success_rate=0.50,
            )
        )
        query = torch.tensor([1.0, 0.0])

        before = store.retrieve(query, top_k=1)
        self.assertEqual(before[0].procedure_id, "schema_patch_old")

        summary = adapt_procedural_memory_after_feedback(
            store,
            selected_procedure_id="schema_patch_old",
            task_embedding=query,
            success=False,
            expected_family_id="routing",
            learning_rate=0.8,
        )
        after = store.retrieve(query, top_k=1)

        self.assertEqual(after[0].procedure_id, "routing_patch")
        self.assertEqual(summary["failed_updates"], 1)
        self.assertGreater(summary["retrieval_margin_after"], summary["retrieval_margin_before"])


if __name__ == "__main__":
    unittest.main()
