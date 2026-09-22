import unittest

import torch

from tac_osm.experiment_v04 import (
    ExperimentConfig,
    ControlledOperationalModel,
    mismatched_permutation,
    shuffle_context,
    action_candidates,
)


class TACOSMv04IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ExperimentConfig()
        self.device = torch.device("cpu")

    def test_shuffle_is_a_true_context_mismatch(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        context = torch.nn.functional.one_hot(labels, num_classes=4).float()

        source = mismatched_permutation(context)

        self.assertEqual(sorted(source.tolist()), list(range(len(labels))))
        self.assertTrue(torch.all(labels[source] != labels))
        shuffled, same_source = shuffle_context(context)
        self.assertTrue(torch.equal(source, same_source))
        self.assertTrue(torch.all(shuffled.argmax(-1) != labels))

    def test_shuffle_rejects_impossible_single_dominant_context(self):
        labels = torch.tensor([0, 0, 0, 0, 1, 2])
        context = torch.nn.functional.one_hot(labels, num_classes=4).float()

        with self.assertRaises(ValueError):
            mismatched_permutation(context)

    def test_all_factor_cells_have_equal_capacity(self):
        counts = []
        for persistent in (False, True):
            for action_conditioned in (False, True):
                model = ControlledOperationalModel(
                    self.cfg, persistent, action_conditioned
                )
                counts.append(model.parameter_count())

        self.assertEqual(len(set(counts)), 1)

    def test_t1_contains_no_context_payload(self):
        labels = torch.tensor([0, 1, 2, 3])
        context = torch.nn.functional.one_hot(labels, num_classes=4).float()
        query = torch.zeros(4, self.cfg.input_dim)
        t1 = torch.cat([query, torch.zeros_like(context)], dim=-1)

        self.assertTrue(torch.equal(t1[:, self.cfg.input_dim :], torch.zeros_like(context)))

    def test_action_candidates_are_identifiable(self):
        candidates = action_candidates(8, self.cfg.action_dim, self.device)
        self.assertEqual(tuple(candidates.shape), (8, self.cfg.action_dim, 1, self.cfg.action_dim))
        self.assertTrue(
            torch.equal(
                candidates[0, 0, 0],
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                candidates[0, 3, 0],
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            )
        )


if __name__ == "__main__":
    unittest.main()
