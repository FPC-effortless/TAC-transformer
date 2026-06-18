import unittest

import torch

from tac_transformer import (
    LifecyclePhase,
    StructureFamily,
    StructureLifecycleStats,
    StructureMemoryModule,
    StructureMemoryRead,
    StructureMemoryState,
    StructureMemoryWrite,
    StructureObject,
)


class StructureTypesTest(unittest.TestCase):
    def test_lifecycle_phase_members(self):
        expected = {"appear", "survive", "strengthen", "specialize", "merge", "decay", "retire"}
        actual = {phase.value for phase in LifecyclePhase}
        self.assertEqual(actual, expected)

    def test_structure_object_defaults(self):
        obj = StructureObject(structure_id=0)
        self.assertEqual(obj.structure_id, 0)
        self.assertIsNone(obj.family_id)
        self.assertIsNone(obj.key_vector)
        self.assertEqual(obj.usage_count, 0)
        self.assertEqual(obj.success_score, 0.0)
        self.assertEqual(obj.survival_score, 0.0)

    def test_structure_family_defaults(self):
        fam = StructureFamily(family_id=1, name="test")
        self.assertEqual(fam.family_id, 1)
        self.assertEqual(fam.member_ids, [])
        self.assertIsNone(fam.centroid_vector)

    def test_lifecycle_stats_defaults(self):
        stats = StructureLifecycleStats()
        self.assertEqual(stats.usage_count, 0)
        self.assertEqual(stats.success_rate, 0.0)
        self.assertEqual(stats.survival_score, 0.0)


class StructureMemoryModuleConstructionTest(unittest.TestCase):
    def test_construction_default_slots(self):
        module = StructureMemoryModule(d_model=16)
        self.assertEqual(module.n_structure_slots, 64)
        self.assertEqual(module.key_bank.shape, (64, 16))
        self.assertEqual(module.value_bank.shape, (64, 16))

    def test_construction_custom_slots(self):
        module = StructureMemoryModule(d_model=8, n_structure_slots=4)
        self.assertEqual(module.n_structure_slots, 4)
        self.assertEqual(module.key_bank.shape, (4, 8))

    def test_parameters_are_registered(self):
        module = StructureMemoryModule(d_model=8, n_structure_slots=4)
        param_names = {name for name, _ in module.named_parameters()}
        self.assertIn("key_bank", param_names)
        self.assertIn("value_bank", param_names)
        self.assertIn("read_gate.weight", param_names)
        self.assertIn("novelty_gate.weight", param_names)


class StructureMemoryReadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.module = StructureMemoryModule(d_model=8, n_structure_slots=4)

    def test_single_read_returns_read_result(self):
        query = torch.randn(8)
        result = self.module.read(query)
        self.assertIsInstance(result, StructureMemoryRead)
        self.assertIsNotNone(result.structure)
        self.assertIsInstance(result.similarity_score, float)
        self.assertIsInstance(result.read_gate, float)
        self.assertGreaterEqual(result.read_gate, 0.0)
        self.assertLessEqual(result.read_gate, 1.0)

    def test_read_returns_valid_slot(self):
        query = torch.randn(8)
        result = self.module.read(query)
        self.assertGreaterEqual(result.structure.slot_id, 0)
        self.assertLess(result.structure.slot_id, self.module.n_structure_slots)

    def test_read_with_batched_query(self):
        query = torch.randn(3, 8)
        result = self.module.read(query)
        self.assertIsInstance(result, StructureMemoryRead)
        self.assertIsNotNone(result.structure)

    def test_read_skips_retired_slots(self):
        module = StructureMemoryModule(d_model=8, n_structure_slots=4)
        module.retired = [True, True, True, False]
        query = torch.randn(8)
        result = module.read(query)
        self.assertEqual(result.structure.slot_id, 3)


class StructureMemoryWriteTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module = StructureMemoryModule(d_model=8, n_structure_slots=4)

    def test_write_updates_key_and_value_bank(self):
        before_key = self.module.key_bank[0].clone().detach()
        before_val = self.module.value_bank[0].clone().detach()
        update_vec = torch.ones(8) * 10.0
        candidate = StructureMemoryWrite(structure_id=0, update_vector=update_vec, write_gate=0.5)
        self.module.write(candidate)
        self.assertFalse(torch.allclose(self.module.key_bank[0].detach(), before_key))
        self.assertFalse(torch.allclose(self.module.value_bank[0].detach(), before_val))

    def test_write_gate_zero_leaves_bank_unchanged(self):
        before_key = self.module.key_bank[1].clone().detach()
        update_vec = torch.ones(8) * 99.0
        candidate = StructureMemoryWrite(structure_id=1, update_vector=update_vec, write_gate=0.0)
        self.module.write(candidate)
        self.assertTrue(torch.allclose(self.module.key_bank[1].detach(), before_key))

    def test_novelty_gate_returns_scalar_in_range(self):
        q = torch.randn(8)
        c = torch.randn(8)
        gate = self.module.novelty_write_gate(q, c)
        self.assertIsInstance(gate, float)
        self.assertGreaterEqual(gate, 0.0)
        self.assertLessEqual(gate, 1.0)

    def test_novelty_gate_accepts_batched_vectors(self):
        q = torch.randn(3, 8)
        c = torch.randn(3, 8)
        gate = self.module.novelty_write_gate(q, c)
        self.assertIsInstance(gate, float)


class StructureMemoryLifecycleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = StructureMemoryModule(d_model=8, n_structure_slots=4)

    def test_lifecycle_update_increments_usage_count(self):
        self.module.update_lifecycle(0, success=True)
        self.module.update_lifecycle(0, success=True)
        self.assertEqual(self.module._lifecycle_stats[0].usage_count, 2)

    def test_lifecycle_update_tracks_success_rate(self):
        self.module.update_lifecycle(0, success=True)
        self.module.update_lifecycle(0, success=False)
        rate = self.module._lifecycle_stats[0].success_rate
        self.assertAlmostEqual(rate, 0.5, places=5)

    def test_lifecycle_update_tracks_transfer_gain(self):
        self.module.update_lifecycle(0, success=True, transfer_gain=1.0)
        self.module.update_lifecycle(0, success=True, transfer_gain=0.0)
        gain = self.module._lifecycle_stats[0].transfer_gain
        self.assertAlmostEqual(gain, 0.5, places=5)

    def test_lifecycle_update_recalculates_survival_score(self):
        self.module.update_lifecycle(0, success=True, transfer_gain=1.0)
        score = self.module._lifecycle_stats[0].survival_score
        self.assertGreater(score, 0.0)

    def test_score_survival_returns_dict_for_all_slots(self):
        scores = self.module.score_survival()
        self.assertEqual(len(scores), self.module.n_structure_slots)
        for slot_id, score in scores.items():
            self.assertIsInstance(score, float)

    def test_decay_retired_marks_low_survival_structures(self):
        scores = self.module.score_survival()
        low_slots = [sid for sid, s in scores.items() if s < 0.1]
        retired = self.module.decay_retired(threshold=0.1)
        self.assertTrue(all(self.module.retired[sid] for sid in retired))
        for sid in retired:
            self.assertIn(sid, low_slots)

    def test_decay_retired_does_not_mark_high_survival_structures(self):
        for _ in range(200):
            self.module.update_lifecycle(0, success=True, transfer_gain=1.0)
        self.module.decay_retired(threshold=0.1)
        self.assertFalse(self.module.retired[0])

    def test_already_retired_slots_not_double_counted(self):
        self.module.retired[0] = True
        retired = self.module.decay_retired(threshold=0.1)
        self.assertNotIn(0, retired)

    def test_get_state_returns_structure_memory_state(self):
        state = self.module.get_state()
        self.assertIsInstance(state, StructureMemoryState)
        self.assertEqual(len(state.structures), self.module.n_structure_slots)
        self.assertEqual(len(state.lifecycle_stats), self.module.n_structure_slots)

    def test_get_state_reflects_lifecycle_updates(self):
        self.module.update_lifecycle(2, success=True, transfer_gain=0.8)
        state = self.module.get_state()
        self.assertEqual(state.lifecycle_stats[2].usage_count, 1)
        self.assertAlmostEqual(state.lifecycle_stats[2].success_rate, 1.0)


class StructureMemoryForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(99)
        self.module = StructureMemoryModule(d_model=16, n_structure_slots=8)

    def test_forward_returns_correct_shape_1d(self):
        query = torch.randn(16)
        out = self.module(query)
        self.assertEqual(out.shape, (1, 16))

    def test_forward_returns_correct_shape_batched(self):
        query = torch.randn(4, 16)
        out = self.module(query)
        self.assertEqual(out.shape, (4, 16))

    def test_forward_is_differentiable(self):
        query = torch.randn(4, 16, requires_grad=False)
        out = self.module(query)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(self.module.key_bank.grad)
        self.assertIsNotNone(self.module.value_bank.grad)
        self.assertGreater(float(self.module.key_bank.grad.abs().sum()), 0)

    def test_full_mini_forward_pass_round_trip(self):
        torch.manual_seed(11)
        module = StructureMemoryModule(d_model=8, n_structure_slots=4)
        query = torch.randn(8)

        read_result = module.read(query)
        self.assertIsInstance(read_result, StructureMemoryRead)

        update_vec = torch.randn(8)
        gate = module.novelty_write_gate(query, update_vec)
        candidate = StructureMemoryWrite(
            structure_id=read_result.structure.structure_id,
            update_vector=update_vec,
            write_gate=gate,
        )
        module.write(candidate)

        module.update_lifecycle(read_result.structure.structure_id, success=True, transfer_gain=0.5)

        scores = module.score_survival()
        self.assertIn(read_result.structure.structure_id, scores)

        out = module(query)
        self.assertEqual(out.shape, (1, 8))

        state = module.get_state()
        self.assertEqual(len(state.structures), 4)
        self.assertEqual(
            state.lifecycle_stats[read_result.structure.structure_id].usage_count, 1
        )


if __name__ == "__main__":
    unittest.main()
