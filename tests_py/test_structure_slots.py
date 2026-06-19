from __future__ import annotations

import unittest

import torch

from tac_transformer.concept_volumes import ConceptVolumeEncoder
from tac_transformer.structure_routing import TwoLevelStructureRouter
from tac_transformer.structure_slots import (
    SlotConditionedProgramBottleneck,
    SlotExecutionOutput,
    StructureSlotState,
)


_D = 16
_N_SLOTS = 5
_N_PROG = 7
_N_FAM = 3
_BATCH = 2
_SEQ = 4


def _hidden() -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randn(_BATCH, _SEQ, _D)


class TestSlotConditionedProgramBottleneck(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = SlotConditionedProgramBottleneck(
            d_model=_D,
            n_structure_slots=_N_SLOTS,
            n_programs=_N_PROG,
        )

    def test_construction_registers_slot_and_specialist_banks(self):
        self.assertEqual(self.module.slot_bank.shape, (_N_SLOTS, _D))
        self.assertEqual(self.module.specialist_bank.shape, (_N_PROG, _D))
        names = {name for name, _ in self.module.named_parameters()}
        self.assertIn("slot_bank", names)
        self.assertIn("specialist_bank", names)

    def test_forward_returns_shape_preserving_output(self):
        out = self.module(_hidden())
        self.assertIsInstance(out, SlotExecutionOutput)
        self.assertIsInstance(out.slot_state, StructureSlotState)
        self.assertEqual(out.hidden.shape, (_BATCH, _SEQ, _D))
        self.assertEqual(out.slot_state.slot_embeddings.shape, (_BATCH, _SEQ, _D))
        self.assertEqual(out.slot_state.slot_weights.shape, (_BATCH, _SEQ, _N_SLOTS))
        self.assertEqual(out.slot_state.slot_id.shape, (_BATCH, _SEQ))
        self.assertEqual(out.specialist_probs.shape, (_BATCH, _SEQ, _N_PROG))
        self.assertEqual(out.specialist_context.shape, (_BATCH, _SEQ, _D))
        self.assertEqual(out.gate.shape, (_BATCH, _SEQ, _D))

    def test_probabilities_sum_to_one(self):
        out = self.module(_hidden())
        self.assertTrue(
            torch.allclose(
                out.slot_state.slot_weights.sum(dim=-1),
                torch.ones(_BATCH, _SEQ),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                out.specialist_probs.sum(dim=-1),
                torch.ones(_BATCH, _SEQ),
                atol=1e-5,
            )
        )

    def test_accepts_two_level_route(self):
        hidden = _hidden()
        encoder = ConceptVolumeEncoder(_D, _N_FAM)
        router = TwoLevelStructureRouter(_D, _N_PROG, _N_FAM)
        route = router(hidden, encoder(hidden))
        out = self.module(hidden, route=route)
        self.assertTrue(torch.allclose(out.specialist_probs, route.specialist_probs))
        self.assertEqual(out.slot_state.specialist_id.shape, route.specialist_id.shape)

    def test_accepts_external_specialist_probs(self):
        hidden = _hidden()
        probs = torch.softmax(torch.randn(_BATCH, _SEQ, _N_PROG), dim=-1)
        out = self.module(hidden, specialist_probs=probs)
        self.assertTrue(torch.allclose(out.specialist_probs, probs))

    def test_load_balance_loss_is_scalar(self):
        module = SlotConditionedProgramBottleneck(
            d_model=_D,
            n_structure_slots=_N_SLOTS,
            n_programs=_N_PROG,
            load_balance_weight=0.1,
        )
        out = module(_hidden())
        self.assertEqual(out.auxiliary_loss.shape, ())
        self.assertGreaterEqual(float(out.auxiliary_loss), 0.0)

    def test_backward_through_slot_bottleneck(self):
        out = self.module(_hidden())
        loss = out.hidden.sum() + out.auxiliary_loss
        loss.backward()
        self.assertIsNotNone(self.module.slot_bank.grad)
        self.assertIsNotNone(self.module.specialist_bank.grad)
        self.assertGreater(float(self.module.slot_bank.grad.abs().sum()), 0.0)

    def test_invalid_specialist_prob_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module(_hidden(), specialist_probs=torch.randn(_BATCH, _SEQ, _N_PROG + 1))

    def test_invalid_constructor_values_are_rejected(self):
        with self.assertRaises(ValueError):
            SlotConditionedProgramBottleneck(_D, 0, _N_PROG)
        with self.assertRaises(ValueError):
            SlotConditionedProgramBottleneck(_D, _N_SLOTS, 0)
        with self.assertRaises(ValueError):
            SlotConditionedProgramBottleneck(_D, _N_SLOTS, _N_PROG, -0.1)


if __name__ == "__main__":
    unittest.main()
