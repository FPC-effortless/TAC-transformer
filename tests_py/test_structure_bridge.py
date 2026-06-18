from __future__ import annotations

import unittest

import torch

from kaggle.benchmark_real003_structure_to_behavior import (
    run_real003_structure_to_behavior_smoke,
)
from tac_transformer.structure_bridge import (
    GatedResidualStructureBridge,
    LinearStructureBridge,
    MLPStructureBridge,
    OracleStructureBridge,
    StructureBridgeOutput,
    build_structure_bridge,
)
from tac_transformer.structure_memory import StructureMemoryModule


_D = 16
_BATCH = 2
_SEQ = 3


def _hidden() -> torch.Tensor:
    torch.manual_seed(21)
    return torch.randn(_BATCH, _SEQ, _D)


def _structure() -> torch.Tensor:
    torch.manual_seed(22)
    return torch.randn(_BATCH, _SEQ, _D)


class TestStructureBridgeVariants(unittest.TestCase):
    def test_linear_bridge_preserves_hidden_shape(self):
        bridge = LinearStructureBridge(_D)
        out = bridge(_hidden(), _structure())
        self.assertIsInstance(out, StructureBridgeOutput)
        self.assertEqual(out.hidden.shape, (_BATCH, _SEQ, _D))
        self.assertEqual(out.bridge_delta.shape, (_BATCH, _SEQ, _D))
        self.assertIsNone(out.gate)

    def test_mlp_bridge_preserves_hidden_shape(self):
        bridge = MLPStructureBridge(_D)
        out = bridge(_hidden(), _structure())
        self.assertEqual(out.hidden.shape, (_BATCH, _SEQ, _D))
        self.assertEqual(out.bridge_delta.shape, (_BATCH, _SEQ, _D))

    def test_gated_residual_bridge_returns_gate_in_range(self):
        bridge = GatedResidualStructureBridge(_D)
        out = bridge(_hidden(), _structure())
        self.assertEqual(out.hidden.shape, (_BATCH, _SEQ, _D))
        self.assertIsNotNone(out.gate)
        self.assertTrue((out.gate >= 0.0).all())
        self.assertTrue((out.gate <= 1.0).all())

    def test_oracle_bridge_maps_labels_to_hidden_adapter(self):
        bridge = OracleStructureBridge(_D, n_oracle_structures=4)
        oracle_ids = torch.tensor([[0, 1, 2], [1, 2, 3]])
        out = bridge(_hidden(), oracle_ids)
        self.assertEqual(out.hidden.shape, (_BATCH, _SEQ, _D))
        self.assertEqual(out.structure_vector.shape, (_BATCH, _SEQ, _D))
        self.assertIsNone(out.gate)

    def test_broadcasts_batched_structure_memory_read(self):
        bridge = GatedResidualStructureBridge(_D)
        hidden = _hidden()
        structure_vector = torch.randn(_BATCH, _D)
        out = bridge(hidden, structure_vector)
        self.assertEqual(out.hidden.shape, hidden.shape)
        self.assertEqual(out.structure_vector.shape, hidden.shape)

    def test_structure_memory_to_bridge_composition(self):
        hidden = _hidden()
        memory = StructureMemoryModule(d_model=_D, n_structure_slots=4)
        read_vector = memory(hidden.reshape(-1, _D)).reshape(_BATCH, _SEQ, _D)
        bridge = LinearStructureBridge(_D)
        out = bridge(hidden, read_vector)
        self.assertEqual(out.hidden.shape, hidden.shape)

    def test_backward_through_bridge(self):
        bridge = GatedResidualStructureBridge(_D)
        out = bridge(_hidden(), _structure())
        loss = out.hidden.sum()
        loss.backward()
        grads = [p.grad for p in bridge.parameters()]
        self.assertTrue(any(g is not None and float(g.abs().sum()) > 0.0 for g in grads))

    def test_builder_returns_requested_variant(self):
        self.assertIsInstance(build_structure_bridge("linear", _D), LinearStructureBridge)
        self.assertIsInstance(build_structure_bridge("mlp", _D), MLPStructureBridge)
        self.assertIsInstance(
            build_structure_bridge("gated_residual", _D),
            GatedResidualStructureBridge,
        )
        self.assertIsInstance(
            build_structure_bridge("oracle", _D, n_oracle_structures=3),
            OracleStructureBridge,
        )

    def test_invalid_shapes_are_rejected(self):
        bridge = LinearStructureBridge(_D)
        with self.assertRaises(ValueError):
            bridge(_hidden(), torch.randn(_BATCH, _SEQ, _D + 1))
        with self.assertRaises(ValueError):
            build_structure_bridge("bad", _D)
        with self.assertRaises(ValueError):
            OracleStructureBridge(_D, n_oracle_structures=0)


class TestREAL003BenchmarkSmoke(unittest.TestCase):
    def test_real003_smoke_reports_all_bridge_variants(self):
        result = run_real003_structure_to_behavior_smoke()
        self.assertEqual(result["status"], "passed")
        self.assertIn("frozen_structure_encoder_linear_bridge", result["variants"])
        self.assertIn("frozen_structure_encoder_mlp_bridge", result["variants"])
        self.assertIn("end_to_end_gated_residual_bridge", result["variants"])
        self.assertIn("oracle_structure_label_bridge", result["variants"])
        for variant in result["variants"].values():
            self.assertTrue(variant["shape_ok"])
            self.assertFalse(variant["direct_logit_injection"])


if __name__ == "__main__":
    unittest.main()
