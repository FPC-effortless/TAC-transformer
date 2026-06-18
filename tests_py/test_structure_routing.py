from __future__ import annotations

import unittest

import torch

from tac_transformer import TACConfig
from tac_transformer.concept_volumes import ConceptVolumeEncoder, ConceptVolumeOutput
from tac_transformer.structure_routing import (
    SpecialistRouter,
    StructureFamilyRouter,
    TwoLevelStructureRoute,
    TwoLevelStructureRouter,
)


_D = 16
_N_FAM = 4
_N_PROG = 8
_BATCH = 2
_SEQ = 3


def _hidden(batch: int = _BATCH, seq: int = _SEQ, d: int = _D) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(batch, seq, d)


def _concept_output(
    batch: int = _BATCH,
    seq: int = _SEQ,
    d: int = _D,
    n_fam: int = _N_FAM,
) -> ConceptVolumeOutput:
    encoder = ConceptVolumeEncoder(d, n_fam)
    with torch.no_grad():
        return encoder(_hidden(batch, seq, d))


class TestConceptVolumeEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.encoder = ConceptVolumeEncoder(_D, _N_FAM)

    def test_output_shapes(self):
        hidden = _hidden()
        out = self.encoder(hidden)
        self.assertEqual(out.concept_embedding.shape, (_BATCH, _SEQ, _D))
        self.assertEqual(out.family_logits.shape, (_BATCH, _SEQ, _N_FAM))
        self.assertEqual(out.family_probs.shape, (_BATCH, _SEQ, _N_FAM))
        self.assertEqual(out.family_id.shape, (_BATCH, _SEQ))

    def test_family_probs_sum_to_one(self):
        out = self.encoder(_hidden())
        totals = out.family_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(totals, torch.ones_like(totals), atol=1e-5))

    def test_family_id_is_valid_index(self):
        out = self.encoder(_hidden())
        self.assertTrue((out.family_id >= 0).all())
        self.assertTrue((out.family_id < _N_FAM).all())

    def test_family_centroid_bank_shape(self):
        self.assertEqual(self.encoder.family_centroid_bank.shape, (_N_FAM, _D))

    def test_no_rms_norm_variant(self):
        encoder = ConceptVolumeEncoder(_D, _N_FAM, use_rms_norm=False)
        out = encoder(_hidden())
        self.assertIsNone(encoder.norm)
        self.assertEqual(out.concept_embedding.shape, (_BATCH, _SEQ, _D))

    def test_backward_through_encoder(self):
        hidden = _hidden()
        out = self.encoder(hidden)
        loss = out.family_logits.sum()
        loss.backward()
        self.assertIsNotNone(self.encoder.projection.weight.grad)
        self.assertGreater(float(self.encoder.projection.weight.grad.abs().sum()), 0)


class TestStructureFamilyRouter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.router = StructureFamilyRouter(_D, _N_FAM)

    def test_output_shapes(self):
        h = _hidden()
        c = _concept_output()
        family_id, family_probs, load_loss = self.router(h, c.concept_embedding)
        self.assertEqual(family_id.shape, (_BATCH, _SEQ))
        self.assertEqual(family_probs.shape, (_BATCH, _SEQ, _N_FAM))
        self.assertEqual(load_loss.shape, ())

    def test_family_probs_sum_to_one(self):
        h = _hidden()
        c = _concept_output()
        _, probs, _ = self.router(h, c.concept_embedding)
        totals = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(totals, torch.ones_like(totals), atol=1e-5))

    def test_load_balance_loss_zero_when_weight_is_zero(self):
        h = _hidden()
        c = _concept_output()
        _, _, loss = self.router(h, c.concept_embedding, load_balance_weight=0.0)
        self.assertEqual(float(loss), 0.0)

    def test_load_balance_loss_positive_when_weight_nonzero(self):
        torch.manual_seed(99)
        router = StructureFamilyRouter(_D, _N_FAM)
        h = _hidden()
        c = _concept_output()
        _, _, loss = router(h, c.concept_embedding, load_balance_weight=1.0)
        self.assertGreaterEqual(float(loss), 0.0)


class TestSpecialistRouter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.router = SpecialistRouter(_D, _N_PROG, _N_FAM)

    def _family_id(self) -> torch.Tensor:
        return torch.randint(0, _N_FAM, (_BATCH, _SEQ))

    def _family_emb(self) -> torch.Tensor:
        return torch.randn(_BATCH, _SEQ, _D)

    def test_output_shapes(self):
        specialist_id, probs, loss = self.router(
            _hidden(), self._family_emb(), self._family_id()
        )
        self.assertEqual(specialist_id.shape, (_BATCH, _SEQ))
        self.assertEqual(probs.shape, (_BATCH, _SEQ, _N_PROG))
        self.assertEqual(loss.shape, ())

    def test_specialist_probs_sum_to_one(self):
        _, probs, _ = self.router(_hidden(), self._family_emb(), self._family_id())
        totals = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(totals, torch.ones_like(totals), atol=1e-5))

    def test_load_balance_loss_zero_when_weight_is_zero(self):
        _, _, loss = self.router(
            _hidden(), self._family_emb(), self._family_id(), load_balance_weight=0.0
        )
        self.assertEqual(float(loss), 0.0)

    def test_family_conditioning_changes_specialist_logits(self):
        h = _hidden()
        fam_emb_a = torch.zeros(_BATCH, _SEQ, _D)
        fam_emb_b = torch.ones(_BATCH, _SEQ, _D)
        fid = self._family_id()
        _, probs_a, _ = self.router(h, fam_emb_a, fid)
        _, probs_b, _ = self.router(h, fam_emb_b, fid)
        self.assertFalse(torch.allclose(probs_a, probs_b))


class TestTwoLevelStructureRouter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.router = TwoLevelStructureRouter(
            d_model=_D,
            n_programs=_N_PROG,
            n_structure_families=_N_FAM,
            family_route_loss_weight=0.0,
            specialist_route_loss_weight=0.0,
        )
        self.encoder = ConceptVolumeEncoder(_D, _N_FAM)

    def _concept(self) -> ConceptVolumeOutput:
        return self.encoder(_hidden())

    def test_output_shapes(self):
        route = self.router(_hidden(), self._concept())
        self.assertIsInstance(route, TwoLevelStructureRoute)
        self.assertEqual(route.family_id.shape, (_BATCH, _SEQ))
        self.assertEqual(route.family_probs.shape, (_BATCH, _SEQ, _N_FAM))
        self.assertEqual(route.specialist_id.shape, (_BATCH, _SEQ))
        self.assertEqual(route.specialist_probs.shape, (_BATCH, _SEQ, _N_PROG))
        self.assertEqual(route.route_loss.shape, ())

    def test_route_loss_zero_when_weights_zero(self):
        route = self.router(_hidden(), self._concept())
        self.assertEqual(float(route.route_loss), 0.0)

    def test_route_loss_positive_when_weights_nonzero(self):
        torch.manual_seed(77)
        router = TwoLevelStructureRouter(
            d_model=_D,
            n_programs=_N_PROG,
            n_structure_families=_N_FAM,
            family_route_loss_weight=1.0,
            specialist_route_loss_weight=1.0,
        )
        encoder = ConceptVolumeEncoder(_D, _N_FAM)
        route = router(_hidden(), encoder(_hidden()))
        self.assertGreaterEqual(float(route.route_loss), 0.0)

    def test_backward_through_route(self):
        route = self.router(_hidden(), self._concept())
        loss = route.specialist_probs.sum()
        loss.backward()
        self.assertIsNotNone(self.router.specialist_router.specialist_head.weight.grad)

    def test_family_probs_sum_to_one(self):
        route = self.router(_hidden(), self._concept())
        totals = route.family_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(totals, torch.ones_like(totals), atol=1e-5))

    def test_specialist_probs_sum_to_one(self):
        route = self.router(_hidden(), self._concept())
        totals = route.specialist_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(totals, torch.ones_like(totals), atol=1e-5))


class TestLegacyPathDisabled(unittest.TestCase):
    def test_default_config_has_legacy_routing_type(self):
        cfg = TACConfig(vocab_size=32)
        self.assertEqual(cfg.structure_routing_type, "legacy")
        self.assertEqual(cfg.n_structure_families, 0)

    def test_two_level_config_accepted(self):
        cfg = TACConfig(
            vocab_size=32,
            n_structure_families=4,
            structure_routing_type="two_level",
            family_route_loss_weight=0.01,
            specialist_route_loss_weight=0.01,
        )
        self.assertEqual(cfg.structure_routing_type, "two_level")
        self.assertEqual(cfg.n_structure_families, 4)

    def test_invalid_structure_routing_type_is_rejected(self):
        from tac_transformer import TACTransformerLM
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    structure_routing_type="bad_value",
                )
            )

    def test_negative_n_structure_families_is_rejected(self):
        from tac_transformer import TACTransformerLM
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_structure_families=-1,
                )
            )


class TestEndToEndTwoLevelRoute(unittest.TestCase):
    def test_small_batch_full_forward(self):
        torch.manual_seed(42)
        d, n_fam, n_prog = 32, 3, 6
        encoder = ConceptVolumeEncoder(d, n_fam)
        router = TwoLevelStructureRouter(
            d_model=d,
            n_programs=n_prog,
            n_structure_families=n_fam,
            family_route_loss_weight=0.1,
            specialist_route_loss_weight=0.1,
        )

        hidden = torch.randn(1, 5, d)
        concept = encoder(hidden)
        route = router(hidden, concept)

        self.assertEqual(route.family_id.shape, (1, 5))
        self.assertEqual(route.specialist_id.shape, (1, 5))
        self.assertEqual(route.specialist_probs.shape, (1, 5, n_prog))
        self.assertGreaterEqual(float(route.route_loss), 0.0)

        (route.specialist_probs.sum() + route.route_loss).backward()
        self.assertIsNotNone(router.family_router.family_head.weight.grad)
        self.assertIsNotNone(router.specialist_router.specialist_head.weight.grad)
        self.assertIsNotNone(encoder.projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
