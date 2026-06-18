from __future__ import annotations

import unittest

from tac_transformer import (
    TAC_SCM_V02_ARCHITECTURE,
    best_chunked_recall_tac_config,
    best_tac_config,
    tac_scm_v02_config,
)


class TestTACSCMPresets(unittest.TestCase):
    def test_best_tac_config_remains_legacy_structure_lane(self):
        cfg = best_tac_config(vocab_size=64, d_model=32, n_heads=4, n_layers=1)
        self.assertEqual(cfg.structure_routing_type, "legacy")
        self.assertEqual(cfg.n_structure_families, 0)

    def test_chunked_recall_alias_remains_legacy_baseline(self):
        cfg = best_chunked_recall_tac_config(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_layers=1,
        )
        self.assertEqual(cfg.structure_routing_type, "legacy")
        self.assertEqual(cfg.n_structure_families, 0)

    def test_tac_scm_v02_config_enables_two_level_structure_lane(self):
        cfg = tac_scm_v02_config(vocab_size=64, d_model=32, n_heads=4, n_layers=1)
        self.assertEqual(cfg.structure_routing_type, "two_level")
        self.assertGreater(cfg.n_structure_families, 0)
        self.assertGreater(cfg.n_structure_slots, 0)
        self.assertGreater(cfg.family_route_loss_weight, 0.0)
        self.assertGreater(cfg.specialist_route_loss_weight, 0.0)
        self.assertEqual(cfg.memory_adapter_type, "gated_residual")

    def test_tac_scm_v02_overrides_are_allowed(self):
        cfg = tac_scm_v02_config(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_layers=1,
            n_structure_families=3,
        )
        self.assertEqual(cfg.n_structure_families, 3)

    def test_architecture_constant_is_explicitly_nonempty(self):
        self.assertEqual(TAC_SCM_V02_ARCHITECTURE["structure_routing_type"], "two_level")
        self.assertIn("n_structure_families", TAC_SCM_V02_ARCHITECTURE)


if __name__ == "__main__":
    unittest.main()
