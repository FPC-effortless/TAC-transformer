import unittest

import torch

from kaggle.train_best_tac_agentic import set_learning_rate
from tac_transformer import (
    TACConfig,
    TACTransformerLM,
    VanillaTransformerLM,
    best_tac_config,
)
from tac_transformer.optimization import (
    TACOptimizerConfig,
    build_tac_optimizer,
    tac_optimizer_param_groups,
)


class TACOptimizerTest(unittest.TestCase):
    def test_tac_optimizer_groups_cover_each_trainable_parameter_once(self):
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=512,
                d_model=64,
                n_heads=4,
                n_layers=1,
                n_programs=8,
                max_seq_len=32,
            )
        )

        groups = tac_optimizer_param_groups(
            model,
            TACOptimizerConfig(
                learning_rate=2e-4,
                weight_decay=0.1,
                identity_lr_mult=0.75,
                router_lr_mult=1.25,
                memory_lr_mult=0.5,
            ),
        )

        grouped_param_ids = [
            id(param)
            for group in groups
            for param in group["params"]
        ]
        trainable_param_ids = [
            id(param)
            for param in model.parameters()
            if param.requires_grad
        ]
        group_names = {group["tac_group"] for group in groups}

        self.assertCountEqual(grouped_param_ids, trainable_param_ids)
        self.assertEqual(len(grouped_param_ids), len(set(grouped_param_ids)))
        self.assertIn("identity_decay", group_names)
        self.assertIn("router_decay", group_names)
        self.assertIn("memory_decay", group_names)
        self.assertIn("head_decay", group_names)
        self.assertIn("core_decay", group_names)

    def test_tac_optimizer_applies_lr_multipliers_and_no_decay(self):
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=512,
                d_model=64,
                n_heads=4,
                n_layers=1,
                n_programs=8,
                max_seq_len=32,
            )
        )

        optimizer = build_tac_optimizer(
            model,
            TACOptimizerConfig(
                learning_rate=2e-4,
                weight_decay=0.1,
                core_lr_mult=1.0,
                identity_lr_mult=0.75,
                router_lr_mult=1.25,
                memory_lr_mult=0.5,
                head_lr_mult=1.5,
            ),
        )
        groups = {group["tac_group"]: group for group in optimizer.param_groups}

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertAlmostEqual(groups["core_decay"]["lr"], 2e-4)
        self.assertAlmostEqual(groups["identity_decay"]["lr"], 1.5e-4)
        self.assertAlmostEqual(groups["router_decay"]["lr"], 2.5e-4)
        self.assertAlmostEqual(groups["memory_decay"]["lr"], 1e-4)
        self.assertAlmostEqual(groups["head_decay"]["lr"], 3e-4)
        self.assertEqual(groups["identity_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(groups["router_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(groups["core_decay"]["weight_decay"], 0.1)

    def test_tac_optimizer_falls_back_for_vanilla_model(self):
        model = VanillaTransformerLM(
            TACConfig(
                vocab_size=512,
                d_model=64,
                n_heads=4,
                n_layers=1,
                max_seq_len=32,
            )
        )

        groups = tac_optimizer_param_groups(
            model,
            TACOptimizerConfig(learning_rate=3e-4, weight_decay=0.01),
        )
        grouped_param_ids = [
            id(param)
            for group in groups
            for param in group["params"]
        ]
        trainable_param_ids = [
            id(param)
            for param in model.parameters()
            if param.requires_grad
        ]
        group_names = {group["tac_group"] for group in groups}

        self.assertCountEqual(grouped_param_ids, trainable_param_ids)
        self.assertFalse(any(name.startswith("identity") for name in group_names))
        self.assertFalse(any(name.startswith("router") for name in group_names))
        self.assertIn("core_decay", group_names)
        self.assertIn("head_decay", group_names)

    def test_tac_optimizer_can_be_used_for_a_training_step(self):
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=128,
                d_model=32,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=16,
            )
        )
        optimizer = build_tac_optimizer(
            model,
            TACOptimizerConfig(learning_rate=1e-3, weight_decay=0.01),
        )
        input_ids = torch.randint(0, 128, (2, 12))
        labels = torch.randint(0, 128, (2, 12))

        output = model(input_ids, labels=labels)
        self.assertIsNotNone(output.loss)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        optimizer.step()

        state = optimizer.state_dict()
        self.assertGreater(len(state["state"]), 0)

    def test_kaggle_scheduler_preserves_tac_lr_multipliers(self):
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=128,
                d_model=32,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=16,
            )
        )
        optimizer = build_tac_optimizer(
            model,
            TACOptimizerConfig(
                learning_rate=2e-4,
                weight_decay=0.01,
                identity_lr_mult=0.5,
                router_lr_mult=1.5,
            ),
        )
        args = type(
            "Args",
            (),
            {
                "learning_rate": 2e-4,
                "warmup_steps": 100,
                "steps": 1000,
                "min_lr_ratio": 0.1,
            },
        )()

        set_learning_rate(optimizer, 50, args)
        groups = {group["tac_group"]: group for group in optimizer.param_groups}

        self.assertAlmostEqual(groups["core_decay"]["lr"], 1e-4)
        self.assertAlmostEqual(groups["identity_decay"]["lr"], 5e-5)
        self.assertAlmostEqual(groups["router_decay"]["lr"], 1.5e-4)


if __name__ == "__main__":
    unittest.main()
