import unittest

import torch

from tac_transformer import best_tac_config
from tac_transformer.agentic import (
    AgenticControlBatcher,
    AgenticController,
    RecurrentAgenticBaseline,
    benchmark_agentic_control,
)
from tac_transformer.model import TACTransformerLM


class AgenticArchitectureTest(unittest.TestCase):
    def test_agentic_batcher_hides_value_and_balances_action_target(self):
        batcher = AgenticControlBatcher(
            vocab_size=48,
            seq_len=8,
            num_actions=4,
            seed=13,
        )

        batch = batcher.next_batch(batch_size=16)

        self.assertEqual(batch.context_inputs.shape, (16, 8))
        self.assertEqual(batch.query_inputs.shape, (16, 8))
        self.assertTrue(torch.equal(batch.context_inputs[:, 1], batch.query_inputs[:, 1]))
        for row, value in enumerate(batch.context_inputs[:, 2].tolist()):
            self.assertNotIn(value, batch.query_inputs[row].tolist())
        self.assertGreaterEqual(int(batch.action_targets.min()), 0)
        self.assertLess(int(batch.action_targets.max()), 4)

    def test_agentic_controller_returns_action_world_reward_and_reflection_losses(self):
        config = best_tac_config(
            vocab_size=48,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
        )
        model = AgenticController(
            TACTransformerLM(config),
            num_actions=4,
            use_world_model=True,
            use_reward_model=True,
            use_reflection=True,
            use_memory_action_readout=True,
            use_recurrent_state=True,
            use_modular_cognition=True,
            use_memory_stores=True,
            use_planner=True,
            use_orchestration=True,
        )
        batch = AgenticControlBatcher(
            vocab_size=48,
            seq_len=8,
            num_actions=4,
            seed=21,
        ).next_batch(batch_size=3)

        output = model(
            batch,
            world_loss_weight=0.25,
            reward_loss_weight=0.1,
            reflection_loss_weight=0.1,
            memory_action_loss_weight=1.0,
            planner_loss_weight=1.0,
            orchestration_loss_weight=1.0,
        )
        output.loss.backward()

        self.assertEqual(output.action_logits.shape, (3, 4))
        self.assertEqual(output.next_observation_logits.shape, (3, 48))
        self.assertEqual(output.reward_logits.shape, (3, 2))
        self.assertEqual(output.reflection_logits.shape, (3, 2))
        self.assertIn("action", output.losses)
        self.assertIn("world", output.losses)
        self.assertIn("memory_action", output.losses)
        self.assertIn("planning", output.losses)
        self.assertIn("orchestration", output.losses)
        self.assertIn("reward", output.losses)
        self.assertIn("reflection", output.losses)
        self.assertIsNotNone(model.action_head.weight.grad)

    def test_agentic_benchmark_reports_carry_reset_shuffle_and_baseline(self):
        config = best_tac_config(
            vocab_size=48,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
        )

        result = benchmark_agentic_control(
            config,
            num_actions=4,
            steps=1,
            batch_size=2,
            eval_batches=1,
            eval_batch_size=2,
            learning_rate=1e-3,
            seed=5,
            device="cpu",
        )

        self.assertIn("decision", result)
        self.assertIn("carry", result["tac"]["eval"])
        self.assertIn("reset", result["tac"]["eval"])
        self.assertIn("shuffled", result["tac"]["eval"])
        self.assertIn("carry", result["baseline"]["eval"])
        self.assertIn("action_accuracy", result["tac"]["eval"]["carry"])

    def test_agentic_benchmark_reports_contrastive_loss_and_recurrent_baseline(self):
        config = best_tac_config(
            vocab_size=48,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
        )

        result = benchmark_agentic_control(
            config,
            num_actions=4,
            steps=1,
            batch_size=2,
            eval_batches=1,
            eval_batch_size=2,
            learning_rate=1e-3,
            seed=9,
            device="cpu",
            use_memory_action_readout=True,
            memory_action_loss_weight=1.0,
            memory_action_contrastive_weight=1.0,
            include_recurrent_baseline=True,
        )

        self.assertIn("memory_action_contrastive_loss", result["tac"]["train"])
        self.assertIn("recurrent_baseline", result)
        self.assertIn("carry", result["recurrent_baseline"]["eval"])

    def test_recurrent_agentic_baseline_carries_context_state(self):
        batcher = AgenticControlBatcher(
            vocab_size=48,
            seq_len=8,
            num_actions=4,
            seed=31,
        )
        batch = batcher.next_batch(batch_size=3)
        model = RecurrentAgenticBaseline(
            vocab_size=48,
            d_model=16,
            num_actions=4,
        )

        carry_logits = model(batch, mode="carry")
        reset_logits = model(batch, mode="reset")

        self.assertEqual(carry_logits.shape, (3, 4))
        self.assertEqual(reset_logits.shape, (3, 4))
        difference = (carry_logits - reset_logits).abs().sum().detach()
        self.assertGreater(float(difference), 0.0)
