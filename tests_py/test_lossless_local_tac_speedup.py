import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from kaggle import train_best_tac_agentic
from tac_transformer import TACAuxiliaryOutput, TACConfig, TACOutput, TACTransformerLM
from tac_transformer.training import forward_language_model_window


class _RecordingChunkedModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 11):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.calls = []

    def forward(
        self,
        input_ids,
        *,
        identity_states=None,
        labels=None,
        collect_auxiliary=True,
        **_kwargs,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.detach().clone(),
                "identity_states": identity_states,
                "collect_auxiliary": collect_auxiliary,
                "collect_metrics": _kwargs.get("collect_metrics", True),
            }
        )
        logits = F.one_hot(
            input_ids.remainder(self.config.vocab_size),
            num_classes=self.config.vocab_size,
        ).to(torch.float32)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
            )
        zero = logits.new_zeros(())
        aux_value = logits.new_tensor(1.0 if collect_auxiliary else 0.0)
        aux = TACAuxiliaryOutput(
            coherence=zero.reshape(1, 1, 1),
            program_activations=zero.reshape(1, 1),
            selected_program_mask=zero.reshape(1, 1),
            used_energy=zero.reshape(1),
            attention_probs=zero.reshape(1, 1, 1, 1),
            losses={"coherence": aux_value},
            metrics={"collected_auxiliary": aux_value},
        )
        return TACOutput(
            logits=logits,
            identity_states=[f"state_{len(self.calls)}"],
            aux=aux,
            loss=loss,
        )


class LosslessLocalTacSpeedupTests(unittest.TestCase):
    def test_trainer_metric_collection_schedule_matches_observable_steps(self):
        args = SimpleNamespace(
            log_every=50,
            eval_every=100,
            checkpoint_every=25,
            steps=120,
        )
        specialization_checkpoints = {40}

        self.assertFalse(
            train_best_tac_agentic.should_collect_train_metrics(
                args,
                step=1,
                specialization_checkpoints=specialization_checkpoints,
            )
        )
        self.assertTrue(
            train_best_tac_agentic.should_collect_train_metrics(
                args,
                step=25,
                specialization_checkpoints=specialization_checkpoints,
            )
        )
        self.assertTrue(
            train_best_tac_agentic.should_collect_train_metrics(
                args,
                step=40,
                specialization_checkpoints=specialization_checkpoints,
            )
        )
        self.assertTrue(
            train_best_tac_agentic.should_collect_train_metrics(
                args,
                step=50,
                specialization_checkpoints=specialization_checkpoints,
            )
        )
        self.assertTrue(
            train_best_tac_agentic.should_collect_train_metrics(
                args,
                step=100,
                specialization_checkpoints=specialization_checkpoints,
            )
        )
        self.assertTrue(
            train_best_tac_agentic.should_collect_train_metrics(
                args,
                step=120,
                specialization_checkpoints=specialization_checkpoints,
            )
        )

    def test_chunked_forward_skips_unused_context_auxiliary_collection(self):
        model = _RecordingChunkedModel()
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        labels = torch.tensor([[1, 2, 3, 4, 5, 6]])

        output, loss, logits = forward_language_model_window(
            model,
            input_ids,
            labels,
            chunked_state_within_batch=True,
        )

        self.assertEqual(
            [call["collect_auxiliary"] for call in model.calls],
            [False, True],
        )
        self.assertIsNone(model.calls[0]["identity_states"])
        self.assertEqual(model.calls[1]["identity_states"], ["state_1"])
        self.assertEqual(output.identity_states, ["state_2"])
        self.assertAlmostEqual(float(output.aux.losses["coherence"]), 1.0)

        expected_logits = torch.cat(
            [
                F.one_hot(input_ids[:, :3], num_classes=model.config.vocab_size).float(),
                F.one_hot(input_ids[:, 3:], num_classes=model.config.vocab_size).float(),
            ],
            dim=1,
        )
        expected_context_loss = F.cross_entropy(
            expected_logits[:, :3].reshape(-1, model.config.vocab_size),
            labels[:, :3].reshape(-1),
        )
        expected_query_loss = F.cross_entropy(
            expected_logits[:, 3:].reshape(-1, model.config.vocab_size),
            labels[:, 3:].reshape(-1),
        )
        expected_loss = (expected_context_loss * 3 + expected_query_loss * 3) / 6

        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(loss, expected_loss)

    def test_chunked_forward_can_defer_query_metrics_while_keeping_auxiliary_losses(self):
        model = _RecordingChunkedModel()
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        labels = torch.tensor([[1, 2, 3, 4, 5, 6]])

        output, loss, logits = forward_language_model_window(
            model,
            input_ids,
            labels,
            chunked_state_within_batch=True,
            collect_metrics=False,
        )

        self.assertEqual(
            [call["collect_auxiliary"] for call in model.calls],
            [False, True],
        )
        self.assertEqual(
            [call["collect_metrics"] for call in model.calls],
            [False, False],
        )
        self.assertAlmostEqual(float(output.aux.losses["coherence"]), 1.0)
        self.assertEqual(logits.shape, (1, 6, model.config.vocab_size))
        self.assertGreater(float(loss), 0.0)

    def test_chunked_forward_can_skip_query_auxiliary_for_cadence_runs(self):
        model = _RecordingChunkedModel()
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        labels = torch.tensor([[1, 2, 3, 4, 5, 6]])

        output, loss, logits = forward_language_model_window(
            model,
            input_ids,
            labels,
            chunked_state_within_batch=True,
            collect_auxiliary=False,
            collect_metrics=False,
        )

        self.assertEqual(
            [call["collect_auxiliary"] for call in model.calls],
            [False, False],
        )
        self.assertEqual(
            [call["collect_metrics"] for call in model.calls],
            [False, False],
        )
        self.assertAlmostEqual(float(output.aux.losses["coherence"]), 0.0)
        self.assertEqual(logits.shape, (1, 6, model.config.vocab_size))
        self.assertGreater(float(loss), 0.0)

    def test_non_auxiliary_chunked_forward_keeps_energy_cost_gradient_path(self):
        config = TACConfig(
            vocab_size=64,
            d_model=32,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=16,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            position_type="rope",
            program_compute_type="linear_expert",
            routing_type="base_semantic",
            routing_top_k=2,
            memory_read_type="content_addressed",
            content_store_size=4,
            content_read_steps=2,
            content_read_gate_type="cue_match",
            program_memory_update_type="program_conditioned",
            memory_allocation_type="creb",
            memory_allocation_k=2,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        labels = torch.tensor([[2, 3, 4, 5, 6, 7]])

        _, loss, _ = forward_language_model_window(
            model,
            input_ids,
            labels,
            chunked_state_within_batch=True,
            collect_auxiliary=False,
            collect_metrics=False,
        )
        loss.backward()

        grad = model.blocks[-1].identity_field.raw_energy_costs.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.allclose(grad, torch.zeros_like(grad)))

    def test_auxiliary_loss_cadence_helper_defaults_to_every_step(self):
        default_args = train_best_tac_agentic.parse_args(["--scale", "smoke"])
        cadence_args = train_best_tac_agentic.parse_args(
            ["--scale", "smoke", "--aux-loss-cadence", "4"]
        )

        self.assertEqual(default_args.aux_loss_cadence, 1)
        self.assertTrue(
            train_best_tac_agentic.should_collect_auxiliary_losses(
                default_args,
                step=1,
            )
        )
        self.assertFalse(
            train_best_tac_agentic.should_collect_auxiliary_losses(
                cadence_args,
                step=3,
            )
        )
        self.assertTrue(
            train_best_tac_agentic.should_collect_auxiliary_losses(
                cadence_args,
                step=4,
            )
        )

    def test_real_tac_context_auxiliary_skip_preserves_query_outputs(self):
        torch.manual_seed(191)
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            position_type="rope",
            program_compute_type="linear_expert",
            routing_type="base_semantic",
            routing_top_k=2,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_read_type="content_addressed",
            content_read_steps=2,
            content_read_gate_type="synthesis",
            memory_adapter_type="gated_residual",
            identity_attention_type="identity_first",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]])
        labels = torch.tensor([[3, 5, 7, 9, 11, 13, 15, 17]])
        split = input_ids.shape[1] // 2

        context_full = model(
            input_ids[:, :split],
            labels=labels[:, :split],
            collect_auxiliary=True,
        )
        query_full = model(
            input_ids[:, split:],
            labels=labels[:, split:],
            identity_states=context_full.identity_states,
        )
        context_fast = model(
            input_ids[:, :split],
            labels=labels[:, :split],
            collect_auxiliary=False,
        )
        query_fast = model(
            input_ids[:, split:],
            labels=labels[:, split:],
            identity_states=context_fast.identity_states,
        )

        torch.testing.assert_close(query_fast.logits, query_full.logits)
        torch.testing.assert_close(query_fast.loss, query_full.loss)
        self.assertEqual(len(context_fast.identity_states), len(context_full.identity_states))
        for fast_state, full_state in zip(
            context_fast.identity_states,
            context_full.identity_states,
        ):
            for field_name in (
                "stability",
                "program_memory",
                "content_cues",
                "content_values",
                "content_mask",
            ):
                torch.testing.assert_close(
                    getattr(fast_state, field_name),
                    getattr(full_state, field_name),
                )

    def test_real_tac_metric_deferral_preserves_logits_loss_and_auxiliary_losses(self):
        torch.manual_seed(192)
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            position_type="rope",
            program_compute_type="linear_expert",
            routing_type="base_semantic",
            routing_top_k=2,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_read_type="content_addressed",
            content_read_steps=2,
            content_read_gate_type="synthesis",
            memory_adapter_type="gated_residual",
            identity_attention_type="identity_first",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16]])
        labels = torch.tensor([[4, 6, 8, 10, 12, 14, 16, 18]])

        full_output, full_loss, full_logits = forward_language_model_window(
            model,
            input_ids,
            labels,
            chunked_state_within_batch=True,
            collect_metrics=True,
        )
        deferred_output, deferred_loss, deferred_logits = forward_language_model_window(
            model,
            input_ids,
            labels,
            chunked_state_within_batch=True,
            collect_metrics=False,
        )

        torch.testing.assert_close(deferred_logits, full_logits)
        torch.testing.assert_close(deferred_loss, full_loss)
        for name, full_aux_loss in full_output.aux.losses.items():
            torch.testing.assert_close(
                deferred_output.aux.losses[name],
                full_aux_loss,
            )
        self.assertEqual(
            float(deferred_output.aux.metrics["program_memory_cosine"]),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
