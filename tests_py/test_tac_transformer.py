import unittest
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from zipfile import ZipFile

import torch

from tac_transformer import (
    AUTHORITY_EXACT_MEMORY_INDEX,
    AUTHORITY_SYSTEM2_VERIFY_INDEX,
    ContentWritePolicy,
    IdentityState,
    TACConfig,
    TACTransformerLM,
    VanillaTransformerLM,
    best_chunked_memory_training_kwargs,
    best_tac_config,
    run5_capability_config,
    run5_capability_training_kwargs,
    run5b_capability_config,
    run5b_capability_training_kwargs,
)
from tac_transformer.data import (
    dedupe_prepared_jsonl,
    normalize_template_text,
    prepare_jsonl_dataset,
    sanitize_training_text,
    serialize_record,
)
from tac_transformer.hard_agentic_data import generate_hard_agentic_records, hard_record_to_jsonl
from kaggle import build_hard_agentic_corpus
from kaggle import analyze_program_specialization
from kaggle import audit_content_memory_causality
from kaggle import evaluate_checkpoint_harder_matrix
from kaggle import evaluate_forced_programs
from kaggle import inspect_identity_memory
from kaggle import make_agentic_training_bundle
from kaggle import train_vanilla_baseline
from kaggle import analyze_routing_collapse
from kaggle import benchmark_harder_research_matrix
from kaggle import benchmark_program_specialization_objectives
from tac_transformer.knowledge_work import estimate_tokens, generate_knowledge_work_records, record_to_jsonl
from kaggle.train_tac_synthetic import parse_args
from kaggle import train_best_tac_agentic
from tac_transformer.model import (
    IdentityAugmentedSelfAttention,
    IdentityState,
    RMSNorm,
    SwiGLUFeedForward,
)
from tac_transformer.training import (
    ChunkedRecallBatcher,
    JsonlLabeledTextBatcher,
    JsonlTextBatcher,
    TokenizedMemmapBatcher,
    apply_memory_read_logits,
    SyntheticProgramBatcher,
    benchmark_chunked_memory,
    benchmark_data_energy_efficiency,
    benchmark_synthetic,
    build_tokenized_memmap_from_jsonl,
    count_parameters,
    estimate_tac_parameter_count,
    estimate_vanilla_parameter_count,
    parameter_matched_baseline_config,
    category_route_loss,
    category_program_mi_loss,
    selected_program_mi_loss,
    train_chunked_memory,
    train_synthetic,
    _default_aux_weights,
)
from tac_transformer.evaluation import benchmark_effectiveness, evaluate_state_interventions
from tac_transformer.capability import (
    CAPABILITY_SANITY_VARIANTS,
    aggregate_capability_sanity_results,
    aggregate_evolutionary_search_results,
    aggregate_external_run5b_validation,
    aggregate_routing_pressure_phase_results,
    aggregate_run5_pathfinder_results,
    build_routing_pressure_phase_variants,
    build_run5_pathfinder_variants,
    category_program_mi_bits_from_probs,
    format_evolutionary_search_markdown,
    format_external_run5b_validation_markdown,
    run_capability_sanity_matrix,
    run_capability_variant_settings,
    run_routing_pressure_phase_matrix,
)


class TACTransformerArchitectureTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            energy_budget=2.2,
            beta=1.75,
        )

    def test_language_model_forward_returns_logits_state_and_auxiliary_losses(self):
        model = TACTransformerLM(self.config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])

        output = model(input_ids)

        self.assertEqual(output.logits.shape, (2, 5, 32))
        self.assertEqual(len(output.identity_states), 1)
        self.assertEqual(output.identity_states[0].stability.shape, (2, 6))
        self.assertEqual(output.aux.coherence.shape, (2, 5, 5))
        self.assertEqual(output.aux.selected_program_mask.shape, (2, 6))
        self.assertEqual(output.aux.token_program_activations.shape, (2, 5, 6))
        self.assertEqual(output.aux.token_selected_program_mask.shape, (2, 5, 6))
        self.assertLessEqual(
            float(output.aux.used_energy.max().detach()),
            self.config.energy_budget + 1e-5,
        )
        self.assertGreaterEqual(float(output.aux.losses["coherence"].detach()), 0)
        self.assertGreaterEqual(float(output.aux.losses["energy"].detach()), 0)
        self.assertGreaterEqual(float(output.aux.losses["program_reuse"].detach()), 0)

    def test_routing_top_k_cannot_exceed_program_count(self):
        with self.assertRaisesRegex(ValueError, "routing_top_k must be <= n_programs"):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_heads=4,
                    n_layers=1,
                    n_programs=2,
                    max_seq_len=12,
                    routing_top_k=3,
                )
            )

    def test_identity_coherence_modulates_attention_probabilities(self):
        attention = IdentityAugmentedSelfAttention(d_model=8, n_heads=2)
        hidden = torch.randn(1, 3, 8)
        coherence = torch.zeros(1, 3, 3)
        coherence[:, 0, 2] = 1.0

        baseline = attention(hidden, coherence=coherence, beta=0.0).attention_probs
        modulated = attention(hidden, coherence=coherence, beta=5.0).attention_probs

        self.assertGreater(
            float(modulated[0, :, 0, 2].mean().detach()),
            float(baseline[0, :, 0, 2].mean().detach()),
        )

    def test_identity_state_persists_across_forward_calls(self):
        model = TACTransformerLM(self.config)
        first_ids = torch.tensor([[1, 2, 3, 4]])
        second_ids = torch.tensor([[4, 3, 2, 1]])

        first = model(first_ids)
        continued = model(second_ids, identity_states=first.identity_states)
        fresh = model(second_ids)

        self.assertFalse(
            torch.allclose(
                continued.identity_states[0].stability,
                fresh.identity_states[0].stability,
            )
        )

    def test_identity_conditioned_decision_continuity_biases_carried_routes(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            routing_type="energy",
            energy_budget=1.35,
            decision_continuity_strength=20.0,
            decision_continuity_decay=0.0,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        with torch.no_grad():
            model.token_embedding.weight.zero_()
            model.position_embedding.weight.zero_()
            identity = model.blocks[0].identity_field
            identity.program_embeddings.zero_()
            identity.program_embeddings[0, 0] = 1.0
            identity.program_embeddings[5, 1] = 1.0

        input_ids = torch.tensor([[1, 2, 3, 4]])
        zeros = torch.zeros(1, config.n_programs)
        memory = torch.zeros(1, config.n_programs, config.d_model)
        prior_decision = torch.zeros(1, config.n_programs)
        prior_decision[:, 5] = 1.0
        carried_state = IdentityState(
            stability=zeros,
            program_memory=memory,
            decision_memory=prior_decision,
        )
        unconditioned_state = IdentityState(stability=zeros, program_memory=memory)

        continued = model(input_ids, identity_states=[carried_state])
        unconditioned = model(input_ids, identity_states=[unconditioned_state])

        self.assertEqual(float(continued.aux.selected_program_mask[0, 5].detach()), 1.0)
        self.assertEqual(float(unconditioned.aux.selected_program_mask[0, 0].detach()), 1.0)
        self.assertEqual(float(unconditioned.aux.selected_program_mask[0, 5].detach()), 0.0)
        self.assertIn("decision_continuity", continued.aux.losses)
        self.assertIn("decision_continuity_agreement", continued.aux.metrics)
        self.assertGreater(
            float(continued.aux.metrics["decision_continuity_agreement"].detach()),
            0.9,
        )
        self.assertIsNotNone(continued.identity_states[0].decision_memory)

    def test_forward_can_skip_auxiliary_diagnostics_for_inference(self):
        model = TACTransformerLM(self.config)
        output = model(torch.tensor([[1, 2, 3, 4]]), collect_auxiliary=False)

        self.assertEqual(output.logits.shape, (1, 4, self.config.vocab_size))
        self.assertEqual(len(output.identity_states), 1)
        self.assertEqual(float(output.aux.losses["coherence"].detach()), 0.0)
        self.assertIn("active_expert_fraction", output.aux.metrics)

    def test_run5b_plus_program_embed_dim_forward_wires_data_energy(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            program_embed_dim=8,
            activation_l1_weight=0.05,
            identity_norm_floor_weight=0.1,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

        output = model(input_ids, labels=input_ids)
        continued = model(input_ids, identity_states=output.identity_states)

        self.assertEqual(output.logits.shape, (2, 4, config.vocab_size))
        self.assertEqual(output.aux.data_energy.shape, (2, 4))
        self.assertIn("data_energy", output.aux.losses)
        self.assertIn("activation_l1", output.aux.losses)
        self.assertIn("identity_norm_floor", output.aux.losses)
        self.assertEqual(
            output.identity_states[0].decision_memory_ebm.shape,
            (2, config.n_programs, config.program_embed_dim),
        )
        self.assertTrue(torch.isfinite(output.aux.losses["data_energy"]))
        self.assertTrue(torch.isfinite(continued.aux.losses["ebm_decision_continuity"]))
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_run5b_plus_default_aux_weights_include_new_losses(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            program_embed_dim=8,
            activation_l1_weight=0.05,
            identity_norm_floor_weight=0.1,
            decision_continuity_loss_weight=0.2,
        )

        weights = _default_aux_weights(config)

        self.assertEqual(weights["data_energy"], 1.0)
        self.assertAlmostEqual(weights["activation_l1"], 0.05)
        self.assertAlmostEqual(weights["identity_norm_floor"], 0.1)
        self.assertAlmostEqual(weights["ebm_decision_continuity"], 0.2)

    def test_global_attention_token_ids_extend_sliding_window(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            attention_window_size=2,
            global_attention_token_ids=(9,),
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[9, 1, 2, 3, 4]])

        output = model(input_ids)

        self.assertFalse(torch.isneginf(output.aux.attention_probs[0, :, 4, 0]).any())

    def test_identity_compressed_attention_reads_program_memory_slots(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            identity_attention_type="compressed_memory",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 2, 3, 4, 5]]))
        query = model(
            torch.tensor([[5, 4, 3, 2]]),
            identity_states=context.identity_states,
        )
        reset = model(torch.tensor([[5, 4, 3, 2]]))
        loss = query.logits.mean()
        loss.backward()

        self.assertEqual(query.logits.shape, (1, 4, 32))
        self.assertEqual(query.aux.attention_probs.shape[-1], 4 + config.n_programs)
        self.assertEqual(reset.aux.attention_probs.shape[-1], 4)
        self.assertFalse(torch.allclose(query.logits, reset.logits))
        self.assertIsNotNone(model.blocks[0].attention.key_value.weight.grad)

    def test_local_attention_window_masks_tokens_outside_window(self):
        dense_attention = IdentityAugmentedSelfAttention(
            d_model=8,
            n_heads=2,
            causal=True,
        )
        attention = IdentityAugmentedSelfAttention(
            d_model=8,
            n_heads=2,
            causal=True,
            attention_window_size=2,
        )
        attention.load_state_dict(dense_attention.state_dict())
        hidden = torch.randn(1, 5, 8)

        output = attention(hidden)
        dense = dense_attention(hidden)
        expected_logits = dense.attention_logits[..., -2:].clone()
        for index in range(hidden.shape[1]):
            start = max(0, index - 1)
            selected = dense.attention_logits[..., index, start : index + 1]
            expected_logits[..., index, -selected.shape[-1] :] = selected
            if selected.shape[-1] < 2:
                expected_logits[..., index, : 2 - selected.shape[-1]] = float("-inf")
        expected_probs = torch.softmax(expected_logits, dim=-1)

        self.assertEqual(output.attention_logits.shape, (1, 2, 5, 2))
        self.assertTrue(torch.isneginf(output.attention_logits[0, :, 0, 0]).all())
        self.assertTrue(torch.allclose(output.attention_logits, expected_logits, atol=1e-6))
        self.assertTrue(torch.allclose(output.attention_probs, expected_probs, atol=1e-6))

    def test_local_attention_window_supports_identity_sparse_mask(self):
        dense_attention = IdentityAugmentedSelfAttention(
            d_model=8,
            n_heads=2,
            causal=True,
        )
        attention = IdentityAugmentedSelfAttention(
            d_model=8,
            n_heads=2,
            causal=True,
            attention_window_size=2,
        )
        attention.load_state_dict(dense_attention.state_dict())
        hidden = torch.randn(1, 5, 8)
        sparse_mask = torch.tensor(
            [
                [
                    [True, False, False, False, False],
                    [True, True, False, False, False],
                    [False, True, True, False, False],
                    [False, False, True, True, False],
                    [False, False, False, True, True],
                ]
            ]
        )

        output = attention(hidden, identity_sparse_mask=sparse_mask)
        dense = dense_attention(hidden, identity_sparse_mask=sparse_mask)
        expected_logits = dense.attention_logits[..., -2:].clone()
        for index in range(hidden.shape[1]):
            start = max(0, index - 1)
            selected = dense.attention_logits[..., index, start : index + 1]
            expected_logits[..., index, -selected.shape[-1] :] = selected
            if selected.shape[-1] < 2:
                expected_logits[..., index, : 2 - selected.shape[-1]] = float("-inf")
        expected_probs = torch.softmax(expected_logits, dim=-1)

        self.assertEqual(output.attention_logits.shape, (1, 2, 5, 2))
        self.assertTrue(torch.allclose(output.attention_logits, expected_logits, atol=1e-6))
        self.assertTrue(torch.allclose(output.attention_probs, expected_probs, atol=1e-6))

    def test_identity_sparse_attention_masks_cross_program_edges(self):
        attention = IdentityAugmentedSelfAttention(d_model=8, n_heads=2)
        hidden = torch.randn(1, 4, 8)
        sparse_mask = torch.tensor(
            [
                [
                    [True, True, False, False],
                    [True, True, False, False],
                    [False, False, True, True],
                    [False, False, True, True],
                ]
            ]
        )

        output = attention(hidden, identity_sparse_mask=sparse_mask)

        self.assertTrue(torch.isneginf(output.attention_logits[0, :, 0, 2]).all())
        self.assertTrue(torch.isneginf(output.attention_logits[0, :, 2, 0]).all())
        self.assertEqual(float(output.attention_probs[0, :, 0, 2].sum().detach()), 0.0)
        self.assertEqual(float(output.attention_probs[0, :, 2, 0].sum().detach()), 0.0)

    def test_identity_first_attention_uses_identity_aware_key_values(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            identity_attention_type="identity_first",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        output = model(torch.tensor([[1, 2, 3, 4, 5]]))

        loss = output.logits.mean() + sum(output.aux.losses.values())
        loss.backward()

        self.assertEqual(output.logits.shape, (1, 5, 32))
        self.assertIsNotNone(model.blocks[0].attention.identity_key_value)
        self.assertIsNotNone(model.blocks[0].attention.identity_key_value.weight.grad)
        self.assertGreater(
            float(model.blocks[0].attention.identity_key_value.weight.grad.abs().sum()),
            0,
        )
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_sparse_compressed_attention_keeps_memory_slots(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            identity_attention_type="coherence_sparse_compressed",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 2, 3, 4, 5]]))
        query = model(
            torch.tensor([[5, 4, 3, 2]]),
            identity_states=context.identity_states,
        )

        self.assertEqual(query.aux.attention_probs.shape[-1], 4 + config.n_programs)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_causal_model_prefix_logits_ignore_future_tokens(self):
        model = TACTransformerLM(self.config)
        base = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
        changed_future = torch.tensor([[1, 5, 9, 13, 4, 6, 8, 10]])

        base_logits = model(base).logits[:, :4, :]
        changed_logits = model(changed_future).logits[:, :4, :]

        self.assertTrue(
            torch.allclose(base_logits, changed_logits, atol=1e-6),
            "causal prefix logits changed when only future tokens changed",
        )

    def test_forward_pass_supports_backpropagation_to_program_embeddings(self):
        model = TACTransformerLM(self.config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        output = model(input_ids)
        loss = (
            output.logits.mean()
            + output.aux.losses["coherence"]
            + output.aux.losses["energy"]
            + output.aux.losses["program_reuse"]
        )
        loss.backward()

        grad = model.blocks[0].identity_field.program_embeddings.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_parameter_count_reports_trainable_parameters(self):
        model = TACTransformerLM(self.config)
        counts = count_parameters(model)

        self.assertGreater(counts["total"], 0)
        self.assertEqual(counts["total"], counts["trainable"])
        self.assertGreater(counts["identity_field"], 0)

    def test_modern_backbone_switches_use_rmsnorm_and_swiglu(self):
        modern_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            norm_type="rmsnorm",
            mlp_type="swiglu",
        )
        model = TACTransformerLM(modern_config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        output = model(input_ids)
        loss = output.logits.mean() + sum(output.aux.losses.values())
        loss.backward()

        self.assertIsInstance(model.blocks[0].norm_attention, RMSNorm)
        self.assertIsInstance(model.blocks[0].mlp, SwiGLUFeedForward)
        self.assertEqual(output.logits.shape, (1, 5, 32))
        self.assertGreater(count_parameters(model)["total"], count_parameters(TACTransformerLM(self.config))["total"])
        self.assertIsNotNone(model.blocks[0].mlp.up_gate.weight.grad)

    def test_vanilla_baseline_supports_modern_backbone_switches(self):
        modern_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            norm_type="rmsnorm",
            mlp_type="swiglu",
        )
        model = VanillaTransformerLM(modern_config)
        output = model(torch.tensor([[1, 2, 3, 4]]))

        self.assertIsInstance(model.blocks[0].norm_attention, RMSNorm)
        self.assertIsInstance(model.blocks[0].mlp, SwiGLUFeedForward)
        self.assertEqual(output.logits.shape, (1, 4, 32))
        self.assertEqual(output.identity_states, [])

    def test_invalid_backbone_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    norm_type="batchnorm",
                )
            )

        with self.assertRaises(ValueError):
            VanillaTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    mlp_type="relu",
                )
            )

    def test_rope_position_encoding_drops_learned_position_parameters(self):
        learned_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
        )
        rope_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            position_type="rope",
        )
        learned = TACTransformerLM(learned_config)
        rope = TACTransformerLM(rope_config)

        output = rope(torch.tensor([[1, 2, 3, 4, 5]]))

        self.assertIsNone(rope.position_embedding)
        self.assertEqual(output.logits.shape, (1, 5, 32))
        self.assertEqual(
            count_parameters(learned)["total"] - count_parameters(rope)["total"],
            learned_config.max_seq_len * learned_config.d_model,
        )

    def test_rope_position_encoding_preserves_causal_prefix_behavior(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            position_type="rope",
        )
        model = TACTransformerLM(config)
        base = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
        changed_future = torch.tensor([[1, 5, 9, 13, 4, 6, 8, 10]])

        base_logits = model(base).logits[:, :4, :]
        changed_logits = model(changed_future).logits[:, :4, :]

        self.assertTrue(torch.allclose(base_logits, changed_logits, atol=1e-6))

    def test_rope_scaling_controls_preserve_shape_and_parameter_count(self):
        base_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=32,
            position_type="rope",
        )
        scaled_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=32,
            position_type="rope",
            rope_scaling_type="linear",
            original_context_length=16,
            target_context_length=32,
        )
        base = TACTransformerLM(base_config)
        scaled = TACTransformerLM(scaled_config)
        scaled.load_state_dict(base.state_dict())
        input_ids = torch.arange(16).unsqueeze(0) % 32

        base_logits = base(input_ids).logits
        scaled_logits = scaled(input_ids).logits

        self.assertEqual(base_logits.shape, scaled_logits.shape)
        self.assertEqual(count_parameters(base)["total"], count_parameters(scaled)["total"])
        self.assertFalse(torch.allclose(base_logits[:, -1, :], scaled_logits[:, -1, :]))

    def test_invalid_rope_scaling_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_heads=4,
                    max_seq_len=12,
                    position_type="rope",
                    rope_scaling_type="bad",
                )
            )

    def test_vanilla_baseline_supports_rope_position_encoding(self):
        model = VanillaTransformerLM(
            TACConfig(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                max_seq_len=12,
                position_type="rope",
            )
        )
        output = model(torch.tensor([[1, 2, 3, 4]]))

        self.assertIsNone(model.position_embedding)
        self.assertEqual(output.logits.shape, (1, 4, 32))

    def test_invalid_position_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    position_type="sinusoidal",
                )
            )

    def test_grouped_query_attention_reduces_parameters_and_runs_tac(self):
        full_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
        )
        gqa_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
        )
        full_model = TACTransformerLM(full_config)
        gqa_model = TACTransformerLM(gqa_config)

        output = gqa_model(torch.tensor([[1, 2, 3, 4, 5]]))

        self.assertEqual(gqa_model.blocks[0].attention.n_kv_heads, 2)
        self.assertEqual(output.logits.shape, (1, 5, 32))
        self.assertLess(
            count_parameters(gqa_model)["total"],
            count_parameters(full_model)["total"],
        )
        self.assertEqual(
            estimate_tac_parameter_count(gqa_config),
            count_parameters(gqa_model)["total"],
        )

    def test_parameter_matched_baseline_preserves_valid_rope_grouped_heads(self):
        config = TACConfig(
            vocab_size=64,
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            n_programs=16,
            max_seq_len=16,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            position_type="rope",
        )

        matched = parameter_matched_baseline_config(config)
        model = VanillaTransformerLM(matched)

        self.assertEqual(matched.d_model % matched.n_heads, 0)
        self.assertEqual((matched.d_model // matched.n_heads) % 2, 0)
        self.assertEqual(matched.n_heads % matched.n_kv_heads, 0)
        self.assertGreater(count_parameters(model)["total"], 0)

    def test_grouped_query_attention_preserves_causal_prefix_behavior(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            position_type="rope",
        )
        model = TACTransformerLM(config)
        base = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
        changed_future = torch.tensor([[1, 5, 9, 13, 4, 6, 8, 10]])

        base_logits = model(base).logits[:, :4, :]
        changed_logits = model(changed_future).logits[:, :4, :]

        self.assertTrue(torch.allclose(base_logits, changed_logits, atol=1e-6))

    def test_vanilla_baseline_supports_grouped_query_attention(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=1,
            max_seq_len=12,
        )
        model = VanillaTransformerLM(config)
        output = model(torch.tensor([[1, 2, 3, 4]]))

        self.assertEqual(model.blocks[0].attention.n_kv_heads, 1)
        self.assertEqual(output.logits.shape, (1, 4, 32))
        self.assertEqual(
            estimate_vanilla_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_invalid_grouped_query_attention_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_heads=4,
                    n_kv_heads=3,
                )
            )

        with self.assertRaises(ValueError):
            VanillaTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_heads=4,
                    n_kv_heads=0,
                )
            )

    def test_routed_program_experts_add_trainable_compute_to_tac(self):
        legacy_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
        )
        expert_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            program_compute_type="linear_expert",
        )
        legacy_model = TACTransformerLM(legacy_config)
        expert_model = TACTransformerLM(expert_config)

        output = expert_model(torch.tensor([[1, 2, 3, 4, 5]]))
        loss = output.logits.mean() + sum(output.aux.losses.values())
        loss.backward()

        expert_layer = expert_model.blocks[0].identity_field
        self.assertEqual(output.logits.shape, (1, 5, 32))
        self.assertIsNotNone(expert_layer.program_expert_weight.grad)
        self.assertGreater(float(expert_layer.program_expert_weight.grad.abs().sum()), 0)
        self.assertGreater(
            count_parameters(expert_model)["total"],
            count_parameters(legacy_model)["total"],
        )
        self.assertEqual(
            estimate_tac_parameter_count(expert_config),
            count_parameters(expert_model)["total"],
        )

    def test_coalition_context_modulates_linear_experts_from_program_memory(self):
        base_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            program_compute_type="linear_expert",
            routing_top_k=2,
            coalition_context_type="none",
        )
        coalition_config = replace(
            base_config,
            coalition_context_type="program_memory",
            coalition_context_scale=0.5,
        )
        base_model = TACTransformerLM(base_config)
        coalition_model = TACTransformerLM(coalition_config)
        compatible = {
            key: value
            for key, value in base_model.state_dict().items()
            if key in coalition_model.state_dict()
            and coalition_model.state_dict()[key].shape == value.shape
        }
        coalition_model.load_state_dict(compatible, strict=False)
        identity = coalition_model.blocks[0].identity_field
        self.assertIsNotNone(identity.coalition_context_projection)

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        state = IdentityState(
            stability=torch.ones(1, 6),
            program_memory=torch.randn(1, 6, 16),
        )
        base_output = base_model(input_ids, identity_states=[state])
        coalition_output = coalition_model(input_ids, identity_states=[state])
        loss = coalition_output.logits.mean()
        loss.backward()

        self.assertIn("coalition_context_norm", coalition_output.aux.metrics)
        self.assertGreater(
            float(coalition_output.aux.metrics["coalition_context_norm"].detach()),
            0.0,
        )
        self.assertFalse(torch.allclose(base_output.logits, coalition_output.logits))
        self.assertIsNotNone(identity.coalition_context_projection.weight.grad)
        self.assertGreater(
            float(identity.coalition_context_projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertEqual(
            estimate_tac_parameter_count(coalition_config),
            count_parameters(coalition_model)["total"],
        )

    def test_graph_coalition_context_modulates_program_specific_experts(self):
        base_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            program_compute_type="linear_expert",
            routing_top_k=2,
            coalition_context_type="program_memory_task_graph",
            coalition_context_scale=0.5,
        )
        model = TACTransformerLM(base_config)
        identity = model.blocks[0].identity_field
        self.assertIsNotNone(identity.coalition_context_projection)
        self.assertIsNotNone(identity.coalition_source_key_projection)
        self.assertIsNotNone(identity.coalition_source_value_projection)
        self.assertIsNotNone(identity.coalition_target_query_projection)
        self.assertIsNotNone(identity.coalition_task_query_projection)

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        state = IdentityState(
            stability=torch.ones(1, 6),
            program_memory=torch.randn(1, 6, 16),
        )
        output = model(input_ids, identity_states=[state])
        loss = output.logits.mean()
        loss.backward()

        self.assertIn("coalition_context_norm", output.aux.metrics)
        self.assertGreater(
            float(output.aux.metrics["coalition_context_norm"].detach()),
            0.0,
        )
        self.assertIsNotNone(identity.coalition_source_key_projection.weight.grad)
        self.assertGreater(
            float(identity.coalition_source_key_projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(identity.coalition_source_value_projection.weight.grad)
        self.assertGreater(
            float(identity.coalition_source_value_projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(identity.coalition_target_query_projection.weight.grad)
        self.assertGreater(
            float(identity.coalition_target_query_projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(identity.coalition_task_query_projection.weight.grad)
        self.assertGreater(
            float(identity.coalition_task_query_projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertEqual(
            estimate_tac_parameter_count(base_config),
            count_parameters(model)["total"],
        )

    def test_invalid_coalition_context_options_are_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    coalition_context_type="dense_graph",
                )
            )
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    coalition_context_scale=-0.1,
                )
            )

    def test_sparse_linear_program_experts_report_active_compute_proxy(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            program_compute_type="sparse_linear_expert",
        )
        model = TACTransformerLM(config)

        output = model(torch.tensor([[1, 2, 3, 4, 5]]))
        loss = output.logits.mean() + sum(output.aux.losses.values())
        loss.backward()

        expert_layer = model.blocks[0].identity_field
        self.assertIn("active_expert_parameters", output.aux.metrics)
        self.assertGreater(float(output.aux.metrics["active_expert_parameters"]), 0)
        self.assertLess(
            float(output.aux.metrics["active_expert_parameters"]),
            float(output.aux.metrics["total_expert_parameters"]),
        )
        self.assertIsNotNone(expert_layer.program_expert_weight.grad)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_low_rank_linear_program_experts_train_with_reduced_parameters(self):
        full_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            program_compute_type="linear_expert",
        )
        low_rank_config = replace(
            full_config,
            program_compute_type="low_rank_linear_expert",
            program_expert_rank=5,
        )
        full_model = TACTransformerLM(full_config)
        low_rank_model = TACTransformerLM(low_rank_config)

        output = low_rank_model(torch.tensor([[1, 2, 3, 4, 5]]))
        loss = output.logits.mean() + sum(output.aux.losses.values())
        loss.backward()

        expert_layer = low_rank_model.blocks[0].identity_field
        self.assertEqual(output.logits.shape, (1, 5, 32))
        self.assertIsNotNone(expert_layer.program_expert_down.grad)
        self.assertIsNotNone(expert_layer.program_expert_up.grad)
        self.assertGreater(float(expert_layer.program_expert_down.grad.abs().sum()), 0)
        self.assertGreater(float(expert_layer.program_expert_up.grad.abs().sum()), 0)
        self.assertLess(
            count_parameters(low_rank_model)["identity_field"],
            count_parameters(full_model)["identity_field"],
        )
        self.assertEqual(
            estimate_tac_parameter_count(low_rank_config),
            count_parameters(low_rank_model)["total"],
        )

    def test_low_rank_linear_expert_rank_is_validated(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_heads=4,
                    program_compute_type="low_rank_linear_expert",
                    program_expert_rank=0,
                )
            )

    def test_sparse_linear_program_experts_match_dense_routed_outputs(self):
        dense_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            program_compute_type="linear_expert",
        )
        sparse_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            program_compute_type="sparse_linear_expert",
        )
        dense_model = TACTransformerLM(dense_config)
        sparse_model = TACTransformerLM(sparse_config)
        sparse_model.load_state_dict(dense_model.state_dict())
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        dense_output = dense_model(input_ids)
        sparse_output = sparse_model(input_ids)

        self.assertTrue(torch.allclose(dense_output.logits, sparse_output.logits, atol=1e-6))
        self.assertLess(float(sparse_output.aux.metrics["active_expert_fraction"]), 1.0)

    def test_sparse_linear_expert_uses_batched_selected_expert_dispatch(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            program_compute_type="sparse_linear_expert",
        )
        layer = TACTransformerLM(config).blocks[0].identity_field
        hidden = torch.randn(2, 5, 16)
        routed_weights = torch.zeros(2, 5, 6)
        routed_weights[:, :, 0] = 0.25
        routed_weights[:, :, 2] = 0.75

        context = layer._compute_sparse_program_context(hidden, routed_weights)

        self.assertEqual(context.shape, hidden.shape)
        self.assertEqual(layer.latest_sparse_dispatch_size, 2)

    def test_vanilla_baseline_ignores_program_expert_compute_parameters(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            program_compute_type="linear_expert",
        )
        model = VanillaTransformerLM(config)
        output = model(torch.tensor([[1, 2, 3, 4]]))

        self.assertEqual(output.logits.shape, (1, 4, 32))
        self.assertEqual(count_parameters(model)["identity_field"], 0)
        self.assertEqual(
            estimate_vanilla_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_invalid_program_compute_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    program_compute_type="moe",
                )
            )

    def test_identity_sink_programs_are_always_selected_without_spending_route_energy(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            energy_budget=1.0,
            n_sink_programs=2,
        )
        model = TACTransformerLM(config)

        output = model(torch.tensor([[1, 2, 3, 4, 5]]))

        selected = output.aux.selected_program_mask[0]
        self.assertTrue(torch.equal(selected[:2], torch.ones(2)))
        used_energy = float(output.aux.used_energy.max().detach())
        self.assertLessEqual(used_energy, config.energy_budget + 1e-5)
        self.assertIn("sink_programs", output.aux.metrics)
        self.assertEqual(float(output.aux.metrics["sink_programs"]), 2.0)

    def test_invalid_identity_sink_program_count_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_programs=6,
                    n_sink_programs=7,
                )
            )

    def test_expert_choice_routing_reports_balanced_program_load(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            routing_type="expert_choice",
        )
        model = TACTransformerLM(config)

        output = model(torch.tensor([[1, 2, 3, 4, 5, 6]]))

        self.assertEqual(output.logits.shape, (1, 6, 32))
        self.assertIn("routing_load_std", output.aux.metrics)
        self.assertEqual(float(output.aux.metrics["routing_type"]), 1.0)
        used_energy = float(output.aux.used_energy.max().detach())
        self.assertLessEqual(used_energy, config.energy_budget + 1e-5)

    def test_hash_routing_is_deterministic_for_same_inputs(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            routing_type="hash",
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        first = model(input_ids)
        second = model(input_ids)

        self.assertTrue(
            torch.equal(first.aux.selected_program_mask, second.aux.selected_program_mask)
        )
        self.assertEqual(float(first.aux.metrics["routing_type"]), 3.0)

    def test_sparse_ensemble_routing_extends_base_anchor(self):
        base_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="base",
        )
        ensemble_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="sparse_ensemble",
            routing_top_k=3,
        )
        base = TACTransformerLM(base_config)
        ensemble = TACTransformerLM(ensemble_config)
        ensemble.load_state_dict(base.state_dict())
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        base_output = base(input_ids)
        ensemble_output = ensemble(input_ids)

        self.assertEqual(float(ensemble_output.aux.metrics["routing_type"]), 4.0)
        self.assertTrue(
            torch.all(ensemble_output.aux.selected_program_mask >= base_output.aux.selected_program_mask)
        )
        self.assertGreater(
            float(ensemble_output.aux.selected_program_mask.sum().detach()),
            float(base_output.aux.selected_program_mask.sum().detach()),
        )
        used_energy = float(ensemble_output.aux.used_energy.max().detach())
        self.assertLessEqual(used_energy, ensemble_config.energy_budget + 1e-5)

    def test_base_semantic_routing_adds_activation_conditioned_programs(self):
        base_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="base",
            energy_budget=4.0,
        )
        semantic_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="base_semantic",
            routing_top_k=2,
            routing_load_balance_weight=0.05,
            energy_budget=4.0,
        )
        base = TACTransformerLM(base_config)
        semantic = TACTransformerLM(semantic_config)
        semantic.load_state_dict(base.state_dict(), strict=False)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        base_output = base(input_ids)
        semantic_output = semantic(input_ids)

        self.assertEqual(float(semantic_output.aux.metrics["routing_type"]), 5.0)
        self.assertTrue(
            torch.all(
                semantic_output.aux.token_selected_program_mask
                >= base_output.aux.token_selected_program_mask
            )
        )
        self.assertGreater(
            float(semantic_output.aux.token_selected_program_mask.sum().detach()),
            float(base_output.aux.token_selected_program_mask.sum().detach()),
        )
        self.assertIn("routing_load_balance", semantic_output.aux.losses)
        self.assertIn("routing_load_balance", semantic_output.aux.metrics)
        used_energy = float(semantic_output.aux.used_energy.max().detach())
        self.assertLessEqual(used_energy, semantic_config.energy_budget + 1e-5)

    def test_base_semantic_routing_can_target_program_family(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="base_semantic",
            routing_top_k=2,
            semantic_route_allowed_programs=(2, 4, 6),
            semantic_route_suppressed_programs=(4,),
            energy_budget=4.0,
        )
        base_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="base",
            energy_budget=4.0,
        )
        base = TACTransformerLM(base_config)
        model = TACTransformerLM(config)
        model.load_state_dict(base.state_dict(), strict=False)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        base_mask = base(input_ids).aux.token_selected_program_mask
        targeted_mask = model(input_ids).aux.token_selected_program_mask
        semantic_extra = (targeted_mask - base_mask).clamp_min(0.0)

        blocked = [0, 1, 3, 4, 5, 7]
        self.assertEqual(float(semantic_extra[..., blocked].sum().detach()), 0.0)
        self.assertGreater(float(semantic_extra[..., [2, 6]].sum().detach()), 0.0)

    def test_base_semantic_routing_rejects_invalid_program_filters(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_programs=4,
                    routing_type="base_semantic",
                    semantic_route_allowed_programs=(4,),
                )
            )
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_programs=4,
                    routing_type="base_semantic",
                    semantic_route_suppressed_programs=(1, 1),
                )
            )

    def test_base_semantic_soft_routing_uses_differentiable_semantic_mask(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="base_semantic_soft",
            routing_top_k=2,
            routing_load_balance_weight=0.05,
            energy_budget=4.0,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        output = model(input_ids)
        mask = output.aux.token_selected_program_mask

        self.assertEqual(float(output.aux.metrics["routing_type"]), 6.0)
        self.assertTrue(torch.any((mask > 0.0) & (mask < 1.0)))
        self.assertIn("routing_load_balance", output.aux.losses)
        used_energy = float(output.aux.used_energy.max().detach())
        self.assertLessEqual(used_energy, config.energy_budget + 1e-5)

    def test_authority_gated_routing_reports_epistemic_route_signals(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="authority_gated",
            routing_top_k=2,
            routing_load_balance_weight=0.05,
            energy_budget=4.0,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])

        output = model(input_ids)

        self.assertEqual(output.logits.shape, (2, 6, 32))
        self.assertEqual(float(output.aux.metrics["routing_type"]), 7.0)
        self.assertEqual(output.aux.authority_logits.shape, (2, 5))
        self.assertEqual(output.aux.authority_probs.shape, (2, 5))
        self.assertEqual(output.aux.authority_indices.shape, (2,))
        self.assertEqual(output.aux.verifier_required.shape, (2,))
        self.assertEqual(output.aux.halt_probability.shape, (2,))
        self.assertTrue(output.aux.verifier_required.dtype == torch.bool)
        self.assertTrue(
            torch.allclose(
                output.aux.authority_probs.sum(dim=-1),
                torch.ones(2),
                atol=1e-5,
            )
        )
        self.assertTrue(
            bool(
                torch.all(
                    (output.aux.halt_probability >= 0.0)
                    & (output.aux.halt_probability <= 1.0)
                ).item()
            )
        )
        self.assertIn("authority_verifier_required_rate", output.aux.metrics)
        self.assertIn("authority_halt_probability", output.aux.metrics)
        self.assertIn("routing_load_balance", output.aux.losses)
        used_energy = float(output.aux.used_energy.max().detach())
        self.assertLessEqual(used_energy, config.energy_budget + 1e-5)

    def test_authority_gated_routing_receives_language_loss_gradients(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="authority_gated",
            routing_top_k=2,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        labels = torch.tensor([[2, 3, 4, 5, 6, 7]])

        output = model(input_ids, labels=labels)
        self.assertIsNotNone(output.loss)
        output.loss.backward()

        authority_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "authority_" in name
        ]
        self.assertTrue(authority_grads)
        self.assertTrue(any(grad is not None for grad in authority_grads))
        self.assertGreater(
            sum(
                float(grad.abs().sum())
                for grad in authority_grads
                if grad is not None
            ),
            0.0,
        )

    def test_authority_gated_accepts_supervised_authority_targets(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="authority_gated",
            routing_top_k=2,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
        authority_mode_targets = torch.tensor(
            [AUTHORITY_EXACT_MEMORY_INDEX, AUTHORITY_SYSTEM2_VERIFY_INDEX]
        )
        verifier_required_targets = torch.tensor([False, True])
        halt_targets = torch.tensor([0.0, 1.0])

        output = model(
            input_ids,
            authority_mode_targets=authority_mode_targets,
            verifier_required_targets=verifier_required_targets,
            authority_halt_targets=halt_targets,
        )

        self.assertIn("authority_mode", output.aux.losses)
        self.assertIn("authority_verifier_required", output.aux.losses)
        self.assertIn("authority_halt", output.aux.losses)
        authority_loss = (
            output.aux.losses["authority_mode"]
            + output.aux.losses["authority_verifier_required"]
            + output.aux.losses["authority_halt"]
        )
        self.assertTrue(authority_loss.requires_grad)
        authority_loss.backward()

        authority_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "authority_" in name
        ]
        self.assertGreater(
            sum(
                float(grad.abs().sum())
                for grad in authority_grads
                if grad is not None
            ),
            0.0,
        )

    def test_routing_load_balance_loss_trains_semantic_route(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="base_semantic",
            routing_top_k=2,
            routing_load_balance_weight=1.0,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

        output = model(input_ids)
        balance_loss = output.aux.losses["routing_load_balance"]
        self.assertTrue(balance_loss.requires_grad)
        balance_loss.backward()

        semantic_route_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "program_embeddings" in name
        ]
        self.assertTrue(semantic_route_grads)
        self.assertGreater(
            sum(
                float(grad.abs().sum())
                for grad in semantic_route_grads
                if grad is not None
            ),
            0.0,
        )

    def test_category_route_loss_prefers_category_program_mapping(self):
        good = torch.tensor(
            [
                [[0.9, 0.1, 0.1], [0.8, 0.1, 0.1]],
                [[0.1, 0.9, 0.1], [0.1, 0.8, 0.1]],
            ]
        )
        bad = torch.tensor(
            [
                [[0.1, 0.9, 0.1], [0.1, 0.8, 0.1]],
                [[0.9, 0.1, 0.1], [0.8, 0.1, 0.1]],
            ]
        )
        categories = torch.tensor([0, 1])

        good_loss = benchmark_program_specialization_objectives.category_route_loss(
            good,
            categories,
            n_categories=2,
        )
        bad_loss = benchmark_program_specialization_objectives.category_route_loss(
            bad,
            categories,
            n_categories=2,
        )

        self.assertLess(float(good_loss), float(bad_loss))

    def test_category_program_mi_loss_prefers_category_separable_routes(self):
        separated = torch.full((4, 3, 4), 0.01)
        separated[0:2, :, 0] = 0.95
        separated[2:4, :, 1] = 0.95
        collapsed = torch.full((4, 3, 4), 0.01)
        collapsed[:, :, 0] = 0.95
        categories = torch.tensor([0, 0, 1, 1])

        separated_loss = category_program_mi_loss(
            separated,
            categories,
            n_categories=2,
        )
        collapsed_loss = category_program_mi_loss(
            collapsed,
            categories,
            n_categories=2,
        )

        self.assertLess(float(separated_loss), float(collapsed_loss))

    def test_selected_program_mi_loss_prefers_record_level_category_separation(self):
        program_activations = torch.full((4, 4), 0.01)
        program_activations[0:2, 0] = 0.95
        program_activations[2:4, 1] = 0.95
        separated_mask = torch.zeros(4, 4)
        separated_mask[0:2, 0] = 1.0
        separated_mask[2:4, 1] = 1.0
        collapsed_mask = torch.zeros(4, 4)
        collapsed_mask[:, 0] = 1.0
        categories = torch.tensor([0, 0, 1, 1])

        separated_loss = selected_program_mi_loss(
            program_activations,
            separated_mask,
            categories,
            n_categories=2,
        )
        collapsed_loss = selected_program_mi_loss(
            program_activations,
            collapsed_mask,
            categories,
            n_categories=2,
        )

        self.assertLess(float(separated_loss), float(collapsed_loss))

    def test_invalid_routing_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    routing_type="lottery",
                )
            )

    def test_gated_identity_state_update_adds_trainable_gates(self):
        fixed_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
        )
        gated_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            state_update_type="gated",
        )
        fixed_model = TACTransformerLM(fixed_config)
        gated_model = TACTransformerLM(gated_config)
        first = gated_model(torch.tensor([[1, 2, 3, 4, 5]]))
        second = gated_model(
            torch.tensor([[5, 4, 3, 2, 1]]),
            identity_states=first.identity_states,
        )
        loss = second.logits.mean() + sum(second.aux.losses.values())
        loss.backward()

        identity = gated_model.blocks[0].identity_field
        self.assertEqual(second.logits.shape, (1, 5, 32))
        self.assertIsNotNone(identity.stability_gate.weight.grad)
        self.assertIsNotNone(identity.memory_gate.weight.grad)
        self.assertGreater(float(identity.stability_gate.weight.grad.abs().sum()), 0)
        self.assertGreater(float(identity.memory_gate.weight.grad.abs().sum()), 0)
        self.assertGreater(
            count_parameters(gated_model)["total"],
            count_parameters(fixed_model)["total"],
        )
        self.assertEqual(
            estimate_tac_parameter_count(gated_config),
            count_parameters(gated_model)["total"],
        )

    def test_gated_identity_state_preserves_causal_prefix_behavior(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            state_update_type="gated",
        )
        model = TACTransformerLM(config)
        base = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
        changed_future = torch.tensor([[1, 5, 9, 13, 4, 6, 8, 10]])

        base_logits = model(base).logits[:, :4, :]
        changed_logits = model(changed_future).logits[:, :4, :]

        self.assertTrue(torch.allclose(base_logits, changed_logits, atol=1e-6))

    def test_invalid_state_update_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    state_update_type="gru",
                )
            )

    def test_novelty_gated_memory_write_adds_trainable_gate(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        first = model(torch.tensor([[1, 2, 3, 4, 5]]))
        second = model(
            torch.tensor([[5, 4, 3, 2, 1]]),
            identity_states=first.identity_states,
        )
        loss = second.identity_states[0].program_memory.mean()
        loss.backward()

        identity = model.blocks[0].identity_field
        self.assertIsNotNone(identity.memory_novelty_gate.weight.grad)
        self.assertGreater(float(identity.memory_novelty_gate.weight.grad.abs().sum()), 0)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_memory_separation_loss_reports_program_memory_interference(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_separation_weight=0.01,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        output = model(torch.tensor([[1, 2, 3, 4, 5]]))
        loss = output.aux.losses["separation"]
        loss.backward()

        identity = model.blocks[0].identity_field
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertIn("program_memory_cosine", output.aux.metrics)
        self.assertIn("program_ortho", output.aux.metrics)
        self.assertGreaterEqual(
            float(output.aux.metrics["program_memory_cosine"].detach()),
            0.0,
        )
        self.assertAlmostEqual(
            float(output.aux.metrics["program_ortho"].detach()),
            float(loss.detach()),
            places=6,
        )
        self.assertIsNotNone(identity.program_update.weight.grad)

    def test_program_conditioned_memory_update_makes_program_specific_candidates(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            program_memory_update_type="program_conditioned",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        identity = model.blocks[0].identity_field
        pooled_hidden = torch.randn(2, 1, 16)

        candidates = identity._candidate_program_memory(pooled_hidden)
        loss = candidates.mean()
        loss.backward()

        self.assertEqual(candidates.shape, (2, 6, 16))
        self.assertFalse(torch.allclose(candidates[:, 0, :], candidates[:, 1, :]))
        self.assertIsNotNone(identity.program_conditioned_update)
        self.assertIsNotNone(identity.program_conditioned_update.weight.grad)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_invalid_program_memory_update_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    program_memory_update_type="slot_noise",
                )
            )

    def test_content_anti_collapse_losses_and_reconsolidation_are_reported(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=6,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            content_read_steps=2,
            content_read_gate_type="synthesis",
            content_cue_separation_weight=0.005,
            content_gate_entropy_weight=0.005,
            content_reconsolidate=True,
            content_reconsolidate_rate=0.1,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 7, 11, 4, 5, 6]]))
        query = torch.tensor([[2, 7, 3, 4, 5, 6]])

        output = model(query, identity_states=context.identity_states)

        self.assertIn("content_cue_separation", output.aux.losses)
        self.assertIn("content_gate_entropy", output.aux.losses)
        self.assertIn("content_gate_entropy", output.aux.metrics)
        self.assertIn("content_cue_cosine", output.aux.metrics)
        self.assertIn("content_reconsolidation_gate", output.aux.metrics)
        self.assertGreaterEqual(
            float(output.aux.losses["content_cue_separation"].detach()),
            0.0,
        )
        self.assertGreaterEqual(
            float(output.aux.losses["content_gate_entropy"].detach()),
            0.0,
        )
        self.assertGreater(
            float(output.aux.metrics["content_reconsolidation_gate"].detach()),
            0.0,
        )

        before = context.identity_states[0].content_cues
        after = output.identity_states[0].content_cues
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertFalse(torch.allclose(before, after))

    def test_invalid_memory_write_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    memory_write_type="surprise_only",
                )
            )

    def test_reconsolidation_updates_read_memory_with_trainable_gate(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_reconsolidate=True,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        first = model(torch.tensor([[1, 2, 3, 4, 5]]))
        second = model(
            torch.tensor([[5, 4, 3, 2, 1]]),
            identity_states=first.identity_states,
        )
        loss = second.identity_states[0].program_memory.mean()
        loss.backward()

        identity = model.blocks[0].identity_field
        self.assertIsNotNone(identity.memory_reconsolidate_gate.weight.grad)
        self.assertGreater(
            float(identity.memory_reconsolidate_gate.weight.grad.abs().sum()),
            0.0,
        )
        self.assertIn("memory_reconsolidation_gate", second.aux.metrics)
        self.assertGreater(
            float(second.aux.metrics["memory_reconsolidation_gate"].detach()),
            0.0,
        )
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_invalid_reconsolidation_gate_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    reconsolidate_gate_type="attention",
                )
            )

    def test_creb_memory_allocation_tracks_age_and_dead_program_rate(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_allocation_type="creb",
            memory_allocation_k=2,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        first = model(torch.tensor([[1, 2, 3, 4, 5]]))
        second = model(
            torch.tensor([[5, 4, 3, 2, 1]]),
            identity_states=first.identity_states,
        )
        state = second.identity_states[0]

        self.assertIsNotNone(state.program_age)
        self.assertEqual(state.program_age.shape, state.stability.shape)
        self.assertIn("memory_allocation_dead_rate", second.aux.metrics)
        self.assertIn("memory_allocation_age", second.aux.metrics)
        self.assertIn("memory_allocation_load_std", second.aux.metrics)
        self.assertGreaterEqual(
            float(second.aux.metrics["memory_allocation_dead_rate"].detach()),
            0.0,
        )
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_creb_load_penalty_avoids_recently_overused_programs(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_allocation_type="creb",
            memory_allocation_k=1,
            creb_alpha=0.0,
            creb_beta=0.0,
            creb_gamma=0.0,
            creb_delta=10.0,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        overused = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        previous = IdentityState(
            stability=torch.zeros(1, 4),
            program_memory=torch.zeros(1, 4, 16),
            program_age=torch.zeros(1, 4),
            program_write_frequency=overused,
        )

        output = model(
            torch.tensor([[1, 2, 3, 4]]),
            identity_states=[previous],
        )
        state = output.identity_states[0]

        self.assertIsNotNone(state.program_write_frequency)
        self.assertLess(float(state.program_write_frequency[0, 0].detach()), 1.0)
        self.assertGreater(
            float(state.program_write_frequency[0, 1:].max().detach()),
            0.0,
        )
        self.assertIn("memory_allocation_write_frequency", output.aux.metrics)

    def test_invalid_memory_allocation_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    memory_allocation_type="random_walk",
                )
            )

    def test_invalid_memory_allocation_k_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    memory_allocation_k=0,
                )
            )

    def test_hierarchical_identity_memory_adds_stable_and_archival_tiers(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            state_update_type="gated",
            memory_tier_type="hierarchical",
            memory_read_type="program_memory",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)

        first = model(torch.tensor([[1, 2, 3, 4, 5]]))
        second = model(
            torch.tensor([[5, 4, 3, 2, 1]]),
            identity_states=first.identity_states,
        )
        state = second.identity_states[0]
        read_logits = model.memory_read_logits(
            torch.tensor([2]),
            second.identity_states,
        )

        self.assertIsNotNone(state.stable_program_memory)
        self.assertIsNotNone(state.archival_program_memory)
        self.assertEqual(state.stable_program_memory.shape, state.program_memory.shape)
        self.assertEqual(state.archival_program_memory.shape, state.program_memory.shape)
        self.assertEqual(read_logits.shape, (1, 32))
        self.assertIn("memory_tiers", second.aux.metrics)
        self.assertEqual(float(second.aux.metrics["memory_tiers"]), 3.0)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_pattern_completion_memory_read_uses_stored_engram_patterns(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            state_update_type="gated",
            routing_type="sparse_ensemble",
            routing_top_k=2,
            memory_read_type="pattern_completion",
            pattern_store_size=3,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 2, 3, 4]]))
        state = context.identity_states[0]

        self.assertIsNotNone(state.engram_patterns)
        self.assertIsNotNone(state.engram_values)
        self.assertIsNotNone(state.engram_mask)
        self.assertEqual(state.engram_patterns.shape, (1, 3, 4))
        self.assertEqual(state.engram_values.shape, (1, 3, 16))
        self.assertIn("pattern_completion_hit", context.aux.metrics)

        query = torch.tensor([[4, 3, 2, 1]])
        carried = model(query, identity_states=context.identity_states)
        blank_state = IdentityState(
            stability=state.stability,
            program_memory=state.program_memory,
            engram_patterns=torch.zeros_like(state.engram_patterns),
            engram_values=torch.zeros_like(state.engram_values),
            engram_mask=torch.zeros_like(state.engram_mask),
        )
        blank = model(query, identity_states=[blank_state])

        self.assertGreater(
            float(carried.aux.metrics["pattern_completion_hit"].detach()),
            0.0,
        )
        self.assertFalse(torch.allclose(carried.logits, blank.logits))

    def test_content_addressed_memory_stores_hidden_cue_value_pairs(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 7, 11, 4]]))
        state = context.identity_states[0]

        self.assertIsNotNone(state.content_cues)
        self.assertIsNotNone(state.content_values)
        self.assertIsNotNone(state.content_mask)
        self.assertEqual(state.content_cues.shape, (1, 4, 16))
        self.assertEqual(state.content_values.shape, (1, 4, 16))

        query = torch.tensor([[2, 7, 3, 5]])
        carried = model(query, identity_states=context.identity_states)
        blank_state = IdentityState(
            stability=state.stability,
            program_memory=state.program_memory,
            content_cues=torch.zeros_like(state.content_cues),
            content_values=torch.zeros_like(state.content_values),
            content_mask=torch.zeros_like(state.content_mask),
        )
        blank = model(query, identity_states=[blank_state])
        read_vector = model.memory_read_vector(
            torch.tensor([7]),
            context.identity_states,
        )

        self.assertIn("content_addressed_hit", carried.aux.metrics)
        self.assertGreater(
            float(carried.aux.metrics["content_addressed_hit"].detach()),
            0.0,
        )
        self.assertEqual(read_vector.shape, (1, 16))
        self.assertFalse(torch.allclose(carried.logits, blank.logits))

    def test_content_read_query_top_k_limits_content_memory_queries(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            content_read_query_top_k=2,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 7, 11, 4]]))

        output = model(
            torch.tensor([[2, 7, 3, 5, 6]]),
            identity_states=context.identity_states,
        )

        self.assertEqual(output.logits.shape, (1, 5, 32))
        self.assertEqual(
            float(output.aux.metrics["content_read_queries"].detach()),
            2.0,
        )
        self.assertAlmostEqual(
            float(output.aux.metrics["content_read_query_fraction"].detach()),
            2.0 / 5.0,
        )
        self.assertAlmostEqual(
            float(output.aux.metrics["content_read_skipped_fraction"].detach()),
            3.0 / 5.0,
        )

    def test_default_content_read_reports_full_query_count(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 7, 11, 4]]))

        output = model(
            torch.tensor([[2, 7, 3, 5, 6]]),
            identity_states=context.identity_states,
        )

        self.assertEqual(
            float(output.aux.metrics["content_read_queries"].detach()),
            5.0,
        )
        self.assertEqual(
            float(output.aux.metrics["content_read_query_fraction"].detach()),
            1.0,
        )

    def test_content_write_mask_limits_full_prefill_content_writes(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            content_reconsolidate=False,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        masked = model(
            torch.tensor([[1, 7, 11, 4]]),
            content_write_mask=torch.tensor([[True, False, True]]),
        )
        empty = model(
            torch.tensor([[1, 7, 11, 4]]),
            content_write_mask=torch.tensor([[False, False, False]]),
        )

        self.assertEqual(float(masked.identity_states[0].content_mask.sum()), 2.0)
        self.assertEqual(float(empty.identity_states[0].content_mask.sum()), 0.0)

    def test_masked_prefill_query_skip_write_policy_composes_sparse_prefill_and_decode_skip(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            content_reconsolidate=False,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context_tokens = torch.tensor([[1, 7, 11, 4]])
        write_mask = torch.tensor([[True, False, True]])

        explicit = model(
            context_tokens,
            content_write_mask=write_mask,
            update_content_memory=True,
        )
        policy = model(
            context_tokens,
            content_write_mask=write_mask,
            write_policy=ContentWritePolicy.MASKED_PREFILL_QUERY_SKIP,
        )

        self.assertTrue(
            torch.allclose(
                explicit.identity_states[0].content_cues,
                policy.identity_states[0].content_cues,
            )
        )
        self.assertTrue(
            torch.equal(
                explicit.identity_states[0].content_mask,
                policy.identity_states[0].content_mask,
            )
        )

        query = torch.tensor([[7]])
        skipped = model(
            query,
            identity_states=policy.identity_states,
            write_policy=ContentWritePolicy.MASKED_PREFILL_QUERY_SKIP,
        )
        explicit_skip = model(
            query,
            identity_states=policy.identity_states,
            update_content_memory=False,
        )

        self.assertTrue(
            torch.allclose(
                skipped.identity_states[0].content_cues,
                policy.identity_states[0].content_cues,
            )
        )
        self.assertTrue(
            torch.allclose(
                skipped.identity_states[0].content_values,
                policy.identity_states[0].content_values,
            )
        )
        self.assertTrue(
            torch.equal(
                skipped.identity_states[0].content_mask,
                policy.identity_states[0].content_mask,
            )
        )
        self.assertIsNotNone(skipped.aux.token_program_activations)
        self.assertEqual(skipped.aux.token_program_activations.shape[:2], (1, 1))
        self.assertTrue(torch.allclose(skipped.logits, explicit_skip.logits))

    def test_decode_state_skip_preserves_decode_logits_and_recurrent_state(self):
        torch.manual_seed(7)
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="content_addressed",
            memory_write_type="novelty_gated",
            memory_allocation_type="creb",
            memory_reconsolidate=True,
            content_store_size=4,
            content_reconsolidate=True,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 7, 11, 4]]), write_policy=ContentWritePolicy.DENSE)
        previous = context.identity_states
        query = torch.tensor([[7]])

        query_skip = model(
            query,
            identity_states=previous,
            write_policy=ContentWritePolicy.QUERY_SKIP,
        )
        state_skip = model(
            query,
            identity_states=previous,
            write_policy=ContentWritePolicy.DECODE_STATE_SKIP,
        )

        self.assertTrue(torch.allclose(query_skip.logits, state_skip.logits, atol=1e-6))
        before = previous[0]
        after = state_skip.identity_states[0]
        for name in (
            "stability",
            "program_memory",
            "stable_program_memory",
            "archival_program_memory",
            "program_age",
            "program_write_frequency",
            "engram_patterns",
            "engram_values",
            "content_cues",
            "content_values",
        ):
            expected = getattr(before, name)
            actual = getattr(after, name)
            if expected is None:
                self.assertIsNone(actual)
            else:
                self.assertTrue(torch.allclose(actual, expected), name)
        for name in ("engram_mask", "content_mask"):
            expected = getattr(before, name)
            actual = getattr(after, name)
            if expected is None:
                self.assertIsNone(actual)
            else:
                self.assertTrue(torch.equal(actual, expected), name)

    def test_invalid_content_write_policy_is_rejected(self):
        config = TACConfig(vocab_size=32, d_model=16, n_heads=4, n_layers=1)
        model = TACTransformerLM(config)

        with self.assertRaisesRegex(ValueError, "write_policy"):
            model(torch.tensor([[1, 2, 3]]), write_policy="surprise_me")

    def test_invalid_content_write_mask_shape_is_rejected(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            memory_read_type="content_addressed",
            content_store_size=4,
        )
        model = TACTransformerLM(config)

        with self.assertRaisesRegex(ValueError, "content_write_mask"):
            model(
                torch.tensor([[1, 7, 11, 4]]),
                content_write_mask=torch.tensor([[True, False]]),
            )

    def test_iterative_content_addressed_read_adds_second_lookup_gate(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=6,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            content_read_steps=2,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 7, 11, 4, 5, 6]]))
        query = torch.tensor([[2, 7, 3, 4, 5, 6]])

        output = model(query, identity_states=context.identity_states)

        self.assertIn("content_addressed_hit", output.aux.metrics)
        self.assertGreater(
            float(output.aux.metrics["content_addressed_hit"].detach()),
            0.0,
        )
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_confidence_gated_iterative_content_read_adds_no_gate_parameters(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=6,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            content_read_steps=2,
            content_read_gate_type="confidence",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 7, 11, 4, 5, 6]]))
        query = torch.tensor([[2, 7, 3, 4, 5, 6]])

        output = model(query, identity_states=context.identity_states)

        self.assertIn("content_addressed_hit", output.aux.metrics)
        self.assertIsNone(model.blocks[0].identity_field.content_read_blend_gate)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_confidence_margin_iterative_content_read_adds_no_gate_parameters(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=6,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            content_read_steps=2,
            content_read_gate_type="confidence_margin",
            content_read_confidence_margin=0.05,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 7, 11, 4, 5, 6]]))
        query = torch.tensor([[2, 7, 3, 4, 5, 6]])

        output = model(query, identity_states=context.identity_states)

        self.assertIn("content_addressed_hit", output.aux.metrics)
        self.assertIn("content_synthesis_gate", output.aux.metrics)
        self.assertIsNone(model.blocks[0].identity_field.content_read_blend_gate)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_cue_match_iterative_content_read_adds_no_gate_parameters(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=6,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            content_read_steps=2,
            content_read_gate_type="cue_match",
            content_read_cue_match_threshold=0.65,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 7, 11, 4, 5, 6]]))
        query = torch.tensor([[2, 7, 3, 4, 5, 6]])

        output = model(query, identity_states=context.identity_states)

        self.assertIn("content_addressed_hit", output.aux.metrics)
        self.assertIn("content_synthesis_gate", output.aux.metrics)
        self.assertIsNone(model.blocks[0].identity_field.content_read_blend_gate)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_synthesis_gated_iterative_content_read_reports_synthesis_use(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=6,
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_read_type="content_addressed",
            content_store_size=4,
            content_read_steps=2,
            content_read_gate_type="synthesis",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        context = model(torch.tensor([[1, 7, 11, 4, 5, 6]]))
        query = torch.tensor([[2, 7, 3, 4, 5, 6]])

        output = model(query, identity_states=context.identity_states)

        self.assertIn("content_addressed_hit", output.aux.metrics)
        self.assertIn("content_synthesis_gate", output.aux.metrics)
        self.assertIn("content_gate_entropy", output.aux.metrics)
        self.assertGreaterEqual(
            float(output.aux.metrics["content_synthesis_gate"].detach()),
            0.0,
        )
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_invalid_content_read_steps_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    content_read_steps=0,
                )
            )

    def test_invalid_content_read_query_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    content_read_query_top_k=0,
                )
            )

    def test_invalid_content_read_gate_type_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    content_read_steps=2,
                    content_read_gate_type="oracle",
                )
            )
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    content_read_steps=2,
                    content_read_gate_type="confidence_margin",
                    content_read_confidence_margin=-0.1,
                )
            )
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    content_read_steps=2,
                    content_read_gate_type="cue_match",
                    content_read_cue_match_threshold=-0.1,
                )
            )

    def test_invalid_memory_tier_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    memory_tier_type="fractal",
                )
            )

    def test_product_key_memory_lookup_adds_sparse_trainable_memory_layer(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            memory_lookup_type="product_key",
            memory_lookup_slots=12,
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)

        output = model(torch.tensor([[1, 2, 3, 4, 5]]))
        loss = output.logits.mean() + sum(output.aux.losses.values())
        loss.backward()

        identity = model.blocks[0].identity_field
        self.assertEqual(output.logits.shape, (1, 5, 32))
        self.assertIn("memory_lookup_slots", output.aux.metrics)
        self.assertEqual(float(output.aux.metrics["memory_lookup_slots"]), 12.0)
        self.assertIsNotNone(identity.memory_lookup_values.grad)
        self.assertGreater(float(identity.memory_lookup_values.grad.abs().sum()), 0)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_invalid_memory_lookup_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    memory_lookup_type="sql",
                )
            )

    def test_dual_stream_residual_adds_separate_content_and_identity_gates(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            residual_stream_type="dual_stream",
            program_compute_type="linear_expert",
        )
        model = TACTransformerLM(config)

        output = model(torch.tensor([[1, 2, 3, 4, 5]]))
        loss = output.logits.mean() + sum(output.aux.losses.values())
        loss.backward()
        block = model.blocks[0]

        self.assertEqual(output.logits.shape, (1, 5, 32))
        self.assertIsNotNone(block.content_stream_gate.weight.grad)
        self.assertIsNotNone(block.identity_stream_gate.weight.grad)
        self.assertEqual(float(output.aux.metrics["residual_streams"]), 2.0)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_invalid_residual_stream_option_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    residual_stream_type="braided",
                )
            )

    def test_multi_token_prediction_heads_add_trainable_future_objective(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=12,
            n_prediction_heads=3,
            multi_token_loss_weight=0.5,
        )
        model = TACTransformerLM(config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[2, 3, 4, 5, 6]])

        output = model(input_ids, labels=labels)
        loss = output.loss + output.aux.losses["multi_token"]
        loss.backward()

        self.assertEqual(len(output.multi_token_logits), 2)
        self.assertEqual(output.multi_token_logits[0].shape, (1, 5, 32))
        self.assertGreater(float(output.aux.losses["multi_token"].detach()), 0)
        self.assertIsNotNone(model.multi_token_heads[0].weight.grad)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_invalid_multi_token_prediction_head_count_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_prediction_heads=0,
                )
            )

    def test_best_tac_preset_matches_research_matrix_winner(self):
        config = best_tac_config(
            vocab_size=64,
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_programs=16,
            max_seq_len=16,
            beta=1.5,
            energy_budget=4.0,
        )
        training = best_chunked_memory_training_kwargs()
        model = TACTransformerLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        output = model(input_ids)

        self.assertEqual(config.n_kv_heads, 2)
        self.assertEqual(config.norm_type, "rmsnorm")
        self.assertEqual(config.mlp_type, "swiglu")
        self.assertEqual(config.position_type, "rope")
        self.assertEqual(config.program_compute_type, "linear_expert")
        self.assertEqual(config.routing_type, "base")
        self.assertEqual(config.state_update_type, "gated")
        self.assertEqual(config.memory_write_type, "novelty_gated")
        self.assertEqual(config.memory_tier_type, "flat")
        self.assertEqual(config.memory_lookup_type, "none")
        self.assertEqual(config.memory_read_type, "content_addressed")
        self.assertEqual(config.content_store_size, 8)
        self.assertEqual(config.content_read_steps, 2)
        self.assertEqual(config.content_read_gate_type, "synthesis")
        self.assertEqual(config.memory_adapter_type, "gated_residual")
        self.assertEqual(config.identity_attention_type, "identity_first")
        self.assertIsNone(config.attention_window_size)
        self.assertEqual(config.residual_stream_type, "single")
        self.assertEqual(config.sequence_mixer_type, "attention")
        self.assertEqual(config.state_mixer_kernel_size, 4)
        self.assertEqual(config.n_prediction_heads, 1)
        self.assertFalse(config.detach_identity_state)
        self.assertEqual(training["value_loss_weight"], 3.0)
        self.assertEqual(training["memory_read_loss_weight"], 3.0)
        self.assertEqual(training["memory_adapter_weight"], 6.0)
        self.assertEqual(output.logits.shape, (2, 8, 64))
        self.assertEqual(float(output.aux.metrics["routing_type"]), 2.0)
        self.assertIsNotNone(model.blocks[0].attention.identity_key_value)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_hybrid_sequence_mixer_adds_trainable_state_path(self):
        config = TACConfig(
            vocab_size=48,
            d_model=24,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            n_programs=8,
            max_seq_len=8,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            position_type="rope",
            program_compute_type="linear_expert",
            routing_type="hash",
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_read_type="program_memory",
            memory_adapter_type="gated_residual",
            sequence_mixer_type="hybrid",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        output = model(input_ids, labels=input_ids)
        output.loss.backward()

        self.assertEqual(output.logits.shape, (2, 8, config.vocab_size))
        self.assertEqual(float(output.aux.metrics["sequence_mixer_type"]), 3.0)
        self.assertIsNotNone(model.blocks[0].state_mixer.out_proj.weight.grad)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_alternating_sequence_mixer_replaces_attention_on_odd_layers(self):
        config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=3,
            n_programs=6,
            max_seq_len=8,
            sequence_mixer_type="alternating",
        )
        model = VanillaTransformerLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        output = model(input_ids)

        self.assertIsNotNone(model.blocks[0].attention)
        self.assertIsNone(model.blocks[1].attention)
        self.assertIsNotNone(model.blocks[1].state_mixer)
        self.assertEqual(output.logits.shape, (2, 8, config.vocab_size))
        self.assertEqual(
            estimate_vanilla_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_invalid_sequence_mixer_type_is_rejected(self):
        with self.assertRaises(ValueError):
            TACTransformerLM(
                TACConfig(
                    vocab_size=32,
                    d_model=16,
                    sequence_mixer_type="mystery",
                )
            )

    def test_recurrent_sequence_mixers_are_trainable_and_counted(self):
        for mixer_type, metric_id in {
            "selective_state": 5.0,
            "rwkv": 6.0,
            "xlstm": 7.0,
        }.items():
            with self.subTest(mixer_type=mixer_type):
                config = TACConfig(
                    vocab_size=32,
                    d_model=16,
                    n_heads=4,
                    n_layers=1,
                    n_programs=6,
                    max_seq_len=8,
                    sequence_mixer_type=mixer_type,
                )
                model = TACTransformerLM(config)
                input_ids = torch.randint(0, config.vocab_size, (2, 8))

                output = model(input_ids, labels=input_ids)
                output.loss.backward()

                self.assertEqual(output.logits.shape, (2, 8, config.vocab_size))
                self.assertEqual(
                    float(output.aux.metrics["sequence_mixer_type"]),
                    metric_id,
                )
                self.assertIsNotNone(model.blocks[0].state_mixer.out_proj.weight.grad)
                self.assertEqual(
                    estimate_tac_parameter_count(config),
                    count_parameters(model)["total"],
                )

    def test_chunked_recall_batcher_places_hidden_value_target_in_query_labels(self):
        batcher = ChunkedRecallBatcher(vocab_size=40, seq_len=8, seed=5)

        batch = batcher.next_batch(batch_size=3)

        self.assertEqual(batch.context_inputs.shape, (3, 8))
        self.assertEqual(batch.context_write_mask.shape, (3, 7))
        self.assertEqual(batch.query_inputs.shape, (3, 8))
        self.assertTrue(torch.equal(batch.query_labels[:, batch.value_label_index], batch.value_targets))
        self.assertTrue(torch.equal(batch.context_inputs[:, 1], batch.query_inputs[:, 1]))
        self.assertTrue(torch.equal(batch.context_inputs[:, 2], batch.value_targets))
        self.assertTrue(batch.context_write_mask[:, 1].all())
        self.assertEqual(int(batch.context_write_mask.sum().item()), 3)
        for row_index, value in enumerate(batch.value_targets.tolist()):
            self.assertNotIn(value, batch.query_inputs[row_index].tolist())

    def test_chunked_recall_batcher_supports_harder_task_variants(self):
        for variant in ["multi_key", "delayed_query", "noisy_key", "multi_hop"]:
            with self.subTest(variant=variant):
                batcher = ChunkedRecallBatcher(
                    vocab_size=48,
                    seq_len=12,
                    seed=5,
                    task_variant=variant,
                )

                batch = batcher.next_batch(batch_size=4)

                self.assertEqual(batch.context_inputs.shape, (4, 12))
                self.assertEqual(batch.context_write_mask.shape, (4, 11))
                self.assertEqual(batch.query_inputs.shape, (4, 12))
                self.assertTrue(
                    torch.equal(
                        batch.query_labels[:, batch.value_label_index],
                        batch.value_targets,
                    )
                )
                for row_index, value in enumerate(batch.value_targets.tolist()):
                    self.assertNotIn(value, batch.query_inputs[row_index].tolist())
                if variant == "multi_key":
                    self.assertGreaterEqual(batch.context_inputs.shape[1], 8)
                    context_keys = batch.context_inputs[:, 1:8:2]
                    query_keys = batch.query_inputs[:, 1]
                    for row_index, query_key in enumerate(query_keys.tolist()):
                        self.assertIn(query_key, context_keys[row_index].tolist())
                    self.assertTrue(batch.context_write_mask[:, 1].all())
                    self.assertTrue(batch.context_write_mask[:, 3].all())
                if variant == "delayed_query":
                    self.assertGreater(batch.value_label_index, 2)
                    self.assertTrue(batch.context_write_mask[:, 1].all())
                if variant == "noisy_key":
                    self.assertFalse(
                        torch.equal(
                            batch.context_inputs[:, 1],
                            batch.query_inputs[:, 1],
                        )
                    )
                    self.assertTrue(batch.context_write_mask[:, 1].all())
                if variant == "multi_hop":
                    self.assertTrue(
                        torch.equal(
                            batch.context_inputs[:, 2],
                            batch.context_inputs[:, 3],
                        )
                    )
                    self.assertTrue(batch.context_write_mask[:, 1].all())
                    self.assertTrue(batch.context_write_mask[:, 3].all())
                    self.assertEqual(int(batch.context_write_mask.sum().item()), 8)

    def test_chunked_memory_trainer_and_benchmark_return_scorecard(self):
        config = TACConfig(
            vocab_size=40,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            state_update_type="gated",
        )
        model = TACTransformerLM(config)
        batcher = ChunkedRecallBatcher(vocab_size=40, seq_len=8, seed=17)

        train_metrics = train_chunked_memory(
            model,
            batcher,
            steps=1,
            batch_size=2,
            learning_rate=1e-3,
            device="cpu",
        )
        result = benchmark_chunked_memory(
            config,
            steps=1,
            batch_size=2,
            eval_batches=1,
            eval_batch_size=2,
            learning_rate=1e-3,
            seed=17,
            device="cpu",
            task_variant="multi_key",
        )
        decoded = json.loads(json.dumps(result))

        self.assertGreater(train_metrics["loss"], 0)
        self.assertIn("decision", result)
        self.assertIn("value_accuracy_delta", result["tac"]["chunked_probe"])
        self.assertIn("carry", result["tac"]["chunked_probe"])
        self.assertIn("reset", result["tac"]["chunked_probe"])
        self.assertIn("shuffled", result["tac"]["chunked_probe"])
        self.assertIn(
            "content_read_query_fraction",
            result["tac"]["chunked_probe"]["carry"],
        )
        self.assertIn(
            "content_read_skipped_fraction",
            result["tac"]["chunked_probe"]["carry"],
        )
        self.assertIn(result["decision"]["status"], {"effective", "inconclusive"})
        self.assertEqual(decoded["steps"], 1)
        self.assertEqual(decoded["task_variant"], "multi_key")

    def test_data_energy_efficiency_benchmark_reports_budget_curve(self):
        config = TACConfig(
            vocab_size=40,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            position_type="rope",
            program_compute_type="linear_expert",
            state_update_type="gated",
            memory_write_type="novelty_gated",
            memory_read_type="program_memory",
            memory_adapter_type="gated_residual",
            detach_identity_state=False,
        )

        result = benchmark_data_energy_efficiency(
            config,
            budgets=[1],
            batch_size=2,
            learning_rate=1e-3,
            eval_batches=1,
            eval_batch_size=2,
            seed=91,
            value_loss_weight=1.0,
            memory_read_loss_weight=1.0,
            memory_adapter_weight=2.0,
            match_baseline_parameters=True,
        )

        self.assertEqual(result["budgets"], [1])
        self.assertEqual(len(result["budget_results"]), 1)
        budget = result["budget_results"][0]
        self.assertIn("data_efficiency", budget)
        self.assertIn("energy_efficiency", budget)
        self.assertGreater(budget["data_efficiency"]["train_tokens"], 0)
        self.assertIn("tokens_per_second_ratio", budget["energy_efficiency"])

    def test_chunked_memory_value_loss_weight_changes_reported_objective(self):
        config = TACConfig(
            vocab_size=40,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            state_update_type="gated",
        )

        torch.manual_seed(29)
        unweighted_model = TACTransformerLM(config)
        unweighted = train_chunked_memory(
            unweighted_model,
            ChunkedRecallBatcher(vocab_size=40, seq_len=8, seed=19),
            steps=1,
            batch_size=2,
            learning_rate=1e-3,
            value_loss_weight=0.0,
            device="cpu",
        )

        torch.manual_seed(29)
        weighted_model = TACTransformerLM(config)
        weighted = train_chunked_memory(
            weighted_model,
            ChunkedRecallBatcher(vocab_size=40, seq_len=8, seed=19),
            steps=1,
            batch_size=2,
            learning_rate=1e-3,
            value_loss_weight=3.0,
            device="cpu",
        )

        self.assertIn("value_loss", weighted)
        self.assertGreater(weighted["value_loss"], 0)
        self.assertGreater(weighted["loss"], unweighted["loss"])

    def test_direct_memory_readout_returns_logits_and_trains_context_memory(self):
        config = TACConfig(
            vocab_size=40,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="program_memory",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        batch = ChunkedRecallBatcher(vocab_size=40, seq_len=8, seed=31).next_batch(
            batch_size=2
        )

        context = model(batch.context_inputs, labels=batch.context_labels)
        read_logits = model.memory_read_logits(
            batch.query_inputs[:, 1],
            context.identity_states,
        )
        loss = torch.nn.functional.cross_entropy(read_logits, batch.value_targets)
        loss.backward()

        identity = model.blocks[0].identity_field
        self.assertEqual(read_logits.shape, (2, 40))
        self.assertIsNotNone(identity.program_update.weight.grad)
        self.assertGreater(float(identity.program_update.weight.grad.abs().sum()), 0)

    def test_content_addressed_memory_readout_can_decode_without_lm_head(self):
        config = TACConfig(
            vocab_size=16,
            d_model=8,
            n_heads=2,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="content_addressed",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.token_embedding.weight.zero_()
            model.token_embedding.weight[5, 0] = 1.0
            model.token_embedding.weight[9, 0] = 1.0
            model.token_embedding.weight[9, 1] = 1.0
        state = IdentityState(
            stability=torch.ones(1, 4),
            program_memory=torch.zeros(1, 4, 8),
            content_cues=model.token_embedding(torch.tensor([[5]])).detach(),
            content_values=model.token_embedding(torch.tensor([[9]])).detach(),
            content_mask=torch.ones(1, 1),
        )

        read_logits = model.memory_read_logits(
            torch.tensor([5]),
            [state],
        )

        self.assertEqual(read_logits.shape, (1, 16))
        self.assertGreater(
            float(read_logits[0, 9].detach()),
            float(read_logits[0, 4].detach()),
        )
        self.assertEqual(int(read_logits.argmax(dim=-1).item()), 9)

    def test_cue_match_memory_readout_follows_stored_token_chain(self):
        config = TACConfig(
            vocab_size=40,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="content_addressed",
            content_read_steps=2,
            content_read_gate_type="cue_match",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        batch = ChunkedRecallBatcher(
            vocab_size=40,
            seq_len=8,
            seed=91,
            task_variant="multi_hop",
        ).next_batch(batch_size=4)

        context = model(
            batch.context_inputs,
            labels=batch.context_labels,
            content_write_mask=batch.context_write_mask,
        )
        read_logits = model.memory_read_logits(
            batch.query_inputs[:, 1],
            context.identity_states,
        )

        self.assertIsNotNone(context.identity_states[-1].content_cue_token_ids)
        self.assertIsNotNone(context.identity_states[-1].content_value_token_ids)
        self.assertTrue(torch.equal(read_logits.argmax(dim=-1), batch.value_targets))

    def test_chunked_memory_reports_direct_memory_read_metrics(self):
        config = TACConfig(
            vocab_size=40,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="program_memory",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)

        metrics = train_chunked_memory(
            model,
            ChunkedRecallBatcher(vocab_size=40, seq_len=8, seed=41),
            steps=1,
            batch_size=2,
            learning_rate=1e-3,
            memory_read_loss_weight=2.0,
            device="cpu",
        )

        self.assertIn("memory_read_loss", metrics)
        self.assertIn("memory_read_accuracy", metrics)
        self.assertGreater(metrics["memory_read_loss"], 0)

    def test_memory_read_injection_changes_value_token_prediction(self):
        batcher = ChunkedRecallBatcher(vocab_size=40, seq_len=8, seed=51)
        batch = batcher.next_batch(batch_size=2)
        query_logits = torch.zeros(2, 8, 40)
        query_logits[:, batch.value_label_index, 5] = 2.0
        memory_logits = torch.zeros(2, 40)
        memory_logits[0, int(batch.value_targets[0])] = 8.0
        memory_logits[1, int(batch.value_targets[1])] = 8.0

        injected = apply_memory_read_logits(
            query_logits,
            memory_logits,
            value_label_index=batch.value_label_index,
            weight=1.0,
        )

        predictions = injected[:, batch.value_label_index, :].argmax(dim=-1)
        self.assertTrue(torch.equal(predictions, batch.value_targets))

    def test_residual_memory_adapter_returns_logits_and_trains(self):
        config = TACConfig(
            vocab_size=40,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="program_memory",
            memory_adapter_type="residual",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        batch = ChunkedRecallBatcher(vocab_size=40, seq_len=8, seed=61).next_batch(
            batch_size=2
        )
        context = model(batch.context_inputs, labels=batch.context_labels)
        query = model(
            batch.query_inputs,
            labels=batch.query_labels,
            identity_states=context.identity_states,
        )
        memory_vector = model.memory_read_vector(
            batch.query_inputs[:, 1],
            context.identity_states,
        )
        adapted_logits = model.memory_adapted_logits(
            query.hidden_states,
            memory_vector,
            value_label_index=batch.value_label_index,
            weight=1.0,
        )
        loss = torch.nn.functional.cross_entropy(
            adapted_logits[:, batch.value_label_index, :],
            batch.value_targets,
        )
        loss.backward()

        self.assertEqual(adapted_logits.shape, query.logits.shape)
        self.assertIsNotNone(model.memory_adapter.weight.grad)
        self.assertGreater(float(model.memory_adapter.weight.grad.abs().sum()), 0)

    def test_gated_residual_memory_adapter_returns_logits_and_trains(self):
        config = TACConfig(
            vocab_size=40,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=8,
            state_update_type="gated",
            memory_read_type="program_memory",
            memory_adapter_type="gated_residual",
            detach_identity_state=False,
        )
        model = TACTransformerLM(config)
        batch = ChunkedRecallBatcher(vocab_size=40, seq_len=8, seed=71).next_batch(
            batch_size=2
        )
        context = model(batch.context_inputs, labels=batch.context_labels)
        query = model(
            batch.query_inputs,
            labels=batch.query_labels,
            identity_states=context.identity_states,
        )
        memory_vector = model.memory_read_vector(
            batch.query_inputs[:, 1],
            context.identity_states,
        )
        adapted_logits = model.memory_adapted_logits(
            query.hidden_states,
            memory_vector,
            value_label_index=batch.value_label_index,
            weight=1.5,
        )
        loss = torch.nn.functional.cross_entropy(
            adapted_logits[:, batch.value_label_index, :],
            batch.value_targets,
        )
        loss.backward()

        self.assertEqual(adapted_logits.shape, query.logits.shape)
        self.assertIsNotNone(model.memory_adapter[0].weight.grad)
        self.assertIsNotNone(model.memory_adapter_gate.weight.grad)
        self.assertGreater(float(model.memory_adapter[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.memory_adapter_gate.weight.grad.abs().sum()), 0)
        self.assertEqual(
            estimate_tac_parameter_count(config),
            count_parameters(model)["total"],
        )

    def test_vanilla_baseline_matches_language_model_contract(self):
        model = VanillaTransformerLM(self.config)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[2, 3, 4, 5, 6]])

        output = model(input_ids, labels=labels)
        counts = count_parameters(model)

        self.assertEqual(output.logits.shape, (1, 5, 32))
        self.assertIsNotNone(output.loss)
        self.assertEqual(output.identity_states, [])
        self.assertEqual(counts["identity_field"], 0)
        self.assertGreater(counts["total"], 0)

    def test_parameter_estimators_match_model_counts(self):
        tac_count = count_parameters(TACTransformerLM(self.config))["total"]
        vanilla_count = count_parameters(VanillaTransformerLM(self.config))["total"]
        rich_identity_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=64,
            max_seq_len=12,
        )
        authority_config = TACConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            max_seq_len=12,
            routing_type="authority_gated",
            routing_top_k=2,
        )
        rich_tac_count = estimate_tac_parameter_count(rich_identity_config)
        rich_vanilla_count = estimate_vanilla_parameter_count(rich_identity_config)
        matched = parameter_matched_baseline_config(rich_identity_config)
        matched_count = estimate_vanilla_parameter_count(matched)

        self.assertEqual(estimate_tac_parameter_count(self.config), tac_count)
        self.assertEqual(estimate_vanilla_parameter_count(self.config), vanilla_count)
        self.assertEqual(
            estimate_tac_parameter_count(authority_config),
            count_parameters(TACTransformerLM(authority_config))["total"],
        )
        self.assertLess(
            abs(rich_tac_count - matched_count),
            abs(rich_tac_count - rich_vanilla_count),
        )

    def test_vanilla_baseline_same_backbone_preserves_comparison_shape(self):
        args = train_vanilla_baseline.parse_args(
            [
                "--scale",
                "smoke",
                "--baseline-mode",
                "same_backbone",
                "--steps",
                "1",
            ]
        )
        scale = train_vanilla_baseline.resolved_scale(args)
        comparison, baseline = train_vanilla_baseline.build_vanilla_baseline_config(
            args,
            scale,
        )

        self.assertEqual(baseline.d_model, comparison.d_model)
        self.assertEqual(baseline.n_layers, comparison.n_layers)
        self.assertEqual(baseline.max_seq_len, comparison.max_seq_len)
        self.assertEqual(baseline.vocab_size, comparison.vocab_size)

    def test_vanilla_baseline_parameter_matched_moves_toward_tac_budget(self):
        args = train_vanilla_baseline.parse_args(
            [
                "--scale",
                "smoke",
                "--baseline-mode",
                "parameter_matched",
                "--steps",
                "1",
            ]
        )
        scale = train_vanilla_baseline.resolved_scale(args)
        comparison, baseline = train_vanilla_baseline.build_vanilla_baseline_config(
            args,
            scale,
        )
        same_backbone_gap = abs(
            estimate_tac_parameter_count(comparison)
            - estimate_vanilla_parameter_count(comparison)
        )
        matched_gap = abs(
            estimate_tac_parameter_count(comparison)
            - estimate_vanilla_parameter_count(baseline)
        )

        self.assertLessEqual(matched_gap, same_backbone_gap)
        self.assertEqual(baseline.max_seq_len, comparison.max_seq_len)

    def test_authority_gated_is_in_promotion_benchmark_matrices(self):
        harder_variant = benchmark_harder_research_matrix.VARIANTS["authority_gated"]
        specialization_variant = (
            benchmark_program_specialization_objectives.VARIANTS["authority_gated"]
        )

        self.assertEqual(harder_variant["routing_type"], "authority_gated")
        self.assertEqual(harder_variant["routing_top_k"], 2)
        self.assertEqual(
            specialization_variant["overrides"]["routing_type"],
            "authority_gated",
        )
        self.assertEqual(specialization_variant["overrides"]["routing_top_k"], 2)

    def test_kaggle_default_model_is_150m_parameters(self):
        args = parse_args([])
        config = TACConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_programs=args.n_programs,
            max_seq_len=args.seq_len,
            beta=args.beta,
            energy_budget=args.energy_budget,
        )
        counts = count_parameters(TACTransformerLM(config))

        self.assertEqual(counts["total"], 150_002_688)
        self.assertEqual(counts["trainable"], 150_002_688)
        self.assertEqual(counts["identity_field"], 10_630_656)

    def test_best_tac_agentic_kaggle_config_uses_winning_architecture(self):
        args = train_best_tac_agentic.parse_args(["--scale", "smoke"])
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.vocab_size, 512)
        self.assertEqual(config.norm_type, "rmsnorm")
        self.assertEqual(config.mlp_type, "swiglu")
        self.assertEqual(config.position_type, "rope")
        self.assertEqual(config.program_compute_type, "linear_expert")
        self.assertEqual(config.routing_type, "base")
        self.assertEqual(config.memory_read_type, "content_addressed")
        self.assertEqual(config.content_store_size, 8)
        self.assertEqual(config.content_read_steps, 2)
        self.assertEqual(config.content_read_gate_type, "synthesis")
        self.assertEqual(config.memory_separation_weight, 0.01)
        self.assertEqual(config.content_cue_separation_weight, 0.005)
        self.assertEqual(config.content_gate_entropy_weight, 0.005)
        self.assertTrue(config.content_reconsolidate)
        self.assertEqual(config.content_reconsolidate_rate, 0.1)
        self.assertEqual(config.identity_attention_type, "identity_first")
        self.assertEqual(config.memory_adapter_type, "gated_residual")
        self.assertFalse(config.detach_identity_state)

    def test_best_tac_agentic_accepts_specialization_training_knobs(self):
        args = train_best_tac_agentic.parse_args(
            [
                "--scale",
                "smoke",
                "--program-compute-type",
                "low_rank_linear_expert",
                "--program-expert-rank",
                "5",
                "--mlp-ratio",
                "7",
                "--routing-type",
                "base_semantic",
                "--routing-top-k",
                "2",
                "--routing-load-balance-weight",
                "0.05",
                "--semantic-route-allowed-programs",
                "2",
                "4",
                "6",
                "8",
                "--semantic-route-suppressed-programs",
                "0",
                "15",
                "--category-route-weight",
                "0.1",
            ]
        )
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.program_compute_type, "low_rank_linear_expert")
        self.assertEqual(config.program_expert_rank, 5)
        self.assertEqual(config.mlp_ratio, 7)
        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.routing_top_k, 2)
        self.assertAlmostEqual(config.routing_load_balance_weight, 0.05)
        self.assertEqual(config.semantic_route_allowed_programs, (2, 4, 6, 8))
        self.assertEqual(config.semantic_route_suppressed_programs, (0, 15))
        self.assertAlmostEqual(args.category_route_weight, 0.1)

    def test_jsonl_labeled_text_batcher_returns_domain_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            rows = [
                {"domain": "tool_choice", "text": "choose shell tool"},
                {"domain": "repair_after_failure", "text": "repair failing test"},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            batcher = JsonlLabeledTextBatcher(
                path,
                seq_len=12,
                vocab_size=512,
                seed=7,
            )
            input_ids, labels, category_ids = batcher.next_batch(4)

        self.assertEqual(input_ids.shape, (4, 12))
        self.assertEqual(labels.shape, (4, 12))
        self.assertEqual(category_ids.shape, (4,))
        self.assertEqual(sorted(batcher.categories), ["repair_after_failure", "tool_choice"])
        self.assertTrue(torch.all(category_ids >= 0))

    def test_best_tac_agentic_discovers_local_hard_corpus_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hard_dir = tmp_path / "runs" / "prepared_corpus_agentic_hard"
            hard_dir.mkdir(parents=True)
            (hard_dir / "train.prepared.jsonl").write_text('{"text":"hard"}\n', encoding="utf-8")
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                discovered = train_best_tac_agentic.discover_prepared_jsonl(
                    "train.prepared.jsonl"
                )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(discovered, Path("runs") / "prepared_corpus_agentic_hard" / "train.prepared.jsonl")

    def test_agentic_training_bundle_instructions_use_run4_semantic_selected_mi(self):
        instructions = make_agentic_training_bundle._instructions()

        self.assertIn("tac-hard-agentic-corpus", instructions)
        self.assertIn("identity-first TAC", instructions)
        self.assertIn("--steps 20000", instructions)
        self.assertIn("best_tac_agentic_run4_semantic_selected_mi", instructions)
        self.assertIn("--routing-type base_semantic", instructions)
        self.assertIn("--routing-top-k 2", instructions)
        self.assertIn("--routing-load-balance-weight 0.05", instructions)
        self.assertIn("--category-route-weight 0.5", instructions)
        self.assertIn("--category-route-objective selected_mi", instructions)
        self.assertIn("--precision fp32", instructions)
        self.assertIn("--min-healthy-gradient-norm 1e-12", instructions)
        self.assertIn("--fail-on-unhealthy-optimization", instructions)
        self.assertIn("--specialization-checkpoints 2000 5000 10000 20000", instructions)
        self.assertIn("--specialization-checkpoint-max-records-per-category 16", instructions)
        self.assertIn("--analyze-specialization-at-end", instructions)
        self.assertIn("--specialization-max-records-per-category 64", instructions)
        self.assertIn("analyze_program_specialization.py", instructions)
        self.assertIn("inspect_identity_memory.py", instructions)
        self.assertIn("evaluate_checkpoint_harder_matrix.py", instructions)
        self.assertIn("without fixed program IDs", instructions)
        self.assertNotIn("--auto-resume", instructions)
        self.assertIn("glob(\"**/best-tac-agentic-training-bundle.zip\")", instructions)
        self.assertIn("glob(\"**/kaggle/train_best_tac_agentic.py\")", instructions)
        self.assertIn("searches `/kaggle/input` recursively", instructions)
        self.assertNotIn("--train-jsonl /kaggle/input/tac-hard-agentic-corpus/train.prepared.jsonl", instructions)
        self.assertNotIn("--eval-jsonl /kaggle/input/tac-hard-agentic-corpus/eval.prepared.jsonl", instructions)
        self.assertNotIn("tac-1b-agentic-corpus", instructions)

    def test_agentic_training_bundle_imports_after_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "bundle"
            build_result = subprocess.run(
                [
                    sys.executable,
                    str(Path(make_agentic_training_bundle.__file__)),
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(build_result.returncode, 0, build_result.stderr)
            zip_path = output_dir / "best-tac-agentic-training-bundle.zip"
            extract_dir = tmp_path / "extract"
            with ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)

            import_result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import tac_transformer; print(tac_transformer.run5_capability_config(vocab_size=256).n_programs)",
                ],
                cwd=extract_dir,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(import_result.returncode, 0, import_result.stderr)
        self.assertIn("12", import_result.stdout)

    def test_best_tac_agentic_auto_resume_prefers_output_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "out"
            input_dir = tmp_path / "input" / "previous-output"
            output_dir.mkdir()
            input_dir.mkdir(parents=True)
            output_last = output_dir / "last.pt"
            input_last = input_dir / "last.pt"
            output_last.write_bytes(b"output")
            input_last.write_bytes(b"input")

            resolved = train_best_tac_agentic.resolve_resume_checkpoint(
                None,
                output_dir,
                auto_resume=True,
                input_roots=[tmp_path / "input"],
            )

        self.assertEqual(resolved, output_last)

    def test_best_tac_agentic_auto_resume_falls_back_to_attached_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "out"
            input_dir = tmp_path / "input" / "previous-output"
            output_dir.mkdir()
            input_dir.mkdir(parents=True)
            input_last = input_dir / "last.pt"
            input_last.write_bytes(b"input")

            resolved = train_best_tac_agentic.resolve_resume_checkpoint(
                None,
                output_dir,
                auto_resume=True,
                input_roots=[tmp_path / "input"],
            )

        self.assertEqual(resolved, input_last)

    def test_best_tac_agentic_missing_explicit_resume_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                train_best_tac_agentic.resolve_resume_checkpoint(
                    tmp_path / "missing.pt",
                    tmp_path / "out",
                    auto_resume=False,
                )

    def test_best_tac_agentic_kaggle_smoke_train_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.prepared.jsonl"
            eval_path = tmp_path / "eval.prepared.jsonl"
            text = serialize_record(
                {
                    "record_id": "agent_1",
                    "prompt": "Use the calculator tool, verify, and answer.",
                    "plan": [{"tool": "calculator", "args": {"expr": "2+2"}}],
                    "tool_results": [{"tool": "calculator", "result": "4"}],
                    "target_plan": "call calculator; verify result; answer",
                    "final_answer": "4",
                    "success": True,
                }
            )
            rows = "\n".join(json.dumps({"text": text}) for _ in range(4)) + "\n"
            train_path.write_text(rows, encoding="utf-8")
            eval_path.write_text(rows, encoding="utf-8")

            train_best_tac_agentic.main(
                [
                    "--scale",
                    "smoke",
                    "--d-model",
                    "32",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--n-programs",
                    "4",
                    "--seq-len",
                    "16",
                    "--batch-size",
                    "1",
                    "--steps",
                    "2",
                    "--eval-every",
                    "1",
                    "--eval-batches",
                    "1",
                    "--checkpoint-every",
                    "1",
                    "--train-jsonl",
                    str(train_path),
                    "--eval-jsonl",
                    str(eval_path),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--device",
                    "cpu",
                    "--precision",
                    "fp32",
                ]
            )

            self.assertTrue((tmp_path / "out" / "last.pt").exists())
            self.assertTrue((tmp_path / "out" / "run_manifest.json").exists())
            self.assertTrue((tmp_path / "out" / "final_summary.json").exists())
            manifest = json.loads(
                (tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["train_records"], 4)
            self.assertEqual(manifest["eval_records"], 4)
            self.assertGreater(manifest["tokens_per_optimizer_step"], 0)
            summary = json.loads(
                (tmp_path / "out" / "final_summary.json").read_text(encoding="utf-8")
            )
            latest = summary["latest_metrics"]
            self.assertIn("gradient_norm", latest)
            self.assertIn("grad_scaler_scale", latest)
            self.assertIn("tokens_seen", latest)
            self.assertIn("sequences_seen", latest)
            self.assertIn("epoch_equivalent", latest)
            self.assertIn("aux_loss_separation", latest)
            self.assertIn("weighted_aux_loss_separation", latest)
            self.assertIn("metric_content_gate_entropy", latest)
            self.assertIn("metric_program_ortho", latest)
            self.assertIn("metric_routing_load_std", latest)
            self.assertIn("eval", latest)
            self.assertIn("metric_content_gate_entropy", latest["eval"])
            self.assertIn("aux_loss_separation", latest["eval"])
            self.assertIn("metric_program_ortho", latest["eval"])

    def test_best_tac_agentic_category_route_smoke_logs_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.prepared.jsonl"
            eval_path = tmp_path / "eval.prepared.jsonl"
            rows = [
                {"domain": "tool_choice", "text": "choose calculator and verify"},
                {"domain": "repair_after_failure", "text": "repair failing test then verify"},
                {"domain": "tool_choice", "text": "choose shell for evidence"},
                {"domain": "repair_after_failure", "text": "diagnose error and patch"},
            ]
            payload = "\n".join(json.dumps(row) for row in rows) + "\n"
            train_path.write_text(payload, encoding="utf-8")
            eval_path.write_text(payload, encoding="utf-8")

            train_best_tac_agentic.main(
                [
                    "--scale",
                    "smoke",
                    "--d-model",
                    "32",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--n-programs",
                    "4",
                    "--seq-len",
                    "16",
                    "--batch-size",
                    "1",
                    "--steps",
                    "2",
                    "--eval-every",
                    "1",
                    "--eval-batches",
                    "1",
                    "--checkpoint-every",
                    "1",
                    "--train-jsonl",
                    str(train_path),
                    "--eval-jsonl",
                    str(eval_path),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--device",
                    "cpu",
                    "--precision",
                    "fp32",
                    "--routing-type",
                    "base_semantic",
                    "--routing-top-k",
                    "2",
                    "--routing-load-balance-weight",
                    "0.05",
                    "--category-route-weight",
                    "0.1",
                    "--category-route-objective",
                    "selected_mi",
                ]
            )

            manifest = json.loads(
                (tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (tmp_path / "out" / "final_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["category_route_weight"], 0.1)
        self.assertEqual(manifest["category_route_objective"], "selected_mi")
        self.assertEqual(
            manifest["category_route_categories"],
            ["repair_after_failure", "tool_choice"],
        )
        latest = summary["latest_metrics"]
        self.assertIn("category_route_loss", latest)
        self.assertIn("weighted_category_route_loss", latest)
        self.assertIsInstance(latest["category_route_loss"], float)

    def test_best_tac_agentic_runs_specialization_gate_at_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.prepared.jsonl"
            eval_path = tmp_path / "eval.prepared.jsonl"
            specialization_path = tmp_path / "hard_agentic_eval.generated.jsonl"
            base_text = serialize_record(
                {
                    "record_id": "agent_1",
                    "prompt": "Choose the calculator tool, verify, and answer.",
                    "plan": [{"tool": "calculator", "args": {"expr": "3+4"}}],
                    "tool_results": [{"tool": "calculator", "result": "7"}],
                    "target_plan": "call calculator; verify result; answer",
                    "final_answer": "7",
                    "success": True,
                }
            )
            rows = "\n".join(json.dumps({"text": base_text}) for _ in range(4)) + "\n"
            train_path.write_text(rows, encoding="utf-8")
            eval_path.write_text(rows, encoding="utf-8")
            specialization_rows = [
                {
                    "record_id": "tool_1",
                    "domain": "tool_choice",
                    "text": '<record type="hard_tool_choice"><goal>choose calculator</record>',
                },
                {
                    "record_id": "repair_1",
                    "domain": "repair_after_failure",
                    "text": '<record type="hard_repair_after_failure"><goal>repair failing test</record>',
                },
            ]
            specialization_path.write_text(
                "\n".join(json.dumps(row) for row in specialization_rows) + "\n",
                encoding="utf-8",
            )

            train_best_tac_agentic.main(
                [
                    "--scale",
                    "smoke",
                    "--d-model",
                    "32",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--n-programs",
                    "4",
                    "--seq-len",
                    "16",
                    "--batch-size",
                    "1",
                    "--steps",
                    "2",
                    "--eval-every",
                    "1",
                    "--eval-batches",
                    "1",
                    "--checkpoint-every",
                    "1",
                    "--train-jsonl",
                    str(train_path),
                    "--eval-jsonl",
                    str(eval_path),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--device",
                    "cpu",
                    "--precision",
                    "fp32",
                    "--analyze-specialization-at-end",
                    "--specialization-jsonl",
                    str(specialization_path),
                    "--specialization-max-records-per-category",
                    "1",
                    "--specialization-knockout-programs",
                    "0",
                    "1",
                    "--specialization-device",
                    "cpu",
                ]
            )

            report_path = tmp_path / "out" / "specialization" / "program_specialization.json"
            csv_path = tmp_path / "out" / "specialization" / "program_attribution.csv"
            self.assertTrue(report_path.exists())
            self.assertTrue(csv_path.exists())
            summary = json.loads(
                (tmp_path / "out" / "final_summary.json").read_text(encoding="utf-8")
            )
            analysis = summary["specialization_analysis"]
            self.assertEqual(analysis["records"], 2)
            self.assertEqual(analysis["report"], str(report_path))
            self.assertEqual(analysis["attribution_csv"], str(csv_path))
            self.assertIn("mi_bits", analysis)
            self.assertIn("top_ablation_loss_deltas", analysis)

    def test_best_tac_agentic_skips_end_specialization_on_time_stop(self):
        args = type(
            "Args",
            (),
            {
                "analyze_specialization_at_end": True,
                "skip_end_specialization_on_time_stop": True,
            },
        )()

        self.assertFalse(
            train_best_tac_agentic.should_run_end_specialization(
                args,
                {"stopped_for_time": True},
            )
        )
        self.assertTrue(
            train_best_tac_agentic.should_run_end_specialization(
                args,
                {"stopped_for_time": False},
            )
        )

    def test_best_tac_agentic_loads_existing_specialization_checkpoint_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "summaries.jsonl"
            rows = [
                {"label": "step_2000", "checkpoint_step": 2000},
                {"label": "step_5000", "checkpoint_step": 5000},
            ]
            summary_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            loaded = train_best_tac_agentic.load_specialization_checkpoint_summaries(
                summary_path
            )

        self.assertEqual([row["checkpoint_step"] for row in loaded], [2000, 5000])

    def test_best_tac_agentic_runs_periodic_specialization_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.prepared.jsonl"
            eval_path = tmp_path / "eval.prepared.jsonl"
            specialization_path = tmp_path / "hard_agentic_eval.generated.jsonl"
            rows = [
                {"domain": "tool_choice", "text": "choose calculator and verify"},
                {"domain": "repair_after_failure", "text": "repair failing test"},
                {"domain": "tool_choice", "text": "choose shell for evidence"},
                {"domain": "repair_after_failure", "text": "diagnose and patch"},
            ]
            payload = "\n".join(json.dumps(row) for row in rows) + "\n"
            train_path.write_text(payload, encoding="utf-8")
            eval_path.write_text(payload, encoding="utf-8")
            specialization_path.write_text(payload, encoding="utf-8")

            train_best_tac_agentic.main(
                [
                    "--scale",
                    "smoke",
                    "--d-model",
                    "32",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--n-programs",
                    "4",
                    "--seq-len",
                    "16",
                    "--batch-size",
                    "1",
                    "--steps",
                    "2",
                    "--eval-every",
                    "1",
                    "--eval-batches",
                    "1",
                    "--checkpoint-every",
                    "1",
                    "--train-jsonl",
                    str(train_path),
                    "--eval-jsonl",
                    str(eval_path),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--device",
                    "cpu",
                    "--precision",
                    "fp32",
                    "--specialization-jsonl",
                    str(specialization_path),
                    "--specialization-checkpoints",
                    "1",
                    "--specialization-checkpoint-max-records-per-category",
                    "1",
                    "--specialization-device",
                    "cpu",
                ]
            )

            checkpoint_dir = (
                tmp_path
                / "out"
                / "specialization_checkpoints"
                / "step_000001"
            )
            self.assertTrue((checkpoint_dir / "checkpoint.pt").exists())
            self.assertTrue((checkpoint_dir / "program_specialization.json").exists())
            self.assertTrue((checkpoint_dir / "program_attribution.csv").exists())
            summary_path = tmp_path / "out" / "specialization_checkpoints" / "summaries.jsonl"
            self.assertTrue(summary_path.exists())
            summary = json.loads(
                (tmp_path / "out" / "final_summary.json").read_text(encoding="utf-8")
            )

        checkpoints = summary["specialization_checkpoints"]
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["checkpoint_step"], 1)
        self.assertFalse(checkpoints[0]["run_knockouts"])
        self.assertIn("mi_bits", checkpoints[0])

    def test_identity_memory_inspector_decodes_checkpoint_content_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"
            config = TACConfig(
                vocab_size=512,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=16,
                state_update_type="gated",
                memory_write_type="novelty_gated",
                memory_read_type="content_addressed",
                content_store_size=4,
                content_read_steps=2,
                content_read_gate_type="synthesis",
                identity_attention_type="identity_first",
                memory_adapter_type="gated_residual",
                detach_identity_state=False,
            )
            model = TACTransformerLM(config)
            torch.save(
                {
                    "step": 3,
                    "best_eval_loss": 1.25,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metrics": {"loss": 1.25},
                },
                checkpoint_path,
            )

            report = inspect_identity_memory.inspect_checkpoint_memory(
                checkpoint_path,
                prompt="Use calculator, verify result, then answer.",
                max_slots=3,
                top_k=2,
                device="cpu",
            )

            self.assertEqual(report["checkpoint_step"], 3)
            self.assertEqual(report["config"]["identity_attention_type"], "identity_first")
            self.assertIn("program_memory_cosine", report["metrics"])
            self.assertEqual(len(report["layers"]), 1)
            self.assertIn("programs", report["layers"][0])
            self.assertGreater(len(report["layers"][0]["content_slots"]), 0)
            slot = report["layers"][0]["content_slots"][0]
            self.assertIn("cue_top_tokens", slot)
            self.assertIn("value_top_tokens", slot)
            self.assertIn("decoded_value", slot)

    def test_checkpoint_harder_matrix_evaluator_reports_carry_reset_shuffle(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"
            config = TACConfig(
                vocab_size=512,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=16,
                state_update_type="gated",
                memory_write_type="novelty_gated",
                memory_read_type="content_addressed",
                content_store_size=4,
                identity_attention_type="identity_first",
                memory_adapter_type="gated_residual",
                detach_identity_state=False,
            )
            model = TACTransformerLM(config)
            torch.save(
                {
                    "step": 4500,
                    "best_eval_loss": 0.155,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metrics": {"loss": 0.155},
                },
                checkpoint_path,
            )

            report = evaluate_checkpoint_harder_matrix.evaluate_checkpoint_harder_matrix(
                checkpoint_path,
                tasks=["noisy_key"],
                seeds=[11],
                eval_batches=1,
                eval_batch_size=2,
                device="cpu",
            )

            self.assertEqual(report["checkpoint_step"], 4500)
            self.assertEqual(report["tasks"], ["noisy_key"])
            self.assertEqual(len(report["runs"]), 1)
            probe = report["runs"][0]["probe"]
            self.assertIn("carry", probe)
            self.assertIn("reset", probe)
            self.assertIn("shuffled", probe)
            self.assertIn("value_accuracy_delta", probe)
            self.assertIn("program_memory_cosine", probe["carry"])

    def test_content_read_query_gating_report_includes_task_decisions(self):
        from argparse import Namespace
        from experiments import benchmark_content_read_query_gating_capability as bench

        def run(task, variant, carry, reset, shuffled, read_fraction):
            return {
                "task": task,
                "variant": variant,
                "decision": {"status": "effective"},
                "tac": {
                    "train": {"tokens_per_second": 100.0},
                    "chunked_probe": {
                        "carry": {
                            "value_accuracy": carry,
                            "content_read_query_fraction": read_fraction,
                            "content_read_skipped_fraction": 1.0 - read_fraction,
                            "tokens_per_second": 90.0,
                        },
                        "reset": {"value_accuracy": reset},
                        "shuffled": {"value_accuracy": shuffled},
                    },
                },
            }

        args = Namespace(
            tasks=["noisy_key", "multi_hop"],
            seeds=[11],
            top_k_values=[0, 2],
            seq_len=8,
            vocab_size=40,
            d_model=32,
            n_heads=4,
            n_layers=1,
            n_programs=8,
            steps=120,
            batch_size=16,
            eval_batches=4,
            eval_batch_size=16,
            learning_rate=5e-4,
            preservation_tolerance=0.02,
            min_full_carry=0.20,
            min_full_state_utility=0.05,
            max_read_fraction=0.50,
            device="cpu",
            torch_threads=1,
        )
        aggregate = bench.aggregate_results(
            [
                run("noisy_key", "full_read", 0.30, 0.05, 0.04, 1.0),
                run("noisy_key", "top_k_2", 0.29, 0.05, 0.04, 0.25),
                run("multi_hop", "full_read", 0.06, 0.03, 0.02, 1.0),
                run("multi_hop", "top_k_2", 0.06, 0.03, 0.02, 0.25),
            ],
            args,
        )

        self.assertEqual(
            aggregate["task_decisions"]["noisy_key"]["top_k_2"]["status"],
            "preserved",
        )
        self.assertEqual(
            aggregate["task_decisions"]["multi_hop"]["top_k_2"]["status"],
            "blocked_by_full_read_capability",
        )
        markdown = bench.format_markdown(aggregate)
        self.assertIn("## Per-Task Preservation Decisions", markdown)
        self.assertIn("noisy_key", markdown)
        self.assertIn("multi_hop", markdown)

    def test_coalition_ablation_selects_best_accepted_variant(self):
        from experiments import benchmark_coalition_routing_ablation as bench

        by_task = {
            "single_key": {
                "variants": {
                    "current_parallel_topk": {
                        "mean_carry_value_accuracy": 0.80,
                        "mean_coalition_context_norm": 0.0,
                    },
                    "coalition_program_memory": {
                        "mean_carry_value_accuracy": 0.79,
                        "mean_coalition_context_norm": 0.1,
                    },
                    "coalition_program_memory_graph": {
                        "mean_carry_value_accuracy": 0.81,
                        "mean_coalition_context_norm": 0.2,
                    },
                },
            },
            "multi_hop": {
                "variants": {
                    "current_parallel_topk": {
                        "mean_carry_value_accuracy": 0.20,
                        "mean_coalition_context_norm": 0.0,
                    },
                    "coalition_program_memory": {
                        "mean_carry_value_accuracy": 0.19,
                        "mean_coalition_context_norm": 0.1,
                    },
                    "coalition_program_memory_graph": {
                        "mean_carry_value_accuracy": 0.24,
                        "mean_coalition_context_norm": 0.2,
                    },
                },
            },
        }

        decision = bench._decision(
            by_task,
            min_multihop_gain=0.02,
            max_direct_regression=0.02,
        )

        self.assertEqual(decision["status"], "promote_candidate")
        self.assertEqual(decision["accepted_variant"], "coalition_program_memory_graph")
        self.assertAlmostEqual(decision["single_key_accuracy_delta"], 0.01)
        self.assertAlmostEqual(decision["multi_hop_accuracy_delta"], 0.04)

    def test_forced_program_evaluator_compares_every_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "best.pt"
            data_path = tmp_path / "eval.jsonl"
            config = TACConfig(
                vocab_size=512,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=16,
                state_update_type="gated",
                memory_write_type="novelty_gated",
                memory_read_type="content_addressed",
                content_store_size=4,
                identity_attention_type="identity_first",
                memory_adapter_type="gated_residual",
                detach_identity_state=False,
            )
            model = TACTransformerLM(config)
            torch.save(
                {
                    "step": 9,
                    "best_eval_loss": 1.5,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metrics": {"loss": 1.5},
                },
                checkpoint_path,
            )
            rows = [
                {
                    "record_id": "r1",
                    "domain": "tool_choice",
                    "text": '<record type="hard_tool_choice"><goal>choose shell</record>',
                },
                {
                    "record_id": "r2",
                    "domain": "repair_after_failure",
                    "text": '<record type="hard_repair_after_failure"><goal>repair</record>',
                },
            ]
            data_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = evaluate_forced_programs.evaluate_forced_programs(
                checkpoint_path,
                data_path,
                max_records_per_category=1,
                batch_size=2,
                programs=[0, 1],
                device="cpu",
            )

            self.assertEqual(report["checkpoint_step"], 9)
            self.assertEqual(report["records"], 2)
            self.assertIn("natural", report)
            self.assertEqual(len(report["forced_programs"]), 2)
            self.assertEqual(report["forced_programs"][0]["program"], 0)
            self.assertIn("forced_loss_range", report["summary"])
            self.assertIn("forced_loss_variance", report["summary"])
            self.assertIn("loss_delta_vs_natural", report["forced_programs"][0])
            self.assertIn("category_program_rankings", report)
            self.assertIn("tool_choice", report["category_program_rankings"])
            self.assertIn("top_program_counts", report["natural"])

    def test_program_specialization_analysis_reports_attribution_mi_and_knockouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "best.pt"
            data_path = tmp_path / "hard_eval.jsonl"
            config = TACConfig(
                vocab_size=512,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=16,
                state_update_type="gated",
                memory_write_type="novelty_gated",
                memory_read_type="content_addressed",
                content_store_size=4,
                identity_attention_type="identity_first",
                memory_adapter_type="gated_residual",
                detach_identity_state=False,
            )
            model = TACTransformerLM(config)
            torch.save(
                {
                    "step": 12,
                    "best_eval_loss": 2.0,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metrics": {"loss": 2.0},
                },
                checkpoint_path,
            )
            rows = [
                {
                    "record_id": "r1",
                    "domain": "tool_choice",
                    "text": '<record type="hard_tool_choice"><goal>choose shell</record>',
                },
                {
                    "record_id": "r2",
                    "domain": "tool_choice",
                    "text": '<record type="hard_tool_choice"><goal>choose python</record>',
                },
                {
                    "record_id": "r3",
                    "domain": "repair_after_failure",
                    "text": '<record type="hard_repair_after_failure"><goal>repair test</record>',
                },
                {
                    "record_id": "r4",
                    "domain": "repair_after_failure",
                    "text": '<record type="hard_repair_after_failure"><goal>fix schema</record>',
                },
            ]
            data_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = analyze_program_specialization.analyze_program_specialization(
                checkpoint_path,
                data_path,
                max_records_per_category=2,
                top_k=2,
                knockout_programs=[0, 1],
                device="cpu",
            )

            self.assertEqual(report["checkpoint_step"], 12)
            self.assertEqual(set(report["categories"]), {"repair_after_failure", "tool_choice"})
            self.assertEqual(len(report["records"]), 4)
            self.assertIn("top_programs", report["records"][0])
            self.assertIn("mutual_information", report)
            self.assertIn("mi_bits", report["mutual_information"])
            self.assertIn("activation_histogram", report)
            self.assertIn("tool_choice", report["activation_histogram"]["by_category"])
            self.assertIn("token_mutual_information", report)
            self.assertIn("category_route_histogram", report)
            self.assertIn("specialization_metrics", report)
            self.assertIn("program_memory_summary", report)
            self.assertEqual(
                len(report["program_memory_summary"]["programs"]),
                config.n_programs,
            )
            self.assertIn(
                "mean_vector",
                report["program_memory_summary"]["programs"][0],
            )
            tool_matrix = report["activation_histogram"]["by_category"]["tool_choice"]
            self.assertIn("token_top_program_counts", tool_matrix)
            self.assertIn("mean_token_activation_probabilities", tool_matrix)
            self.assertIn("mean_token_selected_frequencies", tool_matrix)
            self.assertEqual(len(report["ablations"]), 2)
            self.assertIn("loss_delta", report["ablations"][0])
            self.assertIn("by_category", report["ablations"][0])

    def test_program_specialization_analysis_can_emit_token_csv_without_knockouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "best.pt"
            data_path = tmp_path / "hard_eval.jsonl"
            token_csv_path = tmp_path / "token_routes.csv"
            config = TACConfig(
                vocab_size=128,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=12,
                routing_type="base",
                state_update_type="gated",
                memory_write_type="novelty_gated",
                memory_read_type="content_addressed",
                content_store_size=4,
                identity_attention_type="identity_first",
                memory_adapter_type="gated_residual",
                detach_identity_state=False,
            )
            model = TACTransformerLM(config)
            torch.save(
                {
                    "step": 12,
                    "best_eval_loss": 2.0,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metrics": {"loss": 2.0},
                },
                checkpoint_path,
            )
            rows = [
                {
                    "record_id": "r1",
                    "domain": "tool_choice",
                    "text": '<record type="hard_tool_choice"><goal>choose shell</record>',
                }
            ]
            data_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = analyze_program_specialization.analyze_program_specialization(
                checkpoint_path,
                data_path,
                max_records_per_category=None,
                top_k=2,
                run_knockouts=False,
                capture_token_rows=True,
                device="cpu",
            )
            analyze_program_specialization.write_token_csv(report, token_csv_path)

            self.assertEqual(report["ablations"], [])
            self.assertTrue(token_csv_path.exists())
            csv_text = token_csv_path.read_text(encoding="utf-8")
            self.assertIn("token_route_entropy_bits", csv_text)
            self.assertIn("selected_top_program_prob", csv_text)
            self.assertIn("selected_programs", csv_text)

    def test_content_memory_causal_audit_reports_randomized_query_verdict(self):
        config = best_tac_config(
            vocab_size=40,
            max_seq_len=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=4,
        )

        report = audit_content_memory_causality.run_causal_audit(
            config,
            steps=1,
            batch_size=2,
            learning_rate=1e-3,
            eval_batches=1,
            eval_batch_size=2,
            seed=5,
            device="cpu",
            training_kwargs={
                "value_loss_weight": 0.0,
                "memory_read_loss_weight": 0.0,
                "memory_injection_weight": 0.0,
                "memory_adapter_weight": 0.0,
            },
        )

        self.assertIn(report["verdict"], {"pass", "fail"})
        self.assertIn("randomized_query_carry", report)
        self.assertIn("normal_carry", report)
        self.assertIn("chance_accuracy_estimate", report)

    def test_routing_collapse_analyzer_flags_base_final_token_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "best.pt"
            data_path = tmp_path / "hard_eval.jsonl"
            config = TACConfig(
                vocab_size=128,
                d_model=32,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=16,
                routing_type="base",
                state_update_type="gated",
                memory_write_type="novelty_gated",
                memory_read_type="content_addressed",
                content_store_size=4,
                identity_attention_type="identity_first",
                memory_adapter_type="gated_residual",
                detach_identity_state=False,
            )
            model = TACTransformerLM(config)
            torch.save(
                {
                    "step": 12,
                    "best_eval_loss": 2.0,
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metrics": {"loss": 2.0},
                },
                checkpoint_path,
            )
            rows = [
                {
                    "record_id": "r1",
                    "domain": "tool_choice",
                    "text": '<record type="hard_tool_choice"><goal>choose shell</record>',
                },
                {
                    "record_id": "r2",
                    "domain": "repair_after_failure",
                    "text": '<record type="hard_repair_after_failure"><goal>repair test</record>',
                },
            ]
            data_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = analyze_routing_collapse.analyze_routing_collapse(
                checkpoint_path,
                data_path,
                max_records_per_category=1,
                top_k=2,
                forced_programs=[0, 3],
                knockout_programs=[3],
                device="cpu",
            )

            self.assertEqual(report["checkpoint_step"], 12)
            self.assertTrue(report["position_artifacts"]["base_final_token_artifact"])
            self.assertEqual(report["position_artifacts"]["fixed_padded_final_program"], 3)
            self.assertEqual(report["selected_program_distribution"]["program_entropy_bits"], 0.0)
            self.assertIn("mean_entropy_bits", report["raw_activation_summary"])
            self.assertIn("raw_activation_argmax_mi", report)
            self.assertEqual([entry["program"] for entry in report["forced_programs"]], [0, 3])
            self.assertEqual(report["knockouts"][0]["program"], 3)

    def test_synthetic_trainer_updates_model_and_reports_metrics(self):
        batcher = SyntheticProgramBatcher(vocab_size=32, seq_len=9, seed=11)
        model = TACTransformerLM(self.config)

        metrics = train_synthetic(
            model,
            batcher,
            steps=2,
            batch_size=4,
            learning_rate=1e-3,
            aux_weights={"coherence": 0.01, "program_reuse": 0.01, "energy": 0.01},
            device="cpu",
        )

        self.assertEqual(metrics["steps"], 2)
        self.assertGreater(metrics["loss"], 0)
        self.assertGreaterEqual(metrics["tokens_per_second"], 0)

    def test_synthetic_benchmark_compares_tac_and_vanilla_baseline(self):
        result = benchmark_synthetic(
            self.config,
            steps=1,
            batch_size=2,
            eval_batches=1,
            eval_batch_size=2,
            learning_rate=1e-3,
            seed=19,
            match_baseline_parameters=True,
        )

        self.assertIn("tac", result)
        self.assertIn("baseline", result)
        self.assertGreater(result["tac"]["parameter_counts"]["identity_field"], 0)
        self.assertEqual(result["baseline"]["parameter_counts"]["identity_field"], 0)
        self.assertGreater(result["tac"]["final_eval"]["loss"], 0)
        self.assertGreater(result["baseline"]["final_eval"]["loss"], 0)

    def test_prepare_jsonl_dataset_serializes_rows_and_caps_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "prepared.jsonl"
            rows = [
                {
                    "record_id": f"r{index}",
                    "prompt": "Run tests",
                    "final_answer": "Tests passed",
                    "target_plan": "[{\"opcode\":\"RUN_TESTS\"}]",
                    "plan": [{"opcode": "RUN_TESTS"}],
                    "tool_results": [],
                    "domain": "testing",
                    "source": "unit",
                    "success": True,
                }
                for index in range(4)
            ]
            rows.append({"record_id": "missing", "prompt": "No answer", "target_plan": "[]"})
            input_path.write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )

            stats = prepare_jsonl_dataset(input_path, output_path, duplicate_cap=2)
            prepared = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(stats["read"], 5)
            self.assertEqual(stats["written"], 2)
            self.assertEqual(stats["duplicate_capped"], 2)
            self.assertEqual(stats["missing_final_answer"], 1)
            self.assertIn("<prompt>", prepared[0]["text"])
            self.assertIn("<final_answer>", prepared[0]["text"])

    def test_prepare_jsonl_dataset_caps_generic_records_by_serialized_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.json"
            output_path = Path(tmp) / "prepared.jsonl"
            input_path.write_text(
                json.dumps(
                    [
                        {"layer": "L0", "input_state": [1], "output_state": [2]},
                        {"layer": "L0", "input_state": [1], "output_state": [2]},
                        {"layer": "L0", "input_state": [3], "output_state": [4]},
                    ]
                ),
                encoding="utf-8",
            )

            stats = prepare_jsonl_dataset(input_path, output_path, duplicate_cap=1)
            prepared = output_path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(stats["read"], 3)
            self.assertEqual(stats["written"], 2)
            self.assertEqual(stats["duplicate_capped"], 1)
            self.assertEqual(len(prepared), 2)

    def test_dedupe_prepared_jsonl_caps_normalized_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "prepared.jsonl"
            output_path = Path(tmp) / "deduped.jsonl"
            rows = [
                {
                    "record_id": f"r{index}",
                    "text": f'<record id="r{index}"><goal>Read file {index}</record>',
                }
                for index in range(5)
            ]
            input_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            stats = dedupe_prepared_jsonl(
                input_path,
                output_path,
                exact_cap=1,
                template_cap=2,
            )
            kept = output_path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(stats["read"], 5)
            self.assertEqual(stats["written"], 2)
            self.assertEqual(stats["template_capped"], 3)
            self.assertEqual(len(kept), 2)
            self.assertEqual(
                normalize_template_text(rows[0]["text"]),
                normalize_template_text(rows[3]["text"]),
            )

    def test_jsonl_text_batcher_reads_prepared_text_for_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared_path = Path(tmp) / "prepared.jsonl"
            prepared_path.write_text(
                json.dumps({"text": serialize_record({"prompt": "abc", "final_answer": "def", "target_plan": "[]"})})
                + "\n",
                encoding="utf-8",
            )
            batcher = JsonlTextBatcher(prepared_path, seq_len=8, vocab_size=512, seed=3)
            inputs, labels = batcher.next_batch(batch_size=2)

            self.assertEqual(inputs.shape, (2, 8))
            self.assertEqual(labels.shape, (2, 8))
            self.assertLess(int(inputs.max()), 512)

    def test_tokenized_memmap_batcher_reads_manifest_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prepared_path = tmp_path / "prepared.jsonl"
            prepared_path.write_text(
                "\n".join(
                    [
                        json.dumps({"domain": "testing", "text": "abcdef"}),
                        json.dumps({"domain": "rag", "text": "uvwxyz"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = build_tokenized_memmap_from_jsonl(
                prepared_path,
                tmp_path / "tokenized",
                vocab_size=512,
            )
            batcher = TokenizedMemmapBatcher.from_manifest(
                tmp_path / "tokenized" / "manifest.json",
                seq_len=4,
                seed=3,
                include_categories=True,
            )

            inputs, labels = batcher.next_batch(batch_size=2)
            labeled_inputs, labeled_labels, categories = batcher.next_labeled_batch(batch_size=2)

            self.assertEqual(inputs.shape, (2, 4))
            self.assertEqual(labels.shape, (2, 4))
            self.assertEqual(labeled_inputs.shape, (2, 4))
            self.assertEqual(labeled_labels.shape, (2, 4))
            self.assertEqual(categories.shape, (2,))
            self.assertEqual(manifest["records"], 2)
            self.assertLess(int(inputs.max()), 512)
            batcher.close()

    def test_sanitize_training_text_masks_secret_like_values(self):
        text = 'export GEOLOC_TOKEN="sk_live_x123"; password: securepassword123'
        sanitized = sanitize_training_text(text)

        self.assertNotIn("sk_live_x123", sanitized)
        self.assertNotIn("securepassword123", sanitized)
        self.assertIn("<API_KEY>", sanitized)
        self.assertIn("<PASSWORD>", sanitized)

    def test_knowledge_work_generator_emits_training_jsonl(self):
        record = next(generate_knowledge_work_records(seed=5))
        line = record_to_jsonl(record)
        row = json.loads(line)

        self.assertIn(row["domain"], {
            "rag_multi_hop",
            "agentic_tool_use",
            "knowledge_work_synthesis",
            "coding_testing",
            "spreadsheet_analysis",
            "research_brief",
        })
        self.assertIn("<record", row["text"])
        self.assertGreater(estimate_tokens(row["text"]), 20)

    def test_hard_agentic_generator_emits_counterfactual_and_repair_records(self):
        generator = generate_hard_agentic_records(seed=101)
        records = [next(generator) for _ in range(20)]
        text = "\n".join(record.text for record in records)
        line = hard_record_to_jsonl(records[0])
        row = json.loads(line)

        self.assertIn(row["source"], {"hard_agentic_curriculum"})
        self.assertIn("<record", row["text"])
        self.assertIn("<target_plan>", text)
        self.assertIn("<target_action>", text)
        self.assertIn("<shuffled_memory>", text)
        self.assertIn("schema_mismatch", text)

    def test_build_hard_agentic_corpus_writes_manifest_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_dir = tmp_path / "base"
            base_dir.mkdir()
            base_train = base_dir / "train.prepared.jsonl"
            base_eval = base_dir / "eval.prepared.jsonl"
            base_train.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "record_id": f"base_{index}",
                            "domain": "base",
                            "text": f'<record id="base_{index}"><goal>Read file {index}</record>',
                        }
                    )
                    for index in range(8)
                )
                + "\n",
                encoding="utf-8",
            )
            base_eval.write_text(
                json.dumps({"record_id": "eval_1", "domain": "eval", "text": "<record>eval</record>"})
                + "\n",
                encoding="utf-8",
            )

            output_dir = tmp_path / "hard"
            build_hard_agentic_corpus.main(
                [
                    "--base-dir",
                    str(base_dir),
                    "--output-dir",
                    str(output_dir),
                    "--hard-train-records",
                    "12",
                    "--hard-eval-records",
                    "6",
                    "--template-cap",
                    "2",
                ]
            )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue((output_dir / "train.prepared.jsonl").exists())
            self.assertTrue((output_dir / "eval.prepared.jsonl").exists())
            self.assertEqual(manifest["parts"]["hard_agentic_train"]["stats"]["written"], 12)
            self.assertEqual(manifest["parts"]["hard_agentic_eval"]["stats"]["written"], 6)
            self.assertGreater(manifest["train_records"], 12)

    def test_state_intervention_probe_reports_memory_effectiveness_metrics(self):
        model = TACTransformerLM(self.config)
        batcher = SyntheticProgramBatcher(vocab_size=32, seq_len=9, seed=23)

        metrics = evaluate_state_interventions(
            model,
            batcher,
            batches=2,
            batch_size=3,
            device="cpu",
        )

        self.assertIn("carry", metrics)
        self.assertIn("reset", metrics)
        self.assertIn("shuffled", metrics)
        self.assertIn("memory_carry_delta", metrics)
        self.assertIn("state_shuffle_penalty", metrics)
        self.assertIn("routing_entropy", metrics["carry"])
        self.assertIn("active_programs", metrics["carry"])
        self.assertGreater(metrics["carry"]["tokens_per_second"], 0)
        self.assertGreaterEqual(metrics["carry"]["active_programs"], 0)

    def test_effectiveness_benchmark_returns_explicit_json_serializable_scorecard(self):
        result = benchmark_effectiveness(
            self.config,
            steps=1,
            batch_size=2,
            eval_batches=1,
            eval_batch_size=2,
            probe_batches=2,
            learning_rate=1e-3,
            seed=31,
            device="cpu",
            match_baseline_parameters=True,
        )

        decoded = json.loads(json.dumps(result))

        self.assertIn("decision", result)
        self.assertIn(result["decision"]["status"], {"effective", "inconclusive"})
        self.assertIn("tac", result)
        self.assertIn("baseline", result)
        self.assertIn("state_probe", result["tac"])
        self.assertIn("parameter_counts", result["baseline"])
        self.assertIn("memory_carry_delta", result["tac"]["state_probe"])
        self.assertIn("state_shuffle_penalty", result["tac"]["state_probe"])
        self.assertEqual(decoded["decision"]["status"], result["decision"]["status"])

    def test_capability_sanity_matrix_reports_run5_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "capability"
            result = run_capability_sanity_matrix(
                output_dir=output_dir,
                variants=["vanilla_10m_proxy", "tac_base_proxy", "tac_semantic_low_weight"],
                seeds=[3],
                train_records=6,
                eval_records=4,
                steps=1,
                seq_len=16,
                batch_size=2,
                eval_batches=1,
                eval_batch_size=2,
                d_model=32,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                device="cpu",
            )

            decoded = json.loads(json.dumps(result))
            self.assertIn("run5_gate", decoded)
            self.assertIn(decoded["run5_gate"]["status"], {"pass", "blocked", "inconclusive"})
            self.assertTrue((output_dir / "capability_sanity_matrix.json").exists())
            self.assertTrue((output_dir / "RESULTS.md").exists())
            self.assertEqual(set(result["aggregate"]), {
                "vanilla_10m_proxy",
                "tac_base_proxy",
                "tac_semantic_low_weight",
            })
            semantic = result["aggregate"]["tac_semantic_low_weight"]
            self.assertEqual(semantic["category_route_weight"], 0.05)

    def test_capability_sanity_aggregate_detects_objective_regression(self):
        rows = [
            {
                "variant": "vanilla_10m_proxy",
                "seed": 1,
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.0, "accuracy": 0.2},
                "train": {"tokens_per_second": 100.0},
            },
            {
                "variant": "tac_base_proxy",
                "seed": 1,
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.2, "accuracy": 0.18},
                "train": {"tokens_per_second": 60.0},
                "mean_category_route_loss": 0.0,
            },
            {
                "variant": "tac_semantic_low_weight",
                "seed": 1,
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 6.1, "accuracy": 0.01},
                "train": {"tokens_per_second": 55.0},
                "mean_category_route_loss": -0.01,
            },
        ]

        aggregate = aggregate_capability_sanity_results(rows)

        self.assertIn("tac_semantic_low_weight", CAPABILITY_SANITY_VARIANTS)
        self.assertEqual(aggregate["run5_gate"]["status"], "blocked")
        self.assertIn("semantic objective", aggregate["run5_gate"]["reason"])

    def test_run5_capability_preset_uses_low_weight_semantic_routing_under_identity_share_gate(self):
        config = run5_capability_config(vocab_size=512)
        training = run5_capability_training_kwargs()
        counts = count_parameters(TACTransformerLM(config))

        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.routing_top_k, 2)
        self.assertEqual(config.n_programs, 12)
        self.assertAlmostEqual(config.routing_load_balance_weight, 0.05)
        self.assertAlmostEqual(training["category_route_weight"], 0.1)
        self.assertEqual(training["category_route_objective"], "mi")
        self.assertLessEqual(counts["identity_field"] / counts["total"], 0.5)

    def test_train_best_tac_agentic_can_select_run5_capability_preset(self):
        args = train_best_tac_agentic.parse_args([
            "--preset",
            "run5_capability",
            "--scale",
            "smoke",
        ])
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.n_programs, 12)
        self.assertEqual(config.routing_top_k, 2)
        self.assertAlmostEqual(args.category_route_weight, 0.1)
        self.assertEqual(args.category_route_objective, "mi")
        self.assertEqual(args.warmup_steps, 2000)

    def test_run5b_capability_preset_forces_fp32_optimizer_health_gate(self):
        config = run5b_capability_config(vocab_size=512)
        training = run5b_capability_training_kwargs()
        counts = count_parameters(TACTransformerLM(config))

        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.routing_top_k, 2)
        self.assertEqual(config.n_programs, 12)
        self.assertAlmostEqual(config.routing_load_balance_weight, 0.05)
        self.assertAlmostEqual(training["category_route_weight"], 0.1)
        self.assertEqual(training["category_route_objective"], "mi")
        self.assertEqual(training["warmup_steps"], 2000)
        self.assertEqual(training["precision"], "fp32")
        self.assertAlmostEqual(training["min_healthy_gradient_norm"], 1e-12)
        self.assertEqual(training["fail_on_unhealthy_optimization"], 1)
        self.assertLessEqual(counts["identity_field"] / counts["total"], 0.5)

    def test_train_best_tac_agentic_can_select_run5b_capability_preset(self):
        args = train_best_tac_agentic.parse_args([
            "--preset",
            "run5b_capability",
            "--scale",
            "smoke",
        ])
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.routing_type, "base_semantic")
        self.assertEqual(config.n_programs, 12)
        self.assertEqual(config.routing_top_k, 2)
        self.assertAlmostEqual(args.category_route_weight, 0.1)
        self.assertEqual(args.category_route_objective, "mi")
        self.assertEqual(args.warmup_steps, 2000)
        self.assertEqual(args.precision, "fp32")
        self.assertAlmostEqual(args.min_healthy_gradient_norm, 1e-12)
        self.assertTrue(args.fail_on_unhealthy_optimization)

    def test_train_best_tac_agentic_accepts_program_conditioned_memory_flags(self):
        args = train_best_tac_agentic.parse_args([
            "--preset",
            "run5b_capability",
            "--scale",
            "smoke",
            "--program-memory-update-type",
            "program_conditioned",
            "--memory-allocation-type",
            "creb",
            "--memory-allocation-k",
            "6",
            "--memory-separation-weight",
            "0.1",
        ])
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.program_memory_update_type, "program_conditioned")
        self.assertEqual(config.memory_allocation_type, "creb")
        self.assertEqual(config.memory_allocation_k, 6)
        self.assertAlmostEqual(config.memory_separation_weight, 0.1)

    def test_train_best_tac_agentic_accepts_content_read_flags(self):
        args = train_best_tac_agentic.parse_args([
            "--preset",
            "run5b_capability",
            "--scale",
            "smoke",
            "--memory-read-type",
            "content_addressed",
            "--content-read-steps",
            "2",
            "--content-read-gate-type",
            "cue_match",
            "--content-read-confidence-margin",
            "0.07",
            "--content-read-cue-match-threshold",
            "0.66",
            "--content-read-query-top-k",
            "8",
            "--coalition-context-type",
            "program_memory",
            "--coalition-context-scale",
            "0.5",
        ])
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.memory_read_type, "content_addressed")
        self.assertEqual(config.content_read_steps, 2)
        self.assertEqual(config.content_read_gate_type, "cue_match")
        self.assertAlmostEqual(config.content_read_confidence_margin, 0.07)
        self.assertAlmostEqual(config.content_read_cue_match_threshold, 0.66)
        self.assertEqual(config.content_read_query_top_k, 8)
        self.assertEqual(config.coalition_context_type, "program_memory")
        self.assertAlmostEqual(config.coalition_context_scale, 0.5)

    def test_train_best_tac_agentic_accepts_identity_path_ablation_flags(self):
        args = train_best_tac_agentic.parse_args([
            "--preset",
            "run5b_capability",
            "--scale",
            "smoke",
            "--memory-adapter-type",
            "none",
            "--identity-attention-type",
            "none",
            "--program-residual-scale",
            "0.0",
            "--coherence-attention-scale",
            "0.0",
        ])
        scale = train_best_tac_agentic.resolved_scale(args)
        config = train_best_tac_agentic.build_training_config(args, scale)

        self.assertEqual(config.memory_adapter_type, "none")
        self.assertEqual(config.identity_attention_type, "none")
        self.assertAlmostEqual(config.program_residual_scale, 0.0)
        self.assertAlmostEqual(config.coherence_attention_scale, 0.0)

    def test_train_best_tac_agentic_scales_auxiliary_pressure(self):
        args = train_best_tac_agentic.parse_args([
            "--preset",
            "run5b_capability",
            "--scale",
            "smoke",
            "--aux-loss-scale",
            "0.25",
            "--aux-loss-warmup-steps",
            "100",
        ])

        self.assertAlmostEqual(train_best_tac_agentic.aux_loss_multiplier(args, step=0), 0.0)
        self.assertAlmostEqual(train_best_tac_agentic.aux_loss_multiplier(args, step=50), 0.125)
        self.assertAlmostEqual(train_best_tac_agentic.aux_loss_multiplier(args, step=100), 0.25)
        self.assertAlmostEqual(train_best_tac_agentic.aux_loss_multiplier(args, step=200), 0.25)

    def test_train_best_tac_agentic_derives_warmup_from_ratio(self):
        args = train_best_tac_agentic.parse_args([
            "--preset",
            "run5b_capability",
            "--scale",
            "smoke",
            "--steps",
            "300",
            "--warmup-ratio",
            "0.05",
        ])

        self.assertEqual(args.warmup_steps, 15)
        self.assertAlmostEqual(args.warmup_ratio, 0.05)

    def test_train_best_tac_agentic_delays_semantic_route_pressure(self):
        args = train_best_tac_agentic.parse_args([
            "--preset",
            "run5b_capability",
            "--scale",
            "smoke",
            "--category-route-weight",
            "0.1",
            "--category-route-start-step",
            "100",
            "--category-route-warmup-steps",
            "200",
            "--semantic-routing-start-step",
            "100",
        ])

        self.assertEqual(
            train_best_tac_agentic.effective_routing_mode(args, step=0),
            ("base", 1),
        )
        self.assertEqual(
            train_best_tac_agentic.effective_routing_mode(args, step=100),
            ("base_semantic", 2),
        )
        self.assertAlmostEqual(train_best_tac_agentic.category_route_multiplier(args, step=0), 0.0)
        self.assertAlmostEqual(train_best_tac_agentic.category_route_multiplier(args, step=100), 0.0)
        self.assertAlmostEqual(train_best_tac_agentic.category_route_multiplier(args, step=200), 0.05)
        self.assertAlmostEqual(train_best_tac_agentic.category_route_multiplier(args, step=300), 0.1)

    def test_train_best_tac_agentic_applies_semantic_routing_schedule_to_model(self):
        args = train_best_tac_agentic.parse_args([
            "--preset",
            "run5b_capability",
            "--scale",
            "smoke",
            "--semantic-routing-start-step",
            "10",
        ])
        model = TACTransformerLM(
            TACConfig(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                routing_type="base_semantic",
                routing_top_k=2,
            )
        )

        train_best_tac_agentic.apply_semantic_routing_schedule(
            model,
            args,
            step=1,
            final_routing_type="base_semantic",
            final_routing_top_k=2,
        )
        self.assertEqual(model.config.routing_type, "base")
        self.assertEqual(model.config.routing_top_k, 1)

        train_best_tac_agentic.apply_semantic_routing_schedule(
            model,
            args,
            step=10,
            final_routing_type="base_semantic",
            final_routing_top_k=2,
        )
        self.assertEqual(model.config.routing_type, "base_semantic")
        self.assertEqual(model.config.routing_top_k, 2)

    def test_best_tac_agentic_run5b_smoke_persists_optimizer_health(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.prepared.jsonl"
            eval_path = tmp_path / "eval.prepared.jsonl"
            rows = [
                {"domain": "tool_choice", "text": "choose calculator and verify result"},
                {"domain": "repair_after_failure", "text": "repair failing test then verify"},
                {"domain": "tool_choice", "text": "choose shell command for evidence"},
                {"domain": "repair_after_failure", "text": "diagnose error and patch"},
            ]
            payload = "\n".join(json.dumps(row) for row in rows) + "\n"
            train_path.write_text(payload, encoding="utf-8")
            eval_path.write_text(payload, encoding="utf-8")

            train_best_tac_agentic.main(
                [
                    "--preset",
                    "run5b_capability",
                    "--scale",
                    "smoke",
                    "--d-model",
                    "32",
                    "--n-heads",
                    "4",
                    "--n-layers",
                    "1",
                    "--seq-len",
                    "16",
                    "--batch-size",
                    "1",
                    "--steps",
                    "2",
                    "--eval-every",
                    "1",
                    "--eval-batches",
                    "1",
                    "--checkpoint-every",
                    "1",
                    "--train-jsonl",
                    str(train_path),
                    "--eval-jsonl",
                    str(eval_path),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--device",
                    "cpu",
                ]
            )

            manifest = json.loads(
                (tmp_path / "out" / "run_manifest.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (tmp_path / "out" / "final_summary.json").read_text(encoding="utf-8")
            )
            latest = summary["latest_metrics"]
            health = latest["optimization_health"]

            self.assertEqual(manifest["precision"], "fp32")
            self.assertEqual(manifest["optimization_health"]["min_gradient_norm"], 1e-12)
            self.assertEqual(latest["grad_scaler_scale"], 1.0)
            self.assertGreater(latest["gradient_norm"], 0.0)
            self.assertEqual(health["status"], "passed")
            self.assertGreater(health["gradient_norm"], 0.0)

    def test_pathfinder_variant_grid_covers_identity_and_semantic_sweeps(self):
        variants = build_run5_pathfinder_variants(
            program_counts=[8, 12],
            semantic_weights=[0.0, 0.05, 0.2],
            include_vanilla=True,
        )

        self.assertIn("vanilla_10m_proxy", variants)
        self.assertIn("tac_base_p8", variants)
        self.assertIn("tac_semantic_w0p05_p12", variants)
        self.assertIn("tac_semantic_w0p2_p12", variants)
        self.assertEqual(variants["tac_semantic_w0p05_p12"]["n_programs"], 12)
        self.assertEqual(variants["tac_semantic_w0p05_p12"]["routing_type"], "base_semantic")
        self.assertAlmostEqual(
            variants["tac_semantic_w0p05_p12"]["category_route_weight"],
            0.05,
        )

    def test_routing_pressure_phase_grid_is_run3_run4_controlled(self):
        variants = build_routing_pressure_phase_variants(
            semantic_weights=[0.0, 0.01, 0.05, 0.5],
            n_programs=32,
        )

        self.assertEqual(variants["tac_base_run3_control"]["routing_type"], "base")
        self.assertEqual(variants["tac_base_run3_control"]["routing_top_k"], 1)
        self.assertAlmostEqual(variants["tac_base_run3_control"]["category_route_weight"], 0.0)
        self.assertEqual(variants["tac_semantic_mi_w0p01"]["routing_type"], "base_semantic")
        self.assertEqual(variants["tac_semantic_mi_w0p01"]["routing_top_k"], 2)
        self.assertAlmostEqual(variants["tac_semantic_mi_w0p01"]["category_route_weight"], 0.01)
        self.assertAlmostEqual(variants["tac_semantic_mi_w0p5"]["category_route_weight"], 0.5)

    def test_routing_pressure_phase_rejects_label_routing_collapse(self):
        rows = [
            {
                "variant": "tac_base_run3_control",
                "seed": 11,
                "model_type": "tac",
                "routing_type": "base",
                "category_route_weight": 0.0,
                "initial_eval": {"loss": 6.2},
                "final_eval": {"loss": 0.18, "accuracy": 0.93, "perplexity": 1.2},
                "train": {"tokens_per_second": 1000.0},
                "route_eval": {"selected_mi_bits": 0.0, "route_entropy_bits": 3.0},
                "parameter_counts": {"total": 100, "identity_field": 40},
                "memory_health": {"program_memory_cosine": 0.56},
                "config": {"n_programs": 32},
            },
            {
                "variant": "tac_semantic_mi_w0p5",
                "seed": 11,
                "model_type": "tac",
                "routing_type": "base_semantic",
                "category_route_weight": 0.5,
                "initial_eval": {"loss": 6.3},
                "final_eval": {"loss": 6.42, "accuracy": 0.002, "perplexity": 620.0},
                "train": {"tokens_per_second": 900.0},
                "route_eval": {"selected_mi_bits": 0.53, "route_entropy_bits": 0.9},
                "parameter_counts": {"total": 100, "identity_field": 40},
                "memory_health": {"program_memory_cosine": 0.96},
                "config": {"n_programs": 32},
            },
            {
                "variant": "tac_semantic_mi_w0p05",
                "seed": 11,
                "model_type": "tac",
                "routing_type": "base_semantic",
                "category_route_weight": 0.05,
                "initial_eval": {"loss": 6.2},
                "final_eval": {"loss": 0.22, "accuracy": 0.91, "perplexity": 1.25},
                "train": {"tokens_per_second": 930.0},
                "route_eval": {"selected_mi_bits": 0.08, "route_entropy_bits": 2.4},
                "parameter_counts": {"total": 100, "identity_field": 40},
                "memory_health": {"program_memory_cosine": 0.62},
                "config": {"n_programs": 32},
            },
        ]

        result = aggregate_routing_pressure_phase_results(
            rows,
            max_loss_gap_vs_base=0.1,
            max_program_memory_cosine=0.85,
        )

        self.assertEqual(result["recommendation"]["variant"], "tac_semantic_mi_w0p05")
        self.assertEqual(result["aggregate"]["tac_semantic_mi_w0p5"]["phase"], "label_routing_collapse")
        self.assertIn("capability gap", result["rejected"]["tac_semantic_mi_w0p5"][0])
        self.assertIn("program-memory cosine", " ".join(result["rejected"]["tac_semantic_mi_w0p5"]))

    def test_pathfinder_can_include_program_conditioned_creb_mutation(self):
        variants = build_run5_pathfinder_variants(
            program_counts=[12],
            semantic_weights=[0.1],
            include_vanilla=False,
            include_authority=False,
            include_memory_mutations=True,
        )
        name = "tac_program_conditioned_creb_k6_w0p1_p12"
        settings = variants[name]

        self.assertEqual(settings["program_memory_update_type"], "program_conditioned")
        self.assertEqual(settings["memory_allocation_type"], "creb")
        self.assertEqual(settings["memory_allocation_k"], 6)
        self.assertAlmostEqual(settings["memory_separation_weight"], 0.1)
        self.assertEqual(settings["evidence_alias"], "program_conditioned_creb_k6_task_memsep")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.jsonl"
            eval_path = tmp_path / "eval.jsonl"
            payload = "\n".join(
                json.dumps(row)
                for row in [
                    {"domain": "alpha", "text": "abcdefghi" * 4},
                    {"domain": "beta", "text": "jklmnopqr" * 4},
                ]
            ) + "\n"
            train_path.write_text(payload, encoding="utf-8")
            eval_path.write_text(payload, encoding="utf-8")

            result = run_capability_variant_settings(
                name,
                settings,
                train_jsonl=train_path,
                eval_jsonl=eval_path,
                seed=5,
                steps=0,
                seq_len=8,
                batch_size=1,
                eval_batches=1,
                eval_batch_size=1,
                learning_rate=1e-3,
                vocab_size=512,
                d_model=24,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                device="cpu",
            )

        self.assertEqual(result["config"]["program_memory_update_type"], "program_conditioned")
        self.assertEqual(result["config"]["memory_allocation_type"], "creb")
        self.assertEqual(result["config"]["memory_allocation_k"], 6)
        self.assertAlmostEqual(result["config"]["memory_separation_weight"], 0.1)

    def test_pathfinder_runner_uses_candidate_program_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.jsonl"
            eval_path = tmp_path / "eval.jsonl"
            rows = [
                {"domain": "alpha", "text": "abcdefghi" * 4},
                {"domain": "beta", "text": "jklmnopqr" * 4},
            ]
            payload = "\n".join(json.dumps(row) for row in rows) + "\n"
            train_path.write_text(payload, encoding="utf-8")
            eval_path.write_text(payload, encoding="utf-8")
            settings = build_run5_pathfinder_variants(
                program_counts=[12],
                semantic_weights=[0.05],
                include_vanilla=False,
                include_authority=False,
            )["tac_semantic_w0p05_p12"]

            result = run_capability_variant_settings(
                "tac_semantic_w0p05_p12",
                settings,
                train_jsonl=train_path,
                eval_jsonl=eval_path,
                seed=5,
                steps=0,
                seq_len=8,
                batch_size=1,
                eval_batches=1,
                eval_batch_size=1,
                learning_rate=1e-3,
                vocab_size=512,
                d_model=24,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                device="cpu",
            )

        self.assertEqual(result["config"]["n_programs"], 12)

    def test_category_program_mi_bits_from_probs_detects_specialization(self):
        probs = torch.tensor(
            [
                [0.95, 0.05],
                [0.90, 0.10],
                [0.10, 0.90],
                [0.05, 0.95],
            ],
            dtype=torch.float32,
        )
        categories = torch.tensor([0, 0, 1, 1])

        mi = category_program_mi_bits_from_probs(probs, categories, n_categories=2)

        self.assertGreater(mi, 0.5)

    def test_pathfinder_aggregate_rejects_high_identity_share_even_with_good_loss(self):
        rows = [
            {
                "variant": "vanilla_10m_proxy",
                "seed": 1,
                "parameter_counts": {"total": 100, "identity_field": 0},
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.0, "accuracy": 0.2, "perplexity": 55.0},
                "train": {"tokens_per_second": 100.0},
                "route_eval": {"selected_mi_bits": 0.0, "activation_mi_bits": 0.0},
            },
            {
                "variant": "tac_semantic_w0p2_p32",
                "seed": 1,
                "parameter_counts": {"total": 100, "identity_field": 70},
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 3.9, "accuracy": 0.21, "perplexity": 50.0},
                "train": {"tokens_per_second": 70.0},
                "route_eval": {"selected_mi_bits": 0.1, "activation_mi_bits": 0.2},
            },
            {
                "variant": "tac_semantic_w0p05_p12",
                "seed": 1,
                "parameter_counts": {"total": 100, "identity_field": 45},
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.1, "accuracy": 0.2, "perplexity": 60.0},
                "train": {"tokens_per_second": 80.0},
                "route_eval": {"selected_mi_bits": 0.08, "activation_mi_bits": 0.15},
            },
        ]

        result = aggregate_run5_pathfinder_results(rows)

        self.assertEqual(result["recommendation"]["variant"], "tac_semantic_w0p05_p12")
        self.assertIn("identity share", result["rejected"]["tac_semantic_w0p2_p32"][0])

    def test_evolutionary_search_selects_candidate_with_capability_and_health(self):
        rows = [
            {
                "variant": "vanilla_same_backbone",
                "model_type": "vanilla",
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.20, "accuracy": 0.18},
                "train": {"tokens_per_second": 300.0},
                "parameter_counts": {"total": 100, "identity_field": 0},
            },
            {
                "variant": "tac_big_identity",
                "model_type": "tac",
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 3.90, "accuracy": 0.23},
                "train": {"tokens_per_second": 120.0},
                "parameter_counts": {"total": 100, "identity_field": 62},
                "program_memory_cosine": 0.30,
                "dead_program_fraction": 0.05,
                "routed_is_best_fraction": 0.35,
            },
            {
                "variant": "tac_memory_collapsed",
                "model_type": "tac",
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.00, "accuracy": 0.22},
                "train": {"tokens_per_second": 160.0},
                "parameter_counts": {"total": 100, "identity_field": 40},
                "program_memory_cosine": 0.92,
                "dead_program_fraction": 0.0,
                "routed_is_best_fraction": 0.40,
            },
            {
                "variant": "program_conditioned_creb_k6_task_memsep",
                "model_type": "tac",
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.08, "accuracy": 0.21},
                "train": {"tokens_per_second": 180.0},
                "parameter_counts": {"total": 100, "identity_field": 42},
                "route_eval": {"selected_mi_bits": 0.04, "activation_mi_bits": 0.03},
                "program_memory_cosine": 0.21,
                "dead_program_fraction": 0.08,
                "routed_is_best_fraction": 0.34,
            },
        ]

        result = aggregate_evolutionary_search_results(rows)

        self.assertEqual(
            result["recommendation"]["variant"],
            "program_conditioned_creb_k6_task_memsep",
        )
        self.assertEqual(result["recommendation"]["decision"], "promote_for_longer_validation")
        self.assertIn("identity share", result["rejected"]["tac_big_identity"][0])
        self.assertIn("program memory cosine", result["rejected"]["tac_memory_collapsed"][0])
        self.assertIn("longer validation", result["next_actions"][0])

        markdown = format_evolutionary_search_markdown(result)
        self.assertIn("Evolutionary TAC Search", markdown)
        self.assertIn("program_conditioned_creb_k6_task_memsep", markdown)

    def test_evolutionary_search_merges_rows_by_evidence_alias(self):
        rows = [
            {
                "variant": "vanilla_same_backbone",
                "model_type": "vanilla",
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.20, "accuracy": 0.18},
                "train": {"tokens_per_second": 300.0},
                "parameter_counts": {"total": 100, "identity_field": 0},
            },
            {
                "variant": "tac_program_conditioned_creb_k6_w0p1_p12",
                "evidence_alias": "program_conditioned_creb_k6_task_memsep",
                "model_type": "tac",
                "initial_eval": {"loss": 6.0},
                "final_eval": {"loss": 4.15, "accuracy": 0.22},
                "train": {"tokens_per_second": 170.0},
                "parameter_counts": {"total": 100, "identity_field": 42},
            },
            {
                "variant": "program_conditioned_creb_k6_task_memsep",
                "eval_write_stats": {
                    "program_memory_cosine": 0.21,
                    "dead_program_fraction": 0.08,
                },
                "counterfactual_reconstruction": {
                    "routed_is_best_fraction": 0.34,
                },
            },
        ]

        result = aggregate_evolutionary_search_results(rows)
        row = result["aggregate"]["program_conditioned_creb_k6_task_memsep"]

        self.assertEqual(row["variant"], "program_conditioned_creb_k6_task_memsep")
        self.assertAlmostEqual(row["mean_final_loss"], 4.15)
        self.assertAlmostEqual(row["program_memory_cosine"], 0.21)
        self.assertAlmostEqual(row["dead_program_fraction"], 0.08)
        self.assertAlmostEqual(row["routed_is_best_fraction"], 0.34)
        self.assertEqual(result["recommendation"]["variant"], "program_conditioned_creb_k6_task_memsep")

    def test_external_run5b_validation_promotes_capable_memory_healthy_tac(self):
        same_backbone = _external_vanilla_summary(best_loss=0.109, accuracy=0.953)
        parameter_matched = _external_vanilla_summary(best_loss=0.105, accuracy=0.955)
        tac = _external_tac_summary(
            best_loss=0.18,
            program_memory_cosine=0.22,
            specialization_mi=0.08,
            knockout_delta=0.004,
        )

        result = aggregate_external_run5b_validation(
            tac,
            same_backbone,
            parameter_matched,
            tac_manifest={
                "precision": "fp32",
                "config": {
                    "program_memory_update_type": "program_conditioned",
                    "memory_allocation_type": "creb",
                    "memory_allocation_k": 6,
                },
            },
        )

        self.assertEqual(result["decision"]["status"], "promote")
        self.assertAlmostEqual(result["tac"]["same_backbone_loss_gap"], 0.071)
        self.assertEqual(result["tac"]["program_memory_update_type"], "program_conditioned")

        markdown = format_external_run5b_validation_markdown(result)
        self.assertIn("External Run 5B TAC Validation", markdown)
        self.assertIn("Decision: `promote`", markdown)

    def test_external_run5b_validation_uses_standalone_specialization_report(self):
        same_backbone = _external_vanilla_summary(best_loss=0.109, accuracy=0.953)
        parameter_matched = _external_vanilla_summary(best_loss=0.105, accuracy=0.955)
        tac = _external_tac_summary(
            best_loss=0.18,
            program_memory_cosine=0.22,
            specialization_mi=0.0,
            knockout_delta=0.0,
        )
        specialization_report = {
            "checkpoint_step": 6000,
            "records": [{"category": "tool_choice"} for _ in range(3)],
            "mutual_information": {
                "mi_bits": 0.04,
                "normalized_mi": 0.2,
            },
            "ablations": [
                {"program": 0, "loss_delta": 0.002},
                {"program": 1, "loss_delta": -0.001},
            ],
            "specialization_metrics": {
                "knockout_selectivity": [
                    {"program": 0, "selectivity_span": 0.03},
                ],
            },
        }

        result = aggregate_external_run5b_validation(
            tac,
            same_backbone,
            parameter_matched,
            specialization_report=specialization_report,
        )

        self.assertEqual(result["decision"]["status"], "promote")
        self.assertEqual(result["specialization"]["source"], "standalone_report")
        self.assertEqual(result["specialization"]["label"], "checkpoint_step_6000")
        self.assertEqual(result["specialization"]["records"], 3)
        self.assertTrue(result["specialization"]["run_knockouts"])
        self.assertAlmostEqual(result["specialization"]["max_knockout_loss_delta"], 0.002)
        self.assertAlmostEqual(
            result["specialization"]["max_knockout_selectivity_span"],
            0.03,
        )

    def test_external_run5b_validation_rejects_capability_gap(self):
        same_backbone = _external_vanilla_summary(best_loss=0.109, accuracy=0.953)
        parameter_matched = _external_vanilla_summary(best_loss=0.105, accuracy=0.955)
        tac = _external_tac_summary(
            best_loss=0.90,
            program_memory_cosine=0.20,
            specialization_mi=0.12,
            knockout_delta=0.005,
        )

        result = aggregate_external_run5b_validation(tac, same_backbone, parameter_matched)

        self.assertEqual(result["decision"]["status"], "reject")
        self.assertIn("same-backbone vanilla", result["decision"]["reason"])


def _external_vanilla_summary(*, best_loss: float, accuracy: float) -> dict:
    return {
        "completed_steps": 20000,
        "target_steps": 20000,
        "stopped_for_time": False,
        "best_eval_loss": best_loss,
        "latest_metrics": {
            "eval": {
                "loss": best_loss + 0.02,
                "accuracy": accuracy,
            }
        },
    }


def _external_tac_summary(
    *,
    best_loss: float,
    program_memory_cosine: float,
    specialization_mi: float,
    knockout_delta: float,
) -> dict:
    return {
        "completed_steps": 20000,
        "target_steps": 20000,
        "stopped_for_time": False,
        "best_eval_loss": best_loss,
        "latest_metrics": {
            "next_token_loss": best_loss + 0.03,
            "program_memory_cosine": program_memory_cosine,
            "gradient_norm": 0.1,
            "grad_scaler_scale": 1.0,
            "optimization_health": {"status": "passed"},
        },
        "specialization_analysis": {
            "enabled": True,
            "label": "end",
            "records": 384,
            "run_knockouts": True,
            "mi_bits": specialization_mi,
            "normalized_mi": 0.1,
            "top_ablation_loss_deltas": [
                {"program": 3, "loss_delta": knockout_delta}
            ],
        },
    }


if __name__ == "__main__":
    unittest.main()
