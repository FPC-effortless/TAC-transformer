import unittest

import torch

from experiments.benchmark_content_update_frequency import (
    CONTENT_UPDATE_SCHEDULES,
    BoundaryWriteGate,
    ChainVerifierGate,
    ProtectedPathQuotaGate,
    aggregate_content_update_results,
    apply_chain_verifier_gate,
    boundary_gate_features,
    bridge_verifier_features,
    chain_verifier_features,
    calibrated_boundary_content_write_mask,
    counterfactual_second_read_is_better,
    deterministic_gate_seed,
    evaluate_content_update_schedule,
    format_content_update_markdown,
    hybrid_boundary_gate_features,
    hybrid_ranked_boundary_content_write_mask,
    is_direct_written_answer,
    mixed_path_structural_content_write_mask,
    path_aligned_content_write_mask,
    predicted_token_reaches_written_target,
    prediction_error_content_write_mask,
    protected_path_quota_content_write_mask,
    retrieval_aware_content_write_mask,
    run_query,
    run_context_with_update_schedule,
    should_update_event_error_segment,
    should_update_segment,
    sparse_graph_path_tokens,
    structural_pair_content_write_mask,
    train_boundary_write_gate,
    train_hybrid_ranked_boundary_write_gate,
    train_mixed_path_structural_hybrid_write_gate,
    train_path_aligned_hybrid_write_gate,
    train_ranked_boundary_write_gate,
)
from tac_transformer.model import TACAuxiliaryOutput
from tac_transformer.training import ChunkedRecallBatch, ChunkedRecallBatcher
from tac_transformer import TACTransformerLM, best_tac_config


class AdapterProbeModel:
    def __init__(self):
        self.adapter_calls = 0
        self.config = type("Config", (), {"memory_read_type": "content_addressed"})()

    def __call__(
        self,
        input_ids,
        *,
        labels=None,
        identity_states=None,
        collect_auxiliary=True,
        update_content_memory=True,
    ):
        del labels, identity_states, collect_auxiliary, update_content_memory
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 5)
        hidden = torch.zeros(batch, seq_len, 4)
        aux = TACAuxiliaryOutput(
            coherence=torch.zeros(batch, seq_len, seq_len),
            program_activations=torch.zeros(batch, 2),
            selected_program_mask=torch.zeros(batch, 2),
            used_energy=torch.zeros(batch),
            attention_probs=torch.zeros(batch, 1, seq_len, seq_len),
            losses={},
            metrics={},
        )
        return type(
            "Output",
            (),
            {"logits": logits, "hidden_states": hidden, "aux": aux, "loss": None},
        )()

    def memory_read_vector(self, query_tokens, states):
        del states
        return torch.zeros(query_tokens.shape[0], 4)

    def lm_head(self, memory_vector):
        logits = torch.zeros(memory_vector.shape[0], 5)
        logits[:, 3] = 2.0
        return logits

    def memory_adapted_logits(
        self,
        hidden_states,
        memory_vector,
        *,
        value_label_index,
        weight,
    ):
        del memory_vector, weight
        self.adapter_calls += 1
        logits = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 5)
        logits[:, value_label_index, 3] = 4.0
        return logits


class ChainProbeModel:
    def __init__(self):
        self.config = type("Config", (), {"memory_read_type": "content_addressed"})()
        self.read_tokens: list[int] = []

    def __call__(
        self,
        input_ids,
        *,
        labels=None,
        identity_states=None,
        collect_auxiliary=True,
        update_content_memory=True,
    ):
        del labels, identity_states, collect_auxiliary, update_content_memory
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 12)
        hidden = torch.zeros(batch, seq_len, 12)
        aux = TACAuxiliaryOutput(
            coherence=torch.zeros(batch, seq_len, seq_len),
            program_activations=torch.zeros(batch, 2),
            selected_program_mask=torch.zeros(batch, 2),
            used_energy=torch.zeros(batch),
            attention_probs=torch.zeros(batch, 1, seq_len, seq_len),
            losses={},
            metrics={},
        )
        return type(
            "Output",
            (),
            {"logits": logits, "hidden_states": hidden, "aux": aux, "loss": None},
        )()

    def memory_read_vector(self, query_tokens, states):
        del states
        self.read_tokens.extend(int(token) for token in query_tokens.detach().cpu())
        vector = torch.zeros(query_tokens.shape[0], 12)
        for row, token in enumerate(query_tokens.tolist()):
            if token == 4:
                vector[row, 7] = 1.0
            elif token == 7:
                vector[row, 9] = 1.0
        return vector

    def lm_head(self, memory_vector):
        return memory_vector


class CounterfactualChainProbeModel(ChainProbeModel):
    def memory_read_vector(self, query_tokens, states):
        del states
        self.read_tokens.extend(int(token) for token in query_tokens.detach().cpu())
        vector = torch.zeros(query_tokens.shape[0], 12)
        for row, token in enumerate(query_tokens.tolist()):
            if token == 4:
                vector[row, 7] = 1.0
            elif token == 7:
                vector[row, 9] = 4.0
            elif token == 8:
                vector[row, 9] = 4.0
        return vector


class ContentUpdateFrequencyTest(unittest.TestCase):
    def test_segment_schedule_helper_handles_periodic_and_never(self):
        self.assertTrue(should_update_segment(1, 0))
        self.assertTrue(should_update_segment(2, 0))
        self.assertFalse(should_update_segment(2, 1))
        self.assertTrue(should_update_segment(2, 2))
        self.assertFalse(should_update_segment(0, 0))

    def test_event_error_helper_seeds_then_uses_previous_loss_threshold(self):
        self.assertTrue(should_update_event_error_segment(3.0, 0, None))
        self.assertFalse(should_update_event_error_segment(3.0, 1, 2.99))
        self.assertTrue(should_update_event_error_segment(3.0, 2, 3.0))

    def test_context_schedule_controls_content_store_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        input_ids = torch.tensor([[0, 5, 6, 7, 8, 9, 10, 11]])
        labels = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])

        full = run_context_with_update_schedule(
            model,
            input_ids,
            labels,
            CONTENT_UPDATE_SCHEDULES["full_update"],
            segment_len=4,
        )
        none = run_context_with_update_schedule(
            model,
            input_ids,
            labels,
            CONTENT_UPDATE_SCHEDULES["no_content_updates"],
            segment_len=4,
        )

        self.assertGreater(float(full.states[0].content_mask.sum()), 0.0)
        self.assertEqual(float(none.states[0].content_mask.sum()), 0.0)
        self.assertEqual(full.context_update_fraction, 1.0)
        self.assertEqual(none.context_update_fraction, 0.0)

    def test_event_error_context_schedule_can_skip_low_error_segments(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        result = run_context_with_update_schedule(
            model,
            torch.tensor([[0, 5, 6, 7, 8, 9, 10, 11]]),
            torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]]),
            CONTENT_UPDATE_SCHEDULES["event_error_ge_6p0"],
            segment_len=2,
        )

        self.assertGreater(result.context_update_fraction, 0.0)
        self.assertLess(result.context_update_fraction, 1.0)

    def test_prediction_error_write_mask_uses_full_context_probe(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        mask = prediction_error_content_write_mask(
            model,
            torch.tensor([[0, 5, 6, 7]]),
            torch.tensor([[5, 6, 7, 8]]),
            threshold=100.0,
        )

        self.assertEqual(mask.shape, (1, 3))
        self.assertTrue(bool(mask[0, 0]))
        self.assertEqual(int(mask.sum()), 1)

    def test_masked_full_context_schedule_writes_sparse_pairs_without_segmenting(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        result = run_context_with_update_schedule(
            model,
            torch.tensor([[0, 5, 6, 7]]),
            torch.tensor([[5, 6, 7, 8]]),
            CONTENT_UPDATE_SCHEDULES["event_mask_ge_6p0"],
            segment_len=2,
        )

        self.assertEqual(result.output.logits.shape[:2], (1, 4))
        self.assertGreater(result.context_update_fraction, 0.0)
        self.assertLessEqual(result.context_update_fraction, 1.0)

    def test_retrieval_aware_write_mask_respects_top_fraction(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        mask = retrieval_aware_content_write_mask(
            model,
            torch.tensor([[0, 5, 6, 7, 8]]),
            torch.tensor([[5, 6, 7, 8, 9]]),
            write_fraction=0.5,
        )

        self.assertEqual(mask.shape, (1, 4))
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(bool(mask[0, 0]))

    def test_retrieval_aware_write_mask_rejects_invalid_fraction(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
            )
        )

        with self.assertRaisesRegex(ValueError, "write_fraction"):
            retrieval_aware_content_write_mask(
                model,
                torch.tensor([[0, 5, 6, 7]]),
                torch.tensor([[5, 6, 7, 8]]),
                write_fraction=0.0,
            )

    def test_structural_pair_mask_writes_early_odd_cue_positions(self):
        mask = structural_pair_content_write_mask(
            torch.tensor([[0, 5, 6, 7, 8, 9, 10, 11]]),
            max_pairs=3,
        )

        self.assertEqual(mask.shape, (1, 7))
        self.assertEqual(mask[0].nonzero().flatten().tolist(), [1, 3, 5])

    def test_structural_pair_schedule_writes_sparse_full_context_pairs(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        result = run_context_with_update_schedule(
            model,
            torch.tensor([[0, 5, 6, 7, 8, 9, 10, 11]]),
            torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]]),
            CONTENT_UPDATE_SCHEDULES["structural_pair_top_4"],
            segment_len=2,
        )

        self.assertEqual(result.output.logits.shape[:2], (1, 8))
        self.assertAlmostEqual(result.context_update_fraction, 3 / 7)

    def test_boundary_gate_features_expose_position_and_token_signals(self):
        features = boundary_gate_features(torch.tensor([[0, 5, 6, 7]]))

        self.assertEqual(features.shape, (1, 3, 6))
        self.assertEqual(features[0, :, 2].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(features[0, :, 3].tolist(), [1.0, 0.0, 1.0])

    def test_trained_boundary_gate_recovers_structural_top_pairs(self):
        torch.manual_seed(7)
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=19,
            task_variant="multi_key",
        )
        gate = train_boundary_write_gate(
            batcher,
            batches=24,
            batch_size=4,
            max_pairs=3,
            learning_rate=0.5,
            device=torch.device("cpu"),
        )
        input_ids = torch.tensor([[0, 5, 6, 7, 8, 9, 10, 11]])
        learned = calibrated_boundary_content_write_mask(input_ids, gate)
        target = structural_pair_content_write_mask(input_ids, max_pairs=3)

        self.assertTrue(torch.equal(learned, target))

    def test_ranked_boundary_gate_recovers_structural_top_pairs(self):
        torch.manual_seed(7)
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=23,
            task_variant="multi_key",
        )
        gate = train_ranked_boundary_write_gate(
            batcher,
            batches=32,
            batch_size=4,
            max_pairs=3,
            learning_rate=0.3,
            device=torch.device("cpu"),
        )
        input_ids = torch.tensor([[0, 5, 6, 7, 8, 9, 10, 11]])
        learned = calibrated_boundary_content_write_mask(input_ids, gate)
        target = structural_pair_content_write_mask(input_ids, max_pairs=3)

        self.assertTrue(torch.equal(learned, target))

    def test_hybrid_boundary_features_expose_probe_and_recurrence_signals(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
            )
        )
        features = hybrid_boundary_gate_features(
            model,
            torch.tensor([[0, 5, 6, 5, 8]]),
            torch.tensor([[5, 6, 5, 8, 9]]),
        )

        self.assertEqual(features.shape, (1, 4, 11))
        self.assertEqual(features[0, :, 9].tolist(), [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(features[0, :, 10].tolist(), [0.0, 0.0, 1.0, 0.0])

    def test_hybrid_ranked_boundary_gate_recovers_structural_top_pairs(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=29,
            task_variant="multi_key",
        )
        gate = train_hybrid_ranked_boundary_write_gate(
            model,
            batcher,
            batches=32,
            batch_size=4,
            max_pairs=3,
            learning_rate=0.3,
            device=torch.device("cpu"),
        )
        input_ids = torch.tensor([[0, 5, 6, 7, 8, 9, 10, 11]])
        labels = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])
        learned = hybrid_ranked_boundary_content_write_mask(
            model,
            input_ids,
            labels,
            gate,
        )
        target = structural_pair_content_write_mask(input_ids, max_pairs=3)

        self.assertTrue(torch.equal(learned, target))

    def test_path_aligned_mask_selects_answer_edges(self):
        mask = path_aligned_content_write_mask(
            input_ids=torch.tensor(
                [
                    [0, 4, 9, 12, 13],
                    [0, 4, 7, 7, 9],
                    [0, 5, 9, 12, 13],
                ]
            ),
            query_tokens=torch.tensor([4, 4, 6]),
            value_targets=torch.tensor([9, 9, 9]),
        )

        self.assertEqual(
            mask.tolist(),
            [
                [False, True, False, False],
                [False, True, False, True],
                [False, True, False, False],
            ],
        )

    def test_path_aligned_hybrid_gate_recovers_answer_edges(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=41,
            task_variant="multi_hop",
        )
        gate = train_path_aligned_hybrid_write_gate(
            model,
            batcher,
            batches=32,
            batch_size=8,
            max_pairs=4,
            learning_rate=0.25,
            device=torch.device("cpu"),
        )
        batch = batcher.next_batch(4, device=torch.device("cpu"))
        target = path_aligned_content_write_mask(
            batch.context_inputs,
            batch.query_inputs[:, 1],
            batch.value_targets,
        )
        learned = hybrid_ranked_boundary_content_write_mask(
            model,
            batch.context_inputs,
            batch.context_labels,
            gate,
        )

        self.assertFalse(bool(target.logical_and(learned.logical_not()).any()))
        self.assertEqual(learned.sum(dim=1).tolist(), [4, 4, 4, 4])

    def test_mixed_path_structural_mask_combines_targets(self):
        input_ids = torch.tensor([[0, 4, 7, 7, 9, 12, 13, 14]])
        structural = structural_pair_content_write_mask(input_ids, max_pairs=4)
        path = path_aligned_content_write_mask(
            input_ids,
            query_tokens=torch.tensor([4]),
            value_targets=torch.tensor([9]),
        )
        mixed = mixed_path_structural_content_write_mask(
            input_ids,
            query_tokens=torch.tensor([4]),
            value_targets=torch.tensor([9]),
            max_pairs=4,
        )

        self.assertTrue(torch.equal(mixed, structural.logical_or(path)))
        self.assertTrue(bool(mixed[0, 1].item()))
        self.assertTrue(bool(mixed[0, 3].item()))

    def test_mixed_path_structural_hybrid_gate_preserves_path_edges(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=41,
            task_variant="multi_hop",
        )
        gate = train_mixed_path_structural_hybrid_write_gate(
            model,
            batcher,
            batches=32,
            batch_size=8,
            max_pairs=4,
            learning_rate=0.25,
            device=torch.device("cpu"),
        )
        batch = batcher.next_batch(4, device=torch.device("cpu"))
        path = path_aligned_content_write_mask(
            batch.context_inputs,
            batch.query_inputs[:, 1],
            batch.value_targets,
        )
        learned = hybrid_ranked_boundary_content_write_mask(
            model,
            batch.context_inputs,
            batch.context_labels,
            gate,
        )

        self.assertFalse(bool(path.logical_and(learned.logical_not()).any()))
        self.assertEqual(learned.sum(dim=1).tolist(), [4, 4, 4, 4])

    def test_protected_path_quota_mask_reserves_path_slot(self):
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
            )
        )
        hybrid_weight = torch.zeros(11)
        hybrid_weight[2] = 10.0
        path_weight = torch.zeros(11)
        path_weight[3] = 10.0
        gate = ProtectedPathQuotaGate(
            hybrid_gate=BoundaryWriteGate(
                weight=hybrid_weight,
                bias=torch.tensor(0.0),
                max_pairs=3,
            ),
            path_gate=BoundaryWriteGate(
                weight=path_weight,
                bias=torch.tensor(0.0),
                max_pairs=3,
            ),
            max_pairs=3,
            path_quota=1,
        )

        mask = protected_path_quota_content_write_mask(
            model,
            torch.tensor([[0, 5, 6, 7, 8]]),
            torch.tensor([[5, 6, 7, 8, 9]]),
            gate,
        )

        self.assertEqual(mask.sum(dim=1).tolist(), [3])
        self.assertTrue(bool(mask[0, 1].item()))
        self.assertTrue(bool(mask[0, 3].item()))
        self.assertTrue(bool(mask[0, 0].item()) or bool(mask[0, 2].item()))

    def test_calibrated_boundary_schedule_uses_trained_gate(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=19,
            task_variant="multi_key",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="calibrated_boundary_top_4",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_key",
            seed=19,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "calibrated_boundary_full_context")
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_ranked_boundary_schedule_uses_trained_gate(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=23,
            task_variant="multi_key",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="ranked_boundary_top_4",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_key",
            seed=23,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "ranked_boundary_full_context")
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_ranked_boundary_schedule_is_independent_of_prior_gate_training(self):
        def build_model():
            torch.manual_seed(7)
            return TACTransformerLM(
                best_tac_config(
                    vocab_size=32,
                    d_model=16,
                    n_heads=4,
                    n_layers=1,
                    n_programs=4,
                    max_seq_len=8,
                    content_store_size=4,
                    content_read_steps=1,
                )
            )

        def build_batcher():
            return ChunkedRecallBatcher(
                vocab_size=32,
                seq_len=8,
                seed=23,
                task_variant="multi_key",
            )

        alone = evaluate_content_update_schedule(
            build_model(),
            schedule_name="ranked_boundary_top_4",
            batcher=build_batcher(),
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_key",
            seed=23,
            device=torch.device("cpu"),
        )
        model_after_other_gate = build_model()
        evaluate_content_update_schedule(
            model_after_other_gate,
            schedule_name="ranked_boundary_top_6",
            batcher=build_batcher(),
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_key",
            seed=23,
            device=torch.device("cpu"),
        )
        after = evaluate_content_update_schedule(
            model_after_other_gate,
            schedule_name="ranked_boundary_top_4",
            batcher=build_batcher(),
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_key",
            seed=23,
            device=torch.device("cpu"),
        )

        self.assertEqual(alone["carry"]["value_accuracy"], after["carry"]["value_accuracy"])
        self.assertEqual(alone["reset"]["value_accuracy"], after["reset"]["value_accuracy"])
        self.assertEqual(alone["shuffled"]["value_accuracy"], after["shuffled"]["value_accuracy"])

    def test_hybrid_chain_policy_variants_share_write_gate_seed_group(self):
        seed = 31

        self.assertEqual(
            deterministic_gate_seed(seed, "hybrid_ranked_boundary_top_4"),
            deterministic_gate_seed(seed, "hybrid_ranked_boundary_top_4_counterfactual_chain_k2"),
        )
        self.assertEqual(
            deterministic_gate_seed(seed, "hybrid_ranked_boundary_top_4"),
            deterministic_gate_seed(seed, "hybrid_ranked_boundary_top_4_oracle_chain_k2"),
        )

    def test_gate_training_does_not_advance_evaluation_batcher_stream(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        schedule_name = "hybrid_ranked_boundary_top_4_bridge_veto_chain_k2"
        schedule = CONTENT_UPDATE_SCHEDULES[schedule_name]
        original = dict(schedule)
        schedule["gate_train_batches"] = 1
        schedule["chain_gate_train_batches"] = 1
        try:
            batcher = ChunkedRecallBatcher(
                vocab_size=32,
                seq_len=8,
                seed=31,
                task_variant="multi_hop",
            )
            control = ChunkedRecallBatcher(
                vocab_size=32,
                seq_len=8,
                seed=31,
                task_variant="multi_hop",
            )

            evaluate_content_update_schedule(
                model,
                schedule_name=schedule_name,
                batcher=batcher,
                batches=1,
                batch_size=2,
                segment_len=2,
                task="multi_hop",
                seed=31,
                device=torch.device("cpu"),
            )

            control.next_batch(2)
            observed = batcher.next_batch(2)
            expected = control.next_batch(2)
        finally:
            schedule.clear()
            schedule.update(original)

        self.assertTrue(torch.equal(observed.context_inputs, expected.context_inputs))
        self.assertTrue(torch.equal(observed.query_inputs, expected.query_inputs))
        self.assertTrue(torch.equal(observed.value_targets, expected.value_targets))

    def test_hybrid_ranked_boundary_schedule_uses_trained_gate(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=29,
            task_variant="multi_key",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_key",
            seed=29,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_query_evaluation_uses_trained_memory_adapter_readout(self):
        batch = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 4, 3]]),
            context_labels=torch.tensor([[4, 3, 2]]),
            query_inputs=torch.tensor([[1, 4, 3]]),
            query_labels=torch.tensor([[2, 2, 3]]),
            value_targets=torch.tensor([3]),
            value_label_index=2,
        )
        model = AdapterProbeModel()

        row = run_query(
            model,
            batch,
            states=[],
            update_content_memory=False,
            memory_adapter_weight=6.0,
        )

        self.assertEqual(model.adapter_calls, 1)
        self.assertEqual(row["value_accuracy"], 1.0)
        self.assertEqual(row["memory_read_accuracy"], 1.0)

    def test_query_evaluation_can_chain_memory_reads_for_multi_hop(self):
        batch = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 4, 7, 7, 9]]),
            context_labels=torch.tensor([[4, 7, 7, 9, 2]]),
            query_inputs=torch.tensor([[1, 4, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )
        model = ChainProbeModel()

        one_step = run_query(
            model,
            batch,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=1,
        )
        two_step = run_query(
            model,
            batch,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
        )

        self.assertEqual(one_step["memory_read_accuracy"], 0.0)
        self.assertEqual(two_step["memory_read_accuracy"], 1.0)
        self.assertEqual(two_step["value_accuracy"], 1.0)
        self.assertEqual(model.read_tokens, [4, 4, 7])

    def test_query_evaluation_can_gate_chain_on_predicted_written_cue(self):
        model = ChainProbeModel()
        direct = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 7, 9]]),
            context_labels=torch.tensor([[7, 9, 2]]),
            query_inputs=torch.tensor([[1, 7, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )
        multi_hop = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 4, 7, 7, 9]]),
            context_labels=torch.tensor([[4, 7, 7, 9, 2]]),
            query_inputs=torch.tensor([[1, 4, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )

        direct_row = run_query(
            model,
            direct,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="predicted_written_cue",
            context_inputs=direct.context_inputs,
            context_write_mask=torch.tensor([[False, True]]),
        )
        multi_hop_row = run_query(
            model,
            multi_hop,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="predicted_written_cue",
            context_inputs=multi_hop.context_inputs,
            context_write_mask=torch.tensor([[False, True, False, True]]),
        )

        self.assertEqual(direct_row["value_accuracy"], 1.0)
        self.assertEqual(direct_row["memory_chain_fraction"], 0.0)
        self.assertEqual(multi_hop_row["value_accuracy"], 1.0)
        self.assertEqual(multi_hop_row["memory_chain_fraction"], 1.0)
        self.assertEqual(model.read_tokens, [7, 4, 7])

    def test_predicted_token_reaches_written_target_identifies_bridge_paths(self):
        context_inputs = torch.tensor(
            [
                [0, 4, 7, 7, 9],
                [0, 7, 9, 4, 8],
                [0, 4, 7, 7, 9],
            ]
        )
        context_write_mask = torch.tensor(
            [
                [False, True, False, True],
                [False, True, False, True],
                [False, True, False, False],
            ]
        )

        reaches_target = predicted_token_reaches_written_target(
            torch.tensor([7, 9, 7]),
            torch.tensor([9, 9, 9]),
            context_inputs,
            context_write_mask,
        )

        self.assertEqual(reaches_target.tolist(), [True, False, False])

    def test_counterfactual_second_read_requires_confidence_gain(self):
        first_logits = torch.zeros(2, 12)
        second_logits = torch.zeros(2, 12)
        first_logits[0, 7] = 1.0
        second_logits[0, 9] = 4.0
        first_logits[1, 9] = 4.0
        second_logits[1, 2] = 1.0

        accept = counterfactual_second_read_is_better(
            first_logits,
            second_logits,
            min_confidence_gain=0.05,
            min_margin_gain=0.0,
        )

        self.assertEqual(accept.tolist(), [True, False])

    def test_chain_verifier_features_expose_label_free_halt_signals(self):
        first_logits = torch.zeros(2, 12)
        first_logits[0, 9] = 4.0
        first_logits[1, 7] = 4.0
        features = chain_verifier_features(
            first_logits,
            torch.tensor([9, 7]),
            torch.tensor([7, 4]),
            torch.tensor([[0, 7, 9, 0, 0], [0, 4, 7, 7, 9]]),
            torch.tensor([[False, True, False, False], [False, True, False, True]]),
        )

        self.assertEqual(features.shape, (2, 10))
        self.assertEqual(features[:, 3].tolist(), [0.0, 1.0])
        self.assertGreater(features[1, 4], features[0, 4])

    def test_chain_verifier_gate_can_select_bridge_like_first_predictions(self):
        first_logits = torch.zeros(2, 12)
        first_logits[0, 9] = 4.0
        first_logits[1, 7] = 4.0
        features = chain_verifier_features(
            first_logits,
            torch.tensor([9, 7]),
            torch.tensor([7, 4]),
            torch.tensor([[0, 7, 9, 0, 0], [0, 4, 7, 7, 9]]),
            torch.tensor([[False, True, False, False], [False, True, False, True]]),
        )
        weight = torch.zeros(features.shape[-1])
        weight[3] = 8.0
        gate = ChainVerifierGate(weight=weight, bias=torch.tensor(-4.0))

        self.assertEqual(apply_chain_verifier_gate(features, gate).tolist(), [False, True])

    def test_query_evaluation_can_oracle_gate_chain_on_written_target_path(self):
        model = ChainProbeModel()
        direct = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 7, 9]]),
            context_labels=torch.tensor([[7, 9, 2]]),
            query_inputs=torch.tensor([[1, 7, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )
        multi_hop = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 4, 7, 7, 9]]),
            context_labels=torch.tensor([[4, 7, 7, 9, 2]]),
            query_inputs=torch.tensor([[1, 4, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )

        direct_row = run_query(
            model,
            direct,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="oracle_written_target",
            context_inputs=direct.context_inputs,
            context_write_mask=torch.tensor([[False, True]]),
        )
        multi_hop_row = run_query(
            model,
            multi_hop,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="oracle_written_target",
            context_inputs=multi_hop.context_inputs,
            context_write_mask=torch.tensor([[False, True, False, True]]),
        )

        self.assertEqual(direct_row["value_accuracy"], 1.0)
        self.assertEqual(direct_row["memory_chain_fraction"], 0.0)
        self.assertEqual(multi_hop_row["value_accuracy"], 1.0)
        self.assertEqual(multi_hop_row["memory_chain_fraction"], 1.0)
        self.assertEqual(model.read_tokens, [7, 4, 7])

    def test_query_evaluation_can_use_learned_chain_verifier_gate(self):
        model = ChainProbeModel()
        weight = torch.zeros(10)
        weight[3] = 8.0
        gate = ChainVerifierGate(weight=weight, bias=torch.tensor(-4.0))
        direct = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 7, 9]]),
            context_labels=torch.tensor([[7, 9, 2]]),
            query_inputs=torch.tensor([[1, 7, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )
        multi_hop = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 4, 7, 7, 9]]),
            context_labels=torch.tensor([[4, 7, 7, 9, 2]]),
            query_inputs=torch.tensor([[1, 4, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )

        direct_row = run_query(
            model,
            direct,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="learned_verifier",
            context_inputs=direct.context_inputs,
            context_write_mask=torch.tensor([[False, True]]),
            chain_verifier_gate=gate,
        )
        multi_hop_row = run_query(
            model,
            multi_hop,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="learned_verifier",
            context_inputs=multi_hop.context_inputs,
            context_write_mask=torch.tensor([[False, True, False, True]]),
            chain_verifier_gate=gate,
        )

        self.assertEqual(direct_row["memory_chain_fraction"], 0.0)
        self.assertEqual(multi_hop_row["memory_chain_fraction"], 1.0)
        self.assertEqual(model.read_tokens, [7, 4, 7])

    def test_query_evaluation_can_validate_counterfactual_second_read(self):
        model = CounterfactualChainProbeModel()
        direct = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 7, 9]]),
            context_labels=torch.tensor([[7, 9, 2]]),
            query_inputs=torch.tensor([[1, 7, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )
        multi_hop = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 4, 7, 7, 9]]),
            context_labels=torch.tensor([[4, 7, 7, 9, 2]]),
            query_inputs=torch.tensor([[1, 4, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )

        direct_row = run_query(
            model,
            direct,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="counterfactual_confidence",
        )
        multi_hop_row = run_query(
            model,
            multi_hop,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="counterfactual_confidence",
        )

        self.assertEqual(direct_row["value_accuracy"], 1.0)
        self.assertEqual(direct_row["memory_chain_fraction"], 0.0)
        self.assertEqual(multi_hop_row["value_accuracy"], 1.0)
        self.assertEqual(multi_hop_row["memory_chain_fraction"], 1.0)
        self.assertEqual(model.read_tokens, [7, 9, 4, 7])

    def test_bridge_verifier_features_include_second_read_evidence(self):
        first_logits = torch.zeros(1, 12)
        first_logits[0, 7] = 2.0
        second_logits = torch.zeros(1, 12)
        second_logits[0, 9] = 3.0
        features = bridge_verifier_features(
            first_logits,
            second_logits,
            predicted_tokens=torch.tensor([7]),
            second_predictions=torch.tensor([9]),
            query_tokens=torch.tensor([4]),
            context_inputs=torch.tensor([[0, 4, 7, 7, 9]]),
            context_write_mask=torch.tensor([[False, True, False, True]]),
        )

        self.assertEqual(features.shape, (1, 16))
        self.assertEqual(features[0, 3].item(), 1.0)
        self.assertEqual(features[0, 13].item(), 1.0)
        self.assertEqual(features[0, 14].item(), 1.0)

    def test_query_evaluation_can_use_bridge_verifier_gate(self):
        model = ChainProbeModel()
        weight = torch.zeros(16)
        weight[3] = 8.0
        weight[14] = 4.0
        gate = ChainVerifierGate(weight=weight, bias=torch.tensor(-6.0))
        direct = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 7, 9]]),
            context_labels=torch.tensor([[7, 9, 2]]),
            query_inputs=torch.tensor([[1, 7, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )
        multi_hop = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 4, 7, 7, 9]]),
            context_labels=torch.tensor([[4, 7, 7, 9, 2]]),
            query_inputs=torch.tensor([[1, 4, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )

        direct_row = run_query(
            model,
            direct,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="bridge_verifier",
            context_inputs=direct.context_inputs,
            context_write_mask=torch.tensor([[False, True]]),
            chain_verifier_gate=gate,
        )
        multi_hop_row = run_query(
            model,
            multi_hop,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="bridge_verifier",
            context_inputs=multi_hop.context_inputs,
            context_write_mask=torch.tensor([[False, True, False, True]]),
            chain_verifier_gate=gate,
        )

        self.assertEqual(direct_row["memory_chain_fraction"], 0.0)
        self.assertEqual(multi_hop_row["memory_chain_fraction"], 1.0)
        self.assertEqual(model.read_tokens, [7, 9, 4, 7])

    def test_direct_written_answer_veto_preserves_bridge_cues(self):
        direct_veto = is_direct_written_answer(
            predicted_tokens=torch.tensor([9]),
            query_tokens=torch.tensor([7]),
            context_inputs=torch.tensor([[0, 7, 9]]),
            context_write_mask=torch.tensor([[False, True]]),
        )
        bridge_not_vetoed = is_direct_written_answer(
            predicted_tokens=torch.tensor([7]),
            query_tokens=torch.tensor([4]),
            context_inputs=torch.tensor([[0, 4, 7, 7, 9]]),
            context_write_mask=torch.tensor([[False, True, False, True]]),
        )

        self.assertTrue(bool(direct_veto.item()))
        self.assertFalse(bool(bridge_not_vetoed.item()))

    def test_query_evaluation_can_use_bridge_verifier_with_direct_answer_veto(self):
        model = ChainProbeModel()
        weight = torch.zeros(16)
        weight[3] = 8.0
        weight[14] = 4.0
        gate = ChainVerifierGate(weight=weight, bias=torch.tensor(-6.0))
        direct = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 7, 9]]),
            context_labels=torch.tensor([[7, 9, 2]]),
            query_inputs=torch.tensor([[1, 7, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )
        multi_hop = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 4, 7, 7, 9]]),
            context_labels=torch.tensor([[4, 7, 7, 9, 2]]),
            query_inputs=torch.tensor([[1, 4, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )

        direct_row = run_query(
            model,
            direct,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="bridge_verifier_direct_veto",
            context_inputs=direct.context_inputs,
            context_write_mask=torch.tensor([[False, True]]),
            chain_verifier_gate=gate,
        )
        multi_hop_row = run_query(
            model,
            multi_hop,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="bridge_verifier_direct_veto",
            context_inputs=multi_hop.context_inputs,
            context_write_mask=torch.tensor([[False, True, False, True]]),
            chain_verifier_gate=gate,
        )

        self.assertEqual(direct_row["memory_chain_fraction"], 0.0)
        self.assertEqual(multi_hop_row["memory_chain_fraction"], 1.0)
        self.assertEqual(model.read_tokens, [7, 9, 4, 7])

    def test_sparse_graph_path_tokens_follow_latest_written_edges(self):
        tokens, hits, chains = sparse_graph_path_tokens(
            query_tokens=torch.tensor([4, 8]),
            context_inputs=torch.tensor(
                [
                    [0, 4, 6, 4, 7, 7, 9],
                    [0, 8, 3, 3, 5, 8, 6],
                ]
            ),
            context_write_mask=torch.tensor(
                [
                    [False, True, False, True, False, True],
                    [False, True, True, True, False, True],
                ]
            ),
            steps=2,
        )

        self.assertEqual(tokens.tolist(), [9, 6])
        self.assertEqual(hits.tolist(), [True, True])
        self.assertEqual(chains.tolist(), [True, False])

    def test_query_evaluation_can_use_sparse_graph_path_readout(self):
        model = ChainProbeModel()
        direct = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 7, 9]]),
            context_labels=torch.tensor([[7, 9, 2]]),
            query_inputs=torch.tensor([[1, 7, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )
        multi_hop = ChunkedRecallBatch(
            context_inputs=torch.tensor([[0, 4, 7, 7, 9]]),
            context_labels=torch.tensor([[4, 7, 7, 9, 2]]),
            query_inputs=torch.tensor([[1, 4, 3]]),
            query_labels=torch.tensor([[2, 2, 9]]),
            value_targets=torch.tensor([9]),
            value_label_index=2,
        )

        direct_row = run_query(
            model,
            direct,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="sparse_graph_path",
            context_inputs=direct.context_inputs,
            context_write_mask=torch.tensor([[False, True]]),
        )
        multi_hop_row = run_query(
            model,
            multi_hop,
            states=[],
            update_content_memory=False,
            memory_injection_weight=10.0,
            memory_chain_steps=2,
            memory_chain_policy="sparse_graph_path",
            context_inputs=multi_hop.context_inputs,
            context_write_mask=torch.tensor([[False, True, False, True]]),
        )

        self.assertEqual(direct_row["value_accuracy"], 1.0)
        self.assertEqual(direct_row["memory_chain_fraction"], 0.0)
        self.assertEqual(multi_hop_row["value_accuracy"], 1.0)
        self.assertEqual(multi_hop_row["memory_chain_fraction"], 1.0)
        self.assertEqual(model.read_tokens, [7, 4])

    def test_hybrid_ranked_boundary_chain_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4_chain_k2",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertEqual(row["memory_chain_steps"], 2)
        self.assertIn("memory_chain_fraction", row["carry"])
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_hybrid_ranked_boundary_conditional_chain_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4_cond_chain_k2",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertEqual(row["memory_chain_steps"], 2)
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_hybrid_ranked_boundary_oracle_chain_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4_oracle_chain_k2",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertEqual(row["memory_chain_steps"], 2)
        self.assertIn("memory_chain_fraction", row["carry"])
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_hybrid_ranked_boundary_learned_chain_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4_learned_chain_k2",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertEqual(row["memory_chain_steps"], 2)
        self.assertIn("memory_chain_fraction", row["carry"])
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_hybrid_ranked_boundary_thresholded_learned_chain_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4_learned_chain_t2_k2",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertEqual(row["memory_chain_steps"], 2)
        self.assertIn("memory_chain_fraction", row["carry"])
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_hybrid_ranked_boundary_counterfactual_chain_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4_counterfactual_chain_k2",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertEqual(row["memory_chain_steps"], 2)
        self.assertIn("memory_chain_fraction", row["carry"])
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_hybrid_ranked_boundary_bridge_verifier_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4_bridge_chain_k2",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertEqual(row["memory_chain_steps"], 2)
        self.assertIn("memory_chain_fraction", row["carry"])
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_hybrid_ranked_boundary_precision_bridge_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4_bridge_precision_chain_k2",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        schedule = CONTENT_UPDATE_SCHEDULES[
            "hybrid_ranked_boundary_top_4_bridge_precision_chain_k2"
        ]
        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertEqual(row["memory_chain_steps"], 2)
        self.assertIn("memory_chain_fraction", row["carry"])
        self.assertGreater(schedule["chain_gate_negative_weight"], 1.0)
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_hybrid_ranked_boundary_bridge_veto_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4_bridge_veto_chain_k2",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        schedule = CONTENT_UPDATE_SCHEDULES[
            "hybrid_ranked_boundary_top_4_bridge_veto_chain_k2"
        ]
        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertEqual(row["memory_chain_steps"], 2)
        self.assertIn("memory_chain_fraction", row["carry"])
        self.assertEqual(schedule["memory_chain_policy"], "bridge_verifier_direct_veto")
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_hybrid_ranked_boundary_graph_path_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="hybrid_ranked_boundary_top_4_graph_path_k2",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        schedule = CONTENT_UPDATE_SCHEDULES["hybrid_ranked_boundary_top_4_graph_path_k2"]
        self.assertEqual(row["phase"], "hybrid_ranked_boundary_full_context")
        self.assertEqual(row["memory_chain_steps"], 2)
        self.assertEqual(schedule["memory_chain_policy"], "sparse_graph_path")
        self.assertIn("memory_chain_fraction", row["carry"])
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_path_aligned_hybrid_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="path_aligned_hybrid_top_4",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "path_aligned_hybrid_full_context")
        self.assertEqual(row["memory_chain_steps"], 1)
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_mixed_path_structural_hybrid_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="mixed_path_structural_hybrid_top_4",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "mixed_path_structural_hybrid_full_context")
        self.assertEqual(row["memory_chain_steps"], 1)
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_protected_path_quota_hybrid_schedule_uses_sparse_writes(self):
        torch.manual_seed(7)
        model = TACTransformerLM(
            best_tac_config(
                vocab_size=32,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=8,
                content_store_size=4,
                content_read_steps=1,
            )
        )
        batcher = ChunkedRecallBatcher(
            vocab_size=32,
            seq_len=8,
            seed=31,
            task_variant="multi_hop",
        )
        row = evaluate_content_update_schedule(
            model,
            schedule_name="protected_path_quota_hybrid_top_4_q1",
            batcher=batcher,
            batches=1,
            batch_size=2,
            segment_len=2,
            task="multi_hop",
            seed=31,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["phase"], "protected_path_quota_hybrid_full_context")
        self.assertEqual(row["memory_chain_steps"], 1)
        self.assertAlmostEqual(row["context_update_fraction"], 4 / 7)

    def test_aggregate_ranks_query_skip_when_it_preserves_carry_and_improves_speed(self):
        rows = [
            {
                "task": "single_key",
                "seed": 11,
                "schedule": "full_update",
                "phase": "full_window",
                "carry": {"value_accuracy": 0.5, "tokens_per_second": 100.0},
                "reset": {"value_accuracy": 0.1},
                "shuffled": {"value_accuracy": 0.2},
                "context_loss": 2.0,
                "context_update_fraction": 1.0,
                "query_update_fraction": 1.0,
                "carry_reset_delta": 0.4,
                "carry_shuffled_delta": 0.3,
            },
            {
                "task": "single_key",
                "seed": 11,
                "schedule": "query_skip",
                "phase": "full_window",
                "carry": {"value_accuracy": 0.5, "tokens_per_second": 130.0},
                "reset": {"value_accuracy": 0.1},
                "shuffled": {"value_accuracy": 0.2},
                "context_loss": 2.0,
                "context_update_fraction": 1.0,
                "query_update_fraction": 0.0,
                "carry_reset_delta": 0.4,
                "carry_shuffled_delta": 0.3,
            },
            {
                "task": "single_key",
                "seed": 11,
                "schedule": "event_error_ge_4p0",
                "phase": "event_error_context",
                "carry": {"value_accuracy": 0.5, "tokens_per_second": 125.0, "content_addressed_hit": 0.3},
                "reset": {"value_accuracy": 0.1},
                "shuffled": {"value_accuracy": 0.2},
                "context_loss": 2.01,
                "context_update_fraction": 0.5,
                "query_update_fraction": 0.0,
                "carry_reset_delta": 0.4,
                "carry_shuffled_delta": 0.3,
            },
        ]

        aggregate = aggregate_content_update_results(rows)
        markdown = format_content_update_markdown(aggregate)

        self.assertEqual(aggregate["recommendation"]["schedule"], "query_skip")
        self.assertEqual(aggregate["event_error_decision"]["status"], "passed")
        self.assertIn("query_skip", markdown)
        self.assertIn("Sparse Context-Write Gate", markdown)
        self.assertIn("content_update_frequency.v1", aggregate["schema"])

    def test_aggregate_reports_memory_chain_fraction(self):
        rows = [
            {
                "task": "multi_hop",
                "seed": 11,
                "schedule": "full_update",
                "phase": "full_window",
                "carry": {
                    "value_accuracy": 0.5,
                    "tokens_per_second": 100.0,
                    "memory_chain_fraction": 0.0,
                },
                "reset": {"value_accuracy": 0.1},
                "shuffled": {"value_accuracy": 0.2},
                "context_loss": 2.0,
                "context_update_fraction": 1.0,
                "query_update_fraction": 1.0,
                "carry_reset_delta": 0.4,
                "carry_shuffled_delta": 0.3,
            },
            {
                "task": "multi_hop",
                "seed": 11,
                "schedule": "hybrid_ranked_boundary_top_4_bridge_veto_chain_k2",
                "phase": "hybrid_ranked_boundary_full_context",
                "carry": {
                    "value_accuracy": 0.6,
                    "tokens_per_second": 90.0,
                    "memory_chain_fraction": 0.25,
                },
                "reset": {"value_accuracy": 0.1},
                "shuffled": {"value_accuracy": 0.2},
                "context_loss": 2.0,
                "context_update_fraction": 0.25,
                "query_update_fraction": 0.0,
                "carry_reset_delta": 0.5,
                "carry_shuffled_delta": 0.4,
            },
        ]

        aggregate = aggregate_content_update_results(rows)

        self.assertEqual(
            aggregate["by_schedule"][
                "hybrid_ranked_boundary_top_4_bridge_veto_chain_k2"
            ]["mean_memory_chain_fraction"],
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
